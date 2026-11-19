from agent_eval.analysis.signals import leak_check
from agent_eval.tools.web import is_blocked

QUESTION = ("I recently read a research paper but I can't remember its title. The title of this research paper ends "
            "with the word Intervention and the paper was written by four authors one of whom is an extension specialist")


def test_blocklist_catches_dataset_dumps():
    assert is_blocked("https://github.com/LeqsNaN/RARG/blob/main/data/bcplus_qa.jsonl")   # the pilot leak
    assert is_blocked("https://github.com/someone/repo/blob/main/data/questions.jsonl")    # unnamed dump
    assert is_blocked("https://raw.githubusercontent.com/x/y/main/test.parquet")
    assert is_blocked("https://huggingface.co/datasets/Tevatron/browsecomp-plus")
    assert not is_blocked("https://github.com/someone/repo")                                 # ordinary repo page
    assert not is_blocked("https://en.wikipedia.org/wiki/Aziz_Maraka")
    assert not is_blocked("https://bvb.ro/info/Raportari/AST/report.pdf")


def _work(output: str) -> dict:
    return {"question": QUESTION, "trajectory": {"steps": [
        {"step": 0, "tool_calls": [{"tool": "web_search", "tool_output_raw": output}]}]}}


def test_leak_check():
    assert leak_check(_work("[1] x\nURL: https://github.com/a/b/blob/main/data/bcplus_qa.jsonl\nsnippet"))[0]
    assert leak_check(_work("dump: " + QUESTION.upper() + " answer: X"))[0]          # verbatim question text
    ok, reason = leak_check(_work("Blackwell, Trzesniewski & Dweck 2007, Child Development, intervention study"))
    assert not ok and reason is None


def test_search_retries_rate_limits_and_records_status():
    import asyncio

    import httpx

    from agent_eval.tools import web
    from agent_eval.tools.base import ToolContext

    class Flaky(web.SearchBackend):
        name = "flaky"

        def __init__(self, codes):
            self.codes = list(codes)

        async def search(self, query, k):
            if self.codes:
                code = self.codes.pop(0)
                req = httpx.Request("POST", "https://api.example/search")
                raise httpx.HTTPStatusError("x", request=req, response=httpx.Response(code, request=req, headers={"retry-after": "0"}))
            return [{"title": "t", "url": "https://example.org/a", "snippet": "s"}]

    ok = asyncio.run(web.WebSearchTool(Flaky([429, 503]), 0.005).run({"query": "q"}, ToolContext("q", "t")))
    assert not ok.is_error and ok.meta["retried"] == ["429", "503"] and ok.cost_usd == 0.005
    bad = asyncio.run(web.WebSearchTool(Flaky([401]), 0.005).run({"query": "q"}, ToolContext("q", "t")))
    assert bad.is_error and bad.error_type == "SearchHTTP401" and bad.cost_usd == 0.0


def test_malformed_urls_do_not_crash():
    assert not is_blocked("http://[not-an-ipv6/page")
    assert is_blocked("http://[weird/browsecomp/dump")  # substring rules still apply
