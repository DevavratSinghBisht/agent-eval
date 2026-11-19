"""Client-side web search + fetch tools (identical for every model provider).

Search backends are pluggable (Tavily, Exa, mock). All results and fetches pass through a
leak blocklist so pages that republish BrowseComp questions/answers are never returned.
"""

from __future__ import annotations

import asyncio
import io
import os
import re
from urllib.parse import urlparse

import httpx

from ..core.types import ToolSpec
from .base import RawToolOutput, Tool, schema

# Hosts/paths known to mirror benchmark data. Extend via AGENT_EVAL_BLOCKLIST (comma-separated).
DEFAULT_BLOCKLIST = [
    "huggingface.co/datasets",
    "github.com/openai/simple-evals",
    "openaipublic.blob.core.windows.net",
    "browsecomp",
    "browse_comp",
    "browse-comp",
    "bcplus",
    "bc_plus",
    "bc-plus",
    "simple-evals",
    "simple_evals",
    "kaggle.com/datasets",
    "datasets-server.huggingface.co",
]

SEARCH_RETRIES = 3  # retries on 429/5xx/network errors (backoff 2, 4, 8 s or Retry-After)

# Raw data files on code/data hosting sites: benchmark dumps (e.g. a repo's data/bcplus_qa.jsonl)
# live under arbitrary repo names, so match by host + data-file extension rather than by name.
CODE_HOSTS = ("github.com", "githubusercontent.com", "gitlab.com", "gist.github.com", "huggingface.co",
              "bitbucket.org", "codeberg.org", "kaggle.com", "zenodo.org", "figshare.com")
DATA_FILE_RE = re.compile(r"\.(jsonl|json|parquet|arrow|csv|tsv|pkl|zip|gz|zst)(\?|#|$)", re.IGNORECASE)


def blocklist() -> list[str]:
    extra = [s.strip().lower() for s in os.environ.get("AGENT_EVAL_BLOCKLIST", "").split(",") if s.strip()]
    return DEFAULT_BLOCKLIST + extra


def is_dataset_file(url: str) -> bool:
    try:
        p = urlparse(url.lower())
    except ValueError:  # malformed URLs (e.g. stray '[' parsed as IPv6) cannot be dataset-file links
        return False
    return any(p.netloc == h or p.netloc.endswith("." + h) for h in CODE_HOSTS) and bool(DATA_FILE_RE.search(p.path))


def is_blocked(url: str) -> bool:
    u = url.lower()
    return any(b in u for b in blocklist()) or is_dataset_file(u)


# ---- search backends ------------------------------------------------------------------
class SearchBackend:
    name = "base"

    async def search(self, query: str, k: int) -> list[dict[str, str]]:
        raise NotImplementedError


class TavilyBackend(SearchBackend):
    name = "tavily"

    def __init__(self):
        self.key = os.environ.get("TAVILY_API_KEY")
        if not self.key:
            raise RuntimeError("TAVILY_API_KEY is not set")

    async def search(self, query, k):
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {self.key}"},
                json={"query": query, "max_results": k, "search_depth": "basic"},
            )
            r.raise_for_status()
            return [
                {"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("content", "")}
                for x in r.json().get("results", [])
            ]


class ExaBackend(SearchBackend):
    name = "exa"

    def __init__(self):
        self.key = os.environ.get("EXA_API_KEY")
        if not self.key:
            raise RuntimeError("EXA_API_KEY is not set")

    async def search(self, query, k):
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                "https://api.exa.ai/search",
                headers={"x-api-key": self.key},
                json={"query": query, "numResults": k, "contents": {"highlights": True}},
            )
            r.raise_for_status()
            out = []
            for x in r.json().get("results", []):
                hl = x.get("highlights") or []
                out.append({"title": x.get("title") or "", "url": x.get("url", ""), "snippet": " ... ".join(hl)})
            return out


class MockBackend(SearchBackend):
    """Offline backend for tests: returns pages from an in-memory dict {url: (title, text)}."""

    name = "mock"
    pages: dict[str, tuple[str, str]] = {}

    async def search(self, query, k):
        terms = set(re.findall(r"\w+", query.lower()))
        scored = []
        for url, (title, text) in self.pages.items():
            words = set(re.findall(r"\w+", (title + " " + text).lower()))
            scored.append((len(terms & words), url, title, text))
        scored.sort(key=lambda x: -x[0])
        return [{"title": t, "url": u, "snippet": tx[:300]} for s, u, t, tx in scored[:k] if s > 0]


def make_backend(name: str) -> SearchBackend:
    return {"tavily": TavilyBackend, "exa": ExaBackend, "mock": MockBackend}[name]()


# ---- tools ----------------------------------------------------------------------------
class WebSearchTool(Tool):
    def __init__(self, backend: SearchBackend, cost_per_call: float, k: int = 5, snippets_only: bool = False):
        self.backend, self.cost, self.k = backend, cost_per_call, k
        desc = (
            "Search the web. Returns up to {k} results, each with title, URL and a short snippet. "
            "Write specific queries (names, dates, distinctive phrases); reformulate rather than "
            "repeating a query that returned nothing useful. "
        ).format(k=k)
        desc += (
            "Snippets are all you can see; there is no page-fetch tool in this setup."
            if snippets_only
            else "Snippets are short: open promising pages with web_fetch to read their full text."
        )
        self.spec = ToolSpec(
            name="web_search",
            description=desc,
            parameters=schema({"query": {"type": "string", "description": "The search query."}}),
        )

    async def run(self, args, ctx):
        q = (args.get("query") or "").strip()
        if not q:
            return RawToolOutput("Error: query must be a non-empty string.", is_error=True, error_type="BadArgs", deliverable=False)
        # Rate limits (429) and server errors (5xx) are retried with backoff so harness-side throttling
        # does not show up as agent failures; every attempt's status is kept in the tool metadata.
        attempts: list[str] = []
        for attempt in range(SEARCH_RETRIES + 1):
            try:
                results = await self.backend.search(q, self.k + 3)
                break
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                attempts.append(str(code))
                if (code == 429 or code >= 500) and attempt < SEARCH_RETRIES:
                    retry_after = e.response.headers.get("retry-after", "")
                    await asyncio.sleep(float(retry_after) if retry_after.isdigit() else 2 ** (attempt + 1))
                    continue
                return RawToolOutput(
                    f"Error: search failed (HTTP {code}). Try again or rephrase.",
                    is_error=True, error_type=f"SearchHTTP{code}", deliverable=False,
                    meta={"query": q, "attempts": attempts},
                )
            except Exception as e:
                attempts.append(type(e).__name__)
                if attempt < SEARCH_RETRIES:
                    await asyncio.sleep(2 ** (attempt + 1))
                    continue
                return RawToolOutput(
                    f"Error: search failed ({type(e).__name__}). Try again or rephrase.",
                    is_error=True, error_type=f"Search{type(e).__name__}", deliverable=False,
                    meta={"query": q, "attempts": attempts},
                )
        kept = [r for r in results if not is_blocked(r["url"])][: self.k]
        blocked = sum(1 for r in results if is_blocked(r["url"]))
        if not kept:
            text = f"No results for: {q}"
        else:
            text = "\n\n".join(
                f"[{i}] {r['title']}\nURL: {r['url']}\n{r['snippet']}" for i, r in enumerate(kept, 1)
            )
        meta = {"query": q, "n_blocked": blocked, **({"retried": attempts} if attempts else {})}
        return RawToolOutput(text, urls=[r["url"] for r in kept], cost_usd=self.cost, meta=meta)


class WebFetchTool(Tool):
    spec = ToolSpec(
        name="web_fetch",
        description=(
            "Fetch a web page or PDF by URL and return its main text content (navigation and "
            "boilerplate removed). Use URLs from web_search results or links you found in pages."
        ),
        parameters=schema({"url": {"type": "string", "description": "Absolute http(s) URL."}}),
    )

    async def run(self, args, ctx):
        url = (args.get("url") or "").strip()
        if not re.match(r"^https?://", url):
            return RawToolOutput("Error: url must start with http:// or https://", is_error=True, error_type="BadArgs", deliverable=False)
        if is_blocked(url):
            return RawToolOutput("Error: this URL is blocked in this evaluation.", is_error=True, error_type="Blocked", deliverable=False)
        mock = MockBackend.pages.get(url)
        if mock is not None:
            return RawToolOutput(f"URL: {url}\nTitle: {mock[0]}\n\n{mock[1]}", urls=[url])
        try:
            async with httpx.AsyncClient(
                timeout=30, follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; agent-eval research harness)"},
            ) as c:
                r = await c.get(url)
        except Exception as e:
            return RawToolOutput(f"Error: could not fetch {url} ({type(e).__name__}).", is_error=True, error_type=f"Fetch{type(e).__name__}", deliverable=False)
        if r.status_code >= 400:
            return RawToolOutput(f"Error: HTTP {r.status_code} for {url}.", is_error=True, error_type=f"HTTP{r.status_code}", deliverable=False)
        ctype = r.headers.get("content-type", "")
        text = await asyncio.to_thread(_extract, r.content, ctype, url)
        if not text.strip():
            return RawToolOutput(f"URL: {url}\n\n[The page returned no extractable text.]", urls=[url], meta={"empty": True})
        return RawToolOutput(f"URL: {str(r.url)}\n\n{text}", urls=[url, str(r.url)])


def _extract(content: bytes, ctype: str, url: str) -> str:
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    import trafilatura

    html = content.decode("utf-8", errors="replace")
    text = trafilatura.extract(html, include_tables=True, include_links=False, favor_recall=True)
    if text:
        return text
    return re.sub(r"<[^>]+>", " ", html)


def host(url: str) -> str:
    return urlparse(url).netloc.lower()

