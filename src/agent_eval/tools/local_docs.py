"""Local-corpus tools over the BrowseComp-Plus corpus (~100K docs), BM25 via bm25s.

The index is built once (`agent-eval prepare browsecomp_plus`) and cached under data/.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from ..core.types import ToolSpec
from .base import RawToolOutput, Tool, schema

SNIPPET_CHARS = 600


class Corpus:
    _instances: dict[str, "Corpus"] = {}
    _lock = threading.Lock()

    def __init__(self, docs: dict[str, dict[str, str]], retriever=None, docid_order: list[str] | None = None):
        self.docs = docs  # docid -> {"text", "url", "title"}
        self.retriever = retriever
        self.docid_order = docid_order or list(docs)

    # ---- construction ------------------------------------------------------------------
    @classmethod
    def from_records(cls, records: list[dict[str, str]]) -> "Corpus":
        import bm25s

        docs = {str(r["docid"]): {"text": r["text"], "url": r.get("url", ""), "title": _title(r["text"])} for r in records}
        order = list(docs)
        retriever = bm25s.BM25()
        retriever.index(bm25s.tokenize([docs[d]["text"] for d in order], stopwords="en", show_progress=False), show_progress=False)
        return cls(docs, retriever, order)

    @classmethod
    def load(cls, data_dir: str | Path) -> "Corpus":
        key = str(Path(data_dir).resolve())
        with cls._lock:
            if key not in cls._instances:
                import bm25s

                d = Path(data_dir) / "browsecomp_plus"
                docs = {}
                with open(d / "corpus.jsonl", encoding="utf-8") as fh:
                    for line in fh:
                        r = json.loads(line)
                        docs[str(r["docid"])] = {"text": r["text"], "url": r.get("url", ""), "title": _title(r["text"])}
                order = json.loads((d / "docid_order.json").read_text(encoding="utf-8"))
                retriever = bm25s.BM25.load(str(d / "bm25_index"))
                cls._instances[key] = cls(docs, retriever, order)
            return cls._instances[key]

    @staticmethod
    def build_index(data_dir: str | Path) -> None:
        import bm25s

        d = Path(data_dir) / "browsecomp_plus"
        order, texts = [], []
        with open(d / "corpus.jsonl", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                order.append(str(r["docid"]))
                texts.append(r["text"])
        retriever = bm25s.BM25()
        retriever.index(bm25s.tokenize(texts, stopwords="en"))
        retriever.save(str(d / "bm25_index"))
        (d / "docid_order.json").write_text(json.dumps(order), encoding="utf-8")

    # ---- queries -----------------------------------------------------------------------
    def search(self, query: str, k: int) -> list[str]:
        import bm25s

        tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        if not tokens.vocab:
            return []
        results, scores = self.retriever.retrieve(tokens, k=min(k, len(self.docid_order)), show_progress=False)
        return [self.docid_order[int(i)] for i, s in zip(results[0], scores[0]) if s > 0]


def _title(text: str) -> str:
    for line in text.splitlines()[:6]:
        if line.lower().startswith("title:"):
            return line.split(":", 1)[1].strip()
    return text.strip().splitlines()[0][:120] if text.strip() else ""


class SearchDocsTool(Tool):
    def __init__(self, corpus: Corpus, k: int = 5):
        self.corpus, self.k = corpus, k
        self.spec = ToolSpec(
            name="search_docs",
            description=(
                f"Search the local document collection (a fixed corpus of web pages) with BM25 keyword "
                f"matching. Returns up to {k} documents with docid, title, URL and the opening text. "
                "Use distinctive keywords; read promising documents in full with read_doc."
            ),
            parameters=schema({"query": {"type": "string", "description": "Keyword query."}}),
        )

    async def run(self, args, ctx):
        q = (args.get("query") or "").strip()
        if not q:
            return RawToolOutput("Error: query must be a non-empty string.", is_error=True, error_type="BadArgs", deliverable=False)
        ids = await asyncio.to_thread(self.corpus.search, q, self.k)
        if not ids:
            return RawToolOutput(f"No documents matched: {q}", meta={"query": q})
        parts = []
        for i, d in enumerate(ids, 1):
            doc = self.corpus.docs[d]
            parts.append(f"[{i}] docid: {d}\nTitle: {doc['title']}\nURL: {doc['url']}\n{doc['text'][:SNIPPET_CHARS]}")
        return RawToolOutput("\n\n".join(parts), docids=ids, urls=[self.corpus.docs[d]["url"] for d in ids], meta={"query": q})


class ReadDocTool(Tool):
    def __init__(self, corpus: Corpus, allowed: set[str] | None = None):
        self.corpus, self.allowed = corpus, allowed
        self.spec = ToolSpec(
            name="read_doc",
            description="Read the full text of a document from the local collection by its docid.",
            parameters=schema({"docid": {"type": "string", "description": "The docid shown in search results."}}),
        )

    async def run(self, args, ctx):
        d = str(args.get("docid", "")).strip()
        if self.allowed is not None and d not in self.allowed:
            return RawToolOutput(
                f"Error: docid {d!r} is not one of the documents provided for this question.",
                is_error=True, error_type="BadDocid", deliverable=False,
            )
        doc = self.corpus.docs.get(d)
        if doc is None:
            return RawToolOutput(f"Error: no document with docid {d!r}.", is_error=True, error_type="BadDocid", deliverable=False)
        return RawToolOutput(f"docid: {d}\nURL: {doc['url']}\n\n{doc['text']}", docids=[d], urls=[doc["url"]])
