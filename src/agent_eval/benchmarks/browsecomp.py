"""BrowseComp (OpenAI, 2025) — faithful port of openai/simple-evals `browsecomp_eval.py`.

Differences from the reference implementation (both are bug fixes, documented on purpose):
  1. simple-evals returns `match.group(0)` ("correct: yes") and compares it to "yes", so
     nothing is ever scored correct; we compare the captured group instead.
  2. simple-evals' HTML report shows the encrypted answer; we store the decrypted answer.
"""

from __future__ import annotations

import base64
import hashlib
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

CSV_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"

# Verbatim from simple-evals (originally from the HLE repo).
QUERY_TEMPLATE = """
{Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


@dataclass
class Task:
    task_id: str
    question: str
    answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


def derive_key(password: str, length: int) -> bytes:
    key = hashlib.sha256(password.encode()).digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode()


def encrypt(plaintext: str, password: str) -> str:
    data = plaintext.encode()
    key = derive_key(password, len(data))
    return base64.b64encode(bytes(a ^ b for a, b in zip(data, key))).decode()


def csv_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / "browsecomp" / "browse_comp_test_set.csv"


def download(data_dir: str | Path) -> Path:
    p = csv_path(data_dir)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        with httpx.Client(timeout=120, follow_redirects=True) as c:
            r = c.get(CSV_URL)
            r.raise_for_status()
            p.write_bytes(r.content)
    return p


def load_tasks(
    data_dir: str | Path,
    num_examples: int | None = None,
    seed: int = 0,
    task_ids: list[str] | None = None,
) -> list[Task]:
    df = pd.read_csv(download(data_dir))
    tasks = []
    for i, row in enumerate(df.to_dict("records")):
        canary = row.get("canary", "")
        tasks.append(
            Task(
                task_id=f"bc-{i:04d}",
                question=decrypt(row.get("problem", ""), canary),
                answer=decrypt(row.get("answer", ""), canary),
                metadata={"topic": row.get("problem_topic")},
            )
        )
    if task_ids:
        wanted = set(task_ids)
        return [t for t in tasks if t.task_id in wanted]
    if num_examples:
        # same sampling as simple-evals (random.Random(0).sample) when seed == 0
        tasks = random.Random(seed).sample(tasks, num_examples)
    return tasks


GRADE_RE = re.compile(r"correct:\s*\**\s*(yes|no)", re.IGNORECASE)
EXTRACT_RE = re.compile(r"extracted_final_answer:\s*(.*)", re.IGNORECASE)
CONF_RE = re.compile(r"confidence:\s*\**\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_grade(text: str) -> dict[str, Any]:
    m = GRADE_RE.search(text)
    e = EXTRACT_RE.search(text)
    c = CONF_RE.search(text)
    return {
        "verdict": m.group(1).lower() if m else "no",
        "parse_ok": bool(m),
        "extracted_final_answer": e.group(1).strip() if e else None,
        "grader_confidence": float(c.group(1)) if c else None,
    }


RESPONSE_CONF_RE = re.compile(r"Confidence:\s*(\d+(?:\.\d+)?)\s*%?", re.IGNORECASE)


def parse_response_confidence(response: str) -> float | None:
    m = RESPONSE_CONF_RE.search(response)
    return min(100.0, float(m.group(1))) if m else None
