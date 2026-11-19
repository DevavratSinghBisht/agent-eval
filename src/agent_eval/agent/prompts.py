"""Harness-level messages (benchmark-neutral). System and task prompts come from the benchmark
plugin (`Benchmark.system_prompt` / `task_prompt`)."""

from __future__ import annotations

WRAP_UP_NUDGE = (
    "[Budget notice] You have used most of your step/token budget for this task. Stop now "
    "and call final_answer with your best answer and an honest confidence."
)
NO_TOOL_NUDGE = (
    "You replied without calling a tool. If you are done, call final_answer now; otherwise continue "
    "working with the available tools."
)
MAX_TOKENS_NUDGE = "Your previous response was cut off by the output length limit. Be more concise and continue."

SUMMARIZER_PROMPT = """Summarize the following tool output for an agent working on this task:

<task>
{question}
</task>

Keep every fact, name, date, number and URL that could be relevant to the task, verbatim where possible. Drop boilerplate. Do not solve the task yourself.

<tool_output>
{text}
</tool_output>"""
