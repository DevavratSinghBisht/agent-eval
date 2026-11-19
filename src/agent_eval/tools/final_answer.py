from __future__ import annotations

from ..core.types import ToolSpec
from .base import RawToolOutput, Tool, schema


class FinalAnswerTool(Tool):
    """Default termination tool. The run ends when it is called; its arguments are stored in
    ctx.state['final_answer'] and rendered for the grader by `Benchmark.format_response`.

    Benchmarks needing structured answers subclass this with their own `spec` (keep the name
    `final_answer` and store the arguments the same way)."""

    spec = ToolSpec(
        name="final_answer",
        description=(
            "Submit your final answer and end the task. Call this exactly once, when you are done. "
            "Give ONE answer: do not list alternatives or hedge between candidates. "
            "If you could not fully verify it, still give your best answer with a low confidence."
        ),
        parameters=schema(
            {
                "explanation": {"type": "string", "description": "Brief justification, citing the evidence you used."},
                "answer": {"type": "string", "description": "The final answer only."},
                "confidence": {"type": "integer", "description": "Confidence from 0 to 100 that the answer is correct."},
            }
        ),
    )

    async def run(self, args, ctx):
        ctx.state["final_answer"] = dict(args)
        return RawToolOutput("Final answer recorded.", deliverable=False)
