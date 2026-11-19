from __future__ import annotations

from ..core.types import ToolSpec
from .base import RawToolOutput, Tool, schema


class FinalAnswerTool(Tool):
    """Explicit termination tool (Mercor-style). The run ends when this is called; its fields
    map onto BrowseComp's official Explanation / Exact Answer / Confidence response format."""

    spec = ToolSpec(
        name="final_answer",
        description=(
            "Submit your final answer and end the task. Call this exactly once, when you are done. "
            "Give ONE answer: do not list alternatives or hedge between candidates. "
            "If you could not verify an answer, still give your single best guess and a low confidence."
        ),
        parameters=schema(
            {
                "explanation": {"type": "string", "description": "Brief justification, citing the evidence you found."},
                "exact_answer": {"type": "string", "description": "The succinct final answer only."},
                "confidence": {"type": "integer", "description": "Confidence from 0 to 100 that the answer is correct."},
            }
        ),
    )

    async def run(self, args, ctx):
        ctx.state["final_answer"] = {
            "explanation": str(args.get("explanation", "")),
            "exact_answer": str(args.get("exact_answer", "")),
            "confidence": args.get("confidence"),
        }
        return RawToolOutput("Final answer recorded.", deliverable=False)


def format_response(fa: dict | None, fallback_text: str = "") -> str:
    """Render the agent's answer in the official BrowseComp response format for the grader."""
    if not fa:
        return fallback_text or "Explanation: None\nExact Answer: None\nConfidence: 0%"
    conf = fa.get("confidence")
    return (
        f"Explanation: {fa.get('explanation', '')}\n"
        f"Exact Answer: {fa.get('exact_answer', '')}\n"
        f"Confidence: {conf if conf is not None else 100}%"
    )
