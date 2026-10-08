from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# For RISEBench++ (ZH), use instruction_chinese
INSTRUCTION_FIELD = "instruction"

Route = Literal["direct", "web_search", "code_solver"]
Executor = Literal["generative", "programmatic_edit"]
Action = Literal["pass", "refine", "replan"]


@dataclass
class Plan:
    route: Route
    executor: Executor
    rationale: str
    visual_analysis: str
    target_state: str
    edit_prompt: str = ""
    render_request: dict[str, Any] | None = None
    # Which source image the drawing program is applied to. Only meaningful for
    # `programmatic_edit`; a task may require annotating the third input, not the first.
    base_image_index: int = 0
    preserve: list[str] = field(default_factory=list)
    tool_request: dict[str, Any] | None = None
    tool_result: dict[str, Any] | None = None
    tool_attempts: int = 0
    fallbacks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Refinement:
    """One incremental edit.

    Deliberately *not* shaped like a Plan: a refinement has no route, no tool call
    and no re-planning: it only carries the delta and which attempt it started from.
    `executor` is chosen by the verifier rather than the planner, and `render_request`
    holds the drawing program when it chose the deterministic path.
    """

    instruction: str
    executor: Executor
    based_on_attempt: int
    render_request: dict[str, Any] | None = None
    fallback: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RefinementRequest:
    """The edit the verifier is asking for, and how it wants it produced.

    The three parts travel as one object because they are only meaningful together: an
    executor choice or a drawing program with no edit to run is not a decision, it is a
    leftover. Keeping them in a slot that a `pass` or a `replan` leaves empty makes that
    a property of the structure rather than a convention three call sites have to
    remember, so an inapplicable "generative" can never reappear in a trace.

    `executor` is chosen by the verifier rather than the planner: a candidate drawn
    deterministically onto an untouched source is usually repaired best by another
    deterministic drawing, not by a diffusion repaint. `render` carries that
    already-validated drawing program, and is None exactly when `executor` is
    "generative".
    """

    instruction: str
    executor: Executor = "generative"
    render: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Verification:
    action: Action
    reasoning_score: float
    consistency_score: float | None
    quality_score: float | None
    overall_score: float
    analysis: str
    completed_requirements: list[str] = field(default_factory=list)
    missing_requirements: list[str] = field(default_factory=list)
    refinement_request: RefinementRequest | None = None
    pass_rule: str | None = None
    requested_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_enabled(item: dict[str, Any], field_name: str) -> bool:
    value = item.get(field_name)
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def as_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def turn_metadata(item: dict[str, Any], turn_index: int) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in ("category", "subcategory", "task_type"):
        values = as_list(item.get(key, ""))
        value = values[min(turn_index, len(values) - 1)] if values else ""
        result[key] = str(value)
    result["sample_index"] = str(item.get("index", ""))
    result["turn"] = str(turn_index + 1)
    return result


def turn_count(item: dict[str, Any]) -> int:
    return len(as_list(item[INSTRUCTION_FIELD]))


def turn_view(item: dict[str, Any], turn_index: int) -> dict[str, Any]:
    """A single-turn view of a (possibly multi-turn) sample.

    A multi-turn sample is executed as a sequence of independent single-turn tasks,
    each one a full plan-execute-verify loop whose input image is the previous turn's
    accepted output. Every stage therefore sees exactly one instruction and the
    per-turn category that selects the scoring scheme, never the whole chain.
    Sample-level flags such as `consistency_free` are carried through unchanged.
    """
    view = dict(item)
    instructions = as_list(item[INSTRUCTION_FIELD])
    view["instruction"] = str(instructions[min(turn_index, len(instructions) - 1)])
    for key in ("category", "subcategory", "task_type"):
        values = as_list(item.get(key, ""))
        if values:
            view[key] = str(values[min(turn_index, len(values) - 1)])
    return view


def is_logical(item: dict[str, Any]) -> bool:
    return "logical_reasoning" in as_list(item.get("category", ""))
