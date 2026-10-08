from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .executor import Flux2KleinExecutor
from .multimodal import OpenLuxClient, load_local_image
from .planner import Planner
from .schemas import (
    INSTRUCTION_FIELD,
    Plan,
    Refinement,
    RefinementRequest,
    Verification,
    as_list,
    is_enabled,
    turn_count,
    turn_metadata,
    turn_view,
)
from .tools import ImageRenderer, SafeCodeSolver, TavilySearch
from .verifier import Verifier


_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_OUTPUT_CATEGORIES = {
    "causal_reasoning",
    "counterfactual_reasoning",
    "hybrid_reasoning",
    "logical_reasoning",
    "spatial_reasoning",
    "temporal_reasoning",
}


def _validate_identifier(value: Any, field_name: str) -> str:
    text = str(value)
    if not _SAFE_IDENTIFIER.fullmatch(text) or text in {".", ".."}:
        raise ValueError(f"Unsafe {field_name}: {text!r}")
    return text


def _tally(values: Iterable[str]) -> dict[str, int]:
    """Counts keyed in sorted order, so two traces can be diffed directly."""
    return dict(sorted(Counter(values).items()))


def _validate_item_identifiers(item: dict[str, Any]) -> None:
    _validate_identifier(item.get("index", ""), "index")
    categories = [str(value) for value in as_list(item.get("category", ""))]
    if not categories or any(category not in _OUTPUT_CATEGORIES - {"hybrid_reasoning"} for category in categories):
        raise ValueError(f"Unsupported category value: {item.get('category')!r}")


class RiseAgent:
    def __init__(
        self,
        *,
        planner: Planner,
        executor: Flux2KleinExecutor,
        verifier: Verifier,
        data_root: Path,
        output_root: Path,
        renderer: ImageRenderer | None = None,
        seed: int = 42,
        max_replans: int = 1,
        max_refinements: int = 2,
    ) -> None:
        if not 0 <= max_replans <= 1:
            raise ValueError("max_replans must be 0 or 1")
        if not 0 <= max_refinements <= 2:
            raise ValueError("max_refinements must be between 0 and 2")
        self.planner = planner
        self.executor = executor
        self.verifier = verifier
        self.renderer = renderer or ImageRenderer()
        self.data_root = data_root.resolve()
        self.output_root = output_root.resolve()
        self.seed = seed
        self.max_replans = max_replans
        self.max_refinements = max_refinements

    def _data_path(self, relative_path: str) -> Path:
        path = (self.data_root / relative_path).resolve()
        try:
            path.relative_to(self.data_root)
        except ValueError as exc:
            raise ValueError(f"Dataset path escapes the data directory: {relative_path}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Dataset image not found: {path}")
        return path

    def _seed_for(self, sample_index: str, attempt: int, turn: int) -> int:
        digest = hashlib.sha256(f"{self.seed}:{sample_index}:{attempt}:{turn}".encode()).digest()
        return int.from_bytes(digest[:4], "big")

    def _execute(
        self,
        plan: Plan,
        images: list[Image.Image],
        *,
        sample_index: str,
        attempt: int,
        turn: int,
    ) -> Image.Image:
        """Turn a plan into pixels with whichever executor the planner selected."""
        if plan.executor == "programmatic_edit":
            if plan.render_request is None:
                raise ValueError(
                    "Plan selected programmatic_edit but carries no render_request"
                )
            return self.renderer.render(
                images, plan.render_request, base_index=plan.base_image_index
            )
        return self.executor.generate(
            plan.edit_prompt,
            images,
            self._seed_for(sample_index, attempt, turn),
        )

    def _allowed_actions(self, replan_count: int, refinement_count: int) -> list[str]:
        """Which actions the loop controller may pick, given the current budget.

        The budget does not depend on which executor produced the candidate. How a
        refinement is *carried out* does: the verifier decides that per candidate, and
        may repair a deterministically drawn candidate with another drawing program
        instead of a diffusion repaint.

        Replanning is available only before the first refinement, so the loop is always
        one of: replan then refine, or refine only.
        """
        allowed = ["pass"]
        if refinement_count < self.max_refinements:
            allowed.append("refine")
        if replan_count < self.max_replans and refinement_count == 0:
            allowed.append("replan")
        return allowed

    def _plan_and_execute(
        self,
        item: dict[str, Any],
        turn_index: int,
        instruction: str,
        images: list[Image.Image],
        *,
        total_turns: int,
        attempt: int,
        replan_feedback: str = "",
        failed_image: Image.Image | None = None,
    ) -> tuple[Image.Image, Plan]:
        metadata = turn_metadata(item, turn_index)
        metadata["total_turns"] = str(total_turns)
        metadata["is_replan"] = str(bool(replan_feedback))
        plan = self.planner.plan(
            instruction=instruction,
            images=images,
            metadata=metadata,
            replan_feedback=replan_feedback,
            failed_image=failed_image,
        )
        candidate = self._execute(
            plan,
            images,
            sample_index=str(item["index"]),
            attempt=attempt,
            turn=turn_index,
        )
        return candidate, plan

    def _refine(
        self,
        item: dict[str, Any],
        candidate: Image.Image,
        request: RefinementRequest,
        *,
        attempt: int,
        turn_index: int,
        based_on_attempt: int,
    ) -> tuple[Image.Image, Refinement]:
        """Apply one incremental correction to the current candidate.

        The verifier chooses how. A `programmatic_edit` refinement draws its already
        validated program straight onto the candidate, so every pixel it does not touch
        stays byte-identical: that is the point of repairing a deterministically drawn
        candidate this way rather than repainting it. A `generative` refinement hands the
        candidate to the diffusion executor, which is the only one that can invent or
        remove realistic content. Either way the model sees *only* the candidate, so it
        cannot drift back towards the source, and the generative prompt is the verifier's
        terse delta with no wrapper template.

        A drawing program that fails here has already rendered once inside the verifier
        against this exact image, so a failure means something changed underneath us;
        falling back to the diffusion editor keeps the turn alive instead of losing it.
        """
        instruction = request.instruction.strip()
        render_request = request.render
        fallback = ""
        if request.executor == "programmatic_edit" and render_request is not None:
            try:
                refined = self.renderer.render([candidate], render_request, base_index=0)
            except Exception as exc:
                fallback = (
                    f"programmatic refinement failed at execution: {type(exc).__name__}: {exc}"
                )
            else:
                return refined, Refinement(
                    instruction=instruction,
                    executor="programmatic_edit",
                    based_on_attempt=based_on_attempt,
                    render_request=render_request,
                )
        refined = self.executor.generate(
            instruction,
            [candidate],
            self._seed_for(str(item["index"]), attempt, turn_index),
        )
        return refined, Refinement(
            instruction=instruction,
            executor="generative",
            based_on_attempt=based_on_attempt,
            fallback=fallback,
        )

    def _output_path(self, *parts: str) -> Path:
        path = self.output_root.joinpath(*parts).resolve()
        try:
            path.relative_to(self.output_root)
        except ValueError as exc:
            raise ValueError(f"Output path escapes output directory: {path}") from exc
        return path

    def _save_candidate(
        self,
        trace_dir: Path,
        turn_index: int,
        attempt: int,
        kind: str,
        image: Image.Image,
    ) -> Path:
        path = trace_dir / f"turn_{turn_index + 1:02d}_attempt_{attempt:02d}_{kind}.png"
        image.save(path)
        return path

    def _run_turn(
        self,
        item: dict[str, Any],
        turn_index: int,
        input_images: list[Image.Image],
        *,
        total_turns: int,
        trace_dir: Path,
        on_progress: Callable[[dict[str, Any]], None],
    ) -> tuple[Image.Image, dict[str, Any]]:
        """One complete plan-execute-verify loop for a single instruction.

        This is the whole agent for a single-turn sample, and one link of the chain for
        a multi-turn one. The turn is scored against its own instruction and its own
        input images, so a later turn never has to be judged through the lens of the
        earlier ones.
        """
        view = turn_view(item, turn_index)
        instruction = str(view["instruction"])
        candidates: list[dict[str, Any]] = []
        candidate_images: list[Image.Image] = []
        attempt = 0
        replan_count = 0
        refinement_count = 0

        def record(
            image: Image.Image,
            kind: str,
            detail: dict[str, Any],
            executor: str,
            base_image_index: int | None = None,
            previous_render: dict[str, Any] | None = None,
        ) -> Verification:
            path = self._save_candidate(trace_dir, turn_index, attempt, kind, image)
            verification = self.verifier.verify(
                item=view,
                source_images=input_images,
                candidate=image,
                allowed_actions=self._allowed_actions(replan_count, refinement_count),
                source_executor=executor,
                base_image_index=base_image_index,
                previous_render=previous_render,
            )
            candidates.append(
                {
                    "attempt": attempt,
                    "kind": kind,
                    "executor": executor,
                    "image": str(path.relative_to(self.output_root)),
                    **detail,
                    "verification": verification.to_dict(),
                }
            )
            candidate_images.append(image.copy())
            on_progress(
                {
                    "turn": turn_index + 1,
                    "instruction": instruction,
                    "status": "running",
                    "replan_count": replan_count,
                    "refinement_count": refinement_count,
                    "candidates": candidates,
                }
            )
            return verification

        candidate, plan = self._plan_and_execute(
            item,
            turn_index,
            instruction,
            input_images,
            total_turns=total_turns,
            attempt=attempt,
        )
        def canvas_of(current: Plan) -> int | None:
            """Which source image the candidate is an edited copy of, when there is one.

            Only a drawing program edits a specific source in place; a generated image
            is a fresh frame informed by all of them, so it has no single canvas.
            """
            return (
                current.base_image_index
                if current.executor == "programmatic_edit"
                else None
            )

        executor = plan.executor
        render_of_candidate = plan.render_request
        verification = record(
            candidate,
            "initial",
            {"plan": plan.to_dict()},
            executor,
            canvas_of(plan),
            render_of_candidate,
        )

        while verification.action != "pass":
            if verification.action == "replan" and "replan" in self._allowed_actions(
                replan_count, refinement_count
            ):
                replan_count += 1
                previous_attempt, previous_candidate = attempt, candidate
                attempt += 1
                candidate, plan = self._plan_and_execute(
                    item,
                    turn_index,
                    instruction,
                    input_images,
                    total_turns=total_turns,
                    attempt=attempt,
                    replan_feedback=verification.analysis,
                    failed_image=previous_candidate,
                )
                executor = plan.executor
                render_of_candidate = plan.render_request
                verification = record(
                    candidate,
                    "replan",
                    {"plan": plan.to_dict(), "replanned_from_attempt": previous_attempt},
                    executor,
                    canvas_of(plan),
                    render_of_candidate,
                )
                continue
            if verification.action == "refine" and "refine" in self._allowed_actions(
                replan_count, refinement_count
            ):
                if verification.refinement_request is None:
                    raise ValueError("Verification requested refine but carries no request")
                refinement_count += 1
                previous_attempt = attempt
                attempt += 1
                candidate, refinement = self._refine(
                    item,
                    candidate,
                    verification.refinement_request,
                    attempt=attempt,
                    turn_index=turn_index,
                    based_on_attempt=previous_attempt,
                )
                if refinement.executor == "programmatic_edit":
                    render_of_candidate = refinement.render_request
                    canvas = canvas_of(plan)
                else:
                    executor = "generative"
                    render_of_candidate = None
                    canvas = None
                verification = record(
                    candidate,
                    "refine",
                    {"refinement": refinement.to_dict()},
                    executor,
                    canvas,
                    render_of_candidate,
                )
                continue
            break

        best_index = max(
            range(len(candidates)),
            key=lambda index: (candidates[index]["verification"]["overall_score"], -index),
        )
        selected = candidates[best_index]
        selected_rule = selected["verification"]["pass_rule"]
        stop_reason = (
            verification.pass_rule if verification.action == "pass" else "budget_exhausted"
        )
        turn_record = {
            "turn": turn_index + 1,
            "instruction": instruction,
            "category": view.get("category"),
            "status": "completed",
            "passed": selected_rule is not None,
            "pass_rule": selected_rule,
            "stop_reason": stop_reason,
            "selected_attempt": selected["attempt"],
            "selected_score": selected["verification"]["overall_score"],
            "selected_image": selected["image"],
            "replan_count": replan_count,
            "refinement_count": refinement_count,
            "candidates": candidates,
        }
        return candidate_images[best_index], turn_record

    def run_item(self, item: dict[str, Any], overwrite: bool = False) -> dict[str, Any]:
        _validate_item_identifiers(item)
        sample_index = _validate_identifier(item["index"], "index")
        multi_turn = is_enabled(item, "multi_turn")
        output_category = "hybrid_reasoning" if multi_turn else str(item["category"])
        if output_category not in _OUTPUT_CATEGORIES:
            raise ValueError(f"Unsupported output category: {output_category!r}")
        final_dir = self._output_path("images", output_category)
        final_path = self._output_path("images", output_category, f"{sample_index}.png")
        trace_dir = self._output_path("agent_traces", sample_index)
        trace_path = self._output_path("agent_traces", sample_index, "trace.json")
        if final_path.exists() and trace_path.exists() and not overwrite:
            try:
                with trace_path.open(encoding="utf-8") as handle:
                    previous_trace = json.load(handle)
                if (
                    previous_trace.get("status") == "completed"
                    and previous_trace.get("output") == str(final_path)
                ):
                    return {"index": sample_index, "status": "skipped", "output": str(final_path)}
            except (OSError, json.JSONDecodeError):
                pass

        final_dir.mkdir(parents=True, exist_ok=True)
        trace_dir.mkdir(parents=True, exist_ok=True)
        for stale in trace_dir.glob("turn_*.png"):
            stale.unlink()
        (trace_dir / "error.json").unlink(missing_ok=True)

        total_turns = turn_count(item)
        base: dict[str, Any] = {
            "index": sample_index,
            "multi_turn": multi_turn,
            "total_turns": total_turns,
        }
        self._write_trace(trace_path, {**base, "status": "running", "turns": []})

        turn_records: list[dict[str, Any]] = []
        try:
            current_images = [load_local_image(self._data_path(str(path))) for path in item["image"]]

            def on_progress(active: dict[str, Any]) -> None:
                self._write_trace(
                    trace_path,
                    {**base, "status": "running", "turns": turn_records + [active]},
                )

            output: Image.Image | None = None
            for turn_index in range(total_turns):
                output, turn_record = self._run_turn(
                    item,
                    turn_index,
                    current_images,
                    total_turns=total_turns,
                    trace_dir=trace_dir,
                    on_progress=on_progress,
                )
                turn_records.append(turn_record)
                current_images = [output]
            if output is None:
                raise ValueError(f"Sample {item['index']} has no instructions")

            output.save(final_path)
            last_turn = turn_records[-1]
            trace = {
                **base,
                "status": "completed",
                "passed": last_turn["passed"],
                "pass_rules": _tally([last_turn["pass_rule"] or "unresolved"]),
                "stop_reasons": _tally(record["stop_reason"] for record in turn_records),
                "turn_scores": [record["selected_score"] for record in turn_records],
                "final_score": last_turn["selected_score"],
                "output": str(final_path),
                "replan_count": sum(record["replan_count"] for record in turn_records),
                "refinement_count": sum(record["refinement_count"] for record in turn_records),
                "turns": turn_records,
            }
            self._write_trace(trace_path, trace)
            return trace
        except Exception as exc:
            self._write_trace(
                trace_path,
                {
                    **base,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "turns": turn_records,
                },
            )
            raise

    @staticmethod
    def _write_trace(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must contain a list")
    required = {"index", "category", INSTRUCTION_FIELD, "image"}
    for offset, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item {offset} must be an object")
        missing = required - set(item)
        if missing:
            raise ValueError(f"Dataset item {offset} is missing fields: {sorted(missing)}")
        _validate_item_identifiers(item)
        if not isinstance(item["image"], list) or not item["image"]:
            raise ValueError(f"Dataset item {offset} must contain at least one image")
        if is_enabled(item, "multi_turn"):
            turns = as_list(item[INSTRUCTION_FIELD])
            if not turns:
                raise ValueError(f"Dataset item {offset} has no turns")
            for field in ("category", "subcategory", "task_type"):
                if len(as_list(item.get(field, []))) != len(turns):
                    raise ValueError(f"Dataset item {offset} has misaligned multi-turn field: {field}")
    return data


def _select_items(data: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    indices = {value.strip() for value in args.indices.split(",") if value.strip()} if args.indices else None
    selected = []
    for item in data:
        if indices is not None and str(item["index"]) not in indices:
            continue
        categories = [str(value) for value in as_list(item["category"])]
        if args.category and args.category not in categories and args.category != "hybrid_reasoning":
            continue
        if args.category == "hybrid_reasoning" and not is_enabled(item, "multi_turn"):
            continue
        selected.append(item)
    selected = selected[args.start :]
    return selected[: args.limit] if args.limit is not None else selected


_CONFIG_KEYS = {
    "data",
    "input_dir",
    "output_dir",
    "model_path",
    "planner_model",
    "verifier_model",
    "api_base",
    "steps",
    "guidance_scale",
    "cpu_offload",
    "seed",
    "max_replans",
    "max_refinements",
    "category",
    "indices",
    "start",
    "limit",
    "overwrite",
    "fail_fast",
    "dry_run",
}
_SECRET_CONFIG_KEYS = {"api_key", "tavily_api_key"}


def _config_path(argv: list[str] | None) -> Path:
    default_path = Path(__file__).resolve().parent / "agent.cfg"
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=default_path)
    known, _ = parser.parse_known_args(argv)
    return known.config.resolve()


def _load_config_defaults(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Agent config not found: {config_path}")
    config = configparser.ConfigParser(interpolation=None)
    with config_path.open(encoding="utf-8") as handle:
        config.read_file(handle)
    if "agent" not in config:
        raise ValueError(f"Config must contain an [agent] section: {config_path}")
    section = config["agent"]
    unknown = set(section) - _CONFIG_KEYS - _SECRET_CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown agent config keys: {sorted(unknown)}")
    forbidden = set(section) & _SECRET_CONFIG_KEYS
    if forbidden:
        raise ValueError(
            "API keys must be supplied through API_KEY and TAVILY_API_KEY environment variables, "
            f"not config fields: {sorted(forbidden)}"
        )

    defaults: dict[str, Any] = {}
    path_keys = {"data", "input_dir", "output_dir", "model_path"}
    int_keys = {"steps", "seed", "max_replans", "max_refinements", "start", "limit"}
    bool_keys = {"cpu_offload", "overwrite", "fail_fast", "dry_run"}
    for key, raw_value in section.items():
        value = raw_value.strip()
        if not value:
            defaults[key] = None
        elif key in path_keys:
            path = Path(value).expanduser()
            defaults[key] = path if path.is_absolute() else (config_path.parent / path).resolve()
        elif key in int_keys:
            defaults[key] = int(value)
        elif key == "guidance_scale":
            defaults[key] = float(value)
        elif key in bool_keys:
            defaults[key] = section.getboolean(key)
        else:
            defaults[key] = value
    return defaults


def build_parser(argv: list[str] | None = None) -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent.parent
    config_path = _config_path(argv)
    config_defaults = _load_config_defaults(config_path)
    parser = argparse.ArgumentParser(description="RISE++ agentic image-editing baseline")
    parser.add_argument("--config", type=Path, default=config_path)
    parser.add_argument("--data", type=Path, default=base / "data" / "overall_data.json")
    parser.add_argument("--input-dir", type=Path, default=base / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-path", type=Path, default=Path("/pyuanzhang/data/models/FLUX.2-klein-9B"))
    parser.add_argument("--planner-model", default=os.getenv("PLANNER_MODEL") or os.getenv("API_MODEL"))
    parser.add_argument("--verifier-model", default=os.getenv("VERIFIER_MODEL") or os.getenv("API_MODEL"))
    parser.add_argument("--api-base", default=os.getenv("API_BASE", "https://api.openlux.ai/v1"))
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-replans",
        type=int,
        choices=(0, 1),
        default=1,
        help="Maximum replans before refinement begins (after refine, only refine/pass are allowed)",
    )
    parser.add_argument("--max-refinements", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--category")
    parser.add_argument("--indices", help="Comma-separated sample indices")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Validate and list selected samples without loading models",
    )
    parser.set_defaults(**config_defaults)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser(argv).parse_args(argv)
    if args.output_dir is None:
        raise SystemExit("Set output_dir in agent.cfg or pass --output-dir")
    if args.planner_model is None:
        args.planner_model = os.getenv("PLANNER_MODEL") or os.getenv("API_MODEL")
    if args.verifier_model is None:
        args.verifier_model = os.getenv("VERIFIER_MODEL") or os.getenv("API_MODEL")
    data = _load_dataset(args.data)
    selected = _select_items(data, args)
    print(f"Selected {len(selected)} of {len(data)} samples")
    if args.dry_run:
        for item in selected:
            print(item["index"])
        return 0

    api_key = os.getenv("API_KEY", "")
    if not api_key:
        raise SystemExit("API_KEY must be set in the environment")
    if not args.planner_model or not args.verifier_model:
        raise SystemExit("Set API_MODEL or pass both --planner-model and --verifier-model")

    client = OpenLuxClient(api_key=api_key, base_url=args.api_base)
    renderer = ImageRenderer()
    planner = Planner(
        client=client,
        model=args.planner_model,
        search=TavilySearch(os.getenv("TAVILY_API_KEY")),
        solver=SafeCodeSolver(),
        renderer=renderer,
    )
    executor = Flux2KleinExecutor(
        args.model_path,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        cpu_offload=args.cpu_offload,
    )
    verifier = Verifier(client=client, model=args.verifier_model, renderer=renderer)
    agent = RiseAgent(
        planner=planner,
        executor=executor,
        verifier=verifier,
        renderer=renderer,
        data_root=args.input_dir,
        output_root=args.output_dir,
        seed=args.seed,
        max_replans=args.max_replans,
        max_refinements=args.max_refinements,
    )

    failures = 0
    for offset, item in enumerate(selected, 1):
        sample_index = item["index"]
        print(f"[{offset}/{len(selected)}] {sample_index}", flush=True)
        try:
            result = agent.run_item(item, overwrite=args.overwrite)
            print(
                f"  {result['status']} -> {result['output']}"
                + (f" (score={result['final_score']})" if "final_score" in result else "")
                + (
                    f" turns={result['turn_scores']}"
                    if result.get("multi_turn") and "turn_scores" in result
                    else ""
                )
            )
        except Exception as exc:
            failures += 1
            print(f"  failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            error_dir = args.output_dir / "agent_traces" / str(sample_index)
            error_dir.mkdir(parents=True, exist_ok=True)
            RiseAgent._write_trace(
                error_dir / "error.json",
                {"index": sample_index, "status": "failed", "error": f"{type(exc).__name__}: {exc}"},
            )
            if args.fail_fast:
                raise
    print(f"Completed: {len(selected) - failures}; failed: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
