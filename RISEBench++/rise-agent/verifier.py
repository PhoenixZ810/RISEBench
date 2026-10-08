from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable

from PIL import Image

from .multimodal import OpenLuxClient, image_content
from .schemas import RefinementRequest, Verification, as_list, is_enabled, is_logical
from .tools import ImageRenderer


_CONSISTENCY_SYSTEM = """You are the consistency checker inside an image-editing agent. You receive the original source image(s) followed by the agent's candidate output image and the editing instruction(s). Judge only unintended changes outside what the instruction asked for.
For a logical task (`logical_scoring` is true), score 1 if visual style, layout, background, line weight, colors, and unaffected content are preserved; otherwise score 0. Emit exactly the integer 0 or the integer 1 and nothing between them: no fraction, no decimal, no confidence or partial credit. When preservation is imperfect but arguable, you must commit to 0 or 1. Ignore only the marks the instruction asked to add.
For other tasks, score an integer 1-5: 5 perfect preservation, 4 one minor visible difference, 3 one major or a few minor differences, 2 multiple major differences, 1 severe inconsistency. When several source images are supplied, treat all of them jointly as source content and do not punish rearrangement that the instruction required.
If one source image is labelled as the canvas, preservation is judged against that image alone: the candidate is an edited copy of it, and the other sources are reference material whose content is not expected to appear in the output. Do not penalise the candidate for differing from a source that is not the canvas.
Return JSON only: {"score": number, "analysis": "concise evidence"}."""

_QUALITY_SYSTEM = """You are the visual quality checker inside an image-editing agent. You receive the original source image(s) followed by the candidate output image. Judge only clarity, blur, distortion, broken textures, rendering glitches, and physical inaccuracies introduced in edited or newly generated regions. Do not punish defects already present in the source. If counterfactual=true, do not punish intended impossible or nonphysical content. Score an integer 1-5: 5 clean and coherent, 4 minor artifact, 3 noticeable local artifacts, 2 major artifacts, 1 unusable. Return JSON only: {"score": number, "analysis": "concise evidence"}."""

_REASONING_SYSTEM = """You are the correctness checker inside an image-editing agent. You receive the original source image(s), then the candidate output image, then the instruction(s).
No reference answer is available. You must independently work out what the instruction requires from the source image(s) alone, then check whether the candidate realises it. State your own derived expectation in the analysis before judging.
For a logical task (`logical_scoring` is true), score is binary: 1 only when the answer shown is correct, complete, and free of extra answer content, otherwise 0. Emit exactly the integer 0 or the integer 1 and nothing between them: no fraction, no decimal, no confidence or partial credit. A partly correct answer is 0. For other tasks, score is an integer 1-5: 5 fully satisfies every requirement, 4 minor discrepancy, 3 partial, 2 major missing/incorrect details, 1 unrelated or failed.
Return JSON only:
{
  "expected_result": "what the instruction requires, derived from the source images",
  "score": number,
  "analysis": "concise evidence-based assessment",
  "completed_requirements": ["..."],
  "missing_requirements": ["..."]
}"""

_ACTION_SYSTEM = """You are the loop controller for an image-editing agent. The correctness, consistency, and quality dimensions have already been scored independently. Use those immutable findings to choose one allowed action:
- pass when the candidate is good enough to ship. Every applicable score sitting at its maximum always qualifies. Below that, pass is available only if every applicable score reaches its `pass_floor` and you judge that the candidate realises the great majority of the instruction's intent with no conspicuous consistency or quality defect. A score under its `pass_floor` never qualifies, however appealing the image looks.
- refine only when the candidate is partly correct and a localized incremental edit can repair it while preserving successful content.
- replan when it solves the wrong problem or must be regenerated from the original source.
Choose only an action in allowed_actions.

A refinement runs one of two executors, and you choose which with `refine_with`:
- "generative" hands the candidate plus `refinement_instruction` to a diffusion editor. It is the only executor that can invent or remove realistic imagery, but it repaints the whole candidate: line weights, fonts, grid alignment and chart geometry all drift, and content that already works often degrades. This refinement is not free, so only choose it when the expected repair clearly outweighs that damage, and prefer passing an acceptable candidate over chasing a marginal point.
- "programmatic_edit" applies a deterministic drawing program (`render`) directly onto the candidate, so every pixel you do not touch stays byte-identical. It draws the primitives listed in `render_schema`, erases a rectangle back to its sampled surrounding background, recolours a region's background while leaving borders and labels intact, and copies, moves, rotates or mirrors rectangular regions. It cannot invent photographic content, restyle, relight or shade.

"programmatic_edit" is selectable only while `programmatic_refine_available` is true, which is exactly when the candidate was itself drawn deterministically onto an untouched source, so every pixel outside the drawing is still exact. Whenever it is available and the repair is expressible with those primitives - a marker, stroke or label in the wrong place, a missing line, a wrong digit, a mark that must be erased, content that must be moved, rotated or mirrored - choose it: it fixes the defect at no cost to everything already correct, whereas a diffusion repaint of such a candidate destroys the exact fidelity that made it good in the first place. Choose "generative" for such a candidate only when the repair genuinely needs new realistic imagery or a change no primitive can express, and prefer replan over a diffusion repaint whenever replan is allowed.
When `programmatic_refine_available` is false, `refine_with` must be "generative".

`refinement_instruction` is always required for refine. For "generative" it is fed verbatim to the image editor as the whole prompt: write it as one or two terse imperative sentences naming only the concrete visual delta (what to change, where, to what), with no preamble, no restatement of the task and no preservation boilerplate. For "programmatic_edit" write the same terse delta; it is used as the fallback prompt if your drawing program turns out to be unusable.

When you choose "programmatic_edit", also emit `render`, a drawing program applied to the candidate image shown to you:
- Coordinates are normalised fractions of the candidate: x=0 is its left edge, x=1 the right edge, y=0 the top edge, y=1 the bottom edge. Scalars (`width`, `radius`, `size`, `length`) are fractions of the shorter side; sensible defaults are width 0.008 and size 0.05.
- Every `op` name and every field must appear verbatim in the `render_schema` supplied with the task; an unlisted op or a renamed field is rejected outright. All bounding boxes are [x0, y0, x1, y1] corner pairs, never [x, y, width, height].
- Drawing only ever adds ink. Whenever the repair requires something already in the candidate to be gone - a mark in the wrong cell, a wrong digit, a stroke that must be rerouted - first `clear_region` over its bounding box, leaving `color` null so the surrounding background is sampled automatically, then draw the replacement on top. Use `recolor_region` instead when borders, digits or labels inside that rectangle must survive.
- `paste_region` copies a rectangle and can rotate it by a quarter turn or mirror it; set `clear_source` true to move rather than duplicate.
- If `previous_render` is supplied it is the program that produced this candidate: reuse its `grids` declaration and its coordinates instead of re-deriving them, and address cells as integer [grid_name, row, col] references.
- Emit the minimum set of operations that repairs the defect, and make sure they change something: a program that leaves the candidate identical is rejected.

Return JSON only:
{
  "action": "pass|refine|replan",
  "analysis": "brief action rationale",
  "refine_with": "generative|programmatic_edit",
  "refinement_instruction": "terse incremental edit; required for refine",
  "render": {"coord_space": "normalized", "grids": {}, "ops": [...]}
}"""

_REFINE_RENDER_REPAIR_SYSTEM = """The drawing program you proposed as a refinement was rejected by the renderer. Read the error and the `render_schema`, then emit a corrected `render` object obeying the same conventions: normalised coordinates over the candidate image, op names and field names taken verbatim from the schema, bounding boxes as [x0, y0, x1, y1] corner pairs.
If the error says the image was left unchanged, your operations drew nothing visible: check that coordinates lie inside [0,1], that widths and sizes are large enough to see, and that the colour differs from what is already at that location. Keep any `clear_region` operations and `clear_source` flags already present, since they remove content the repair requires to be gone.
If the repair cannot be expressed with these primitives at all, return {"abandon": true} and the refinement will be handed to the diffusion editor instead.
Return JSON only: {"render": {"coord_space": "normalized", "ops": [...]}, "abandon": false}"""


_DISCRETIONARY_PASS_FLOOR = 4.0
_QUALITY_PASS_FLOOR = 3.0
_MAX_REFINE_RENDER_REPAIRS = 1
_REFINE_RENDER_TIMEOUT = 300.0
_REFINE_RENDER_MAX_TOKENS = 16384


def _pass_floors(item: dict[str, Any]) -> dict[str, float | None]:
    """Lowest score per dimension that still leaves `pass` on the table.

    At or above the floor the controller may accept the candidate; below it the loop
    keeps working no matter what the controller asks for. 4 is the lowest rubric band
    that still reads as "one minor difference", i.e. no conspicuous defect. A logical
    sample is scored 0/1 on every dimension, where "nearly right" does not exist, so
    its floor stays at the maximum.
    """
    logical = is_logical(item)
    floor = 1.0 if logical else _DISCRETIONARY_PASS_FLOOR
    quality_floor = 1.0 if logical else _QUALITY_PASS_FLOOR
    return {
        "reasoning_score": floor,
        "consistency_score": None if is_enabled(item, "consistency_free") else floor,
        "quality_score": None if logical else quality_floor,
    }


class Verifier:
    def __init__(
        self,
        client: OpenLuxClient,
        model: str,
        renderer: ImageRenderer | None = None,
    ) -> None:
        self.client = client
        self.model = model
        # The same renderer the planner and runner use, so a drawing program accepted
        # here is exactly the program the runner can execute later.
        self.renderer = renderer or ImageRenderer()

    @staticmethod
    def _source_content(
        source_images: list[Image.Image], base_image_index: int | None = None
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        single = len(source_images) == 1
        for index, image in enumerate(source_images, 1):
            if single:
                label = "Source image:"
            elif base_image_index is not None and index - 1 == base_image_index:
                label = f"Source image {index} (the canvas this candidate edits):"
            else:
                label = f"Source image {index}:"
            content.extend(image_content(image, label))
        return content

    def _judge_dimension(
        self,
        *,
        system_prompt: str,
        task: dict[str, Any],
        source_images: list[Image.Image],
        candidate: Image.Image,
        lower: float,
        upper: float,
        base_image_index: int | None = None,
    ) -> tuple[float, str, dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": json.dumps(task, ensure_ascii=False)}
        ]
        content.extend(self._source_content(source_images, base_image_index))
        content.extend(image_content(candidate, "Candidate output image:"))
        payload = self.client.json_completion(
            model=self.model,
            system_prompt=system_prompt,
            content=content,
        )
        try:
            score = min(upper, max(lower, float(payload.get("score"))))
            analysis = str(payload.get("analysis", ""))
        except (TypeError, ValueError):
            score = lower
            analysis = f"[score parse error: raw value was {payload.get('score')!r}] " + str(payload.get("analysis", ""))
        return score, analysis, payload

    def verify(
        self,
        *,
        item: dict[str, Any],
        source_images: list[Image.Image],
        candidate: Image.Image,
        allowed_actions: Iterable[str],
        source_executor: str,
        base_image_index: int | None = None,
        previous_render: dict[str, Any] | None = None,
    ) -> Verification:
        instructions = as_list(item["instruction"])
        allowed = [action for action in allowed_actions if action in {"pass", "refine", "replan"}]
        if not allowed:
            allowed = ["pass", "refine"]
        logical = is_logical(item)
        consistency_free = is_enabled(item, "consistency_free")
        category_values = [str(value) for value in as_list(item.get("category", ""))]

        consistency_score: float | None = None
        consistency_analysis = "Not applicable (consistency-free sample)."
        quality_score: float | None = None
        quality_analysis = "Not applicable (logical sample)."

        def _judge_consistency() -> None:
            nonlocal consistency_score, consistency_analysis
            consistency_score, consistency_analysis, _ = self._judge_dimension(
                system_prompt=_CONSISTENCY_SYSTEM,
                task={
                    "instructions_in_order": instructions,
                    "logical_scoring": logical,
                    "source_image_count": len(source_images),
                },
                source_images=source_images,
                candidate=candidate,
                lower=0.0 if logical else 1.0,
                upper=1.0 if logical else 5.0,
                base_image_index=base_image_index,
            )

        def _judge_quality() -> None:
            nonlocal quality_score, quality_analysis
            quality_score, quality_analysis, _ = self._judge_dimension(
                system_prompt=_QUALITY_SYSTEM,
                task={"counterfactual": "counterfactual_reasoning" in category_values},
                source_images=source_images,
                candidate=candidate,
                lower=1.0,
                upper=5.0,
                base_image_index=base_image_index,
            )

        reasoning_payload: dict[str, Any] = {}
        reasoning_score: float = 0.0 if logical else 1.0
        reasoning_analysis = ""

        def _judge_reasoning() -> None:
            nonlocal reasoning_score, reasoning_analysis, reasoning_payload
            reasoning_score, reasoning_analysis, reasoning_payload = self._judge_dimension(
                system_prompt=_REASONING_SYSTEM,
                task={
                    "instructions_in_order": instructions,
                    "logical_scoring": logical,
                },
                source_images=source_images,
                candidate=candidate,
                lower=0.0 if logical else 1.0,
                upper=1.0 if logical else 5.0,
                base_image_index=base_image_index,
            )

        tasks = [_judge_reasoning]
        if not consistency_free:
            tasks.append(_judge_consistency)
        if not logical:
            tasks.append(_judge_quality)

        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = [pool.submit(fn) for fn in tasks]
            for fut in futures:
                fut.result()  # re-raise any exception from a scoring call
        completed = reasoning_payload.get("completed_requirements", [])
        missing = reasoning_payload.get("missing_requirements", [])
        if not isinstance(completed, list):
            completed = [str(completed)]
        if not isinstance(missing, list):
            missing = [str(missing)]

        programmatic_refine = source_executor == "programmatic_edit" and "refine" in allowed

        action_task: dict[str, Any] = {
            "allowed_actions": allowed,
            "source_executor": source_executor,
            "programmatic_refine_available": programmatic_refine,
            "max_scores": {
                "reasoning_score": 1 if logical else 5,
                "consistency_score": None if consistency_free else (1 if logical else 5),
                "quality_score": None if logical else 5,
            },
            "pass_floor": _pass_floors(item),
            "scores": {
                "reasoning_score": reasoning_score,
                "consistency_score": consistency_score,
                "quality_score": quality_score,
            },
            "findings": {
                "reasoning": reasoning_analysis,
                "expected_result": str(reasoning_payload.get("expected_result", "")),
                "consistency": consistency_analysis,
                "quality": quality_analysis,
                "completed_requirements": completed,
                "missing_requirements": missing,
            },
        }
        if programmatic_refine:
            action_task["render_schema"] = self.renderer.protocol()
            if previous_render is not None:
                action_task["previous_render"] = previous_render
        action_content: list[dict[str, Any]] = [
            {"type": "text", "text": json.dumps(action_task, ensure_ascii=False)}
        ]
        if programmatic_refine:
            action_content.extend(
                image_content(
                    candidate,
                    "Candidate output image (the canvas a drawing program would be applied to):",
                )
            )
        action_payload, action_note = self._choose_action(
            action_content, action_task, programmatic_refine
        )

        if (
            str(action_payload.get("action", "")).lower() == "pass"
            and set(allowed) != {"pass"}
        ):
            floors = _pass_floors(item)
            acceptable = all(
                floor is None or (value is not None and value >= floor)
                for floor, value in (
                    (floors["reasoning_score"], reasoning_score),
                    (floors["consistency_score"], consistency_score),
                    (floors["quality_score"], quality_score),
                )
            )
            if not acceptable:
                retry_allowed = [a for a in allowed if a != "pass"]
                retry_task = {**action_task, "allowed_actions": retry_allowed}
                retry_content: list[dict[str, Any]] = [
                    {"type": "text", "text": json.dumps(retry_task, ensure_ascii=False)}
                ]
                if programmatic_refine:
                    retry_content.extend(
                        image_content(
                            candidate,
                            "Candidate output image (the canvas a drawing program would be applied to):",
                        )
                    )
                action_payload, action_note = self._choose_action(
                    retry_content, retry_task, programmatic_refine
                )

        refinement_executor = "generative"
        refinement_render: dict[str, Any] | None = None
        render_note = action_note
        wants_programmatic = (
            str(action_payload.get("action", "")).lower() == "refine"
            and str(action_payload.get("refine_with", "generative")).lower() == "programmatic_edit"
        )
        if programmatic_refine and wants_programmatic:
            refinement_render, validation_note = self._validate_refine_render(
                payload=action_payload,
                candidate=candidate,
                action_task=action_task,
            )
            if refinement_render is not None:
                refinement_executor = "programmatic_edit"
            render_note = "; ".join(part for part in (render_note, validation_note) if part)

        combined_payload = {
            "action": action_payload.get("action"),
            "reasoning_score": reasoning_score,
            "consistency_score": consistency_score,
            "quality_score": quality_score,
            "analysis": (
                f"Action: {action_payload.get('analysis', '')}\n"
                f"Reasoning: {reasoning_analysis}\n"
                f"Consistency: {consistency_analysis}\n"
                f"Quality: {quality_analysis}"
                + (f"\nRefinement: {render_note}" if render_note else "")
            ),
            "completed_requirements": completed,
            "missing_requirements": missing,
            "refinement_instruction": action_payload.get("refinement_instruction", ""),
            "refinement_executor": refinement_executor,
            "refinement_render": refinement_render,
        }
        return self._parse(combined_payload, item, allowed)

    def _choose_action(
        self,
        action_content: list[dict[str, Any]],
        action_task: dict[str, Any],
        programmatic_refine: bool,
    ) -> tuple[dict[str, Any], str]:
        """Pick the loop action, retrying without the drawing-program option if needed.

        Asking for a `render` alongside the decision is what makes a programmatic
        refinement possible, but it is also what makes the answer long enough to hit the
        output ceiling: a 25-cell permutation costs thousands of tokens of hidden
        reasoning before the first visible character. When that happens the decision
        itself is still perfectly answerable, so the option is withdrawn and the same
        question asked again. Losing a programmatic refinement is a far smaller cost than
        losing the sample, which is exactly what an exception here would do - the runner
        treats a failed verify as a failed sample.
        """
        try:
            payload = self.client.json_completion(
                model=self.model,
                system_prompt=_ACTION_SYSTEM,
                content=action_content,
                timeout=_REFINE_RENDER_TIMEOUT if programmatic_refine else None,
                max_tokens=_REFINE_RENDER_MAX_TOKENS if programmatic_refine else 4096,
            )
        except Exception as exc:
            if not programmatic_refine:
                raise
            plain_task = {
                key: value
                for key, value in action_task.items()
                if key not in {"render_schema", "previous_render"}
            }
            plain_task["programmatic_refine_available"] = False
            payload = self.client.json_completion(
                model=self.model,
                system_prompt=_ACTION_SYSTEM,
                content=[{"type": "text", "text": json.dumps(plain_task, ensure_ascii=False)}],
            )
            return payload, (
                "programmatic refinement not offered, the drawing-program request failed: "
                f"{type(exc).__name__}: {exc}"
            )
        return payload, ""

    def _validate_refine_render(
        self,
        *,
        payload: dict[str, Any],
        candidate: Image.Image,
        action_task: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str]:
        """Accept the controller's drawing program only if it actually renders.

        Validation is a real render against the very candidate the runner will refine,
        and the renderer is deterministic, so a program accepted here cannot fail later.
        The rendered pixels are discarded: producing the refinement is the runner's job,
        and returning an image from the verifier would put a non-serialisable object into
        the trace. On failure the caller downgrades to the diffusion editor, so a bad
        program costs one repair call rather than the whole turn.
        """
        request = payload.get("render")
        errors: list[str] = []
        for attempt in range(_MAX_REFINE_RENDER_REPAIRS + 1):
            if isinstance(request, dict):
                try:
                    self.renderer.render([candidate], request, base_index=0)
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
                else:
                    return request, (
                        f"programmatic refinement accepted after repair: {' | '.join(errors)}"
                        if errors
                        else ""
                    )
            else:
                errors.append("controller chose programmatic_edit but returned no 'render' object")
            if attempt == _MAX_REFINE_RENDER_REPAIRS:
                break
            repaired = self._repair_refine_render(
                rejected=request,
                error=errors[-1],
                candidate=candidate,
                action_task=action_task,
            )
            if repaired is None:
                break
            request = repaired
        return None, (
            "programmatic refinement rejected, downgraded to generative: " + " | ".join(errors)
        )

    def _repair_refine_render(
        self,
        *,
        rejected: Any,
        error: str,
        candidate: Image.Image,
        action_task: dict[str, Any],
    ) -> dict[str, Any] | None:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "rejected_render": rejected,
                        "error": error,
                        "render_schema": action_task.get("render_schema"),
                        "previous_render": action_task.get("previous_render"),
                        "findings": action_task.get("findings"),
                    },
                    ensure_ascii=False,
                ),
            }
        ]
        content.extend(
            image_content(candidate, "Candidate output image (the canvas to draw on):")
        )
        try:
            payload = self.client.json_completion(
                model=self.model,
                system_prompt=_REFINE_RENDER_REPAIR_SYSTEM,
                content=content,
                timeout=_REFINE_RENDER_TIMEOUT,
                max_tokens=_REFINE_RENDER_MAX_TOKENS,
            )
        except Exception:
            return None
        if payload.get("abandon"):
            return None
        repaired = payload.get("render")
        return repaired if isinstance(repaired, dict) else None

    @staticmethod
    def _parse(payload: dict[str, Any], item: dict[str, Any], allowed: list[str]) -> Verification:
        logical = is_logical(item)
        consistency_free = is_enabled(item, "consistency_free")
        upper, lower = (1.0, 0.0) if logical else (5.0, 1.0)

        def score(name: str, nullable: bool = False) -> float | None:
            raw = payload.get(name)
            if nullable and raw is None:
                return None
            try:
                return min(upper, max(lower, float(raw)))
            except (TypeError, ValueError):
                return None if nullable else lower

        reasoning = score("reasoning_score")
        consistency = score("consistency_score", nullable=consistency_free)
        quality = None if logical else score("quality_score")
        assert reasoning is not None
        if logical:
            consistency = 0.0 if consistency is None else consistency
            overall = 0.3 * (1 + 4 * consistency) + 0.7 * (1 + 4 * reasoning)
        elif consistency_free:
            assert quality is not None
            overall = 0.8 * reasoning + 0.2 * quality
        else:
            consistency = lower if consistency is None else consistency
            assert quality is not None
            overall = 0.3 * consistency + 0.5 * reasoning + 0.2 * quality
        if reasoning == lower:
            overall = max(1.0, overall * 0.5)

        complete = reasoning == upper
        if not consistency_free:
            complete = complete and consistency == upper
        if not logical:
            complete = complete and quality == upper

        # Two distinct gates: `complete` ends the loop on its own, while `acceptable`
        # merely grants the controller permission to stop if it thinks the candidate is
        # good enough. Everything below the floor keeps being worked on regardless of
        # what the controller asked for.
        floors = _pass_floors(item)
        acceptable = all(
            floor is None or (value is not None and value >= floor)
            for floor, value in (
                (floors["reasoning_score"], reasoning),
                (floors["consistency_score"], consistency),
                (floors["quality_score"], quality),
            )
        )

        requested = str(payload.get("action", "refine")).lower()
        if requested not in {"pass", "refine", "replan"}:
            requested = "refine"

        pass_rule: str | None = None
        if complete:
            action, pass_rule = "pass", "complete"
        elif requested == "pass" and acceptable and "pass" in allowed:
            action, pass_rule = "pass", "discretionary"
        elif requested == "pass" and not acceptable and set(allowed) == {"pass"}:
            action, pass_rule = "pass", "budget_exhausted"
        else:
            preferred = "refine" if requested == "pass" else requested
            action = (
                preferred
                if preferred in allowed
                else next(
                    (option for option in ("refine", "replan") if option in allowed),
                    preferred,
                )
            )
        refinement = str(payload.get("refinement_instruction", "")).strip()
        missing = payload.get("missing_requirements", [])
        if not isinstance(missing, list):
            missing = [str(missing)]

        request: RefinementRequest | None = None
        if action == "refine":
            if not refinement:
                refinement = "Fix only: " + "; ".join(str(value) for value in missing)
            render = payload.get("refinement_render")
            if (
                requested == "refine"
                and str(payload.get("refinement_executor", "generative")) == "programmatic_edit"
                and isinstance(render, dict)
            ):
                request = RefinementRequest(refinement, "programmatic_edit", render)
            else:
                request = RefinementRequest(refinement, "generative", None)

        completed = payload.get("completed_requirements", [])
        if not isinstance(completed, list):
            completed = [str(completed)]
        return Verification(
            action=action,  # type: ignore[arg-type]
            reasoning_score=reasoning,
            consistency_score=consistency,
            quality_score=quality,
            overall_score=round(overall, 6),
            analysis=str(payload.get("analysis", "")),
            completed_requirements=[str(value) for value in completed],
            missing_requirements=[str(value) for value in missing],
            refinement_request=request,
            pass_rule=pass_rule,
            requested_action=requested,
        )
