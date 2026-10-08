from __future__ import annotations

import base64
import io
import json
import re
import time
from pathlib import Path
from typing import Any

from PIL import Image


def image_to_data_url(image: Image.Image, max_side: int = 1536) -> str:
    image = image.convert("RGB")
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def image_content(image: Image.Image, label: str | None = None) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if label:
        content.append({"type": "text", "text": label})
    content.append(
        {
            "type": "image_url",
            "image_url": {"url": image_to_data_url(image), "detail": "high"},
        }
    )
    return content


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model response does not contain a JSON object")
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Model response JSON must be an object")
    return value


def _is_timeout(error: Exception) -> bool:
    try:
        from openai import APITimeoutError
    except ImportError:
        return "timed out" in str(error).lower()
    return isinstance(error, APITimeoutError) or "timed out" in str(error).lower()


def _is_bad_request(error: Exception) -> bool:
    """Whether the server rejected the request as malformed (HTTP 4xx other than 429)."""
    try:
        from openai import BadRequestError, UnprocessableEntityError
    except ImportError:
        return False
    return isinstance(error, (BadRequestError, UnprocessableEntityError))


class TruncatedResponse(ValueError):
    """The model hit its output ceiling mid-answer, so the JSON is incomplete.

    Worth its own type because the remedy is the opposite of every other failure's:
    the request was fine and must not be retried unchanged, it simply needs more room.
    """


# Ceiling for the automatic budget escalation below.
_MAX_OUTPUT_TOKENS = 16384

class OpenLuxClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openlux.ai/v1",
        timeout: float = 120.0,
        retries: int = 3,
    ) -> None:
        if not api_key:
            raise ValueError("API_KEY is required")
        if retries < 1:
            raise ValueError("retries must be positive")
        base_url = base_url.rstrip("/")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the `openai` package before running the agent") from exc
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.retries = retries
        self._use_response_format = True

    def json_completion(
        self,
        *,
        model: str,
        system_prompt: str,
        content: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if not model:
            raise ValueError("Planner/verifier model name is required")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        api = self._client.with_options(timeout=timeout) if timeout is not None else self._client
        last_error: Exception | None = None
        attempt = 0
        budget = max_tokens
        while attempt < self.retries:
            attempt += 1
            try:
                kwargs = dict(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=budget,
                )
                if self._use_response_format:
                    try:
                        response = api.chat.completions.create(
                            **kwargs, response_format={"type": "json_object"}
                        )
                    except Exception as exc:
                        if _is_timeout(exc) or not _is_bad_request(exc):
                            raise
                        self._use_response_format = False
                        response = api.chat.completions.create(**kwargs)
                else:
                    response = api.chat.completions.create(**kwargs)
                choice = response.choices[0]
                text = choice.message.content
                if choice.finish_reason == "length":
                    raise TruncatedResponse(
                        f"Model response was cut off at max_tokens={budget} "
                        f"(finish_reason=length, {len(text or '')} visible characters)"
                    )
                if not text:
                    raise ValueError("Model returned an empty response")
                return extract_json(text)
            except Exception as exc:
                last_error = exc
                if _is_timeout(exc) or attempt >= self.retries:
                    break
                if isinstance(exc, TruncatedResponse):
                    if budget >= _MAX_OUTPUT_TOKENS:
                        break
                    budget = min(_MAX_OUTPUT_TOKENS, budget * 2)
                    continue
                time.sleep(2 ** (attempt - 1))
        assert last_error is not None
        raise RuntimeError(
            f"API request failed after {attempt} attempt(s): "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error


def load_local_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()
