from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image


class Flux2KleinExecutor:
    def __init__(
        self,
        model_path: str | Path,
        *,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
        cpu_offload: bool = False,
    ) -> None:
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("guidance_scale must be a finite non-negative number")
        self.model_path = Path(model_path)
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.cpu_offload = cpu_offload
        self._pipe: Any = None
        self._torch: Any = None

    def _load(self) -> None:
        if self._pipe is not None:
            return
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Local FLUX.2 model directory not found: {self.model_path}")
        try:
            import torch
            from diffusers import Flux2KleinPipeline
        except ImportError as exc:
            raise RuntimeError(
                "FLUX.2 Klein requires the versions in `agent/requirements.txt`"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("FLUX.2-klein-9B execution requires a CUDA device")
        self._torch = torch
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self._pipe = Flux2KleinPipeline.from_pretrained(
            str(self.model_path),
            torch_dtype=dtype,
            local_files_only=True,
        )
        if self.cpu_offload:
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe.to("cuda")

    def generate(self, prompt: str, images: list[Image.Image], seed: int) -> Image.Image:
        self._load()
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Edit prompt cannot be empty")
        if not images:
            raise ValueError("At least one source image is required for editing")
        input_image: Image.Image | list[Image.Image]
        input_image = images[0] if len(images) == 1 else images
        result = self._pipe(
            prompt=prompt,
            image=input_image,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=self._torch.Generator(device="cpu").manual_seed(seed),
        )
        if not getattr(result, "images", None):
            raise RuntimeError("FLUX.2 pipeline returned no images")
        return result.images[0].convert("RGB")
