"""Core generation engine for Texture-Craft."""

import base64
import io
import os
import pathlib
import random
import time
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Magenta background used during model training to represent transparent alpha.
BG = np.array([255, 0, 255], dtype=np.int16)
DEFAULT_MODEL_DIR = pathlib.Path("models/minecraft-item-16px")
REPO_ID = "yuuki14202028/minecraft-item-16px"


class TextureEngine:
    _instance: Optional["TextureEngine"] = None

    def __init__(
        self,
        model_path: Optional[str] = None,
        cache_downloaded_model: bool = True,
        device: Optional[str] = None,
    ):
        if device not in (None, "cuda", "cpu"):
            raise ValueError("device must be 'cuda', 'cpu', or None.")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model_path = pathlib.Path(model_path) if model_path else DEFAULT_MODEL_DIR
        self.cache_downloaded_model = cache_downloaded_model
        self.pipe: Optional[StableDiffusionPipeline] = None
        self._load_pipeline()

    @classmethod
    def get_instance(
        cls,
        model_path: Optional[str] = None,
        cache_downloaded_model: bool = True,
        device: Optional[str] = None,
    ) -> "TextureEngine":
        """Return the singleton engine instance, creating it once if needed."""
        if cls._instance is None:
            cls._instance = cls(model_path, cache_downloaded_model, device)
        return cls._instance

    def _resolve_model_source(self) -> tuple[str, Optional[str]]:
        if self.model_path.exists() and any(self.model_path.iterdir()):
            return str(self.model_path), None
        return REPO_ID, os.environ.get("HF_TOKEN")

    def _load_pipeline(self) -> None:
        """Load the model from a local cache or download it from Hugging Face."""
        source, token = self._resolve_model_source()

        print(f"[TextureEngine] Loading pipeline on {self.device} from '{source}'...", flush=True)
        pipe = StableDiffusionPipeline.from_pretrained(
            source,
            token=token,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

        if source == REPO_ID and self.cache_downloaded_model:
            self.model_path.mkdir(parents=True, exist_ok=True)
            self.pipe.save_pretrained(str(self.model_path))

        print(f"[TextureEngine] Pipeline ready on {self.device}!", flush=True)

    @staticmethod
    def fold_to_tile(image: Image.Image, size: int = 16) -> Image.Image:
        """Downsample generated output while preserving transparency."""
        has_alpha = image.mode == "RGBA"
        channels = 4 if has_alpha else 3
        a = np.asarray(image.convert("RGBA" if has_alpha else "RGB"), dtype=np.float32)
        c = a.shape[0] // size
        blocks = a.reshape(size, c, size, c, channels)
        if not has_alpha:
            pixels = blocks.mean(axis=(1, 3))
        else:
            alpha = blocks[..., 3:4] / 255
            alpha_mean = alpha.mean(axis=(1, 3))
            weighted_rgb = (blocks[..., :3] * alpha).mean(axis=(1, 3))
            rgb = np.divide(
                weighted_rgb,
                alpha_mean,
                out=np.zeros_like(weighted_rgb),
                where=alpha_mean > 0,
            )
            pixels = np.concatenate((rgb, alpha_mean * 255), axis=2)

        return Image.fromarray(pixels.round().clip(0, 255).astype(np.uint8))

    @staticmethod
    def key_out(tile: Image.Image, tol: int = 100) -> Image.Image:
        """Remove magenta and uniform edge backgrounds without cutting into sprites."""
        rgb = np.asarray(tile.convert("RGB"), dtype=np.int16)
        height, width = rgb.shape[:2]
        magenta = np.abs(rgb - BG).sum(axis=2) <= tol

        corner_size = max(1, min(16, height // 4, width // 4))
        corners = np.stack(
            (
                rgb[:corner_size, :corner_size].reshape(-1, 3),
                rgb[:corner_size, -corner_size:].reshape(-1, 3),
                rgb[-corner_size:, :corner_size].reshape(-1, 3),
                rgb[-corner_size:, -corner_size:].reshape(-1, 3),
            )
        )
        corner_colors = np.median(corners, axis=1)
        corner_distances = np.abs(
            corner_colors[:, None, :] - corner_colors[None, :, :]
        ).max(axis=2)
        matching_corners = corner_distances <= 24
        background_index = int(np.argmax(matching_corners.sum(axis=1)))
        background_color = corner_colors[background_index]
        uniform_background = matching_corners[background_index].sum() >= 2

        transparent = magenta
        if uniform_background:
            background = np.max(np.abs(rgb - background_color), axis=2) <= 20
            connected = np.zeros((height, width), dtype=bool)
            pending = [
                (0, 0),
                (0, width - 1),
                (height - 1, 0),
                (height - 1, width - 1),
            ]
            while pending:
                row, col = pending.pop()
                if connected[row, col] or not background[row, col]:
                    continue
                connected[row, col] = True
                if row > 0:
                    pending.append((row - 1, col))
                if row + 1 < height:
                    pending.append((row + 1, col))
                if col > 0:
                    pending.append((row, col - 1))
                if col + 1 < width:
                    pending.append((row, col + 1))
            if connected.mean() >= 0.15:
                transparent |= connected

        out = np.dstack(
            [rgb.astype(np.uint8), np.where(transparent, 0, 255).astype(np.uint8)]
        )
        return Image.fromarray(out)

    @staticmethod
    def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
        """Encodes PIL image to standard base64 string."""
        buffered = io.BytesIO()
        image.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @staticmethod
    def image_to_data_uri(image: Image.Image, format: str = "PNG") -> str:
        """Encodes PIL image to browser-renderable data URI."""
        b64 = TextureEngine.image_to_base64(image, format=format)
        return f"data:image/{format.lower()};base64,{b64}"

    @staticmethod
    def contact_sheet(
        images: List[Image.Image], cols: int = 8, cell: int = 64, checker: bool = True
    ) -> Image.Image:
        """Assembles a list of 16x16 tiles onto a checkered preview sheet."""
        if not images:
            return Image.new("RGBA", (cell, cell), (0, 0, 0, 0))
        rows = (len(images) + cols - 1) // cols
        sheet = Image.new("RGBA", (cols * cell, rows * cell), (32, 32, 36, 255))
        if checker:
            c = cell // 8
            for y in range(0, sheet.height, c):
                for x in range(0, sheet.width, c):
                    if (x // c + y // c) % 2:
                        sheet.paste((54, 54, 60, 255), (x, y, x + c, y + c))
        for i, im in enumerate(images):
            tile = im.convert("RGBA").resize((cell, cell), Image.Resampling.NEAREST)
            sheet.alpha_composite(tile, ((i % cols) * cell, (i // cols) * cell))
        return sheet

    @staticmethod
    def _normalize_seed(seed: Optional[int], *, default_max: int = 2**31 - 1) -> int:
        if seed is None or seed < 0:
            return random.randint(0, default_max)
        return int(seed)

    def generate_single(
        self,
        prompt: str,
        negative_prompt: str = "",
        seed: Optional[int] = None,
        steps: int = 30,
        guidance_scale: float = 9.0,
        res: int = 256,
        tile_size: int = 16,
    ) -> Tuple[Image.Image, Image.Image, int, float]:
        """Generate a single 16x16 Minecraft item texture.

        Returns: (tile, raw_image, seed, latency_seconds)
        """
        if self.pipe is None:
            raise RuntimeError("Texture pipeline is not initialized.")

        seed = self._normalize_seed(seed)
        t0 = time.time()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        raw = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            height=int(res),
            width=int(res),
            generator=generator,
        ).images[0]

        tile = self.fold_to_tile(self.key_out(raw), size=tile_size)
        latency = round(time.time() - t0, 3)
        return tile, raw, seed, latency

    def generate_batch(
        self,
        prompt: str,
        variants: int = 1,
        negative_prompt: str = "",
        seed_base: Optional[int] = None,
        steps: int = 30,
        guidance_scale: float = 9.0,
    ) -> List[Dict[str, Any]]:
        """Generate multiple variants for a single prompt."""
        variant_count = max(1, int(variants))
        base_seed = self._normalize_seed(seed_base, default_max=2**30 - 1)

        results: List[Dict[str, Any]] = []
        for index in range(variant_count):
            seed = base_seed + (index * 1000)
            tile, raw, actual_seed, latency = self.generate_single(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                steps=steps,
                guidance_scale=guidance_scale,
            )
            results.append(
                {
                    "variant_index": index + 1,
                    "seed": actual_seed,
                    "latency_sec": latency,
                    "tile_image": tile,
                    "raw_image": raw,
                    "base64_tile": self.image_to_base64(tile),
                    "data_uri_tile": self.image_to_data_uri(tile),
                }
            )
        return results

    def get_system_status(self) -> Dict[str, Any]:
        """Returns hardware and engine status."""
        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else None
        vram_total_gb = None
        vram_allocated_gb = None

        if cuda_ok:
            props = torch.cuda.get_device_properties(0)
            vram_total_gb = round(props.total_memory / (1024**3), 2)
            vram_allocated_gb = round(torch.cuda.memory_allocated(0) / (1024**3), 2)

        return {
            "status": "ready" if self.pipe is not None else "initializing",
            "device": self.device,
            "cuda_available": cuda_ok,
            "gpu_name": gpu_name,
            "vram_total_gb": vram_total_gb,
            "vram_allocated_gb": vram_allocated_gb,
            "model_path": str(self.model_path),
        }
