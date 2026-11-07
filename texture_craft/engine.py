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

# Magenta background used during model training to represent transparent alpha
BG = np.array([255, 0, 255], dtype=np.int16)
DEFAULT_MODEL_DIR = pathlib.Path("models/minecraft-item-16px")
REPO_ID = "yuuki14202028/minecraft-item-16px"


class TextureEngine:
    _instance: Optional["TextureEngine"] = None

    def __init__(self, model_path: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model_path = pathlib.Path(model_path) if model_path else DEFAULT_MODEL_DIR
        self.pipe: Optional[StableDiffusionPipeline] = None
        self._load_pipeline()

    @classmethod
    def get_instance(cls, model_path: Optional[str] = None) -> "TextureEngine":
        """Singleton accessor to ensure model stays in GPU VRAM across calls."""
        if cls._instance is None:
            cls._instance = cls(model_path)
        return cls._instance

    def _load_pipeline(self) -> None:
        """Loads model from local directory or pulls from Hugging Face if missing."""
        hf_token = os.environ.get("HF_TOKEN", None)

        if self.model_path.exists() and any(self.model_path.iterdir()):
            source = str(self.model_path)
            token = None
        else:
            source = REPO_ID
            token = hf_token

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

        # Save to local directory if downloaded from HF
        if source == REPO_ID:
            self.model_path.mkdir(parents=True, exist_ok=True)
            self.pipe.save_pretrained(str(self.model_path))

        print(f"[TextureEngine] Pipeline ready on {self.device}!", flush=True)

    @staticmethod
    def fold_to_tile(image: Image.Image, size: int = 16) -> Image.Image:
        """Folds 256px generated output back to size x size pixel art via block averaging."""
        a = np.asarray(image.convert("RGB"), dtype=np.float32)
        c = a.shape[0] // size
        return Image.fromarray(
            a.reshape(size, c, size, c, 3)
            .mean(axis=(1, 3))
            .round()
            .clip(0, 255)
            .astype(np.uint8)
        )

    @staticmethod
    def key_out(tile: Image.Image, tol: int = 100) -> Image.Image:
        """Chroma-keys magenta background (255, 0, 255) into real RGBA transparency."""
        rgb = np.asarray(tile.convert("RGB"), dtype=np.int16)
        bg = np.abs(rgb - BG).sum(axis=2) <= tol
        out = np.dstack([rgb.astype(np.uint8), np.where(bg, 0, 255).astype(np.uint8)])
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
        """Generates a single 16x16 Minecraft item texture.

        Returns: (16x16 transparent tile, 256x256 raw image, seed, latency_seconds)
        """
        if seed is None or seed < 0:
            seed = random.randint(0, 2**31 - 1)

        t0 = time.time()
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        
        raw = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            height=int(res),
            width=int(res),
            generator=generator,
        ).images[0]

        tile = self.key_out(self.fold_to_tile(raw, size=tile_size))
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
        """Generates multiple variants of a prompt."""
        if seed_base is None or seed_base < 0:
            seed_base = random.randint(0, 2**30 - 1)

        results = []
        for i in range(max(1, int(variants))):
            s = seed_base + (i * 1000)
            tile, raw, seed, latency = self.generate_single(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=s,
                steps=steps,
                guidance_scale=guidance_scale,
            )
            results.append({
                "variant_index": i + 1,
                "seed": seed,
                "latency_sec": latency,
                "tile_image": tile,
                "raw_image": raw,
                "base64_tile": self.image_to_base64(tile),
                "data_uri_tile": self.image_to_data_uri(tile),
            })
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
