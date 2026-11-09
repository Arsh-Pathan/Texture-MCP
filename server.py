"""FastAPI-native REST server for Texture-Craft."""

from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from texture_craft.engine import TextureEngine

app = FastAPI(
    title="Texture-Craft API",
    description="Generate 16x16 Minecraft item textures from text prompts.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    device: str
    cuda_available: bool
    gpu_name: Optional[str] = None
    vram_total_gb: Optional[float] = None
    vram_allocated_gb: Optional[float] = None
    model_path: str


class GenerateTextureRequest(BaseModel):
    prompt: str = Field(..., description="Item prompt, e.g. 'ruby broadsword', 'frost potion'")
    negative_prompt: str = Field("", description="Negative prompt (leave empty for best results)")
    variants: int = Field(1, ge=1, le=8, description="Number of variants to generate (1-8)")
    seed: Optional[int] = Field(None, description="Random seed (-1 or null for random)")
    steps: int = Field(30, ge=10, le=50, description="DPM-Solver diffusion steps")
    guidance_scale: float = Field(9.0, ge=3.0, le=15.0, description="Classifier-free guidance scale")


class TextureVariant(BaseModel):
    variant_index: int
    seed: int
    latency_sec: float
    base64_png: str
    data_uri: str


class GenerateTextureResponse(BaseModel):
    prompt: str
    count: int
    results: List[TextureVariant]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """System health check and GPU telemetry."""
    engine = TextureEngine.get_instance()
    return engine.get_system_status()


@app.post("/v1/textures/generate", response_model=GenerateTextureResponse)
def generate_textures(req: GenerateTextureRequest) -> GenerateTextureResponse:
    """Generate 16x16 transparent PNG textures as base64."""
    engine = TextureEngine.get_instance()
    variants = engine.generate_batch(
        prompt=req.prompt,
        variants=req.variants,
        negative_prompt=req.negative_prompt,
        seed_base=req.seed,
        steps=req.steps,
        guidance_scale=req.guidance_scale,
    )

    results = [
        TextureVariant(
            variant_index=v["variant_index"],
            seed=v["seed"],
            latency_sec=v["latency_sec"],
            base64_png=v["base64_tile"],
            data_uri=v["data_uri_tile"],
        )
        for v in variants
    ]
    return GenerateTextureResponse(prompt=req.prompt, count=len(results), results=results)


def main() -> None:
    import uvicorn

    TextureEngine.get_instance()
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
