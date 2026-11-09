import base64
import io
import json

import pytest
from PIL import Image

import app
from texture_craft.engine import TextureEngine, torch


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        (("", 1, -1, 30, 9.0), "Prompt"),
        (("x" * 301, 1, -1, 30, 9.0), "Prompt"),
        (("sword", 0, -1, 30, 9.0), "Variants"),
        (("sword", 5, -1, 30, 9.0), "Variants"),
        (("sword", 1, -2, 30, 9.0), "Seed"),
        (("sword", 1, 2**31, 30, 9.0), "Seed"),
        (("sword", 1, -1, 9, 9.0), "Steps"),
        (("sword", 1, -1, 30, 16.0), "Guidance scale"),
        (("sword\u0000", 1, -1, 30, 9.0), "control characters"),
    ],
)
def test_generation_input_validation(inputs, message):
    with pytest.raises(ValueError, match=message):
        app.validate_generation_inputs(*inputs)


def test_generation_returns_pngs_without_exposing_server_paths(monkeypatch):
    def fake_generate(prompt, variants, seed, steps, guidance_scale):
        return [
            {
                "tile_image": Image.new("RGBA", (16, 16), (12, 34, 56, 0)),
                "variant_index": index,
                "seed": 123 + index,
                "latency_sec": 0.1,
            }
            for index in range(1, variants + 1)
        ]

    monkeypatch.setattr(app, "_generate_on_gpu", fake_generate)

    response = app.generate_textures("  ruby sword  ", variants=2, seed=123)

    assert response["prompt"] == "ruby sword"
    assert response["count"] == 2
    for texture in response["textures"]:
        png = base64.b64decode(texture["base64_png"], validate=True)
        image = Image.open(io.BytesIO(png))
        assert texture["filename"].startswith("texture-")
        assert texture["mime_type"] == "image/png"
        assert texture["data_uri"].startswith("data:image/png;base64,")
        assert image.size == (16, 16)
        assert image.mode == "RGBA"
    assert "path" not in json.dumps(response).lower()
    assert "texture-craft-" not in json.dumps(response)


def test_gradio_api_exposes_typed_generation_tool():
    api_info = app.demo.get_api_info()
    endpoints = api_info["named_endpoints"]
    endpoint = next(
        value for name, value in endpoints.items() if "generate_textures" in name
    )
    parameters = {parameter["parameter_name"] for parameter in endpoint["parameters"]}

    assert parameters == {"prompt", "variants", "seed", "steps", "guidance_scale"}
    assert endpoint["description"]


def test_engine_can_select_zerogpu_cuda_without_availability_probe(monkeypatch):
    monkeypatch.setattr(TextureEngine, "_load_pipeline", lambda self: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    engine = TextureEngine(device="cuda")

    assert engine.device == "cuda"
    assert engine.dtype == torch.float16


def test_downsampling_preserves_hard_pixels_instead_of_averaging():
    source = Image.new("RGBA", (32, 32))
    for y in range(32):
        for x in range(32):
            color = (255, 255, 255, 255) if (x + y) % 2 else (0, 0, 0, 255)
            source.putpixel((x, y), color)

    tile = TextureEngine.fold_to_tile(source)
    expected = source.resize((16, 16), Image.Resampling.NEAREST)
    tile_pixels = [
        tile.getpixel((x, y))
        for y in range(tile.height)
        for x in range(tile.width)
    ]
    expected_pixels = [
        expected.getpixel((x, y))
        for y in range(expected.height)
        for x in range(expected.width)
    ]

    assert tile.size == (16, 16)
    assert tile_pixels == expected_pixels
    assert set(tile_pixels) <= {(0, 0, 0, 255), (255, 255, 255, 255)}
