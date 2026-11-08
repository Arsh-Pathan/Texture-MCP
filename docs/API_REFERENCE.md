# Texture-Craft REST API Reference

The Texture-Craft REST API is built with **FastAPI** to provide instant, GPU-accelerated 16×16 Minecraft texture generation and automated resource pack synthesis for AI Minecraft plugin generators.

- **Base URL:** `http://127.0.0.1:8000`
- **Interactive Swagger Docs:** `http://127.0.0.1:8000/docs`
- **OpenAPI Schema:** `http://127.0.0.1:8000/openapi.json`

---

## 1. Health & GPU Telemetry

### `GET /health`
Returns the status of the engine, CUDA hardware acceleration, and GPU VRAM usage.

#### Response `200 OK`
```json
{
  "status": "ready",
  "device": "cuda",
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 4050 Laptop GPU",
  "vram_total_gb": 6.0,
  "vram_allocated_gb": 1.78,
  "model_path": "models\\minecraft-item-16px"
}
```

---

## 2. In-Memory Texture Generation

### `POST /v1/textures/generate`
Generates one or more 16×16 pixel-art item textures with real alpha transparency, returning them as base64 PNGs.

#### Request Body
```json
{
  "prompt": "enchanted ruby sword",
  "negative_prompt": "",
  "variants": 2,
  "seed": 42,
  "steps": 30,
  "guidance_scale": 9.0
}
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | `string` | *(Required)* | Text description (e.g. `frost dagger`, `blue mana potion`). |
| `negative_prompt` | `string` | `""` | Leave empty for optimal diffusion quality. |
| `variants` | `integer` | `1` | Number of variations to generate (1 to 8). |
| `seed` | `integer` | `null` | Random seed for reproducibility (`null` or `-1` for random). |
| `steps` | `integer` | `30` | DPM-Solver diffusion steps (15–50). |
| `guidance_scale` | `float` | `9.0` | CFG scale (7.0–12.0 recommended). |

#### Response `200 OK`
```json
{
  "prompt": "enchanted ruby sword",
  "count": 2,
  "results": [
    {
      "variant_index": 1,
      "seed": 42,
      "latency_sec": 1.52,
      "base64_png": "iVBORw0KGgoAAAANSUhEUgAA...",
      "data_uri": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
    },
    {
      "variant_index": 2,
      "seed": 1042,
      "latency_sec": 1.48,
      "base64_png": "iVBORw0KGgoAAAANSUhEUgAA...",
      "data_uri": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
    }
  ]
}
```

---

## 3. Resource Pack Initialization

### `POST /v1/resourcepack/init`
Creates a fresh Minecraft resource pack skeleton with `pack.mcmeta` and required subdirectories.

#### Request Body
```json
{
  "pack_dir": "C:/server/plugins/MyPlugin/resourcepack",
  "pack_name": "MyPluginRP",
  "description": "Custom Items for MyPlugin",
  "pack_format": 34
}
```

| Minecraft Version | `pack_format` Value |
|---|---|
| **1.21 – 1.21.1** | `34` (Default) |
| **1.21.2 – 1.21.3** | `42` |
| **1.21.4 and newer** | `46+` |
| **1.20.5 – 1.20.6** | `32` |
| **1.20.2 – 1.20.4** | `22` |

#### Response `200 OK`
```json
{
  "pack_dir": "C:/server/plugins/MyPlugin/resourcepack",
  "pack_name": "MyPluginRP",
  "mcmeta_path": "C:/server/plugins/MyPlugin/resourcepack/pack.mcmeta",
  "pack_format": 34
}
```

---

## 4. Single Item Resource Pack Generation

### `POST /v1/resourcepack/item`
Generates a texture and writes all necessary resource pack files directly to disk:
1. `assets/<namespace>/textures/item/<item_id>.png` (16×16 RGBA texture)
2. `assets/<namespace>/models/item/<item_id>.json` (item model)
3. For pack format 46+ (Minecraft 1.21.4+), `assets/minecraft/items/<base_item>.json` (CustomModelData range dispatch). Older formats use `assets/minecraft/models/item/<base_item>.json` overrides.

#### Request Body
```json
{
  "pack_dir": "C:/server/plugins/MyPlugin/resourcepack",
  "namespace": "myplugin",
  "item_id": "frost_blade",
  "prompt": "frost blade with icy blue runes",
  "base_vanilla_item": "diamond_sword",
  "custom_model_data": 1001,
  "display_name": "Frost Blade",
  "seed": 42
}
```

#### Response `200 OK`
```json
{
  "namespace": "myplugin",
  "item_id": "frost_blade",
  "resource_location": "myplugin:frost_blade",
  "texture_file": "C:/server/plugins/MyPlugin/resourcepack/assets/myplugin/textures/item/frost_blade.png",
  "model_file": "C:/server/plugins/MyPlugin/resourcepack/assets/myplugin/models/item/frost_blade.json",
  "base_vanilla_item": "diamond_sword",
  "custom_model_data": 1001,
  "vanilla_override_file": "C:/server/plugins/MyPlugin/resourcepack/assets/minecraft/items/diamond_sword.json",
  "base64_preview": "iVBORw0KGgoAAAANSUhEUgAA...",
  "seed": 42,
  "latency_sec": 1.54,
  "java_code_snippet": "ItemStack item = new ItemStack(Material.DIAMOND_SWORD);\nItemMeta meta = item.getItemMeta();\nif (meta != null) {\n    meta.setDisplayName(\"§6Frost Blade\");\n    meta.setCustomModelData(1001);\n    item.setItemMeta(meta);\n}"
}
```

---

## 5. Batch Resource Pack Creation

### `POST /v1/resourcepack/batch`
Creates multiple custom textures, models, and CustomModelData mappings in one single call.

#### Request Body
```json
{
  "pack_dir": "C:/server/plugins/MyPlugin/resourcepack",
  "namespace": "myplugin",
  "init_pack_if_missing": true,
  "pack_name": "MyPluginRP",
  "items": [
    {
      "item_id": "flame_sword",
      "prompt": "fiery burning blade",
      "base_vanilla_item": "diamond_sword",
      "custom_model_data": 1001,
      "display_name": "Flame Sword"
    },
    {
      "item_id": "frost_sword",
      "prompt": "ice sword with crystalline blade",
      "base_vanilla_item": "diamond_sword",
      "custom_model_data": 1002,
      "display_name": "Frost Sword"
    },
    {
      "item_id": "mana_potion",
      "prompt": "cyan mana elixir potion flask",
      "base_vanilla_item": "potion",
      "custom_model_data": 2001,
      "display_name": "Mana Potion"
    }
  ]
}
```

#### Response `200 OK`
```json
{
  "pack_dir": "C:/server/plugins/MyPlugin/resourcepack",
  "total_items": 3,
  "total_latency_sec": 4.61,
  "items": [
    {
      "namespace": "myplugin",
      "item_id": "flame_sword",
      "resource_location": "myplugin:flame_sword",
      "texture_file": "C:/.../flame_sword.png",
      "model_file": "C:/.../flame_sword.json",
      "base_vanilla_item": "diamond_sword",
      "custom_model_data": 1001,
      "vanilla_override_file": "C:/.../diamond_sword.json",
      "base64_preview": "...",
      "seed": 28419204,
      "latency_sec": 1.51,
      "java_code_snippet": "..."
    }
  ]
}
```
