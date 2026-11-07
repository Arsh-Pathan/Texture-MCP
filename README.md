# ✨ Texture-Craft

**GPU-Accelerated 16×16 Minecraft Item Texture Generator & Resource Pack Builder.**  
Exposes a **Native FastAPI REST service** and an **MCP (Model Context Protocol) Server** for AI Minecraft plugin generators and coding agents.

---

## ⚡ What Texture-Craft Does

Texture-Craft uses a fine-tuned Stable Diffusion 1.5 pipeline (`yuuki14202028/minecraft-item-16px`) trained specifically on Minecraft mod item textures.
- Generates **authentic 16×16 RGBA sprites** with clean background transparency (not downscaled illustrations).
- **Fast:** ~1.5 seconds per texture on an RTX 4050 Laptop GPU.
- **Automated Resource Pack Assembly:** Automatically writes textures (`.png`), item models (`.json`), and sorted `CustomModelData` predicate overrides directly to your resource pack folder.
- **Java Code Snippets:** Returns ready-to-use Bukkit/Spigot/Paper code snippets for custom items.

---

## 🚀 Quick Start

### 1. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/Arsh-Pathan/Texture-Craft.git
cd Texture-Craft

# Sync virtual environment using uv
uv sync
```

### 2. Run the Native REST API
```powershell
.\.venv\Scripts\python.exe server.py
```
- **API URL:** `http://127.0.0.1:8000`
- **Interactive Swagger Docs:** `http://127.0.0.1:8000/docs`

### 3. Run the MCP Server (for AI Agents)
```powershell
.\.venv\Scripts\python.exe mcp_server.py --transport stdio
```

---

## 🔌 Integration Options

### 1. REST API
Call Texture-Craft from any language (Python, Node.js, Go, Java):

```bash
# Generate texture & integrate directly into your plugin's resource pack
curl -X POST http://127.0.0.1:8000/v1/resourcepack/item \
  -H "Content-Type: application/json" \
  -d '{
    "pack_dir": "C:/server/plugins/MyPlugin/resourcepack",
    "namespace": "myplugin",
    "item_id": "frost_blade",
    "prompt": "frost blade with icy blue runes",
    "base_vanilla_item": "diamond_sword",
    "custom_model_data": 1001,
    "display_name": "Frost Blade"
  }'
```

See [docs/API_REFERENCE.md](docs/API_REFERENCE.md) for full endpoint specifications.

---

### 2. MCP Server (Cursor, Claude Desktop, Antigravity)
Add Texture-Craft to your agent's config:

```json
{
  "mcpServers": {
    "texture-craft": {
      "command": "C:\\path\\to\\Texture-Craft\\.venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\Texture-Craft\\mcp_server.py"]
    }
  }
}
```

See [docs/MCP_INTEGRATION.md](docs/MCP_INTEGRATION.md) for IDE and agent setup.

---

### 3. AI Plugin Generator Guide
For instructions on connecting Texture-Craft to an AI Minecraft plugin builder, see [docs/PLUGIN_GENERATOR_GUIDE.md](docs/PLUGIN_GENERATOR_GUIDE.md).

---

## 📁 Repository Structure

```
Texture-Craft/
├── texture_craft/                  # Core library
│   ├── engine.py                   # Model pipeline, CUDA inference, 16x16 folding & chroma-key
│   └── resourcepack.py             # Resource pack manager, model JSONs & CustomModelData
├── server.py                       # FastAPI REST API (port 8000, OpenAPI docs at /docs)
├── mcp_server.py                   # Model Context Protocol server (stdio & SSE)
├── models/
│   └── minecraft-item-16px/        # Pre-cached model weights (instant loading)
├── docs/
│   ├── API_REFERENCE.md            # Complete REST API documentation & curl examples
│   ├── MCP_INTEGRATION.md          # Guide for Cursor, Claude Desktop, Antigravity, and agents
│   └── PLUGIN_GENERATOR_GUIDE.md   # Step-by-step guide for AI Minecraft plugin creators
├── pyproject.toml
└── uv.lock
```

---

## 📜 License
MIT License. Model weights inherit CreativeML OpenRAIL-M from Stable Diffusion 1.5.
