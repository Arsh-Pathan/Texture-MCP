# Texture-Craft MCP Integration Guide

Texture-Craft implements the standard **Model Context Protocol (MCP)** specification via `mcp_server.py`. This allows LLM coding agents (e.g., Cursor, Claude Desktop, Antigravity, or custom agent frameworks) to invoke 16×16 texture generation directly via tool calls.

---

## 🛠️ Exposed MCP Tools

| Tool | Parameters | Description |
|---|---|---|
| `generate_texture` | `prompt`, `variants`, `seed`, `steps`, `guidance_scale` | Generates 16×16 transparent pixel-art textures and returns base64 PNGs. |
| `create_resourcepack_item` | `pack_dir`, `item_id`, `prompt`, `namespace`, `base_vanilla_item`, `custom_model_data`, `display_name`, `seed` | Generates texture, writes PNG and model JSON to pack, registers CustomModelData override, and returns Java code snippet. |
| `init_resourcepack` | `pack_dir`, `pack_name`, `description`, `pack_format` | Generates `pack.mcmeta` and standard Minecraft folder hierarchy. |
| `batch_create_resourcepack_items` | `pack_dir`, `namespace`, `items_json` | Batches multiple items into the resource pack in one pass. |
| `get_system_status` | *(none)* | Returns GPU telemetry, VRAM usage, and engine readiness. |

---

## 💻 1. Claude Desktop Integration

Add Texture-Craft to your Claude Desktop configuration (`%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "texture-craft": {
      "command": "C:\\Users\\ArshPathan\\Projects\\CSky-Developments\\sourcecode\\Texture-Craft\\.venv\\Scripts\\python.exe",
      "args": [
        "C:\\Users\\ArshPathan\\Projects\\CSky-Developments\\sourcecode\\Texture-Craft\\mcp_server.py"
      ]
    }
  }
}
```

---

## ⚡ 2. Cursor IDE Integration

In Cursor, open **Settings → Features → MCP Servers → Add New MCP Server**:
- **Name:** `texture-craft`
- **Type:** `command`
- **Command:** `C:\Users\ArshPathan\Projects\CSky-Developments\sourcecode\Texture-Craft\.venv\Scripts\python.exe C:\Users\ArshPathan\Projects\CSky-Developments\sourcecode\Texture-Craft\mcp_server.py`

Alternatively, add it directly to `.cursor/mcp.json` in your project root:
```json
{
  "mcpServers": {
    "texture-craft": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "c:\\Users\\ArshPathan\\Projects\\CSky-Developments\\sourcecode\\Texture-Craft"
    }
  }
}
```

---

## 🌐 3. SSE / HTTP Daemon Mode (for Web & Microservices)

If your plugin generator runs as a web service or in another process, start the MCP server over SSE:

```bash
.\.venv\Scripts\python.exe mcp_server.py --transport sse --port 8001
```

Now any MCP client supporting Server-Sent Events can connect to `http://127.0.0.1:8001/sse`.

---

## 🤖 4. Custom Python LLM Agent Example (LangChain / OpenAI / Anthropic)

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command=r"C:\Users\ArshPathan\Projects\CSky-Developments\sourcecode\Texture-Craft\.venv\Scripts\python.exe",
    args=[r"C:\Users\ArshPathan\Projects\CSky-Developments\sourcecode\Texture-Craft\mcp_server.py"],
)

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List tools
            tools = await session.list_tools()
            print("Available tools:", [t.name for t in tools.tools])
            
            # Generate texture for resource pack
            result = await session.call_tool(
                "create_resourcepack_item",
                arguments={
                    "pack_dir": "./output_pack",
                    "namespace": "rpgplugin",
                    "item_id": "void_scythe",
                    "prompt": "void scythe with purple energy blade",
                    "base_vanilla_item": "netherite_hoe",
                    "custom_model_data": 5001,
                    "display_name": "Void Scythe"
                }
            )
            print("Result:", result.content)

asyncio.run(main())
```
