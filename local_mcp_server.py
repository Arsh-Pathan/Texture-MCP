"""Local stdio MCP server for Texture-Craft PNG generation."""

from typing import Any

from mcp.server.fastmcp import FastMCP

from app import generate_textures

server = FastMCP("TextureCraft")


@server.tool()
def generate_texture(
    prompt: str,
    variants: int = 1,
    seed: int = -1,
    steps: int = 30,
    guidance_scale: float = 9.0,
) -> dict[str, Any]:
    """Generate transparent 16x16 item PNGs and return base64 image data."""
    return generate_textures(prompt, variants, seed, steps, guidance_scale)


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
