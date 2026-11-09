# Plugin Generator Guide

This project is intentionally focused on texture generation only. Use the REST API or MCP tool to generate textures and then integrate them into your own plugin pipeline.

## Typical workflow

1. Call `/v1/textures/generate` or `generate_texture`
2. Read `base64_png` or `data_uri` from the response
3. Save or reuse the image in your downstream plugin code

This repository does not include resource pack generation or CustomModelData helpers.
