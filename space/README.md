---
title: Minecraft Item Textures 16px
emoji: 🗡️
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: true
license: creativeml-openrail-m
models:
  - yuuki14202028/minecraft-item-16px
datasets:
  - OVAWARE/16xModdedMinecraft
tags:
  - minecraft
  - pixel-art
  - stable-diffusion
  - text-to-image
  - texture
  - 16x16
  - sprite
  - gamedev
thumbnail: https://huggingface.co/yuuki14202028/minecraft-item-16px/resolve/main/preview.png
short_description: Real 16x16 Minecraft item textures from text or a sketch
---

Text prompt in, real 16×16 Minecraft mod item texture out — with transparency, ready to drop
into a resource pack.

The model generates at 256×256 on a fixed 16×16 grid; this Space folds it back and keys out the
magenta background so you get the actual sprite rather than a pixel-art-looking illustration.

Three tabs:

- **from text** — a prompt in, textures out
- **from a sketch** — draw a crude 16×16 doodle (or upload one) and the model finishes it. What
  this model gets wrong is the *shape* of unfamiliar objects, not texture or shading, so giving
  it the shape yourself makes prompts land that text alone never does
- **a whole set** — one design plus a list of materials, all sharing a seed, so the silhouettes
  line up across the tier

Model: [yuuki14202028/minecraft-item-16px](https://huggingface.co/yuuki14202028/minecraft-item-16px)
Code: [github.com/yuuki14202028/pixelgen](https://github.com/yuuki14202028/pixelgen)
