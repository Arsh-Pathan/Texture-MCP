# Hugging Face Gradio Space

The Space provides a Gradio interface and one MCP tool, `generate_textures`. The tool generates transparent PNGs only; it has no resource-pack, arbitrary file-write, or path input/output features. Local REST and stdio MCP are separate entry points and remain available.

## Create and deploy the Space

1. Create a new Space on Hugging Face and select the **Gradio** SDK.
2. Select **ZeroGPU** as the Space hardware in the Space's Settings → Hardware page. ZeroGPU is supported by Gradio Spaces and dynamically allocates GPU for inference.
3. Use Python 3.12. The repository README already contains the Space metadata (`sdk: gradio`, `app_file: app.py`), and `requirements.txt` contains the Space dependencies.
4. Add the Hugging Face Space as a Git remote and push this repository's deployment branch:

   ```powershell
   git remote add space https://huggingface.co/spaces/<owner>/<space-name>
   git push space main
   ```

   Authenticate with Hugging Face Git credentials or the `hf` CLI. Do not put an access token in the repository or commit `.env`.
5. Wait for the Space build to complete, then open the Space page and try a short generation prompt.

No Space secrets or environment variables are required for the public model. `HF_TOKEN` may be configured as a Space secret only if model access requirements change; never place it in source control. The code reads `SPACE_ID` as provided by Hugging Face to select Space-specific model startup behavior.

## MCP endpoint and verification

For Space `<owner>/<space-name>`, the base MCP endpoint is:

```text
https://<owner>-<space-name>.hf.space/gradio_api/mcp/
```

The MCP schema endpoint is:

```text
https://<owner>-<space-name>.hf.space/gradio_api/mcp/schema
```

After the Space is running:

1. Open the Space's **Settings** page and inspect its MCP section/tool listing. Confirm `generate_textures` appears with `prompt`, `variants`, `seed`, `steps`, and `guidance_scale` inputs.
2. Request the schema URL above in a browser or with `curl`; its JSON should list the same tool and typed inputs.
3. Configure an MCP client with the base endpoint. The Gradio MCP server provides its supported HTTP transports under that route.

Gradio exposes MCP through `demo.launch(mcp_server=True)`. The UI and MCP tool call the same bounded generation function.

## Hardware and limitations

- ZeroGPU queues requests and applies per-user daily usage quotas; availability and quota depend on Hugging Face account tier and current policy. Check the current [ZeroGPU documentation](https://huggingface.co/docs/hub/spaces-zerogpu) for account eligibility, quota, and queue priority.
- Generation requests are limited to 1–8 variants, 10–50 steps, prompts up to 300 characters, and bounded seeds/guidance values. Larger batches take longer and consume more ZeroGPU quota; allocated duration is estimated from the request and capped at 300 seconds.
- CPU inference is supported by the underlying engine but can be very slow. Free CPU hardware should not be expected to run this diffusion model comfortably.
- Initial startup downloads the model from the Hugging Face cache and explicitly places it on CUDA under ZeroGPU's documented startup CUDA emulation. Space startup does not rely on `torch.cuda.is_available()` to choose a device, and downloads are not copied into the app's local model directory.
- Transparent-background extraction is heuristic; inspect generated textures before use.

## Model licensing

The repository's application code is MIT-licensed. Model weights are from `yuuki14202028/minecraft-item-16px` and inherit Stable Diffusion 1.5 / CreativeML OpenRAIL-M terms. Review the [model repository](https://huggingface.co/yuuki14202028/minecraft-item-16px) and its license, including use restrictions, before hosting or using the weights or generated outputs.

## Local entry points

- `python server.py`: local FastAPI REST server at `http://127.0.0.1:8000`.
- `python local_mcp_server.py`: local stdio MCP server using FastMCP.
- `python app.py`: Gradio UI with Gradio's MCP HTTP server enabled.
