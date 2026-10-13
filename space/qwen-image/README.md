---
title: Qwen-Image-2.1-Turbo
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
short_description: 8-step Qwen-Image-2.1-Turbo, hosted by LLMJob
python_version: "3.12"
startup_duration_timeout: 1h
license: other
license_name: qwen-research
license_link: https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/LICENSE
---

# Qwen-Image-2.1-Turbo

LLMJob's copy of Hugging Face's
[`hugging-apps/qwen-qwen-image-2-1-turbo`](https://huggingface.co/spaces/hugging-apps/qwen-qwen-image-2-1-turbo)
Space. Generation, the safety check and the kernel loading are theirs. The page design and the
examples are ours. It runs on ZeroGPU for now. The next step
is to send generation to the [LLMJob](https://llmjob.com) GPU network, which has no per-user
limits.

To publish or update it, from the repo root:

```
huggingface-cli upload <your-user>/qwen-image-2-1-turbo space/qwen-image . --repo-type space
```

The Space needs ZeroGPU hardware, which needs a PRO account.

Text-to-image generation and multi-image editing with
[`Qwen/Qwen-Image-2.1-Turbo`](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo), the official
8-step distilled checkpoint of [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1),
running on ZeroGPU through `QwenImage21Pipeline` in `diffusers`.

* **Text to image**: leave the input gallery empty and describe what you want.
* **Image editing**: upload up to 10 images and refer to them as `<image1>` … `<image10>`.
* **Transparency**: the model natively decodes RGBA. Ask for it in the prompt.

Sampling follows the model card: the checkpoint's saved 8-step sigma schedule, CFG 1 and prefix KV caching.

## Speed

The transformer blocks and VAE decoder run as ahead-of-time compiled kernels (AOTInductor) from
[`hugging-apps/qwen-image-2-1-aoti`](https://huggingface.co/hugging-apps/qwen-image-2-1-aoti). Turbo has
the same architecture as Qwen-Image-2.1, and weights are runtime inputs to the compiled graph, so the
same artifacts apply. Set `QWEN21_AOTI=0` to run fully eager.

## Prompt enhancement

The optional **Enhance prompt** checkbox calls
[`hugging-apps/qwen-image-2-1-prompt-enhancer`](https://huggingface.co/spaces/hugging-apps/qwen-image-2-1-prompt-enhancer),
which serves the official `Qwen-Image-2.1-PE-T2I` / `PE-I2I` rewriters.

## Safety

Editing requests (any submission with input images) are screened with
[`hfmlsoc/ncii-guard-v02`](https://huggingface.co/hfmlsoc/ncii-guard-v02) before anything is generated.

## Examples

The example prompts are our own, written into `app.py`. They are text-to-image only, so the Space
ships no example images.

## License

> Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026
> Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved.

Research and evaluation use only. See the
[license](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/LICENSE).
