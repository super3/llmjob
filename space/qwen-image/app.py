"""Qwen-Image-2.1-Turbo: 8-step text-to-image generation and image editing on ZeroGPU.

One request runs in three stages:

1. NCII guard (CPU sidecar subprocess). Requests with input images have their prompt screened
   with ``hfmlsoc/ncii-guard-v02`` before any GPU time is reserved.
2. Optional prompt enhancement, delegated over the Gradio API to the companion Space that keeps
   the official ``Qwen-Image-2.1-PE-T2I`` / ``PE-I2I`` rewriters warm.
3. Generation with ``QwenImage21Pipeline`` inside ``@spaces.GPU``, using the checkpoint's saved
   8-step sigma schedule, CFG 1 and the prefix KV cache, as the model card shows.
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import spaces  # noqa: E402  (must precede torch / any CUDA-touching import)

import math  # noqa: E402
import random  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import gradio as gr  # noqa: E402
import torch  # noqa: E402
from diffusers import QwenImage21Pipeline  # noqa: E402
from PIL import Image  # noqa: E402

import ncii_guard  # noqa: E402
from qwen21_aoti import aoti_load_pipeline  # noqa: E402

MODEL_ID = "Qwen/Qwen-Image-2.1-Turbo"
# Compiled for Qwen-Image-2.1. Turbo shares its architecture, and the compiled graphs take the
# weights as runtime inputs, so the same kernels serve this checkpoint.
AOTI_REPO = os.environ.get("QWEN21_AOTI_REPO", "hugging-apps/qwen-image-2-1-aoti")
PE_SPACE_ID = os.environ.get("PE_SPACE_ID", "hugging-apps/qwen-image-2-1-prompt-enhancer")
PE_MAX_NEW_TOKENS = int(os.environ.get("PE_MAX_NEW_TOKENS", "1024"))

MAX_SEED = 2**31 - 1
MAX_INPUT_IMAGES = 10

GUARD_REJECTION_MESSAGE = "prompt invalid based on our classifiers, try again"
GUARD_THRESHOLD = 0.5

# The model card's aspect-ratio presets at the 2048 base resolution.
BASE_RESOLUTION = 2048
ASPECT_RATIOS = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}
RESOLUTION_CHOICES = [("1K (fastest)", 1024), ("1.5K", 1536), ("2K (model card default)", 2048)]
DEFAULT_RESOLUTION = 2048
EAGER_STEP_PENALTY = 1.3  # measured slowdown of the eager path vs AOTI kernels
KV_CACHE_BYTES_PER_TOKEN = 32 * 2 * 4096 * 2
KV_CACHE_BUDGET_GB = 10.0

PROMPT_INFO = (
    "Long, detailed prompts work best. Put any words you want in the image in quotes."
)


def prefix_tokens(n_images: int, resolution: int) -> int:
    return n_images * (int(resolution) // 16) ** 2


def kv_cache_fits(n_images: int, resolution: int) -> bool:
    return prefix_tokens(n_images, resolution) * KV_CACHE_BYTES_PER_TOKEN <= KV_CACHE_BUDGET_GB * 1e9


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------

print(f"[load] {MODEL_ID} …", flush=True)
pipe = QwenImage21Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")
print(f"[load] sample_sigmas={getattr(pipe.config, 'sample_sigmas', None)}", flush=True)

# Tile only decodes above 1536 px so a 2K decode stays inside the 48 GB slice.
pipe.vae.enable_tiling(
    tile_sample_min_height=1536,
    tile_sample_min_width=1536,
    tile_sample_stride_height=1152,
    tile_sample_stride_width=1152,
)

AOTI_LOADED = []
if os.environ.get("QWEN21_AOTI", "1") != "0":
    try:
        AOTI_LOADED, _cfg = aoti_load_pipeline(pipe, AOTI_REPO, token=os.environ.get("HF_TOKEN"))
        print(f"[load] aoti kernels from {AOTI_REPO}: {AOTI_LOADED} (torch {_cfg['torch']})", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[load] aoti unavailable, running eager: {exc!r}", flush=True)
AOTI_BLOCKS = "QwenImage21DecodeBlock" in AOTI_LOADED
print("[load] pipeline ready", flush=True)

print(f"[load] {ncii_guard.GUARD_ID} sidecar …", flush=True)
ncii_guard.start()
print("[load] guard ready", flush=True)


def check_guard(prompt: str) -> None:
    score = ncii_guard.score(prompt or "")
    print(f"[guard] score={score:.3f} threshold={GUARD_THRESHOLD}", flush=True)
    if score >= GUARD_THRESHOLD:
        raise gr.Error(GUARD_REJECTION_MESSAGE)


# --------------------------------------------------------------------------------------
# Prompt enhancement (companion Space)
# --------------------------------------------------------------------------------------

_pe_client = None


def get_pe_client():
    global _pe_client
    if _pe_client is None:
        from gradio_client import Client

        token = os.environ.get("HF_TOKEN")
        _pe_client = Client(PE_SPACE_ID, token=token, httpx_kwargs={"timeout": 900}, verbose=False)
    return _pe_client


def enhance_prompt(prompt: str, image_paths: list[str]) -> tuple[str, str]:
    from gradio_client import handle_file

    started = time.perf_counter()
    result = get_pe_client().predict(
        prompt=prompt,
        image_paths=[handle_file(p) for p in image_paths],
        max_new_tokens=PE_MAX_NEW_TOKENS,
        enable_thinking=False,
        seed=0,
        randomize_seed=True,
        api_name="/enhance",
    )
    rewritten, wh_ratio = str(result[0]).strip(), str(result[1]).strip()
    print(f"[enhance] {time.perf_counter() - started:.1f}s -> ratio={wh_ratio!r}", flush=True)
    return (rewritten or prompt), wh_ratio


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def normalize_gallery(gallery) -> list[str]:
    """Reduce a ``gr.Gallery`` value to a flat list of local file paths."""
    paths: list[str] = []
    for item in gallery or []:
        if isinstance(item, (tuple, list)):
            item = item[0]
        if isinstance(item, dict):
            item = item.get("path") or item.get("image") or item.get("name")
            if isinstance(item, dict):
                item = item.get("path")
        if isinstance(item, Image.Image):
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            item.save(tmp.name)
            item = tmp.name
        if isinstance(item, str) and item:
            paths.append(item)
    return paths


def resolve_size(resolution: int, aspect_ratio: str) -> tuple[int | None, int | None]:
    if not aspect_ratio or aspect_ratio not in ASPECT_RATIOS:
        return None, None
    base_w, base_h = ASPECT_RATIOS[aspect_ratio]
    scale = int(resolution) / BASE_RESOLUTION
    round32 = lambda v: max(32, int(round(v * scale / 32)) * 32)  # noqa: E731
    return round32(base_w), round32(base_h)


def fit_resolution(n_images: int, resolution: int) -> int:
    """Largest resolution tier <= the request at which the condition images stay affordable.

    Measured on this Space: five condition images at 2K exhausted the 48 GB slice inside the
    Qwen3-VL encoder (allocator assert), five at 1.5K took ~99 s. Picking the largest tier whose
    prefix KV cache fits keeps every request memory-safe and fast (1 image -> 2K, 2 -> 1.5K,
    3+ -> 1K); 5-10 images run at 1K without the cache.
    """
    resolution = int(resolution)
    if n_images == 0:
        return resolution
    for tier in (2048, 1536, 1024):
        if tier <= resolution and kv_cache_fits(n_images, tier):
            return tier
    return min(resolution, 1024)


def estimate_duration(prompt="", image_paths=None, seed=0, resolution=DEFAULT_RESOLUTION,
                      aspect_ratio="Auto", *args, **kwargs):
    """GPU seconds to reserve, fitted to `[generate]` timings measured on this Space (AOTI on).

    Text-to-image: 1K 3.4-4.2 s, 2K 18.9 s. One 2K condition image (cached prefix): 30.2 s.
    Five condition images without the cache: 1K 35.2 s, 1.5K 98.8 s.
    """
    try:
        res = int(resolution)
    except (TypeError, ValueError):
        res = DEFAULT_RESOLUTION
    n_images = len(image_paths or [])
    scale = res / 1024.0
    seconds = 3.8 * scale**2.32
    if n_images:
        if kv_cache_fits(n_images, res):
            seconds += n_images * 11.3 * (res / 2048.0) ** 2
        else:
            seconds += n_images * 6.3 * scale**2.56
    if not AOTI_BLOCKS:
        seconds *= EAGER_STEP_PENALTY
    return int(min(180, math.ceil((seconds + 3.0) * 1.15)))


# --------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------


@spaces.GPU(duration=estimate_duration)
def _diffuse(prompt, image_paths, seed, resolution, aspect_ratio, progress=gr.Progress(track_tqdm=True)):
    condition_images = [Image.open(p) for p in image_paths] or None
    width, height = resolve_size(resolution, aspect_ratio)
    use_kv_cache = kv_cache_fits(len(image_paths), int(resolution))

    started = time.perf_counter()
    image = pipe(
        prompt=prompt,
        image=condition_images,
        height=height,
        width=width,
        output_resolution=int(resolution),
        use_kv_cache=use_kv_cache,
        generator=torch.Generator(device="cuda").manual_seed(int(seed) % (MAX_SEED + 1)),
    ).images[0]
    print(
        f"[generate] images={len(image_paths)} res={resolution} ratio={aspect_ratio} size={image.size} "
        f"kv_cache={use_kv_cache} aoti={AOTI_LOADED} elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return image


def generate_image(
    prompt: str = "",
    input_images=None,
    seed: int = 0,
    resolution: int = DEFAULT_RESOLUTION,
    aspect_ratio: str = "Auto",
    progress=gr.Progress(track_tqdm=True),
):
    """Generate or edit an image with Qwen-Image-2.1-Turbo (8 steps).

    With no input images this is text-to-image; with one to ten input images it is image
    editing, and the prompt may refer to them as <image1>…<image10>.

    Args:
        prompt: What to generate, or how to edit the inputs.
        input_images: Up to 10 condition images. Leave empty for text-to-image.
        seed: Random seed.
        resolution: Target side length in pixels (1024, 1536 or 2048); lowered automatically for 2+ input images.
        aspect_ratio: One of the model card's ratios (1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16) or "Auto".

    Returns:
        The generated image as RGBA (the model natively supports transparency).
    """
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Please enter a prompt.")
    image_paths = normalize_gallery(input_images)
    if len(image_paths) > MAX_INPUT_IMAGES:
        raise gr.Error(f"Up to {MAX_INPUT_IMAGES} input images are supported.")
    if image_paths:
        check_guard(prompt)
        fitted = fit_resolution(len(image_paths), resolution)
        if fitted != int(resolution):
            gr.Info(f"{len(image_paths)} input images: rendering at {fitted}px to stay within GPU memory.")
            resolution = fitted
    return _diffuse(prompt, image_paths, seed, resolution, aspect_ratio, progress)


def prepare(prompt, input_images, enhance, seed, randomize_seed, aspect_ratio):
    """Screen the prompt, optionally rewrite it, and resolve the seed, all off-GPU."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Please enter a prompt.")
    image_paths = normalize_gallery(input_images)
    if len(image_paths) > MAX_INPUT_IMAGES:
        raise gr.Error(f"Up to {MAX_INPUT_IMAGES} input images are supported.")
    if image_paths:
        check_guard(prompt)

    resolved_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    final_prompt, rewritten = prompt, ""
    if enhance:
        try:
            final_prompt, wh_ratio = enhance_prompt(prompt, image_paths)
            rewritten = final_prompt
            if wh_ratio in ASPECT_RATIOS and aspect_ratio == "Auto" and not image_paths:
                aspect_ratio = wh_ratio
        except Exception as exc:  # noqa: BLE001
            print(f"[enhance] failed: {exc!r}", flush=True)
            gr.Warning(f"Prompt enhancement unavailable, using the original prompt ({exc}).")
    return final_prompt, rewritten, resolved_seed, aspect_ratio



def run_example(prompt: str, aspect_ratio: str = "1:1"):
    """Text-to-image example at 1K, the fastest tier."""
    return generate_image(prompt, None, 0, 1024, aspect_ratio)


# --------------------------------------------------------------------------------------
# Examples: our own text-to-image prompts. Each one shows something the model is good at:
# text in the image, layout, photo realism, illustration and transparency.
# --------------------------------------------------------------------------------------

EXAMPLES = [
    ("Neon sign", "16:9",
     "A narrow alley in Tokyo at night in the rain. A pink neon sign above a small ramen shop reads "
     "\"OPEN ALL NIGHT\". Reflections on the wet pavement, steam from the doorway, cinematic photo."),
    ("Mars travel poster", "2:3",
     "A 1930s art deco travel poster for Mars. Bold lettering at the top reads \"VISIT OLYMPUS MONS\", "
     "with \"Daily rockets from Earth\" in smaller type at the bottom. A red volcano at sunset, a sleek "
     "silver rocket, warm orange and teal palette."),
    ("Product photo", "4:3",
     "Product photo of a matte black coffee grinder on a walnut kitchen counter. Soft morning light "
     "from a window on the left, a few coffee beans scattered in front, shallow depth of field."),
    ("Recipe card", "4:3",
     "A handwritten recipe card on a wooden kitchen table. The title reads \"Grandma's Lemon Bars\". "
     "Below it, a list of ingredients and three numbered steps in neat blue ink. A lemon and a "
     "rolling pin beside the card, top-down view."),
    ("How a GPU draws", "3:4",
     "A clean flat-design infographic titled \"How a GPU draws a picture\". Four numbered steps with "
     "simple icons: 1. Read the prompt, 2. Start from noise, 3. Remove noise 8 times, 4. Decode the "
     "image. White background, blue accents, readable sans-serif text."),
    ("Watercolor nook", "1:1",
     "A cozy reading nook by a window on a rainy afternoon, a ginger cat asleep on a stack of books, "
     "a steaming mug of tea. Loose watercolor illustration with soft edges."),
    ("Robot sticker", "1:1",
     "This is an RGBA image with transparency. A die-cut sticker of a friendly cartoon robot waving "
     "and holding a wrench, thick white outline, bright colors. The image has alpha channel and the "
     "background is transparent."),
    ("Mountain cabin", "16:9",
     "A small wooden cabin on a snowy mountain ridge at blue hour, warm light glowing from the "
     "windows, the Milky Way overhead. Wide-angle landscape photo."),
]
EXAMPLE_ROWS = [[text, ratio] for _, ratio, text in EXAMPLES]
EXAMPLE_LABELS = [label for label, _, _ in EXAMPLES]


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------

# LLMJob's colors and fonts, from llmjob.com.
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.stone,
    radius_size=gr.themes.sizes.radius_md,
    font=[gr.themes.GoogleFont("Schibsted Grotesk"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#fcfcfb",
    body_background_fill_dark="#121211",
    body_text_color="#0e0f10",
    block_background_fill="#ffffff",
    block_background_fill_dark="#1a1a17",
    block_border_color="#e3e1d9",
    block_border_color_dark="#2c2c28",
    block_label_text_color="#5b5e62",
    input_background_fill="#faf9f6",
    input_background_fill_dark="#161614",
    button_primary_background_fill="#2563eb",
    button_primary_background_fill_hover="#1d4ed8",
    button_primary_text_color="#ffffff",
    button_large_text_weight="600",
)

css = """
#app { margin: 0 auto; max-width: 1180px; padding: 8px 4px 0; }
#hero { padding: 18px 2px 6px; }
#hero h1 { font-size: 34px; font-weight: 800; letter-spacing: -0.02em; margin: 0 0 6px; }
#hero h1 span { color: #2563eb; }
#hero p { color: var(--body-text-color-subdued); font-size: 16px; margin: 0 0 12px; max-width: 760px; }
#hero .links a { margin-right: 14px; font-size: 14px; color: var(--body-text-color-subdued); text-decoration: none; }
#hero .links a:hover { color: #2563eb; }
.pills { display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 14px; }
.pill { font-family: 'IBM Plex Mono', monospace; font-size: 12px; padding: 4px 10px; border-radius: 999px;
        background: #eff4ff; color: #1d4ed8; border: 1px solid #dbe6fb; }
.dark .pill { background: #16213a; color: #9bb8f5; border-color: #24365e; }
#generate { min-height: 52px; font-size: 17px; }
#result { min-height: 560px; }
#footer { text-align: center; color: var(--body-text-color-subdued); font-size: 14px; padding: 18px 0 6px; }
#footer a { color: #2563eb; text-decoration: none; padding: 0; margin: 0; }
"""

HERO = """
<div id="hero">
  <h1>Qwen-Image 2.1 <span>Turbo</span></h1>
  <p>Make an image from a sentence, or edit one you upload. Turbo uses 8 steps instead of 50,
     so a 1K image takes a few seconds.</p>
  <div class="pills">
    <span class="pill">8 steps</span>
    <span class="pill">up to 2K</span>
    <span class="pill">text in images</span>
    <span class="pill">transparent PNG</span>
    <span class="pill">free</span>
  </div>
  <div class="links">
    <a href="https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo" target="_blank">Model</a>
    <a href="https://github.com/QwenLM/Qwen-Image-2.1" target="_blank">GitHub</a>
    <a href="https://qwen.ai/blog?id=qwen-image-2.1" target="_blank">Blog</a>
  </div>
</div>
"""

FOOTER = """
<div id="footer">Hosted by <a href="https://llmjob.com" target="_blank">LLMJob</a>, a network of GPUs that mine when they aren't serving. Want an API for this model? <a href="https://llmjob.com" target="_blank">Get in touch</a>.</div>
"""

with gr.Blocks(title="Qwen-Image 2.1 Turbo · LLMJob") as demo:
    with gr.Column(elem_id="app"):
        gr.HTML(HERO)
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=320):
                prompt = gr.Textbox(label="Prompt", info=PROMPT_INFO, lines=5, max_lines=24,
                                    placeholder="A neon sign in the rain that reads \"OPEN ALL NIGHT\"…")
                aspect_ratio = gr.Radio(label="Shape", choices=["Auto"] + list(ASPECT_RATIOS), value="Auto",
                                        info="Auto follows the input image, or square for text to image.")
                generate_button = gr.Button("Generate", variant="primary", size="lg", elem_id="generate")
                enhance = gr.Checkbox(label="Enhance prompt (slower, expands short prompts)", value=False)
                rewritten_prompt_output = gr.Textbox(label="Rewritten prompt", lines=3, max_lines=20,
                                                     interactive=False, visible=False)
                with gr.Accordion("Edit an image (optional)", open=False):
                    input_images = gr.Gallery(
                        label="Upload up to 10 images, then refer to them as <image1>, <image2>… in the prompt",
                        type="filepath",
                        interactive=True,
                        columns=5,
                        height=220,
                    )
                with gr.Accordion("Advanced settings", open=False):
                    resolution = gr.Radio(label="Resolution", choices=RESOLUTION_CHOICES, value=DEFAULT_RESOLUTION,
                                          info="Lowered automatically when you upload 2 or more images.")
                    with gr.Row():
                        seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                        randomize_seed = gr.Checkbox(label="Random seed", value=True)
                    gr.Markdown("Sampling uses the checkpoint's saved 8-step schedule with CFG 1, as the model card recommends.")
            with gr.Column(scale=6, min_width=320):
                result = gr.Image(label="Result", type="pil", image_mode="RGBA", format="png",
                                  height=560, elem_id="result")

        final_prompt_box = gr.Textbox(visible=False, label="Prompt")

        gr.Examples(
            examples=EXAMPLE_ROWS,
            example_labels=EXAMPLE_LABELS,
            inputs=[prompt, aspect_ratio],
            fn=run_example,
            outputs=[result],
            cache_examples=True,
            cache_mode="lazy",
            label="Try one",
        )
        gr.HTML(FOOTER)

    enhance.change(lambda on: gr.update(visible=on), inputs=enhance, outputs=rewritten_prompt_output,
                   api_visibility="private")
    generate_button.click(
        fn=prepare,
        inputs=[prompt, input_images, enhance, seed, randomize_seed, aspect_ratio],
        outputs=[final_prompt_box, rewritten_prompt_output, seed, aspect_ratio],
    ).then(
        fn=generate_image,
        inputs=[final_prompt_box, input_images, seed, resolution, aspect_ratio],
        outputs=[result],
        api_name="generate",
    )

if __name__ == "__main__":
    demo.launch(mcp_server=True, theme=theme, css=css)
