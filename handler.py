import base64
import io
import os

import requests
import runpod
import torch
from diffusers import AutoPipelineForImage2Image, StableDiffusionXLImg2ImgPipeline
from PIL import Image

MODEL_ID = os.environ.get("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
LORA_PATH = os.environ.get("LORA_PATH", "")
DTYPE = torch.float16

if MODEL_ID.endswith(".safetensors"):
    pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
        MODEL_ID, torch_dtype=DTYPE, use_safetensors=True
    )
else:
    pipe = AutoPipelineForImage2Image.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, variant="fp16", use_safetensors=True
    )

if LORA_PATH:
    pipe.load_lora_weights(LORA_PATH)

pipe.to("cuda")
pipe.enable_vae_tiling()


def load_image(p: dict) -> Image.Image:
    if p.get("image_url"):
        r = requests.get(p["image_url"], timeout=30)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    data = p["image"]
    if "," in data:
        data = data.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def fit(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    s = min(max_side / max(w, h), 1.0)
    w, h = int(w * s) // 8 * 8, int(h * s) // 8 * 8
    return img.resize((max(w, 64), max(h, 64)), Image.LANCZOS)


def handler(job):
    p = job["input"]
    if "prompt" not in p or not (p.get("image") or p.get("image_url")):
        return {"error": "input needs 'prompt' and either 'image_url' or 'image' (base64)"}

    init = fit(load_image(p), int(p.get("max_side", 1024)))
    seed = p.get("seed")
    gen = torch.Generator("cuda").manual_seed(int(seed)) if seed is not None else None

    out = pipe(
        prompt=p["prompt"],
        negative_prompt=p.get("negative_prompt", ""),
        image=init,
        strength=float(p.get("strength", 0.6)),
        guidance_scale=float(p.get("guidance_scale", 6.0)),
        num_inference_steps=int(p.get("steps", 30)),
        generator=gen,
    ).images[0]

    return {"image": to_b64(out), "width": out.width, "height": out.height}


runpod.serverless.start({"handler": handler})
