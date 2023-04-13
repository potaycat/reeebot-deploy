import runpod
import os
import time

sleep_time = int(os.environ.get("SLEEP_TIME", 3))

## load your model(s) into vram here
import torch
from diffusers import DiffusionPipeline, AutoencoderKL, DPMSolverMultistepScheduler
from safetensors.torch import load_file
from sdscripts.networks.lora import create_network_from_weights
import base64
from io import BytesIO


DEVICE = "cuda"
DTYPE = torch.float16


def apply_lora(pipe, lora_path, weight=1.0):
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    unet = pipe.unet
    sd = load_file(lora_path)
    lora_network, sd = create_network_from_weights(
        weight, None, vae, text_encoder, unet, sd
    )
    lora_network.apply_to(text_encoder, unet)
    lora_network.load_state_dict(sd)
    lora_network.to(DEVICE, dtype=DTYPE)


vae = AutoencoderKL.from_pretrained(
    "hakurei/waifu-diffusion", subfolder="vae", torch_dtype=DTYPE
)
scheduler = DPMSolverMultistepScheduler.from_pretrained(
    "andite/anything-v4.0", subfolder="scheduler"
)
pipe = DiffusionPipeline.from_pretrained(
    "ckpt/anything-v4.5",
    vae=vae,
    scheduler=scheduler,
    torch_dtype=DTYPE,
)
# https://github.com/huggingface/diffusers/issues/3064
# pipe.unet.load_attn_procs('LucarioLoRA.safetensors', use_safetensors=True)
pipe.to(DEVICE)

safety_checker = pipe.safety_checker
generator = torch.Generator(DEVICE)


def handler(event):
    print(event)
    # do the things
    x = event["input"]
    prompt = x.get("prompt", "")
    negative_prompt = x.get("negative_prompt", "")
    seed = int(x.get("seed", 0))
    steps = int(x.get("steps", 28))
    cfg_scale = float(x.get("cfg_scale", 7.0))
    safe = bool(x.get("safety_check", True))
    lora = x.get("lora", "LucarioLoRA.safetensors")
    lora_weight = float(x.get("lora_weight", 0.8))

    if safe:
        pipe.safety_checker = safety_checker
    else:
        pipe.safety_checker = None
    apply_lora(pipe, lora, lora_weight)
    out = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        generator=generator.manual_seed(seed),
    )
    buffered = BytesIO()
    out.images[0].save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return_json = {
        "input": x,
        "output": img_str.decode("utf-8"),
        "nsfw_content_detected": str(out.nsfw_content_detected),
    }
    return return_json


runpod.serverless.start({"handler": handler})
