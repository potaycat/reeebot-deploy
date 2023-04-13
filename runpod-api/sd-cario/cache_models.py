from diffusers import (
    DiffusionPipeline,
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
)

try:
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        "andite/anything-v4.0", subfolder="scheduler"
    )
except:
    pass
try:
    vae = AutoencoderKL.from_pretrained("hakurei/waifu-diffusion", subfolder="vae")
except:
    pass
try:
    pipe = DiffusionPipeline.from_pretrained(
        "ckpt/anything-v4.5", vae=vae, scheduler=scheduler
    )
except:
    pass

print("CACHED")
