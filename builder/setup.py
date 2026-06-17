"""
Builder script — runs during Docker image build.
Pre-downloads LTX-2.3 weights so cold starts are ~10-15 seconds.

Models downloaded:
  1. LTX-2.3 main pipeline      ~15GB
  2. LTX-2.3 latent upsampler    ~2GB  (optional Stage 2 HD)
"""
import os
import torch

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

# ── 1. LTX-2.3 main pipeline ─────────────────────────────────────────────────
print("[BUILD] Downloading diffusers/LTX-2.3-Diffusers (~15GB)...")
from diffusers import LTX2Pipeline
pipe = LTX2Pipeline.from_pretrained(
    "diffusers/LTX-2.3-Diffusers",
    torch_dtype=torch.bfloat16,
    token=token,
)
del pipe
print("[BUILD] LTX-2.3 main pipeline done.")

# ── 2. Latent upsampler (Stage 2 HD) ─────────────────────────────────────────
print("[BUILD] Downloading diffusers/LTX-2.3-Latent-Upsampler-Diffusers (~2GB)...")
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
upsampler = LTX2LatentUpsamplerModel.from_pretrained(
    "diffusers/LTX-2.3-Latent-Upsampler-Diffusers",
    torch_dtype=torch.bfloat16,
    token=token,
)
del upsampler
print("[BUILD] Latent upsampler done.")

print("\n[BUILD] All models downloaded. Total ~17GB — build complete.")
