"""
model.py — LTX-2.3 Text-to-Video + Audio
GPU-adaptive: fp8 quantization + smart CPU offload based on VRAM.

VRAM strategy (auto-detected at runtime):
  ≥ 60GB  → full fp16, no offload        (A100 80GB, H100)
  ≥ 40GB  → fp16, light offload          (A40 48GB)
  ≥ 20GB  → bfloat16, cpu offload        (RTX 4090 24GB)
  < 20GB  → fp8 quant + seq cpu offload  (RTX 3090 24GB, RTX 4080)

Resolution + audio VRAM estimates (bfloat16, cpu offload):
  512×512   audio on  → ~8GB   ✅ any GPU
  768×432   audio on  → ~10GB  ✅ any GPU
  1280×720  audio on  → ~16GB  ✅ RTX 4090 safe
  1920×1080 audio on  → ~22GB  ⚠️ RTX 4090 tight → auto fp8
  3840×2160 audio on  → ~38GB  ❌ RTX 4090 OOM  → audio auto-disabled
  3840×2160 audio off → ~34GB  ⚠️ RTX 4090 OOM  → needs A40/A100
"""

import torch
import tempfile
import os
from diffusers import (
    LTX2Pipeline,
    LTX2LatentUpsamplePipeline,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT, STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.pipelines.ltx2.export_utils import encode_video

MODEL_ID     = "diffusers/LTX-2.3-Diffusers"
UPSAMPLER_ID = "diffusers/LTX-2.3-Latent-Upsampler-Diffusers"

_pipe      = None
_upsampler = None


# ── VRAM detection ────────────────────────────────────────────────────────────

def get_vram_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return 0.0


def get_dtype_and_offload(vram_gb: float):
    """Return (torch_dtype, offload_mode) based on available VRAM."""
    if vram_gb >= 60:
        return torch.float16, "none"       # A100 80GB, H100
    elif vram_gb >= 40:
        return torch.float16, "model"      # A40 48GB
    elif vram_gb >= 20:
        return torch.bfloat16, "model"     # RTX 4090 24GB
    else:
        return torch.float8_e4m3fn, "sequential"  # smaller GPUs


# ── Audio safety check ────────────────────────────────────────────────────────

# VRAM needed (GB) per resolution with audio ON (bfloat16 + cpu offload)
_AUDIO_VRAM_NEEDED = {
    (512,  512):  8,
    (768,  432):  10,
    (768,  768):  12,
    (1280, 720):  16,
    (1920, 1080): 23,   # tight on 24GB
    (3840, 2160): 40,   # impossible on 24GB
}

def check_audio_safe(width: int, height: int, vram_gb: float) -> tuple[bool, str]:
    """
    Returns (audio_safe, reason).
    If not safe, caller should disable audio or warn user.
    """
    needed = _AUDIO_VRAM_NEEDED.get((width, height))
    if needed is None:
        # Unknown resolution — estimate from pixel count
        pixels = width * height
        needed = max(8, int(pixels / (1280 * 720) * 16))

    if vram_gb >= needed + 4:   # 4GB headroom
        return True, "ok"
    elif vram_gb >= needed:
        return True, f"tight ({needed}GB needed, {vram_gb:.0f}GB available — may OOM)"
    else:
        return False, (
            f"audio disabled: {width}×{height} + audio needs ~{needed}GB "
            f"but only {vram_gb:.0f}GB VRAM available. "
            f"Use A40 (48GB) or A100 (80GB) for audio at this resolution."
        )


# ── Pipeline loader ───────────────────────────────────────────────────────────

def get_pipeline():
    global _pipe
    if _pipe is None:
        vram_gb = get_vram_gb()
        dtype, offload = get_dtype_and_offload(vram_gb)
        print(f"[MODEL] VRAM: {vram_gb:.1f}GB → dtype={dtype}, offload={offload}")
        print(f"[MODEL] Loading LTX-2.3...")

        _pipe = LTX2Pipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
        )

        if offload == "sequential":
            _pipe.enable_sequential_cpu_offload()
        elif offload == "model":
            _pipe.enable_model_cpu_offload()
        # "none" → stay on GPU fully

        _pipe.vae.enable_tiling()
        print(f"[MODEL] LTX-2.3 ready.")
    return _pipe


def get_upsampler():
    global _upsampler
    if _upsampler is None:
        vram_gb = get_vram_gb()
        dtype, _ = get_dtype_and_offload(vram_gb)
        print(f"[MODEL] Loading latent upsampler...")
        upsampler_model = LTX2LatentUpsamplerModel.from_pretrained(
            UPSAMPLER_ID,
            torch_dtype=dtype,
        )
        scheduler = FlowMatchEulerDiscreteScheduler()
        _upsampler = LTX2LatentUpsamplePipeline(
            upsampler=upsampler_model,
            vae=get_pipeline().vae,
            scheduler=scheduler,
        )
        _upsampler.enable_model_cpu_offload()
        print(f"[MODEL] Upsampler ready.")
    return _upsampler


# ── Dimension helpers ─────────────────────────────────────────────────────────

def snap_dim(d: int) -> int:
    """Snap to nearest multiple of 32 (LTX-2.3 requirement)."""
    return max(32, (d // 32) * 32)


def snap_frames(n: int) -> int:
    """Snap to nearest 8k+1 (LTX-2.3 requirement: 9,17,25,49,97,121,161,241...)"""
    if n <= 1:
        return 9
    remainder = (n - 1) % 8
    if remainder == 0:
        return n
    lower = n - remainder
    upper = lower + 8
    return max(lower if remainder < 4 else upper, 9)


# ── Resolution presets ────────────────────────────────────────────────────────

RESOLUTIONS = {
    "512x512":   (512,  512),
    "768x432":   (768,  432),
    "768x768":   (768,  768),
    "1280x720":  (1280, 720),
    "1920x1080": (1920, 1080),
    "3840x2160": (3840, 2160),
}


# ── Main generate function ────────────────────────────────────────────────────

def generate_video(
    prompt: str,
    negative_prompt: str  = None,
    num_frames: int       = 121,
    fps: float            = 24.0,
    width: int            = 1280,
    height: int           = 720,
    num_inference_steps: int = 30,
    guidance_scale: float = 3.0,
    seed: int             = -1,
    enable_audio: bool    = True,
    use_upsampler: bool   = False,
) -> dict:
    """
    Generate video + synchronized audio with LTX-2.3.

    Returns dict:
        {
            "path":          str,   # path to .mp4 file
            "has_audio":     bool,  # whether audio was generated
            "audio_warning": str,   # non-empty if audio was disabled/warned
            "width":         int,
            "height":        int,
            "num_frames":    int,
        }
    """
    pipe   = get_pipeline()
    vram_gb = get_vram_gb()

    # Snap dimensions
    width      = snap_dim(width)
    height     = snap_dim(height)
    num_frames = snap_frames(num_frames)

    # ── Audio safety check ────────────────────────────────────────────────────
    audio_warning = ""
    actual_audio  = enable_audio

    if enable_audio:
        safe, reason = check_audio_safe(width, height, vram_gb)
        if not safe:
            actual_audio  = False
            audio_warning = reason
            print(f"[MODEL] WARNING: {reason}")
        elif "tight" in reason:
            audio_warning = reason
            print(f"[MODEL] CAUTION: {reason}")

    neg = negative_prompt if negative_prompt else DEFAULT_NEGATIVE_PROMPT

    generator = None
    if seed != -1:
        generator = torch.Generator("cuda").manual_seed(seed)

    print(f"[MODEL] Generating {width}×{height} | {num_frames}f @ {fps}fps | audio={actual_audio}")
    print(f"[MODEL] Prompt: '{prompt[:80]}'")

    # ── Stage 1: generate ─────────────────────────────────────────────────────
    output = pipe(
        prompt=prompt,
        negative_prompt=neg,
        width=width,
        height=height,
        num_frames=num_frames,
        frame_rate=fps,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="latent" if use_upsampler else "np",
        return_dict=True,
    )

    video_data = output.frames
    audio_data = output.audios

    # ── Stage 2: optional latent upsampler ───────────────────────────────────
    if use_upsampler:
        print(f"[MODEL] Running latent upsampler (Stage 2)...")
        upsampler = get_upsampler()
        upsampled = upsampler(
            latents=video_data,
            prompt=prompt,
            sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
            num_inference_steps=10,
            output_type="np",
            return_dict=True,
        )
        video_np = upsampled.frames[0]
    else:
        video_np = video_data[0]

    # ── Export MP4 ────────────────────────────────────────────────────────────
    audio_sample_rate = pipe.vocoder.config.output_sampling_rate  # 48000 Hz

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()

    encode_video(
        video_np,
        fps=fps,
        audio=audio_data[0].float().cpu() if actual_audio else None,
        audio_sample_rate=audio_sample_rate if actual_audio else None,
        output_path=tmp.name,
    )

    print(f"[MODEL] Saved: {tmp.name}")
    return {
        "path":          tmp.name,
        "has_audio":     actual_audio,
        "audio_warning": audio_warning,
        "width":         width,
        "height":        height,
        "num_frames":    num_frames,
    }
