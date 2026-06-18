"""
model.py — LTX-2.3 via official ltx-pipelines package
HuggingFace: Lightricks/LTX-2.3
Requires: Python>=3.12, CUDA>12.7, PyTorch~=2.7
"""
import torch
import tempfile
import numpy as np
import imageio

RESOLUTIONS = {
    "512x512":   (512,  512),
    "768x432":   (768,  432),
    "768x768":   (768,  768),
    "1280x720":  (1280, 720),
    "1920x1080": (1920, 1080),
}

_pipe = None


def get_vram_gb():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return 0.0


def get_pipeline():
    global _pipe
    if _pipe is None:
        from ltx_pipelines.pipeline import LTXPipeline

        vram = get_vram_gb()
        print(f"[MODEL] VRAM: {vram:.1f}GB")
        print(f"[MODEL] Loading Lightricks/LTX-2.3...")

        _pipe = LTXPipeline.from_pretrained(
            "Lightricks/LTX-2.3",
            torch_dtype=torch.bfloat16,
        )
        _pipe.enable_model_cpu_offload()
        print(f"[MODEL] LTX-2.3 ready.")
    return _pipe


def snap_dim(d):
    return max(32, (d // 32) * 32)


def snap_frames(n):
    if n <= 1:
        return 9
    r = (n - 1) % 8
    if r == 0:
        return n
    lower = n - r
    upper = lower + 8
    return max(lower if r < 4 else upper, 9)


def frames_to_mp4(frames_np, fps, output_path):
    """Save frames to mp4 using imageio-ffmpeg."""
    if frames_np.dtype != np.uint8:
        frames_uint8 = (np.clip(frames_np, 0, 1) * 255).astype(np.uint8)
    else:
        frames_uint8 = frames_np

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    )
    for frame in frames_uint8:
        writer.append_data(frame)
    writer.close()
    print(f"[MODEL] Saved {len(frames_uint8)} frames → {output_path}")


def generate_video(
    prompt, negative_prompt=None,
    num_frames=97, fps=24.0,
    width=768, height=432,
    num_inference_steps=30,
    guidance_scale=3.0,
    seed=-1,
    enable_audio=False,
    use_upsampler=False,
):
    pipe    = get_pipeline()
    vram_gb = get_vram_gb()

    width      = snap_dim(width)
    height     = snap_dim(height)
    num_frames = snap_frames(num_frames)

    neg = negative_prompt or "worst quality, inconsistent motion, blurry, jittery, distorted"
    generator = torch.Generator("cuda").manual_seed(seed) if seed != -1 else None

    print(f"[MODEL] Generating {width}x{height} | {num_frames}f @ {fps}fps")
    print(f"[MODEL] Prompt: '{prompt[:80]}'")

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
        output_type="np",
        return_dict=True,
    )

    video_np = output.frames[0]  # (T, H, W, 3)

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()

    frames_to_mp4(video_np, fps=fps, output_path=tmp.name)

    print(f"[MODEL] Done: {tmp.name}")
    return {
        "path":          tmp.name,
        "has_audio":     False,
        "audio_warning": "",
        "num_frames":    num_frames,
    }
