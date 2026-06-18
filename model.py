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

def get_pipeline():
    global _pipe
    if _pipe is None:
        from diffusers import LTXConditionPipeline
        print("[MODEL] Loading LTX-Video-0.9.8...")
        _pipe = LTXConditionPipeline.from_pretrained(
            "Lightricks/LTX-Video-0.9.8-dev",
            torch_dtype=torch.bfloat16,
        )
        _pipe.enable_model_cpu_offload()
        _pipe.vae.enable_tiling()
        print("[MODEL] Ready.")
    return _pipe

def snap_dim(d):
    return max(32, (d // 32) * 32)

def snap_frames(n):
    if n <= 1: return 9
    r = (n - 1) % 8
    if r == 0: return n
    lower = n - r
    return max(lower if r < 4 else lower + 8, 9)

def generate_video(
    prompt, negative_prompt=None,
    num_frames=97, fps=24.0,
    width=768, height=432,
    num_inference_steps=30,
    guidance_scale=3.0,
    seed=-1, enable_audio=False, use_upsampler=False,
):
    pipe = get_pipeline()
    width      = snap_dim(width)
    height     = snap_dim(height)
    num_frames = snap_frames(num_frames)

    neg = negative_prompt or "worst quality, inconsistent motion, blurry, jittery, distorted"
    generator = torch.Generator("cuda").manual_seed(seed) if seed != -1 else None

    print(f"[MODEL] {width}x{height} | {num_frames}f | '{prompt[:60]}'")

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
    )

    frames = output.frames[0]
    if frames.dtype != np.uint8:
        frames = (np.clip(frames, 0, 1) * 255).astype(np.uint8)

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()

    writer = imageio.get_writer(
        tmp.name, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p"
    )
    for f in frames:
        writer.append_data(f)
    writer.close()

    print(f"[MODEL] Saved: {tmp.name}")
    return {
        "path": tmp.name,
        "has_audio": False,
        "audio_warning": "",
        "num_frames": num_frames,
    }
