import torch
import tempfile

RESOLUTIONS = {
    "512x512":   (512,  512),
    "768x432":   (768,  432),
    "768x768":   (768,  768),
    "1280x720":  (1280, 720),
    "1920x1080": (1920, 1080),
}

_pipe      = None
_upsampler = None


def get_vram_gb():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return 0.0


def get_pipeline():
    global _pipe
    if _pipe is None:
        from diffusers import LTX2Pipeline
        vram = get_vram_gb()
        dtype = torch.bfloat16 if vram < 40 else torch.float16
        print(f"[MODEL] VRAM: {vram:.1f}GB | dtype: {dtype}")
        print(f"[MODEL] Loading LTX-2.3...")
        _pipe = LTX2Pipeline.from_pretrained(
            "diffusers/LTX-2.3-Diffusers",
            torch_dtype=dtype,
        )
        _pipe.enable_model_cpu_offload()
        _pipe.vae.enable_tiling()
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


def check_audio_safe(width, height, vram_gb):
    needed = {
        (512,  512):  8,
        (768,  432):  10,
        (768,  768):  12,
        (1280, 720):  16,
        (1920, 1080): 23,
    }.get((width, height), 16)
    if vram_gb >= needed + 4:
        return True, ""
    elif vram_gb >= needed:
        return True, f"tight VRAM ({needed}GB needed, {vram_gb:.0f}GB available)"
    return False, (
        f"audio disabled: {width}x{height} needs ~{needed}GB "
        f"but only {vram_gb:.0f}GB available."
    )


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
    from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT
    from diffusers.pipelines.ltx2.export_utils import encode_video

    pipe    = get_pipeline()
    vram_gb = get_vram_gb()

    width      = snap_dim(width)
    height     = snap_dim(height)
    num_frames = snap_frames(num_frames)

    audio_warning = ""
    actual_audio  = enable_audio
    if enable_audio:
        safe, reason = check_audio_safe(width, height, vram_gb)
        if not safe:
            actual_audio  = False
            audio_warning = reason
            print(f"[MODEL] WARNING: {reason}")

    neg = negative_prompt or DEFAULT_NEGATIVE_PROMPT
    generator = torch.Generator("cuda").manual_seed(seed) if seed != -1 else None

    print(f"[MODEL] Generating {width}x{height} | {num_frames}f | audio={actual_audio}")

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

    video_np   = output.frames[0]
    audio_data = output.audios if actual_audio else None
    audio_sr   = pipe.vocoder.config.output_sampling_rate if actual_audio else None

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()

    encode_video(
        video_np,
        fps=fps,
        audio=audio_data[0].float().cpu() if actual_audio else None,
        audio_sample_rate=audio_sr,
        output_path=tmp.name,
    )

    print(f"[MODEL] Saved: {tmp.name}")
    return {
        "path":          tmp.name,
        "has_audio":     actual_audio,
        "audio_warning": audio_warning,
        "num_frames":    num_frames,
    }
