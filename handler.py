"""
handler.py — RunPod Serverless Handler
LTX-2.3 with GPU-adaptive VRAM management.
Audio auto-disables if resolution exceeds GPU VRAM capacity.
"""
import runpod
from model import generate_video, RESOLUTIONS
from utils.validator import validate_input
from utils.uploader import upload_to_storage


def handler(job):
    """
    RunPod Serverless Handler — LTX-2.3

    Input JSON:
    {
        "prompt":               "A sunset over the ocean, cinematic 4K",
        "negative_prompt":      "shaky, glitchy, low quality",
        "resolution":           "1280x720",
        "num_frames":           121,
        "fps":                  24,
        "num_inference_steps":  30,
        "guidance_scale":       3.0,
        "seed":                 -1,
        "enable_audio":         true,
        "use_upsampler":        false
    }

    Audio notes:
      - enable_audio: true  → audio generated IF VRAM allows
      - If GPU VRAM insufficient, audio auto-disabled & warning returned
      - RTX 4090 (24GB): audio safe up to 1280×720
      - A40 (48GB): audio safe up to 1920×1080
      - A100 (80GB): audio safe at all resolutions including 4K
    """
    job_input = job["input"]

    error = validate_input(job_input)
    if error:
        return {"error": error, "status": "failed"}

    prompt          = job_input.get("prompt", "")
    negative_prompt = job_input.get("negative_prompt", None)
    resolution      = job_input.get("resolution", "1280x720")
    num_frames      = int(job_input.get("num_frames", 121))
    fps             = float(job_input.get("fps", 24.0))
    steps           = int(job_input.get("num_inference_steps", 30))
    guidance_scale  = float(job_input.get("guidance_scale", 3.0))
    seed            = int(job_input.get("seed", -1))
    enable_audio    = bool(job_input.get("enable_audio", True))
    use_upsampler   = bool(job_input.get("use_upsampler", False))

    width, height = RESOLUTIONS.get(resolution, (1280, 720))

    try:
        print(f"[HANDLER] {resolution} | {num_frames}f @ {fps}fps | audio={enable_audio}")
        print(f"[HANDLER] Prompt: '{prompt[:80]}'")

        result = generate_video(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            fps=fps,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            enable_audio=enable_audio,
            use_upsampler=use_upsampler,
        )

        print(f"[HANDLER] Uploading...")
        video_url = upload_to_storage(result["path"])
        print(f"[HANDLER] Done: {video_url}")

        duration = round(result["num_frames"] / fps, 2)

        response = {
            "video_url":       video_url,
            "prompt":          prompt,
            "resolution":      resolution,
            "actual_size":     f"{result['width']}×{result['height']}",
            "num_frames":      result["num_frames"],
            "fps":             fps,
            "duration":        duration,
            "has_audio":       result["has_audio"],
            "used_upsampler":  use_upsampler,
            "model":           "LTX-2.3",
            "status":          "success",
        }

        # Include warning if audio was disabled due to VRAM
        if result["audio_warning"]:
            response["audio_warning"] = result["audio_warning"]

        return response

    except Exception as e:
        print(f"[HANDLER] ERROR: {str(e)}")
        return {"error": str(e), "status": "failed"}


runpod.serverless.start({"handler": handler})
