import runpod
from model import generate_video, RESOLUTIONS
from utils.validator import validate_input
from utils.uploader import upload_to_storage


def handler(job):
    job_input = job["input"]

    error = validate_input(job_input)
    if error:
        return {"error": error, "status": "failed"}

    prompt          = job_input.get("prompt", "")
    negative_prompt = job_input.get("negative_prompt", None)
    resolution      = job_input.get("resolution", "768x432")
    num_frames      = int(job_input.get("num_frames", 97))
    fps             = float(job_input.get("fps", 24.0))
    steps           = int(job_input.get("num_inference_steps", 30))
    guidance_scale  = float(job_input.get("guidance_scale", 3.0))
    seed            = int(job_input.get("seed", -1))
    enable_audio    = bool(job_input.get("enable_audio", False))
    use_upsampler   = bool(job_input.get("use_upsampler", False))

    width, height = RESOLUTIONS.get(resolution, (768, 432))

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
            "video_url":      video_url,
            "prompt":         prompt,
            "resolution":     resolution,
            "num_frames":     result["num_frames"],
            "fps":            fps,
            "duration":       duration,
            "has_audio":      result["has_audio"],
            "model":          "LTX-2.3",
            "status":         "success",
        }

        if result.get("audio_warning"):
            response["audio_warning"] = result["audio_warning"]

        return response

    except Exception as e:
        print(f"[HANDLER] ERROR: {str(e)}")
        return {"error": str(e), "status": "failed"}


runpod.serverless.start({"handler": handler})
