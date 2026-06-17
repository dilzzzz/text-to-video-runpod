ALLOWED_RESOLUTIONS = [
    "512x512",
    "768x432",
    "768x768",
    "1280x720",
    "1920x1080",
    "3840x2160",
]


def validate_input(inp: dict) -> str | None:
    """Validate handler input. Returns error string or None if valid."""

    prompt = inp.get("prompt", "").strip()
    if not prompt:
        return "prompt is required and cannot be empty"
    if len(prompt) > 2000:
        return "prompt must be 2000 characters or less"

    resolution = inp.get("resolution", "1280x720")
    if resolution not in ALLOWED_RESOLUTIONS:
        return f"resolution must be one of: {', '.join(ALLOWED_RESOLUTIONS)}"

    try:
        num_frames = int(inp.get("num_frames", 121))
        if not (9 <= num_frames <= 481):
            return "num_frames must be between 9 and 481"
    except (ValueError, TypeError):
        return "num_frames must be an integer"

    try:
        fps = float(inp.get("fps", 24.0))
        if not (8.0 <= fps <= 60.0):
            return "fps must be between 8 and 60"
    except (ValueError, TypeError):
        return "fps must be a number"

    try:
        steps = int(inp.get("num_inference_steps", 30))
        if not (5 <= steps <= 60):
            return "num_inference_steps must be between 5 and 60"
    except (ValueError, TypeError):
        return "num_inference_steps must be an integer"

    try:
        gs = float(inp.get("guidance_scale", 3.0))
        if not (1.0 <= gs <= 10.0):
            return "guidance_scale must be between 1.0 and 10.0"
    except (ValueError, TypeError):
        return "guidance_scale must be a float"

    return None
