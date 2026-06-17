ALLOWED_RESOLUTIONS = ["512x512","768x432","768x768","1280x720","1920x1080"]

def validate_input(inp):
    prompt = inp.get("prompt","").strip()
    if not prompt:
        return "prompt is required"
    if len(prompt) > 2000:
        return "prompt must be under 2000 chars"
    if inp.get("resolution","768x432") not in ALLOWED_RESOLUTIONS:
        return f"resolution must be one of {ALLOWED_RESOLUTIONS}"
    try:
        n = int(inp.get("num_frames", 97))
        if not (9 <= n <= 481): return "num_frames must be 9–481"
    except: return "num_frames must be integer"
    try:
        s = int(inp.get("num_inference_steps", 30))
        if not (5 <= s <= 60): return "num_inference_steps must be 5–60"
    except: return "num_inference_steps must be integer"
    return None
