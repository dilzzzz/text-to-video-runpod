import os
import runpod

def upload_to_storage(file_path: str) -> str:
    file_name = os.path.basename(file_path)
    with open(file_path, "rb") as f:
        video_bytes = f.read()
    url = runpod.upload_file(
        file=video_bytes,
        file_name=file_name,
        content_type="video/mp4",
    )
    try:
        os.unlink(file_path)
    except OSError:
        pass
    return url
