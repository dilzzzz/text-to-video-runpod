import os
import runpod


def upload_to_storage(file_path: str) -> str:
    """
    Upload video file to RunPod managed storage.
    Returns public URL to the video.

    Alternatively, set BUCKET_ENDPOINT_URL, BUCKET_ACCESS_KEY,
    BUCKET_SECRET_KEY env vars in RunPod to use your own S3 bucket.
    """
    file_name = os.path.basename(file_path)

    with open(file_path, "rb") as f:
        video_bytes = f.read()

    # RunPod built-in bucket upload
    url = runpod.upload_file(
        file=video_bytes,
        file_name=file_name,
        content_type="video/mp4",
    )

    # Clean up temp file
    try:
        os.unlink(file_path)
    except OSError:
        pass

    return url
