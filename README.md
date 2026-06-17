# Text-to-Video Generator — LTX-2.3 on RunPod Serverless

LTX-2.3 (Lightricks) — jointly generates **synchronized video + audio** from a single model.

---

## Architecture

```
prompt ──► LTX2Pipeline ──► video frames + audio waveform ──► encode_video() ──► final.mp4
                ▲
         (optional) Stage 2 latent upsampler for HD (720p → 1080p+)
```

---

## Features

- Native 4K (up to 3840×2160)
- Up to 50fps
- Built-in synchronized audio — no separate audio model needed
- ~17GB model size
- 24–80GB VRAM (A100 80GB recommended)

---

## Quick Start

### 1. GitHub Secrets

`Settings → Secrets and variables → Actions`

| Secret | Value |
|--------|-------|
| `GHCR_TOKEN` | GitHub token (`write:packages`) |
| `HF_TOKEN` | Hugging Face token |

### 2. Build

`Actions → Build and Push Docker Image → Run workflow`

Build time: ~30–40 min (~17GB models)

### 3. RunPod Endpoint Settings

| Field | Value |
|-------|-------|
| Container image | `ghcr.io/YOUR_USERNAME/text-to-video-runpod:latest` |
| GPU | A100 80GB (recommended) or A40 48GB for ≤1080p |
| Container disk | **30 GB** |
| Execution timeout | 600s |

---

## API Examples

### Basic — 720p + audio

```json
{
  "input": {
    "prompt": "A cat walking through a forest, cinematic 4K, golden hour",
    "resolution": "1280x720",
    "num_frames": 121,
    "fps": 24,
    "enable_audio": true
  }
}
```

### 1080p with latent upsampler

```json
{
  "input": {
    "prompt": "A dramatic mountain landscape at sunset, epic cinematic",
    "resolution": "1920x1080",
    "num_frames": 121,
    "fps": 24,
    "num_inference_steps": 40,
    "enable_audio": true,
    "use_upsampler": true
  }
}
```

### Video only (no audio, faster)

```json
{
  "input": {
    "prompt": "Abstract geometric patterns morphing, colorful",
    "resolution": "768x432",
    "num_frames": 97,
    "fps": 24,
    "enable_audio": false
  }
}
```

### Check status

```bash
curl https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/JOB_ID \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Response (COMPLETED)

```json
{
  "status": "COMPLETED",
  "output": {
    "video_url": "https://....mp4",
    "resolution": "1280x720",
    "num_frames": 121,
    "fps": 24.0,
    "duration": 5.04,
    "has_audio": true,
    "model": "LTX-2.3",
    "status": "success"
  }
}
```

---

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Video description (up to 2000 chars) |
| `negative_prompt` | string | LTX default | What to avoid |
| `resolution` | string | `"1280x720"` | See table below |
| `num_frames` | int | `121` | Frames — auto-snapped to 8k+1 |
| `fps` | float | `24.0` | Frame rate (8–60) |
| `num_inference_steps` | int | `30` | Denoising steps (5–60) |
| `guidance_scale` | float | `3.0` | Prompt adherence (1.0–10.0) |
| `seed` | int | `-1` | -1 = random |
| `enable_audio` | bool | `true` | Synchronized audio generation |
| `use_upsampler` | bool | `false` | Stage 2 HD latent upsampler |

---

## Resolutions

| Resolution | Aspect | Notes |
|------------|--------|-------|
| 512×512 | 1:1 | Quick test |
| 768×432 | 16:9 | Fast web |
| 768×768 | 1:1 | Instagram |
| 1280×720 | 16:9 | HD — recommended default |
| 1920×1080 | 16:9 | Full HD — use `use_upsampler: true` |
| 3840×2160 | 16:9 | 4K — needs 80GB VRAM |

---

## Frame count guide (must be 8k+1)

| Duration @ 24fps | num_frames |
|------------------|------------|
| ~2s | 49 |
| ~4s | 97 |
| ~5s | 121 |
| ~8s | 193 |
| ~10s | 241 |

---

## File Structure

```
text-to-video-runpod/
├── handler.py              # RunPod entry point
├── model.py                # LTX-2.3 pipeline (video + audio)
├── Dockerfile
├── requirements.txt
├── builder/
│   └── setup.py            # Pre-downloads LTX-2.3 at build time
├── utils/
│   ├── validator.py
│   └── uploader.py
└── .github/workflows/
    └── docker-build.yml
```

---

## Model

**LTX-2.3** — Lightricks open-source joint audio-video foundation model
HuggingFace: [diffusers/LTX-2.3-Diffusers](https://huggingface.co/diffusers/LTX-2.3-Diffusers)
License: LTX-2 Community License
