# SmolVLA-with-RTC-MuJoCo
Run [SmolVLA](https://huggingface.co/lerobot/smolvla_base) — a compact Vision-Language-Action model — on the SO-ARM robotic simulator using **Real-Time Chunking (RTC)** for smooth, temporally consistent action execution.

---

## Overview

This project provides a controller (`svla_soarm_rtc.py`) that:

- Loads a pretrained SmolVLA policy from HuggingFace
- Runs it inside the `gym_soarm` simulation environment
- Uses **RTC (Real-Time Chunking)** to blend action chunks over time, producing smoother robot motion
- Renders a live dual-camera visualization (diagonal + wrist views) via OpenCV
- Records the episode as an MP4 video

The default task is a **pick-and-place** scenario: the robot arm picks up a colored LEGO brick and places it at a target location.

---

## How It Works

SmolVLA predicts a chunk of 50 future actions at each planning step. Instead of executing each chunk from scratch, **RTC blends the new chunk with leftover actions from the previous plan**, guided by an attention schedule. This produces smoother, more consistent motion compared to naive chunk-switching.

```
Observation (images + joint positions)
        │
        ▼
  SmolVLA Policy  ──► 50-action chunk
        │
        ▼
  RTC Action Queue  ──► blend with previous chunk
        │
        ▼
  Execute action in gym_soarm
```

Replanning happens every `inference_delay` steps (default: 20). Between replanning steps, actions are consumed from the blended queue.

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended) but can work on CPU.

## Usage

### Quick Start

```bash
python svla_soarm_rtc.py
```

This runs the controller with default settings:
- Model: `lerobot/smolvla_base`
- Task: `"blue lego brick"`
- Environment: `gym_soarm/PickAndPlaceCube-v0`
- Max steps: 200
- Saves output to `smolvla_soarm_output.mp4`

### Custom Configuration

Edit the `__main__` block in `svla_soarm_rtc.py` to adjust parameters:

```python
controller = SmolVLAController(
    model_name="lerobot/smolvla_base",
    stats_path="svla_so101_pickplace.json",
    env_name='gym_soarm/PickAndPlaceCube-v0',
    task_instruction="blue lego brick",
    execution_horizon=10,
    max_guidance_weight=5.0,
    prefix_attention_schedule="LINEAR",
    inference_delay=20,
    max_steps=200,
    record_video=True,
    device=None  # auto-detects GPU/CPU
)
results = controller.run()
```

---

## Configuration Parameters

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `"lerobot/smolvla_base"` | HuggingFace model ID for the SmolVLA policy |
| `stats_path` | `"svla_so101_pickplace.json"` | Path to dataset statistics JSON for normalization |
| `env_name` | `"gym_soarm/PickAndPlaceCube-v0"` | Gymnasium environment ID |
| `task_instruction` | `"blue brick"` | Natural language task description |
| `execution_horizon` | `8` | Steps to blend with previous chunk (higher = smoother, less reactive) |
| `max_guidance_weight` | `5.0` | RTC guidance strength (range: 1.0–10.0) |
| `prefix_attention_schedule` | `"EXP"` | Attention schedule for RTC: `"LINEAR"`, `"EXP"`, or `"CONSTANT"` |
| `inference_delay` | `10` | Steps between replanning (lower = more reactive, more compute) |
| `max_steps` | `200` | Maximum episode length |
| `record_video` | `True` | Save episode as MP4 |
| `device` | `None` | Torch device (`"cuda"`, `"cpu"`, or `None` for auto) |

---

## Stats Files

Two dataset statistics files are included for normalization:

| File | Description |
|---|---|
| `svla_so101_pickplace.json` | Statistics for SO-101 pick-and-place dataset |
| `svla_so100_pickplace.json` | Statistics for SO-100 pick-and-place dataset |

These contain per-channel mean/std values used to normalize observations and denormalize predicted actions.

---

## Visualization

During execution, two OpenCV windows are displayed:

- **SmolVLA SO-ARM Control** — side-by-side diagonal and wrist camera views with step/reward/status overlays
- **Joint Information** — panel showing current joint positions (observations) and commanded joint angles (actions) in radians

Press `q` to stop execution early.


---

## Related Work

- [LeRobot](https://github.com/huggingface/lerobot) — HuggingFace robotics library
- [SmolVLA paper](https://huggingface.co/lerobot/smolvla_base) — Compact VLA model
- [gym_soarm](https://github.com/huggingface/gym_soarm) — SO-ARM gymnasium environment
- [Real-Time Chunking (RTC)](https://huggingface.co/blog/smolvla) — Temporal action blending method
