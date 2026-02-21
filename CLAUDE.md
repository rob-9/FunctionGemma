# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hackathon project for hybrid on-device/cloud function calling. A small on-device model (FunctionGemma 270M via Cactus inference engine) handles fast tool-calling, with fallback to Gemini 2.5 Flash (cloud) when confidence is low. The goal is to maximize a composite score balancing **accuracy (F1, 60%)**, **speed (15%)**, and **on-device ratio (25%)**.

## Setup

```bash
cd cactus && source ./setup && cd ..
cactus build --python
cactus download google/functiongemma-270m-it --reconvert
pip install google-genai
```

Requires `GEMINI_API_KEY` environment variable for cloud fallback.

## Commands

```bash
python benchmark.py          # Run full benchmark (30 cases), outputs score
python main.py               # Run example hybrid inference
python submit.py --team "TeamName" --location "City"  # Submit to leaderboard
```

Benchmark results and logs are saved to `logs/<timestamp>/`.

## Architecture

### Key Files

- **`main.py`** — The file to modify. Contains three functions:
  - `generate_cactus()` — On-device inference via FunctionGemma + Cactus FFI
  - `generate_cloud()` — Cloud inference via Gemini 2.5 Flash API
  - `generate_hybrid()` — Routing logic: runs local first, falls back to cloud if confidence < threshold. **This is the function to optimize.** Its signature (inputs/outputs) must not change.

- **`benchmark.py`** — 30 test cases across 3 difficulty levels (easy/medium/hard) with F1 scoring. Contains `compute_total_score()` which weights: easy=20%, medium=30%, hard=50%.

- **`cactus/`** — Submodule containing the Cactus on-device inference engine (C++ with Python ctypes FFI bindings in `cactus/python/src/cactus.py`).

### Inference Flow

1. `generate_hybrid()` always calls `generate_cactus()` first
2. Cactus returns function calls, a confidence score (1.0 - normalized entropy), and optionally a `cloud_handoff` flag
3. If confidence >= threshold, keep on-device result; otherwise call `generate_cloud()`
4. When falling back to cloud, total time includes both local + cloud time

### Scoring Formula

`total_score = sum(difficulty_weight * (0.60 * avg_f1 + 0.15 * time_score + 0.25 * on_device_ratio))`

- `time_score = max(0, 1 - avg_time / 500)` (under 500ms gets full marks)
- Difficulty weights: easy=20%, medium=30%, hard=50%

### Known FunctionGemma Weaknesses

- Produces negative integers for timer/alarm arguments
- Fails on multi-tool calls (hard cases require 2-3 simultaneous function calls)
- Sometimes has high confidence on wrong answers (false confidence)
- Can produce invalid JSON or trigger C++ engine cloud_handoff on high-entropy first tokens
