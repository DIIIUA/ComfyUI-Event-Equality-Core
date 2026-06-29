# Singularity

Singularity is a public-alpha ComfyUI custom node for Wan image-to-video cascade continuation.

It is built for users who want longer Wan I2V / V2V videos without manually extracting a last frame, loading it again, generating another short clip, and stitching everything later.

## Main Feature: Tail 5 Continuation Gate

The main feature in this release is the **Tail 5 Continuation Gate**.

When a cascade segment finishes, Singularity can pause the workflow and show a detached panel:

```text
Source | Tail 1 | Tail 2 | Tail 3 | Tail 4 | Tail 5 | Result
```

You choose the tail frame that should become the source for the next segment, then press `Resume Cascade / Continue`.

This gives you direct control over the continuation point. Instead of blindly continuing from a weak last frame, you can pick the frame with the best pose, face, motion direction, or visual stability.

## What It Does

- Adds one visible ComfyUI node: `Singularity`.
- Generates Wan I2V cascade segments.
- Supports up to five cascade segments in the current public alpha.
- Can pause after cascade 1, 2, 3, or 4.
- Shows five tail-frame candidates at each pause.
- Lets you manually choose the continuation frame.
- Continues the same running workflow after your choice.
- Saves one stitched final video.
- Saves a Markdown runtime report with cascade, motion, seam, and completion evidence.
- Keeps the public default conservative: observe and report first, no active generation math by default.

## Named Function Layers

### Tail 5 Continuation Gate

The pause-and-select layer. It shows Source, five tail candidates, and Result. The selected tail frame becomes the next segment source.

### Same-Run Cascade Stitch

The continuation layer. The node keeps the cascade route inside one run and saves one final stitched video instead of leaving you with separate clips.

### Observer Math Baseline

The default public mode. `Observe Only` records evidence and reports what happened without intentionally mutating tensors.

### Prompt-Pure Strategy Map

Prompt analysis stays report-side. Singularity can inspect prompt structure and relation pressure, but the public default does not inject formula prose into the model-facing prompt.

### Completion Report Gate

The report layer. `CompletionGate = PASS` means the requested route completed and a final video exists. It is structural proof, not a guarantee that the video is visually perfect.

### Seam And Drift Diagnostics

The diagnostics layer. Reports can show cascade seam pressure, visible motion spikes, tail/source continuity pressure, and whether the next segment looks like it re-entered the same event or drifted.

## Current Release

```text
Version: 0.1.1-r178
Visible title: Singularity R178
Release name: Tail 5 Continuation Gate
```

R178 is a continuation-gate stabilization release.

Highlights:

- Friendly public mode names:
  - `Observe Only`
  - `Latent Delta Scale`
  - `Tail Source Reconstruction`
  - `Source Noise Field Shaping`
  - and other research modes.
- Backend still normalizes friendly names to stable canonical values.
- Public controls are grouped visually:
  - `SOURCE`
  - `PROMPT`
  - `TIMELINE`
  - `SAMPLER`
  - `MATH`
  - `DECODE`
  - `OUTPUT`
  - `LAB`
- Internal transport widgets are hidden from the public first surface.
- Native ComfyUI image upload is preserved.
- The Tail 5 panel stays detached under the node and disappears when the run finishes.
- Reports remain enabled by default.
- Public baseline uses `Observe Only` with neutral delta values.

## Public Default Settings

Fresh-node defaults are intended to be safe starter settings:

```text
source_image_file = none
positive_prompt = empty
negative_prompt = built-in Wan-style base negative prompt
cascade_count = 2
pause_after_cascade_1 = true
frames_per_cascade = 49
width = 704
height = 1280
fps = 16
seed = 123
math_control_mode = Observe Only
high_delta_strength = 1.0
low_delta_strength = 1.0
strategy_field_mode = OFF
image_crop = wan_native
save_video = true
save_report = true
sampler_trace_mode = OFF
prompt_transcode_mode = Report Only
formula recommendation = off
```

If `704 x 1280` is too heavy for your GPU, lower the resolution before rendering. A fast comparison size such as `416 x 608` is useful for debugging.

## Basic Workflow

1. Add a fresh `Singularity` node.
2. Connect your model, CLIP, VAE, and source image.
3. Enter your positive prompt.
4. Start with the default two-cascade setup.
5. Queue the workflow.
6. When the Tail 5 Continuation Gate appears, choose the best tail frame.
7. Press `Resume Cascade / Continue`.
8. Wait for the final stitched video and report.

## Required Inputs

- `primary_model`
- `clip`
- `vae`
- `image`
- `source_image_file`

Optional:

- `secondary_model`
- `mask`

In a Wan High/Low setup, `primary_model` is usually the high-noise / structure model and `secondary_model` is usually the low-noise / refinement model. If `secondary_model` is not connected, the node can fall back to the primary model for the second stage.

## Important Controls

### `cascade_count`

How many segments to generate.

```text
1 = one normal clip
2 = one clip plus one continuation
5 = current public-alpha maximum
```

### `pause_after_cascade_1..4`

Controls where the Tail 5 Continuation Gate appears.

For the first test, keep:

```text
cascade_count = 2
pause_after_cascade_1 = true
```

### `frames_per_cascade`

Frames generated per segment. The public default is `49`.

### `math_control_mode`

Default: `Observe Only`.

Public-safe modes:

- `Observe Only`: report evidence, no intentional tensor mutation.
- `Latent Delta Scale`: explicit delta testing mode.
- `Tail Source Reconstruction`: report-only tail/source continuation evidence.

Research modes are available for experiments, but they can change output quality and should not be treated as safe defaults.

### `high_delta_strength` and `low_delta_strength`

Neutral value is `1.0`.

In `Observe Only`, these values are recorded but not used as active generation control.

### `prompt_transcode_mode`

Default: `Report Only`.

`Report Only` keeps the user prompt clean and records prompt/strategy evidence in the report.

### `sampler_trace_mode`

Default: `OFF`.

Enable `Shadow Step Trace` only for diagnostics.

## What The Report Means

Singularity can save a Markdown report beside the video.

Useful report signals:

- `result_status`: whether a final output was produced.
- `CompletionGate`: whether the requested route completed.
- `cascade_progress`: which cascade segments completed.
- `cascade_seam_impulse`: whether the stitch boundary showed a visible jump.
- `tail_next_source_continuity`: whether the next segment needs better source/tail inheritance.
- `seam_phase_classifier`: why the boundary may have jumped.
- `PublicReleaseReadiness`: whether the current settings look public-safe or research-like.

Important: `CompletionGate = PASS` means the structural route completed. It does not mean the video is visually perfect. Always inspect the MP4.

## About The Formula

The project reads generation as an event:

```text
Outcome(t-1) + ObservedBehavior(t-1)
=
Strategy(t)
=
ObservedBehavior(t+1) + Outcome(t+1)
```

For normal users, this means:

```text
source image
+ prompt meaning
+ model interpretation
+ sampler route
+ latent motion
+ visible video
```

should keep describing the same event as the cascade continues.

The formula is not a magic quality button. In the public build, it mostly provides diagnostics and safer structure for experiments.

## Public Alpha Notes

- This is a public alpha, not a final production release.
- Current public cascade limit is 5.
- Infinite cascades and prompt-per-cascade scheduling are future work.
- The node is Wan-first.
- Models are not included.
- Heavy resolutions require more VRAM and time.
- Always inspect the final video.
- Research modes can produce artifacts.

## Install

Place this folder in your ComfyUI `custom_nodes` directory:

```text
ComfyUI/custom_nodes/Singularity
```

Then restart ComfyUI.

## Recommended First Test

```text
cascade_count = 2
pause_after_cascade_1 = true
frames_per_cascade = 49
fps = 16
seed = 123
math_control_mode = Observe Only
high_delta_strength = 1.0
low_delta_strength = 1.0
sampler_trace_mode = OFF
prompt_transcode_mode = Report Only
save_video = true
save_report = true
```

Queue the workflow, choose a tail frame at the pause, continue, then inspect the final MP4.
