# Singularity

Singularity is a public-alpha ComfyUI custom node for Wan image-to-video cascade continuation.

Its main job is simple: generate a Wan video segment, pause at the cascade boundary, show you the source image and three tail-frame candidates, let you choose the frame that should continue the video, resume the same run, and save one final stitched video.

This is built for Wan I2V / V2V users who want longer videos without rebuilding a large continuation workflow by hand after every short clip.

## What You Get

- One visible ComfyUI node: `Singularity`.
- Wan I2V cascade generation.
- Up to five cascade segments in this public alpha.
- Optional pause after cascade 1, 2, 3, and 4.
- A detached Source / Tail 1 / Tail 2 / Tail 3 / Result panel under the node.
- Manual tail-frame selection before continuation.
- Same-run continuation instead of separate unrelated segment renders.
- One final stitched video at the end.
- Markdown reports with cascade, motion, delta, runtime, and completion diagnostics.
- A safer public default where math observation is enabled but delta mutation is off.

## In Plain English

Normally, extending a Wan video looks like this:

```text
make a clip
find a good last frame
load that frame again
generate the next clip
repeat
stitch everything later
```

Singularity moves that loop into one node.

You still choose what looks right. The node just gives you the pause point, the candidate frames, the continue button, the final stitch, and the report evidence.

## Basic Workflow

Connect the node like this:

```text
Wan model route -> Singularity -> saved video + report
```

Required inputs:

- `primary_model`
- `clip`
- `vae`
- `source_image_file`

Optional input:

- `secondary_model`

In a Wan High/Low setup, `primary_model` is usually the high-noise / structure model and `secondary_model` is usually the low-noise / refinement model. If `secondary_model` is not connected, the node can fall back to the primary model for the second stage.

## Public r62 Defaults

The r62 public build starts clean:

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
math_control_mode = OBSERVE_ONLY
high_delta_strength = 1.0
low_delta_strength = 1.0
image_crop = center
save_video = true
save_report = true
sampler_trace_mode = OFF
formula recommendation = visible, experimental, and off by default
```

If `704 x 1280` is too heavy for your GPU, reduce the size before rendering. A lighter test size such as `416 x 608` is useful for fast debugging.

## r62 Pause UI Hotfix

r62 fixes an r60 public-alpha issue where the backend could pause at the cascade boundary, but the detached Source / Tail / Result panel might not appear in modern ComfyUI Desktop.

The node now has two ways to detect a paused cascade:

- the normal ComfyUI websocket event;
- a status-polling fallback that checks the node pause state directly.

This means the `Resume Cascade / Continue` panel should recover even if the frontend misses the first pause event.

The Source / Tail / Result media panel is also rendered immediately for the node. You should see the panel before starting a long cascade; during a pause, the same panel fills with tail candidates and enables continuation.

## Quick Start

1. Add a fresh `Singularity` node.
2. Connect your model, CLIP, and VAE.
3. Upload or choose a source image using `source_image_file`.
4. Enter your positive prompt.
5. Keep the default two-cascade setup for the first test.
6. Queue the workflow.
7. When the pause panel appears, click the tail frame you want.
8. Press `Resume Cascade / Continue`.
9. Wait for the final stitched video and report.

## The Pause Panel

When a pause is reached, Singularity shows a detached panel under the node:

```text
Source | Tail 1 | Tail 2 | Tail 3 | Result
```

What each tile means:

- `Source`: the current source frame for the segment.
- `Tail 1`, `Tail 2`, `Tail 3`: continuation candidates from the end of the segment.
- `Result`: a preview of the stitched video up to the current pause boundary when available.

The selected tail is outlined in green. That selected frame becomes the source for the next cascade segment.

## Main Controls

`source_image_file`

The starting image. Use the native ComfyUI upload button if the image is not already in the input folder.

`positive_prompt`

What you want the video to show.

`negative_prompt`

What you want the video to avoid. r62 includes a simple base negative prompt by default so a new node is not filled with a test scene.

`cascade_count`

How many video segments to generate.

```text
1 = one normal clip
2 = one clip plus one continuation
5 = current public-alpha maximum
```

`pause_after_cascade_1..4`

Where the node should pause and ask you to choose a continuation frame.

Example:

```text
cascade_count = 2
pause_after_cascade_1 = true
```

This means: generate segment 1, pause, choose a tail frame, then continue segment 2.

`frames_per_cascade`

How many frames each segment generates before trimming and stitching.

At `16 fps`:

```text
49 frames = about 3 seconds per segment
121 frames = about 7.5 seconds per segment
```

`width` and `height`

The generation resolution. The r62 default is a 720-class vertical setting (`704 x 1280`). Lower it if you need faster tests or have limited VRAM.

`seed`

The random seed. r62 defaults to `123` so comparisons can start from a stable baseline.

`image_crop`

Controls source-image crop behavior. r62 defaults to `center`.

## Math Controls

Singularity studies generation as an event:

```text
Outcome(t-1) + ObservedBehavior(t-1)
= Strategy(t)
= ObservedBehavior(t+1) + Outcome(t+1)
```

For normal users, this means the node tries to keep the prompt, source image, model behavior, latent motion, and final video understandable as one route.

`math_control_mode`

Controls how active the math layer is.

```text
OBSERVE_ONLY = record reports and diagnostics without intentional tensor mutation
LATENT_DELTA_SCALE = controlled delta-strength research path
DEEP_STEP_DELTA_CONTROL = experimental deep research mode, high risk
```

r62 public default is `OBSERVE_ONLY`.

`high_delta_strength`

Controls the high-stage delta strength when `LATENT_DELTA_SCALE` is enabled.

Simple version: it changes how strongly the first/high stage motion is carried forward.

`low_delta_strength`

Controls the low-stage delta strength when `LATENT_DELTA_SCALE` is enabled.

Simple version: it changes how strongly the refinement stage can reshape the result.

Start with:

```text
high_delta_strength = 1.0
low_delta_strength = 1.0
```

Then change only one value at a time when comparing.

## Drift

Drift means the video starts moving away from what you wanted.

Common drift types:

- Source drift: the video stops respecting the source image.
- Prompt drift: the model ignores the prompt.
- Identity drift: the subject changes too much.
- Motion drift: motion becomes unstable or chaotic.
- Cascade drift: the next segment does not continue cleanly.
- Math drift: research settings break generation instead of guiding it.

Singularity helps expose drift through frame selection and reports. It does not automatically fix every drift case.

## Reports

When `save_report = true`, Singularity writes a markdown report with evidence such as:

- runtime version;
- normalized input settings;
- cascade plan;
- pause boundaries;
- selected tail index;
- segment begin/end records;
- final video path;
- CompletionGate status;
- motion metrics;
- delta diagnostics;
- runtime monitor sidecars where available.

Important: `CompletionGate = PASS` means the structural route completed and a final video exists. It does not mean the video is visually perfect. Always inspect the video.

## r62 UI Fixes

r62 focuses heavily on ComfyUI Desktop / modern frontend behavior:

- the detached Source / Tail / Result panel appears before generation starts;
- the panel uses the stable high overlay layer so the ComfyUI canvas does not hide it;
- stale panels are removed when a new run starts, a source changes, or a paused run is cancelled;
- the native ComfyUI image upload button is kept;
- oversized preview noise is kept under control;
- the research formula-recommendation toggle is visible again but off by default;
- defaults are reset for a clean public node.

## Current Limits

- Public alpha, not final production software.
- Current public cascade limit: 5 segments.
- Infinite cascade and prompt-per-cascade scheduling are future work.
- Wan-first. Other models may need adapters later.
- Heavy resolutions can be slow or memory intensive.
- Experimental math modes can damage output if pushed too hard.

## Installation

Install through ComfyUI-Manager when available, or clone/copy this folder into:

```text
ComfyUI/custom_nodes/Singularity
```

Restart ComfyUI after installing or updating. For ComfyUI Desktop, a full app restart is often the safest way to refresh Python code and frontend JavaScript.

## Best First Test

Use:

```text
cascade_count = 2
pause_after_cascade_1 = true
frames_per_cascade = 49
fps = 16
seed = 123
math_control_mode = OBSERVE_ONLY
save_video = true
save_report = true
```

Then inspect both the final video and the report.

## Short Description

Singularity is a Wan I2V cascade continuation node for ComfyUI. It pauses between cascade stages, lets you choose the continuation frame, resumes the same run, stitches the final video, and writes diagnostics so you can understand what happened instead of guessing.

