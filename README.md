# Singularity

Singularity is a public-alpha ComfyUI custom node for Wan image-to-video cascade continuation.

Its main job is simple: generate a Wan video segment, pause at the cascade boundary, show you the source image and five tail-frame candidates, let you choose the frame that should continue the video, resume the same run, and save one final stitched video.

This is built for Wan I2V / V2V users who want longer videos without rebuilding a large continuation workflow by hand after every short clip.

## What You Get

- One visible ComfyUI node: `Singularity`.
- Wan I2V cascade generation.
- Up to five cascade segments in this public alpha.
- Optional pause after cascade 1, 2, 3, and 4.
- A detached Source / Tail 1 / Tail 2 / Tail 3 / Tail 4 / Tail 5 / Result panel under the node.
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

## Current Release Notes

- The current public stabilization release is `0.1.1-r113`.
- The visible node title includes the R-version: `Singularity R113`.
- r113 is a ComfyUI Desktop / modern frontend hotfix and stabilization release after the r112 widget-order regression.
- r113 keeps ComfyUI's backend widget order stable so saved workflow values stay in the correct fields after restart/save/reload.
- r113 adds a severe widget-drift repair guard for workflows that already show impossible shifted values, such as `width = 49`, `fps = 608`, or `global_steps = NaN`.
- r113 keeps the public starter node clean: no source image, empty positive prompt, built-in Wan-style Chinese negative prompt, reports enabled, two cascades, one pause.
- r113 smoke-test status: two-cascade pause/continue produced `VIDEO`, `CompletionGate = PASS`, `EventCoreBodySummary = PASS`, final stitched video saved, `adjustment_count = 0`, and no severe widget drift repair was triggered.
- The pause panel now exposes five tail-frame slots instead of three, giving you more continuation choices.
- `image_crop = wan_native` keeps the source image as the SourceAnchor until official Wan normalization.
- Prompt transform modes are now prompt-pure: they build a semantic density/context map for the report, but they do not inject formula language into the CLIP prompt.
- Cascade Continue now reuses the current clean prompt StrategyCandidate when the prompt identity is unchanged.
- r90 adds `EventStrategyControlSurfacePlan` and `EventStrategyControlSurfaceApply_*`.
- In r90, `OBSERVE_ONLY`, `LATENT_DELTA_SCALE`, `STRATEGY_PRESSURE_WINDOW`, and `DEEP_STEP_DELTA_CONTROL` are no longer separate math islands. They are policies under one Strategy control surface.
- r89 adds `STRATEGY_PRESSURE_WINDOW`, the first unified functional math surface after the r88 pressure resolver.
- In `STRATEGY_PRESSURE_WINDOW`, `high_delta_strength` and `low_delta_strength` are read as pressure intent and compressed into a small bounded window around `1.0`, so extreme values are studied without instantly replacing the model's native sampler behavior.
- r89 keeps CFG and prompt text clean: it does not write topology prose into the prompt and does not change the sampler step loop.
- r88 adds `EventStrategyReturnPressureResolver`, a report-only topology layer that combines high/low sampler pressure, visible frame motion pressure, late-segment spikes, tail Strategy pressure, object relation pressure, and source-anchor pressure into one Strategy-return pressure vector.
- r88 does not change generation by itself; it tells you which non-text control surface should be tested next after the local sub-strategies return to the global Strategy route.
- r87 protects Continue from partial widget payload drift: if the positive or negative prompt payload looks missing/truncated, the backend reuses the current active prompt carrier instead of falsely treating the next cascade as a changed prompt.
- r84 fixed the r83 pause/continue regression where post-transform prompt checks could stop the cascade before the next segment.
- r85 adds a report-only topology Strategy return map: local prompt/source/sampler/object/tail/frame formulas can be inspected as sub-strategies that must return to the main route Strategy.
- r85 does not change generation by itself; it makes the next math tests more accountable.
- r86 adds `EventPromptPurityLock` and `semantic_density_context_map` so meaning density and context density can be inspected without turning math notes into prompt words.
- Continue identity now compares clean raw/sanitized prompt carriers, so the same user intent can continue without rebuilding a second text layer.
- Continue payload hygiene now protects against missing or truncated positive/negative prompt widget payloads when the same Strategy identity should continue.
- Prompt transform keeps legacy generated-text candidates as report-only evidence; the model-facing positive and negative prompts stay clean.
- If you edit the prompt during a pause, the next cascade intentionally receives the new StrategyCarrier.
- Reports now include relation pressure cards for prompt continuity, low-branch pressure, object carrier identity, tail continuity, frame-spike attribution, SourceAnchor preservation, global Strategy return, and topology Strategy return.

For public packaging, keep the visible default conservative: two cascades, one pause, `OBSERVE_ONLY`, reports enabled, and formula recommendation off.

## Public Defaults

The public build starts clean:

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
image_crop = wan_native
save_video = true
save_report = true
sampler_trace_mode = OFF
formula recommendation = visible, experimental, and off by default
```

If `704 x 1280` is too heavy for your GPU, reduce the size before rendering. A lighter test size such as `416 x 608` is useful for fast debugging.

## r113 Public Stabilization

r113 is the current public stabilization update.

It combines the pause/continue UI work from the earlier public alpha line with the newer Strategy reports and a critical ComfyUI Desktop widget-order fix.

New and stabilized in r113:

- Visible node title: `Singularity R113`.
- Backend widget order is stable again; saved workflow values should remain in the correct fields.
- Public starter node has no bundled image and no positive prompt.
- Built-in Wan-style Chinese negative prompt remains as the starter negative prompt.
- Source / Tail / Result panel remains detached under the node.
- Tail selection now shows five continuation candidate slots.
- Native ComfyUI image upload button is preserved.
- Same-run pause/continue works for two-cascade and multi-cascade workflows.
- Final output is one stitched video.
- Reports include cascade, motion, delta, runtime, Strategy Matrix, Strategy Control Surface, and widget/input normalization evidence.
- Experimental math is available, but the default remains conservative.

The r113 smoke test used:

```text
runtime_version = 0.1.1-r113
result_status = VIDEO
CompletionGate = PASS
cascade_count = 2
completed_segments = 2
frames_per_cascade = 49
width = 416
height = 608
fps = 16
adjustment_count = 0
```

The public package is labeled `0.1.1-r113`.

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
Source | Tail 1 | Tail 2 | Tail 3 | Tail 4 | Tail 5 | Result
```

What each tile means:

- `Source`: the current source frame for the segment.
- `Tail 1` through `Tail 5`: continuation candidates from the end of the segment.
- `Result`: a preview of the stitched video up to the current pause boundary when available.

The selected tail is outlined in green. That selected frame becomes the source for the next cascade segment.

## Main Controls

`source_image_file`

The starting image. Use the native ComfyUI upload button if the image is not already in the input folder.

`positive_prompt`

What you want the video to show.

`negative_prompt`

What you want the video to avoid. r113 includes a simple Wan-style base negative prompt by default so a new node is not filled with a test scene.

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

The generation resolution. The r113 default is a 720-class vertical setting (`704 x 1280`). Lower it if you need faster tests or have limited VRAM.

`seed`

The random seed. r113 defaults to `123` so comparisons can start from a stable baseline.

`image_crop`

Controls where the source-image normalization happens.

```text
wan_native = send the source image directly to WanImageToVideo; Wan owns its normal resize/center crop
disabled = pre-scale inside Singularity without center crop, then send to Wan
center = pre-scale and center-crop inside Singularity, then send to Wan
```

For most users, keep `wan_native`. It matches the normal Wan workflow more closely and avoids an extra resize/crop pass before Wan sees the source image.

## Math Controls

Singularity studies generation as an event:

```text
Outcome(t-1) + ObservedBehavior(t-1)
= Strategy(t)
= ObservedBehavior(t+1) + Outcome(t+1)
```

For normal users, this means the node tries to keep the prompt, source image, model behavior, latent motion, and final video understandable as one route.

`prompt_transcode_mode`

Controls the prompt-topology research map.

```text
REPORT_ONLY = record the map only
TRANSFORM_PROMPT = build the semantic density/context map, but keep the actual CLIP prompt clean
```

Since r86, this does not add topology wording to the prompt. The prompt remains the user's prompt. The math layer sorts meaning density against context density in the report/control space so future active controls can be tested without accidental prompt pollution.

`math_control_mode`

Controls how active the math layer is.

```text
OBSERVE_ONLY = record reports and diagnostics without intentional tensor mutation
LATENT_DELTA_SCALE = controlled delta-strength research path
STRATEGY_PRESSURE_WINDOW = bounded pressure-window policy inside the Strategy Control Surface
DEEP_STEP_DELTA_CONTROL = experimental deep research mode, high risk
```

r113 public default is `OBSERVE_ONLY`.

`high_delta_strength`

Controls the high-stage delta strength when `LATENT_DELTA_SCALE` or `STRATEGY_PRESSURE_WINDOW` is enabled.

Simple version: it changes how strongly the first/high stage motion is carried forward. In `STRATEGY_PRESSURE_WINDOW`, this value is treated as intent and compressed into a safer narrow window.

`low_delta_strength`

Controls the low-stage delta strength when `LATENT_DELTA_SCALE` or `STRATEGY_PRESSURE_WINDOW` is enabled.

Simple version: it changes how strongly the refinement stage can reshape the result. In `STRATEGY_PRESSURE_WINDOW`, this is the main place to test visible refinement pressure without letting the low sampler explode into raw noise. Since r90, the final applied value is chosen by `EventStrategyControlSurfaceApply_low`.

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

## r113 UI And Widget Fixes

r113 focuses heavily on ComfyUI Desktop / modern frontend behavior:

- the detached Source / Tail / Result panel appears before generation starts;
- the panel uses the stable high overlay layer so the ComfyUI canvas does not hide it;
- stale panels are removed when a new run starts, a source changes, or a paused run is cancelled;
- the native ComfyUI image upload button is kept;
- oversized preview noise is kept under control;
- the research formula-recommendation toggle is visible again but off by default;
- backend widget order is preserved to prevent save/reload value drift;
- severe positional widget drift is detected and reset to safe defaults when possible;
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

