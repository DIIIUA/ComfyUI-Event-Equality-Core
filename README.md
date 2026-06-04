# Singularity

Singularity is an experimental ComfyUI custom node for Wan image-to-video cascade generation, manual continuation control, and runtime diagnostics.

Its main feature is simple but uncommon inside ComfyUI: pause a cascade chain, inspect the source frame and tail-frame candidates, choose the continuation frame, continue the same running workflow, and receive one final stitched video instead of a pile of separate clips.

This is a public alpha. It is built for testing, comparison, and long-video experimentation, not for guaranteed production stability.

## In Plain English

Think of Singularity as a long-video helper.

Normally, if you want to extend a Wan video, you generate one clip, find a good last frame, load that frame again, run another clip, and then stitch everything later. Singularity tries to keep that loop inside one ComfyUI node.

The basic flow is:

```text
1. Generate the first segment.
2. Pause.
3. Show the source image and tail-frame candidates.
4. Let you choose the best continuation frame.
5. Continue the same run.
6. Stitch the result into one final video.
```

The node also writes reports so you can compare runs instead of guessing only from the final video.

## What It Does

- Generates Wan I2V video from one external node.
- Supports up to five cascade segments in the current public alpha.
- Can pause at cascade boundaries.
- Shows a detached Source / Tail 1 / Tail 2 / Tail 3 / Result panel under the node.
- Lets the user pick the tail frame used to continue the next cascade.
- Continues the same run after pressing `Resume Cascade / Continue`.
- Produces one stitched final video at the end.
- Writes a markdown report plus runtime monitor sidecars.
- Records CascadePlan, segment begin/end records, motion metrics, delta diagnostics, and CompletionGate status.

## Why This Exists

Long Wan videos often require manually chaining clips, choosing a last frame, reloading it, running a new workflow, and stitching the pieces later. Singularity puts that decision point inside the workflow: the user can stop at a boundary, choose the frame that carries the best continuation strategy, and let the node continue.

The project also studies generation as an event:

```text
Outcome(t-1) + ObservedBehavior(t-1)
= Strategy(t)
= ObservedBehavior(t+1) + Outcome(t+1)
```

In practical terms, the prompt, source image, model interpretation, high/low sampler route, latent evolution, and final visible video should describe the same event. The math layer is meant to observe and gently guide that relation. It should not blindly replace native sampler physics.

## Quick Start

1. Add the `Singularity` node.
2. Connect your Wan route:
   - `primary_model`
   - optional `secondary_model`
   - `clip`
   - `vae`
3. Pick or upload a source image in `source_image_file`.
4. Write your positive and negative prompts.
5. For a first simple test, use:

```text
cascade_count = 1
frames_per_cascade = 49
math_control_mode = LATENT_DELTA_SCALE
high_delta_strength = 1.0
low_delta_strength = 1.0
save_video = true
save_report = true
```

6. For the frame-selection feature, use:

```text
cascade_count = 2
pause_after_cascade_1 = true
frames_per_cascade = 49
```

7. When the panel appears under the node, click the tail frame you want and press `Resume Cascade / Continue`.

## Current Public Alpha Scope

Recommended public use:

```text
cascade_count = 1..5
frames_per_cascade = 49
math_control_mode = LATENT_DELTA_SCALE or OBSERVE_ONLY
high_delta_strength = 1.0
low_delta_strength = 1.0
sampler_trace_mode = OFF
save_video = true
save_report = true
```

Research example:

```text
math_control_mode = LATENT_DELTA_SCALE
high_delta_strength = 0.988
low_delta_strength = 1.0
```

Do not treat that value as universal. It is only a comparison candidate found during local testing.

## What The Main Inputs Mean

### Model Inputs

`primary_model`

The main Wan model input. In a high/low Wan setup, this is usually the high-noise or motion-structure model.

`secondary_model`

Optional second model input. In a high/low Wan setup, this is usually the low-noise or refinement model. If it is not connected, the node can fall back to the primary model for the low phase.

`clip`

The text encoder used for the prompts.

`vae`

The decoder used to turn latent frames into visible frames/video.

### Image And Prompt

`source_image_file`

The starting image. Use the upload button if the image is not already in ComfyUI's input folder.

`positive_prompt`

What you want the video to show.

`negative_prompt`

What you want the video to avoid.

### Cascade Controls

`cascade_count`

How many segments to generate.

```text
1 = one normal clip
2 = first clip + one continuation
5 = current public-alpha maximum
```

`pause_after_cascade_1..4`

Where the node should pause and ask you to choose a continuation frame.

Example:

```text
cascade_count = 2
pause_after_cascade_1 = true
```

This means: generate segment 1, pause, let you choose a tail frame, then continue segment 2.

`frames_per_cascade`

How many frames each segment generates before trim/stitch logic.

At `16 fps`:

```text
49 frames = about 3 seconds per segment
121 frames = about 7.5 seconds per segment
```

### Tail Frame Controls

`selected_tail_index`

The selected tail candidate. The UI usually updates this for you when you click a tail image.

```text
0 = Tail 1
1 = Tail 2
2 = Tail 3
```

`use_formula_recommendation`

Research option. When enabled, the formula recommendation may propose a tail frame. For public use, keep this off unless you are intentionally testing it.

Important: the manual green selection is the user's real choice.

### Math Controls

`math_control_mode`

Chooses how much the math layer is allowed to do.

`high_delta_strength`

Controls the high-stage delta strength. In simple terms: how strongly the first/high stage's movement is carried forward.

`low_delta_strength`

Controls the low-stage delta strength. In simple terms: how strongly the refinement stage is allowed to reshape the result.

Start with:

```text
high_delta_strength = 1.0
low_delta_strength = 1.0
```

Then change only one value at a time when comparing.

### Save And Report

`save_video`

Saves the final output video.

`save_report`

Saves the markdown diagnostic report. Keep this on while testing.

`save_prefix`

The filename prefix for saved outputs.

### Trace Controls

`sampler_trace_mode`

Extra diagnostics for sampler behavior.

```text
OFF = normal public use
SHADOW_STEP_TRACE = diagnostic trace, more report data
```

`sampler_trace_max_steps`

Limits trace size so reports do not grow without control.

## Math Modes

### OBSERVE_ONLY

Records reports and diagnostics without intentionally changing the generation tensors. Use this as a clean baseline.

### LATENT_DELTA_SCALE

Public-safe default path. It preserves the native sampler window and applies controlled delta scaling only through the exposed strengths. Neutral values (`1.0`, `1.0`) are intended to behave as a neutral comparison baseline.

### DEEP_STEP_DELTA_CONTROL

Research mode. It touches the deeper step route and can produce noise or unstable output. Use only for controlled experiments.

## What Drift Means

Drift means the generated result starts separating from the intended route.

Examples:

- Source drift: the video stops respecting the source image.
- Prompt drift: the output moves away from the prompt.
- Identity drift: the subject changes too much over time.
- Motion drift: movement becomes unstable, reversed, or too chaotic.
- Cascade drift: the next segment does not continue the previous segment cleanly.
- Math drift: experimental math starts overriding generation instead of guiding it.

Singularity does not magically remove drift. It gives you evidence for where drift appears.

## CompletionGate

`CompletionGate = PASS` means the structural route completed:

- the requested stages were recorded;
- the cascade route reached a final output;
- `result_status = VIDEO`;
- a final video path exists.

It does not mean the video is visually good. Always inspect the video itself.

Cancelled or no-video runs should not report final PASS.

## Installation

Copy or clone this folder into:

```text
ComfyUI/custom_nodes/Singularity
```

Then restart ComfyUI.

No extra Python requirements are currently needed beyond the ComfyUI runtime and the custom nodes your Wan workflow already uses.

## Known Limits

- Public alpha, not final stable release.
- Wan I2V is the current test route.
- The public cascade limit is 5.
- Full N-cascade policy UI is planned later.
- Model loading, LoRA loading, and Torch Compile are still external workflow responsibilities.
- Deep math is research-only.
- CompletionGate is structural, not aesthetic.

## Suggested Test

Start with:

```text
cascade_count = 2
pause_after_cascade_1 = true
frames_per_cascade = 49
seed = 123
math_control_mode = LATENT_DELTA_SCALE
high_delta_strength = 1.0
low_delta_strength = 1.0
save_video = true
save_report = true
```

When the pause panel appears, choose one of the tail candidates and press `Resume Cascade / Continue`. The final output should be one stitched video.
