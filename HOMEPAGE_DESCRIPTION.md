# Singularity for ComfyUI

Singularity is a public-alpha ComfyUI custom node for Wan I2V / V2V cascade continuation.

Its main feature is the **Tail 5 Continuation Gate**: the workflow can pause between cascade segments, show the source image plus five tail-frame candidates, let you choose the best continuation frame, then continue the same run and save one final stitched video.

In simple terms:

```text
Generate a segment.
Pause.
Pick the best tail frame.
Continue the same run.
Get one stitched final video and a report.
```

## Why This Matters

Long Wan extension workflows usually require a lot of manual work:

- generate a short clip;
- find a good last frame;
- reload that frame as a new source;
- run another clip;
- stitch separate outputs later;
- guess where the continuity broke.

Singularity moves that loop into one node.

You still make the creative decision, but the node gives you the pause point, frame candidates, continue button, final stitch, and diagnostics.

## Key Features

- One visible ComfyUI node: `Singularity`.
- Wan I2V cascade generation.
- Up to five cascade segments in the current public alpha.
- Optional pause after cascade 1, 2, 3, and 4.
- Tail 5 Continuation Gate:
  - `Source`
  - `Tail 1`
  - `Tail 2`
  - `Tail 3`
  - `Tail 4`
  - `Tail 5`
  - `Result`
- Manual tail-frame selection with a green outline.
- Same-run continuation after selecting a frame.
- One final stitched video.
- Markdown runtime report.
- Public-safe default: `Observe Only`, neutral deltas, reports on, trace off.

## Named Function Layers

**Tail 5 Continuation Gate**  
The frame-choice layer. You choose which tail frame should become the next segment source.

**Same-Run Cascade Stitch**  
The final-output layer. The node saves one stitched video instead of leaving you with separate segment clips.

**Observer Math Baseline**  
The safe default. It records what happened without intentionally changing generation tensors.

**Prompt-Pure Strategy Map**  
Prompt analysis stays report-side. The public default does not inject formula text into the model-facing prompt.

**Completion Report Gate**  
The report tells you whether the route completed. `CompletionGate = PASS` means a final video exists; it is not a visual-quality guarantee.

**Seam And Drift Diagnostics**  
The report can show cascade seam jumps, motion spikes, tail/source continuity pressure, and other clues about where the continuation changed.

## R178 Update

Version:

```text
0.1.1-r178
Singularity R178
Tail 5 Continuation Gate
```

R178 updates the public surface:

- cleaner visible mode names;
- grouped UI sections;
- hidden internal transport controls;
- native ComfyUI image upload preserved;
- Tail 5 panel stabilized for pause/continue;
- clean public baseline with no active generation-math mutation;
- improved report evidence for continuation, seam, and public-readiness checks.

## Public Defaults

```text
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
sampler_trace_mode = OFF
prompt_transcode_mode = Report Only
save_video = true
save_report = true
```

If the default resolution is too heavy, lower it before rendering. A small test size such as `416 x 608` is useful for fast comparisons.

## Important Notes

- This is a public alpha.
- The node is Wan-first.
- Models are not included.
- Current public cascade limit is 5.
- Heavy resolutions require more VRAM and time.
- Always inspect the final MP4.
- `CompletionGate = PASS` means the route completed, not that the video is artistically perfect.
- Research modes are available, but they can change or break the output.

## Recommended First Test

```text
cascade_count = 2
pause_after_cascade_1 = true
frames_per_cascade = 49
fps = 16
seed = 123
math_control_mode = Observe Only
sampler_trace_mode = OFF
prompt_transcode_mode = Report Only
save_video = true
save_report = true
```

Run the workflow, choose the best tail frame at the pause, continue, then review the final stitched video and report.
