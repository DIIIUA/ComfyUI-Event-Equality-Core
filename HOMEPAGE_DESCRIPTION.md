# Singularity for ComfyUI

Singularity is a public-alpha ComfyUI custom node for Wan image-to-video cascade continuation.

It is made for users who want longer Wan I2V / V2V videos without manually extracting the last frame, loading it again, making another short clip, and stitching everything later.

In simple terms:

```text
Generate a segment.
Pause.
Look at the last-frame candidates.
Choose the best continuation frame.
Continue the same run.
Get one stitched final video.
```

## Main Feature

The special feature is the pause-and-continue panel.

When a cascade boundary is reached, Singularity shows:

```text
Source | Tail 1 | Tail 2 | Tail 3 | Result
```

You click the tail frame that should become the source for the next cascade, then press `Resume Cascade / Continue`.

This gives you direct control over the continuation point instead of letting the workflow blindly continue from a bad or unstable frame.

## What It Does

- Adds one visible ComfyUI node: `Singularity`.
- Generates Wan I2V video segments.
- Supports up to five cascade segments in the current public alpha.
- Can pause after cascade 1, 2, 3, or 4.
- Lets you manually pick the tail frame for the next segment.
- Continues the same running workflow.
- Saves one final stitched video.
- Saves markdown reports with runtime, cascade, motion, and delta diagnostics.
- Keeps a clean public interface with research-only formula recommendation hidden and disabled.

## Why This Is Useful

Wan extension workflows can become messy very quickly:

- many repeated sampler groups;
- manual last-frame extraction;
- separate output clips;
- manual stitching;
- unclear continuity between segments;
- hard-to-compare prompt, seed, and delta changes.

Singularity turns that into a controlled loop inside one node. You still make the creative decision, but the node handles the pause point, the chosen frame, continuation, final stitching, and report evidence.

## r60 Public Update

This r60 public alpha focuses on making the cascade frame-selection UI usable in modern ComfyUI Desktop and safer for public testing.

New in r60:

- Better detached Source / Tail / Result panel behavior.
- Panel no longer appears over normal ComfyUI modal dialogs.
- Panel hides when internal workflow panels overlap it.
- Stale panels are cleaned up on new runs, source changes, cancelled pauses, and errors.
- Native ComfyUI image upload button is preserved.
- Oversized preview/noise behavior is kept under control.
- Public defaults are reset to a clean two-cascade test route.
- Positive prompt starts empty.
- Source image starts as `none`.
- `OBSERVE_ONLY` is now the public default math mode.
- Delta strengths stay neutral at `1.0 / 1.0`.
- Formula recommendation is hidden and disabled for public use.

## Public Default Settings

The default r60 node is meant to be a clean starter:

```text
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
```

If `704 x 1280` is too heavy for your GPU, lower the resolution before rendering.

## About The Math

The project studies generation as an event:

```text
Outcome(t-1) + ObservedBehavior(t-1)
= Strategy(t)
= ObservedBehavior(t+1) + Outcome(t+1)
```

For a normal user, this means the node tries to make the source image, prompt meaning, model interpretation, sampler route, latent motion, and visible video easier to inspect as one connected process.

The math is not a magic quality button. It is a way to observe, compare, and carefully test how generation changes.

Use `OBSERVE_ONLY` for a clean baseline.

Use `LATENT_DELTA_SCALE` only when intentionally testing delta behavior.

Avoid `DEEP_STEP_DELTA_CONTROL` unless you are doing research and accept that it can break output.

## What Is Drift?

Drift means the video moves away from your intended result.

Examples:

- the source image stops being respected;
- the subject identity changes;
- the prompt becomes ignored;
- motion becomes unstable;
- the next cascade does not continue naturally;
- experimental math settings make the image noisy or broken.

Singularity helps you notice and compare drift. It does not guarantee that every video will be perfect.

## Important Public Alpha Notes

- This is not a final production release.
- Current public cascade limit is 5.
- Infinite cascades and prompt-per-cascade scheduling are future work.
- The node is Wan-first.
- Models are not included.
- Heavy resolutions require more VRAM and time.
- Always inspect the final video, not only the report.
- `CompletionGate = PASS` means the route completed and a final video exists, not that the video is visually perfect.

## Recommended First Test

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

Pick a source image, write a clear prompt, generate, choose a tail frame when the panel appears, continue, then inspect the final stitched video.

## Short Summary

Singularity is a Wan I2V cascade continuation node for ComfyUI. It pauses between cascade stages, lets you choose the continuation frame, resumes the same run, stitches the final video, and writes diagnostics so you can understand the result instead of guessing.
