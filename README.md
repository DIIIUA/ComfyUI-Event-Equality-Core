# Event Horizon / Event Equality Core

Public Alpha `0.1.1-r59`

Event Horizon is a single ComfyUI node for Wan image-to-video workflows. It wraps generation, cascade extension, runtime reporting, motion diagnostics, and Event Equality math records into one visible node.

This release is experimental. It is meant for users and developers who want a compact Wan I2V node that also writes useful evidence about what happened during generation.

For changes since the previous public build, see `CHANGELOG.md`.

## Scientific Foundation & Formalism

The underlying mathematical logic of this node, specifically the `dim=1` Semantic Unfolding and inertia preservation, is based on the **Recursive Strategy Scale Formalism**.
This logic is part of a published scientific work by DIIIUA. A copy of the formal paper ("recursive_strategy_scale_formalism_v0_3_en.pdf") is available on FigShare under a CC BY 4.0 license, which protects the academic and scientific priority of the architecture. Please see the `LICENSE` file for strict rules against commercializing this specific codebase.

## What Is Included

- One visible ComfyUI node: `Event Horizon`
- Wan I2V single-video and cascade generation
- Positive and negative prompt fields with locked readable height
- Optional secondary model branch
- Runtime report output
- Motion math metrics
- RouteMemory, S-Wire, SState, CompletionGate, and conflict records
- Smart cleanup/barrier logging
- Shadow sampler trace records
- Input normalization and integrity reports

The package intentionally does not include handoff archives, old reports, test videos, backups, private workflows, or development-only debug nodes.

## Installation

Copy this folder into:

```text
ComfyUI/custom_nodes/ComfyUI-Event-Equality-Core
```

Restart ComfyUI completely after installing or replacing the node.

In ComfyUI, add:

```text
Event Equality / Event Horizon / Event Horizon
```

## Required Inputs

- `primary_model`: the main Wan model branch.
- `clip`: CLIP/Text encoder connection.
- `vae`: VAE used for decode.
- `source_image_file`: source image loaded from ComfyUI input images.
- `positive_prompt`: what the video should follow.
- `negative_prompt`: what the video should avoid.

Optional:

- `secondary_model`: second Wan model branch for low/secondary sampler stage.
- `image`: direct IMAGE input if a workflow provides one.
- `mask`: optional mask input.

## Main Controls

- `cascade_count`: number of video segments to generate and connect. `1` is a normal single clip. Higher values extend the clip.
- `frames_per_cascade`: frames generated per segment.
- `width`, `height`, `fps`: output dimensions and playback rate.
- `seed`: fixed seed for repeatable comparisons.
- `sampler_name`, `scheduler`, `global_steps`: sampler configuration passed into the Wan sampling path.
- `primary_cfg`, `secondary_cfg`: CFG values for primary and secondary branches.
- `primary_start_step`, `primary_end_step`: step window for the primary/high stage.
- `secondary_start_step`, `secondary_end_step`: step window for the secondary/low stage.
- `decode_tile_size`, `decode_overlap`, `decode_temporal_size`, `decode_temporal_overlap`: tiled decode controls.
- `cleanup_timing`: when the node tries to release memory and record the cleanup boundary.
- `save_video`: writes the generated video.
- `save_report`: writes the markdown report.
- `save_prefix`: file prefix for outputs.

## Math Control Modes

`OBSERVE_ONLY`

Records the Event Equality math without intentionally changing the sampler result. Use this as a baseline when you want to compare behavior.

`LATENT_DELTA_SCALE`

The public default. It treats the high/low latent delta as measured `ObservedBehavior` and applies controlled scaling through the safe native-sampler overlay path. In practical terms, this lets you slightly reduce or amplify how strongly a stage pushes the latent route while still preserving the model sampler as the main generator.

`DEEP_STEP_DELTA_CONTROL`

Experimental. This activates the native step-loop replacement path and can strongly affect output stability. Use only for research sweeps with fixed seeds and saved reports. If the result becomes noisy, lower `high_delta_strength` / `low_delta_strength` or return to `LATENT_DELTA_SCALE`.

## Delta Strength & Inertia

- `high_delta_strength`: scales the observed high-stage latent movement.
- `low_delta_strength`: scales the observed low-stage latent movement.
- `inertia_mass`: applies a Deep EMA (Exponential Moving Average) Momentum buffer to the latent vector path, controlling physical boundary collision preservation. (Range: 0.0 to 1.0, Default: 0.5)

`1.0` delta strength means neutral. Values below `1.0` reduce that stage movement; values above `1.0` amplify it.

### Semantic Normalization (Spatial Unfolding)
`inertia_mass` uses `dim=1` (Channels) for `torch.norm`. This forces the momentum vector to remain unfolded spatially, meaning each individual pixel (like a subject's skin vs a rigid object) maintains its own precise local physics without blending into the whole scene. This prevents "clipping" and structural melting during object interactions.

In Event Equality terms, delta strength and inertia are not just quality sliders. They change measured `ObservedBehavior`, which changes the next latent state and therefore the strategy carrier handed to the next stage.

## Drift

In this node, drift means measurable separation between the intended route and the actual generated route.

Common drift types:

- Source drift: the video stops respecting the input image.
- Prompt drift: the output moves away from the positive prompt or toward the negative prompt.
- Identity drift: the main subject changes too much over time.
- Motion drift: movement becomes unstable, reversed, spiky, or inconsistent.
- Cascade drift: the next cascade segment does not continue the previous segment cleanly.
- Route drift: the recorded stage order, input state, or expected boundary does not match the intended generation route.

The report does not magically prevent drift. It makes drift visible so you can compare runs instead of guessing from the video alone.

## Motion Metrics In The Report

- `frame_delta_norm_mean`: average frame-to-frame movement.
- `frame_delta_norm_std`: how uneven that movement is.
- `frame_delta_spike_ratio`: whether a few frames move much more than the rest.
- `frame_delta_cosine_mean`: whether consecutive movement directions are aligned.
- `frame_delta_reversal_ratio`: how often motion direction reverses.
- `frame_delta_jerk_ratio`: acceleration/jerk proxy.
- `frame_motion_stability_score`: observer-only heuristic for comparing runs.
- `frame_motion_profile`: simple label such as `stable`, `mixed`, or `volatile`.

These are comparison tools, not final quality scores. Always look at the video too.

## CompletionGate

`EventCoreBodyCompletionGate = PASS` means the node completed the expected structural route and wrote the expected records. It does not mean the video is aesthetically good.

A useful run should normally have:

```text
result_status = VIDEO
EventCoreBodyCompletionGate = PASS
saved_video_path is not empty
saved_report_path is not empty
```

## Shadow Sampler Trace

`sampler_trace_mode = SHADOW_STEP_TRACE` records limited step-level shadow information without trying to become a full sampler replacement. `sampler_trace_max_steps` caps how much trace data is written.

This is diagnostic data. It is useful for development and comparison, but it can make reports larger.

## Recommended First Test

Use a fresh `Event Horizon` node after installing.

Suggested conservative test:

```text
cascade_count = 1
frames_per_cascade = 49
width = 416
height = 608
fps = 16
math_control_mode = OBSERVE_ONLY
high_delta_strength = 1.0
low_delta_strength = 1.0
save_video = true
save_report = true
```

After that, compare with:

```text
math_control_mode = LATENT_DELTA_SCALE
high_delta_strength = 0.988 to 1.0
low_delta_strength = 1.0
```

Keep the same seed when comparing runs.

## Known Limits

- Public alpha: expect rough edges.
- Built and tested around Wan I2V-style workflows.
- Fixed-seed reproducibility can change across ComfyUI, model, scheduler, torch, and driver versions.
- `CompletionGate PASS` is structural evidence, not a visual-quality guarantee.
- Deep step control is research-grade and can produce unstable output.
- Full formula coverage at every denoising step is still an active research direction.

## Outputs

The node returns:

- `status`: compact run status.
- `saved_video_path`: video file path when saving succeeds.
- `saved_report_path`: markdown report path when report saving succeeds.
- `report`: report text.
