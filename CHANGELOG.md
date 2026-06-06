# Singularity - Public Changelog

## 0.1.1-r61 Public Alpha Desktop Cascade UI Fix

Status: public alpha update for ComfyUI Desktop / modern frontend cascade continuation.

The main r61 goal is to fix a public r60 regression where the backend could pause correctly but the detached Source / Tail / Result panel might not appear in modern ComfyUI Desktop.

### Emergency Hotfix

- Added a status-polling fallback through `/singularity/cascade/status/{node_id}`.
- The pause UI no longer depends only on one websocket event.
- If ComfyUI Desktop drops or misses the pause event, the frontend can recover the paused state, show the tail frames, and enable `Resume Cascade / Continue`.
- Narrowed the overlay occlusion rules so generic ComfyUI panel containers do not hide the pause UI accidentally.
- Kept the public manual green tail selection route. Formula recommendation remains hidden and disabled.

The broader r61/r60 goal remains to make the manual cascade frame-selection workflow safer for public users: cleaner defaults, less UI overlap, preserved native image upload, and a public interface that does not expose research controls as normal user switches.

### Short Summary

- Updated runtime/package version to `0.1.1-r61`.
- Improved the detached Source / Tail 1 / Tail 2 / Tail 3 / Result pause panel.
- The pause panel now hides behind normal ComfyUI modal/dialog surfaces instead of drawing over them.
- The pause panel also hides when internal workflow panels or drawers overlap it.
- Kept the native ComfyUI `source_image_file` upload button.
- Hid and disabled the public UI for `use_formula_recommendation`; the internal argument remains for old workflow compatibility, but public users no longer see it as a normal option.
- Reset public defaults for a clean starter node:
  - `source_image_file = none`
  - `positive_prompt = empty`
  - built-in base negative prompt
  - `cascade_count = 2`
  - `pause_after_cascade_1 = true`
  - `width = 704`
  - `height = 1280`
  - `seed = 123`
  - `math_control_mode = OBSERVE_ONLY`
  - `image_crop = center`
- Rewrote the README and homepage/CVTI description for normal users, with plain explanations of cascade continuation, drift, math modes, reports, and public-alpha limits.

### What Changed For Users

The node now opens as a clean two-cascade frame-selection test instead of a local development preset. A new user can add the node, connect the model/CLIP/VAE route, choose or upload a source image, write a prompt, generate, choose a tail frame, continue, and inspect one final stitched video.

The math layer is still present, but the public default is observer-first:

```text
math_control_mode = OBSERVE_ONLY
high_delta_strength = 1.0
low_delta_strength = 1.0
```

This means r61 starts as a safer baseline. Users who want to test delta behavior can intentionally switch to `LATENT_DELTA_SCALE`.

### UI Notes

r61 specifically targets the ComfyUI Desktop behavior seen after the modern frontend update. The pause panel is still detached under the node, but it now treats modals, manager windows, workflow panels, sidebars, and drawers as blocking surfaces when they cover the panel area.

### Compatibility Notes

Old workflows that already contain `use_formula_recommendation` should not fail merely because the public UI hides it. The field remains accepted by the Python signature, but new public nodes force the visible user route to manual green tail selection.

### Still Public Alpha

- Current public cascade limit: 5.
- Infinite cascades are not implemented yet.
- Prompt-per-cascade scheduling is future work.
- `CompletionGate = PASS` means the route completed and a final video exists. It does not guarantee visual quality.

## 2026-06-05 - Public Alpha Package Prep

Status: public alpha packaging pass for the active cascade frame-selection build.

- Rewrote the public README in English.
- Updated package metadata to describe the real feature: Wan I2V cascade continuation with manual tail-frame selection and diagnostics.
- Clarified that the public value is frame selection, same-run continuation, final stitched output, and report evidence.
- Clarified that math modes are diagnostic/research controls, not a guarantee of visual quality.
- Confirmed the current delta sweep labels: recent solo runs were high-delta tests, not low-delta tests.
- Prepared the release surface for ComfyUI-Manager registration.

## 2026-06-04 - CascadePlan / Public Alpha Gate Candidate

Status: local release-candidate documentation and telemetry build, installed into the active ComfyUI desktop node. Requires full ComfyUI Desktop restart before physical testing.

- Added `SingularityCascadePlan` as the report-level route contract for cascade execution.
- The plan records requested segment count, final segment index, frames per cascade, pause boundaries, ignored pause flags beyond the selected cascade count, and expected output frames before manual trims.
- Updated CompletionGate semantics: `PASS` now requires the requested cascade route to reach a final video outcome, not merely a structurally coherent partial route.
- Added `cascade_progress`, `route_complete`, `final_output_ok`, and `blocking_reasons` to the Event Core Body summary/gate records.
- Cancelled/no-video paused runs are diagnostic only and should report `CANCELLED` or blocked status instead of false `PASS`.
- Kept the current public-safe cascade limit at 5 while documenting future N-cascade pause policies as a later feature.
- Added internal release readiness docs for a Public Alpha focused on manual frame selection across up to five cascades.

Public Alpha gate:

```text
5-cascade frame selection produces one final stitched VIDEO
report contains SingularityCascadePlan
CompletionGate = PASS only for final VIDEO
cancelled/no-video run does not receive PASS
```

## 2026-06-04 - Local Cascade Resume UI Stabilization

Status: local stabilization build, installed into the active ComfyUI desktop node.

- Confirmed same-run cascade resume: the second cascade continues from the selected tail frame instead of restarting a fresh video.
- Added a detached media overlay below the node with `Source`, three tail candidates, and `Result`.
- Enlarged the detached media overlay so frame choices can be inspected without zooming tightly into the node. The overlay is allowed to be wider than the node itself.
- Kept the Continue button visible only while the cascade is paused; it disappears after the run completes.
- Restored the native ComfyUI image upload button on `source_image_file` so users can browse/upload new source images, not only pick already-known Comfy input assets.
- Added Singularity-specific handling for VHS latent preview: `vhslatentpreview` is kept as a small fixed-size live preview instead of scaling from the full node width.
- Native/frozen image and video previews are hidden in favor of the Singularity media overlay, so the post-run panel reflects source, tail candidates, and result instead of stale noise.
- Fixed media overlay lifecycle after cancelled or restarted cascades: stale Source/Tail/Result panels are now removed on source changes, fresh execution starts, and new pause events so old UI cannot stack over the new run.
- Added real paused-cascade cancel handling: the UI now has `Cancel Pause`, standard ComfyUI interrupt is patched to notify Singularity's backend wait loop, and `/singularity/cascade/cancel` can cancel all active paused cascade states without pressing Continue first.
- Reworked cascade pause boundaries after cascade 2/3/4: every pause now uses the same same-run wait/continue path as cascade 1, trims the selected decoded frame batch in code, keeps all decoded segment batches alive, and runs one final batch concat/video combine instead of saving intermediate PAUSED videos.
- Added pause-time stitched preview media: the `Result` tile now shows a temp-only preview of the already-stitched video from the first frame through the current cascade boundary, while the final saved video is still produced only once at the end.
- Clarified cascade report numbering: the first generated body is now recorded as cascade segment 1, so reports show segments 1-5 instead of only continuation segments 2-5.
- Restored report hygiene for public testing: sampler begin/proposal records now carry explicit statuses, final Event Core Body counts are derived from runtime records instead of writing misleading zeros, and runtime-monitor JSON/CSV/diff sidecars are written next to the saved markdown report.

## 0.1.1-r59 Public Alpha Clean

Compared with the previous public build:

```text
previous public: ComfyUI-Event-Equality-Core_v0.1.1-r56_PublicAlphaCleanReadme.zip
current public:  ComfyUI-Event-Equality-Core_v0.1.1-r59_PublicAlphaClean.zip
```

Status: public alpha / experimental.

The main r59 goal is to keep one clean user-facing node, `Singularity`, while adding stronger diagnostics, safer input handling, and a better placement for the math layer so it does not break the native generation physics.

## Short Summary

Compared with r56:

- Added r57 motion-math metrics.
- Added r58 input-integrity hardening.
- Corrected r59 `LATENT_DELTA_SCALE`: it now uses a safe semantic overlay / native-sampler-preserving path instead of replacing the sampler step loop by default.
- Added the separate experimental `DEEP_STEP_DELTA_CONTROL` mode for deeper math research.
- Cleaned the public package: one external node, no debug nodes, no old workflow aliases, no examples, no checklists, no test artifacts.
- Rewrote the README as public user documentation instead of an internal handoff note.

## Important Workflow Change

r56 public exported:

```text
Singularity
SingularityR56PublicAlpha
EventDebugPing
EventSaveReportToFile
```

r59 public clean exports only:

```text
Singularity
```

In the ComfyUI interface this appears as:

```text
Singularity
```

If an old workflow was saved with a versioned class id such as `SingularityR56PublicAlpha`, remove that old node and add a fresh `Singularity` node. This is intentional: the public package should stay clean and not accumulate internal development aliases.

## What Was Added After r56

### r57 - Motion Math Metrics Diff

r57 adds more detailed motion math after decode.

New metrics:

- `frame_delta_norm_p25`
- `frame_delta_norm_p50`
- `frame_delta_norm_p75`
- `frame_delta_norm_p90`
- `frame_delta_norm_p95`
- `frame_delta_norm_iqr`
- `frame_delta_norm_cv_ratio`
- `frame_delta_p95_to_p50_ratio`
- `frame_delta_jerk_abs_mean`
- `frame_delta_jerk_ratio`
- `frame_motion_stability_score`
- `frame_motion_profile`

What this gives you:

- faster fixed-seed comparison between runs;
- clearer visibility into smoother, sharper, unstable, or spiky motion;
- less accidental mixing between single-run and cascade-run baselines because the runtime signature is stricter;
- more useful reports for behavior analysis, not just pass/fail status.

Important: these metrics are observer-only. They do not mutate tensors and do not improve video quality by themselves.

### r58 - Input Integrity Hardening

r58 adds central input normalization before generation.

Values now normalized and protected:

- `cascade_count`
- `frames_per_cascade`
- `width`
- `height`
- `fps`
- `seed`
- sampler windows: `primary_start_step`, `primary_end_step`, `secondary_start_step`, `secondary_end_step`
- `math_control_mode`
- `high_delta_strength`
- `low_delta_strength`
- `decode_tile_size`
- `decode_overlap`
- `decode_temporal_size`
- `decode_temporal_overlap`
- `image_upscale_method`
- `image_crop`
- `cleanup_timing`
- `video_format`
- `save_prefix`
- `sampler_trace_mode`
- `sampler_trace_max_steps`

What this gives you:

- stale or partially corrupted widget values are less likely to break a run;
- width, height, and decode values are constrained into safe ranges;
- overlap cannot exceed safe tile limits;
- empty or unsafe `save_prefix` values are converted into safe file names;
- sampler and scheduler values are checked against available `KSamplerAdvanced` options when ComfyUI exposes them;
- the report records `EventInputNormalization` and `EventInputNormalizationAdjustments`.

UI hardening:

- prompt boxes are wider and taller;
- positive and negative prompt widgets are protected from overlap during resize/configuration cycles;
- widget order is not sorted, because ComfyUI stores widget values positionally.

### r59 - Strategy Math Native Loop / Overlay Correction

r59 adds a deeper split between safe public math behavior and experimental deep-step math behavior.

Main correction:

```text
LATENT_DELTA_SCALE no longer replaces the native sampler loop by default.
```

Why:

A rough step-by-step sampler loop replacement can turn the output into noise even when the structural gate still shows PASS. That means the math started overriding generation physics instead of helping the model hold the intended semantic strategy.

What changed:

- `LATENT_DELTA_SCALE` now uses the safe `semantic_overlay_native_sampler` path.
- The model's native sampler remains the main generator.
- Delta math is applied as a controlled semantic overlay / post-window relation layer.
- The report now includes `EventMathSamplerPathPolicy`, so the selected math path is visible.
- Fixed false huge `relative_delta` when the high-stage baseline norm is zero.
- High-to-low coupling is no longer incorrectly suppressed by a zero-latent start.
- Added the separate research mode `DEEP_STEP_DELTA_CONTROL`.

## Math Control Modes

### `OBSERVE_ONLY`

Safe baseline mode.

What it does:

- runs generation through the normal path;
- records math/report records;
- should not intentionally mutate generation tensors;
- is suitable for baseline comparison.

When to use it:

- first test after installation;
- checking that model, workflow, image, VAE, CLIP, and save path all work;
- baseline before delta or motion comparisons.

### `LATENT_DELTA_SCALE`

The public default working mode.

What it does:

- measures latent movement as `ObservedBehavior`;
- applies `high_delta_strength` and `low_delta_strength`;
- preserves the native sampler path;
- writes math policy records;
- helps study how changing high/low latent delta strength affects the strategy carrier between stages.

Practical meaning:

- `1.0` = neutral;
- below `1.0` = reduce this stage's delta contribution;
- above `1.0` = amplify this stage's delta contribution.

Formula meaning:

```text
delta strength changes ObservedBehavior
ObservedBehavior changes Outcome
Outcome becomes StrategyCarrier for the next stage
```

So this is not just a quality slider. It is a controlled change to the latent route that gets passed forward.

### `DEEP_STEP_DELTA_CONTROL`

Experimental research mode.

What it does:

- enables the deeper native step-loop control path;
- can intervene more strongly in step dynamics;
- is meant for studying where the math belongs inside the intersection of model output, sampler state, latent state, and prompt strategy.

Risk:

- it can produce noise;
- it requires sweeps over `high_delta_strength` / `low_delta_strength`;
- it is not recommended as a normal public-user mode.

When to use it:

- only for controlled experiments;
- only with a fixed seed;
- always save the report;
- always inspect the actual video, not just PASS/BLOCKED status.

## Drift: What It Means In This Node

Drift is measurable separation between the intended route and the actual generation trajectory.

Drift types:

- Source drift: the video stops respecting the source image.
- Prompt drift: the result moves away from the positive prompt or toward the negative prompt.
- Identity drift: the main object or character changes more than intended.
- Motion drift: movement becomes jerky, spiky, reversed, or too chaotic.
- Cascade drift: the next cascade segment does not continue the previous segment cleanly.
- Route drift: runtime route, stage order, or boundary records do not match the expected logic.
- Math drift: the selected math starts overriding generation instead of refining strategy.

Important:

The report does not remove drift by itself. It shows where drift appears so fixed-seed runs can be compared with evidence.

## Full Node Functionality

### External Node

```text
Singularity
```

This is the only visible node in the public clean package.

It combines:

- source image loading;
- prompt encoding;
- high sampler stage;
- low sampler stage;
- optional cascade continuation;
- VAE decode;
- video saving;
- report saving;
- runtime diagnostics;
- Singularity records.

### Required Inputs

`primary_model`

Main model branch. Used for the high/main sampling stage.

`clip`

Text encoder. Converts positive/negative prompts into conditioning.

`vae`

Used to decode latent frames into images/video.

`source_image_file`

Image picker for the ComfyUI input folder. If the external `image` socket is not connected, the node uses this file picker.

`positive_prompt`

What the model should generate. In the project logic, this is not the whole Strategy; it is a carrier / part of the StrategyCandidate.

`negative_prompt`

What the model should avoid. This is a negative constraint carrier.

`temporal_texture_lock`

Flag for temporal/texture continuity intent. In public alpha it is part of route intent and report context.

### Optional Inputs

`secondary_model`

Optional low/secondary model branch. If it is not connected, the node can fall back to a single branch and record a conflict/warning.

`image`

Direct IMAGE input when the workflow passes an image instead of using the file picker.

`mask`

Optional mask input. Currently used as report/context input for future or partial masked-route scenarios.

## Generation Controls

`cascade_count`

Number of generation segments.

- `1` = single video;
- `2-5` = cascade continuation.

`frames_per_cascade`

Number of frames per cascade segment.

`width`, `height`

Video dimensions. Normalized into a safe step-aligned range.

`fps`

Output video frame rate.

`seed`

Seed for reproducible comparisons. For metric research, keep the same seed across runs.

## Sampler Controls

`sampler_name`

Sampler name. r58+ tries to validate it against available ComfyUI sampler options.

`scheduler`

Scheduler for the sampling path. Also validated/fallbacked when ComfyUI exposes allowed values.

`global_steps`

Total sampling steps.

`primary_cfg`

CFG for the primary/high branch.

`secondary_cfg`

CFG for the secondary/low branch.

`primary_start_step`, `primary_end_step`

Step window for the high sampler stage.

`secondary_start_step`, `secondary_end_step`

Step window for the low sampler stage.

## Delta Controls

`math_control_mode`

Selects the math mode:

- `OBSERVE_ONLY`
- `LATENT_DELTA_SCALE`
- `DEEP_STEP_DELTA_CONTROL`

`high_delta_strength`

Scales high-stage latent delta.

`low_delta_strength`

Scales low-stage latent delta.

Practical recommendation:

Start with:

```text
high_delta_strength = 1.0
low_delta_strength = 1.0
```

For comparison after baseline:

```text
high_delta_strength = 0.988
low_delta_strength = 1.0
```

## Decode Controls

`decode_tile_size`

Tile size for VAE decode.

`decode_overlap`

Overlap between tiles. r58+ prevents it from becoming unsafe relative to tile size.

`decode_temporal_size`

Temporal tile size for video decode.

`decode_temporal_overlap`

Temporal overlap. r58+ constrains it relative to temporal size.

## Image Controls

`image_upscale_method`

Source image resize method:

- `nearest-exact`
- `nearest`
- `bilinear`
- `area`
- `bicubic`
- `lanczos`

`image_crop`

Crop mode:

- `disabled`
- `center`

## Cleanup / Barrier Controls

`cleanup_timing`

Controls cleanup/barrier timing:

- `NONE`
- `BEFORE_GENERATION`
- `BETWEEN_SAMPLERS`
- `AFTER_GENERATION`
- `BEFORE_AND_AFTER`
- `ALL`

What it does:

- records where cleanup boundaries were called;
- attempts to release disposable memory;
- should not destroy tensors needed by the low sampler or final output;
- writes Smart Branch Barrier records into the report.

Formula meaning:

The barrier must not break the StrategyCarrier between the high and low stages.

## Save Controls

`save_video`

Saves the video to ComfyUI output.

`video_format`

Save format:

- `video/h264-mp4`
- `video/h265-mp4`
- `image/webp`
- `image/gif`

`save_report`

Saves the markdown report.

`save_prefix`

Output file prefix. r58+ sanitizes unsafe characters and empty values.

## Sampler Trace Controls

`sampler_trace_mode`

Step trace mode:

- `OFF`
- `SHADOW_STEP_TRACE`

`SHADOW_STEP_TRACE` writes shadow records for step-level behavior analysis, but it should not replace the main sampler output.

`sampler_trace_max_steps`

Limits the number of traced steps so reports do not grow without control.

## Outputs

`status`

Short execution status.

`saved_video_path`

Path to the saved video when `save_video = true` and saving succeeds.

`saved_report_path`

Path to the saved report when `save_report = true` and saving succeeds.

`report`

Report text returned directly from the node output.

## What The Report Can Include

The report can include:

- runtime version/name;
- generation settings snapshot;
- source image load status;
- text encode records;
- high/low sampler records;
- latent boundary math;
- frame motion math;
- cascade boundary math;
- RouteMemory timeline;
- S-Wire state;
- SState snapshots;
- EventConflict records;
- CompletionGate;
- RuntimeMonitor summary;
- optional sidecar paths;
- sampler trace summary;
- input normalization adjustments.

## CompletionGate

`EventCoreBodyCompletionGate = PASS` means structural integrity of the runtime route:

- required stages were recorded;
- stage order was valid;
- report/finalize logic reached completion;
- the node returned the expected outputs.

It does not mean:

- the video is visually good;
- the prompt was understood perfectly;
- drift is absent;
- deep math mode is safe for all cases.

Correct verification:

```text
result_status = VIDEO
EventCoreBodyCompletionGate = PASS
video visually inspected
report metrics compared against baseline
```

## Recommended Test Matrix

### First Run

```text
math_control_mode = OBSERVE_ONLY
high_delta_strength = 1.0
low_delta_strength = 1.0
cascade_count = 1
save_video = true
save_report = true
```

Goal: check that model, VAE, CLIP, image input, and save path work.

### Public Default Run

```text
math_control_mode = LATENT_DELTA_SCALE
high_delta_strength = 1.0
low_delta_strength = 1.0
```

Goal: check the safe overlay path.

### Delta Comparison

```text
math_control_mode = LATENT_DELTA_SCALE
high_delta_strength = 0.988
low_delta_strength = 1.0
same seed as baseline
```

Goal: compare motion metrics and visual output.

### Cascade Test

```text
cascade_count = 5
frames_per_cascade = 49
same seed
```

Goal: check cascade continuation and boundary records.

### Experimental Deep Math Test

```text
math_control_mode = DEEP_STEP_DELTA_CONTROL
high_delta_strength = start below 1.0
low_delta_strength = 1.0
same seed
```

Goal: study deep intervention. If noise appears, return to `LATENT_DELTA_SCALE` or lower strength values.

## Known Limits

- Public alpha, not production-stable.
- The main testbed is currently Wan I2V.
- Step-level formula coverage is still not fully closed.
- Shadow trace is a diagnostic observer, not a full native sampler hook.
- Deep step control is research mode.
- CompletionGate PASS does not replace visual inspection.
- Reproducibility depends on ComfyUI, torch, GPU, drivers, model build, and sampler implementation.

## Package Cleanup In r59 Public

Removed from the public package:

- old examples;
- internal release checklists;
- debug node exports;
- versioned public aliases;
- pycache;
- generated output artifacts;
- handoff/transfer files;
- private workflows;
- old reports and videos.

Kept:

- working node code;
- imported internal modules;
- web UI helper;
- README;
- CHANGELOG;
- license;
- requirements;
- pyproject;
- version file;
- formula integrity note.






