# Singularity - Public Changelog

## 0.1.1-r91 - Public Stabilization

Status: public stabilization release after the r90 Strategy Control Surface runtime gate.

- Promotes the current public build to `0.1.1-r91`.
- Updates the visible node title to `Singularity R91`.
- Keeps the public node surface clean:
  - `source_image_file = none`
  - `positive_prompt = empty`
  - built-in Wan-style Chinese negative prompt remains as the starter negative prompt
  - two-cascade public starter route remains the default
- Cleans the visible `math_control_mode` dropdown so users see only:
  - `OBSERVE_ONLY`
  - `LATENT_DELTA_SCALE`
  - `STRATEGY_PRESSURE_WINDOW`
  - `DEEP_STEP_DELTA_CONTROL`
- Keeps backend compatibility for older workflows that saved lowercase or legacy mode values.
- Confirms the pre-release smoke-test on ComfyUI Desktop immediately before the R91 version bump:
  - tested runtime was the same code path at `0.1.1-r90-dev`
  - `result_status = VIDEO`
  - `EventCoreBodySummary = PASS`
  - two cascades completed with pause/continue
  - final video saved and stitched
  - neutral `STRATEGY_PRESSURE_WINDOW` with `high=1.0`, `low=1.0` produced the same visible output as the neutral r90 baseline (`SSIM = 1.0`, `PSNR = infinite`)
- Keeps r90 math behavior intact:
  - `STRATEGY_PRESSURE_WINDOW` treats strength values as bounded pressure intent
  - neutral `1.0 / 1.0` stays non-mutating
  - prompt text remains clean
  - CFG remains model-native
  - deep sampler step replacement remains isolated to explicit research mode
- This is a stabilization release, not a claim that the experimental math always improves output.

## 0.1.1-r90 Dev - Strategy Control Surface

Status: larger functional consolidation after r89 proved the bounded pressure-window route.

- Adds `EventStrategyControlSurfacePlan`, a single runtime plan that decides how the active math mode is allowed to influence generation.
- Adds `EventStrategyControlSurfaceApply_<branch>`, the single function-level application record for high/low latent transitions.
- Consolidates active math policies:
  - `OBSERVE_ONLY` = record without tensor mutation
  - `LATENT_DELTA_SCALE` = raw branch delta-scale policy
  - `STRATEGY_PRESSURE_WINDOW` = bounded pressure-intent policy
  - `DEEP_STEP_DELTA_CONTROL` = deep research step-loop policy
- Removes the separate r89 pressure-window helper as an independent decision point. The compatibility report records `EventStrategyPressureWindow_high` / `EventStrategyPressureWindow_low` are now emitted from the unified Strategy Control Surface.
- `EventMathControlSummary` now includes the Strategy Control Surface version, status, policy, active flag, and branch policies.
- Formula meaning: high/low sampler pressure no longer has its own local controller. Every active delta change asks the parent `S_global_event_route` control surface for one `effective_strength`.
- Safety meaning: prompt text remains clean, CFG remains model-native outside explicit deep research, and sampler step replacement is still isolated to `DEEP_STEP_DELTA_CONTROL`.

## 0.1.1-r89 Dev - Unified Pressure Window

Status: development functional pass after the r88 Strategy-return pressure resolver.

- Adds `STRATEGY_PRESSURE_WINDOW`, a unified active math mode for the next test round.
- Keeps the native sampler path and CFG behavior; the new mode does not replace the denoising loop.
- Reads `high_delta_strength` and `low_delta_strength` as pressure intent, then compresses that intent into a small bounded window around `1.0`.
- Adds `EventStrategyPressureWindow_high` / `EventStrategyPressureWindow_low` records so reports show:
  - requested branch strength
  - high-to-low coupling multiplier
  - compressed pressure intent
  - max allowed window
  - final effective strength
- Extends the Strategy Matrix so the new pressure-window records count as delta/control evidence.
- Formula meaning: local sampler pressure must return to `S_global_event_route` before it becomes active latent control. This is the first "one functional math surface" after the report-only resolver.
- Safety meaning: prompt text stays clean, topology prose is not injected into the prompt, and extreme values like `0.5` / `1.5` become bounded research nudges rather than raw destructive multipliers.

## 0.1.1-r88 Dev - Strategy Return Pressure Resolver

Status: development topology pass after the r87 Continue payload gate passed.

- Adds `EventStrategyReturnPressureResolver`, a report-only resolver that folds local pressure back into the global Strategy route before proposing the next math surface.
- Reads existing evidence instead of forcing a new controller:
  - `EventPromptCarrierContinuityCard`
  - `EventLowBranchRelationPressureCard`
  - `EventTailStrategyContinuityCard`
  - `EventFrameSpikeAttributionCard`
  - `EventObjectCarrierIdentityCard`
  - `EventTopologyStrategyReturnMap`
- Reports a single pressure vector:
  - prompt carrier pressure
  - high/low sampler pressure
  - visible frame motion pressure
  - late-segment spike pressure
  - seam boundary pressure
  - tail Strategy pressure
  - object relation pressure
  - source anchor pressure
- Adds summary fields for:
  - `strategy_return_resolver_status`
  - `strategy_return_pressure`
  - `strategy_return_primary_attribution`
  - `strategy_return_next_control_surface`
  - `strategy_return_active_control_allowed`
- The resolver stays prompt-pure: it does not inject topology/math prose into CLIP/T5 text.
- The resolver stays observer-only: it does not modify prompts, tensors, sampler steps, deltas, pause routing, or video frames.
- Formula meaning: local high/low, frame-motion, tail, object, source, and prompt sub-strategies are allowed to unfold, but they must return to `S_global_event_route` before any active control is considered.

## 0.1.1-r87 Dev - Continue Payload Guard

Status: development hotfix after the first r86 prompt-purity runtime gate.

- Keeps the r86 prompt-purity law: `TRANSFORM_PROMPT` still builds report/control-space maps and does not inject formula prose into CLIP/T5 text.
- Adds a Continue payload guard for positive prompt truncation, not only negative prompt truncation.
- If the Continue UI sends a prompt payload that looks like a missing or truncated version of the already-active prompt, the backend reuses the current clean StrategyCandidate instead of falsely creating `changed_runtime_strategy`.
- Adds report fields for positive prompt payload hygiene:
  - `positive_payload_missing`
  - `positive_payload_truncated`
  - `positive_prompt_payload_reused_previous_active`
  - `positive_prompt_payload_mismatch_policy`
- New same-prompt match bases can include:
  - `positive_payload_missing_reuse`
  - `positive_payload_truncated_reuse`
  - `prompt_payload_truncated_reuse`
- Formula meaning: a partial widget payload is not a new Strategy. Same user intent must keep the same prompt carrier across pause/continue before sampler math is judged.

## 0.1.1-r86 Dev - Prompt Purity Lock

Status: development correction after prompt-topology tests.

- Added `EventPromptPurityLock`.
- `TRANSFORM_PROMPT` no longer injects formula/topology prose into the CLIP prompt.
- The model-facing positive prompt now stays as the clean user prompt, except when an older generated Strategy tail is detected and stripped as sanitation.
- Added `semantic_density_context_map` to compare:
  - meaning density,
  - context density,
  - density/context balance,
  - object/topology pressure,
  - and next safe control surfaces.
- `EventPromptStrategyTranscodeApply` now reports `semantic_map_only` instead of active prompt rewrite.
- Continue/runtime prompt updates follow the same law: changed prompts are encoded from clean user text, while math remains outside the text route.
- Formula meaning: math, semantics, logic, and Strategy stay free, but they must not be converted into extra prompt words. The prompt is the StrategyCandidate carrier; density sorting belongs in report/control space.

## 0.1.1-r85 Dev - Topology Strategy Return Map

Status: development stabilization pass after the r84 pause/continue gate.

- Added `EventTopologyStrategyReturnMap`, a report-only map that checks how local Strategy collision points return to the global route Strategy.
- The map links prompt/source, prompt polarity, source-to-latent anchoring, high/low sampler pressure, object relation topology, tail continuation, visible frame motion, and final video outcome into one parent route.
- `EventCoreBodySummary` now exposes:
  - `topology_strategy_return_status`
  - `topology_sync_score`
  - `topology_unstable_route_count`
  - `topology_watch_route_count`
  - `topology_primary_pressure_axis`
  - `topology_next_route`
- The prompt strategy packet now includes `strategy_return_contract`, so prompt deconstruction declares the global/local topology before sampler evidence is read.
- No tensors, prompts, sampler steps, or cascade routing are modified by this pass.
- Formula meaning: local formulas may unfold at each carrier collision, but every local Strategy must return to the main StrategyCarrier before data is passed to the next sampler or cascade segment.

## 0.1.1-r84 Dev - Cascade Continue Hotfix

Status: urgent pause/resume runtime fix.

- Fixed a regression in the r83 post-transform prompt-continuity path.
- Continue reached the backend correctly, but the run could fail immediately after pause with `mode_matches` not initialized.
- Moved the mode comparison before post-transform identity checks, so pause/continue can safely decide whether the next cascade reuses the same active StrategyCarrier or receives a changed prompt.
- The visible ComfyUI title now reads `Singularity R84`.
- Formula meaning: local prompt identity checks must return to the main cascade Strategy route; they must never break the route between the selected tail frame and the next segment.

## 0.1.1-r83 Dev - Post-Transform Prompt Continuity

Status: development fix found by the r82 relation cards.

- r82 loaded correctly and generated a VIDEO/PASS report, but `EventPromptCarrierContinuityCard` exposed a hidden issue: Continue could still mark the prompt as `changed_runtime_strategy`.
- Root cause: the backend compared the raw positive widget payload against the already-transformed active StrategyCarrier before checking whether the raw prompt would transform back into the same active carrier.
- Added a post-transform positive identity preview:
  - `positive_payload_transforms_to_current_active`
  - `positive_strategy_identity_matches`
  - `positive_payload_transform_preview_signature`
- If the positive prompt resolves to the same active StrategyCarrier and the negative payload is missing/truncated, Continue can now reuse the previous active negative carrier instead of treating widget payload drift as a real Strategy change.
- Added `EventGlobalStrategyReturnCard` so local prompt/source/low/object/tail/frame cards are explicitly subordinate to the primary route Strategy.
- The global card records divergence flags when local sub-strategies do not return cleanly to the main StrategyCarrier path.
- Formula meaning: unchanged user intent must preserve the same `Strategy(t)` across pause/continue, even when the UI payload carries raw text while the runtime holds a transformed carrier.

## 0.1.1-r82 Dev - Evidence Hygiene + Relation Cards

Status: development report/evidence fix before the next fixed-seed visual tests.

- Aligned stale body/runtime constants across `nodes.py`, `core/execution.py`, `core/cascade.py`, `core/orchestrator.py`, and `core/telemetry.py`.
- The visible ComfyUI title now reads `Singularity R82`.
- Continue payloads now record whether positive and negative prompt fields were actually present.
- If Continue sends the same positive Strategy identity but the negative prompt payload is missing or looks truncated, the backend reuses the current active negative Strategy carrier instead of falsely creating a changed runtime prompt route.
- Added six local report-only relation pressure cards:
  - `EventPromptCarrierContinuityCard`
  - `EventLowBranchRelationPressureCard`
  - `EventObjectCarrierIdentityCard`
  - `EventTailStrategyContinuityCard`
  - `EventFrameSpikeAttributionCard`
  - `EventSourceAnchorPreservationCard`
- `EventCoreBodySummary` now exposes the relation-card count and top card statuses.
- Formula meaning: this release does not add new active sampler control. It gives the next report a cleaner explanation of where Strategy continuity, object relation pressure, low-branch pressure, and seam dynamics diverge.

## 0.1.1-r81 Dev - Visible Node Version Label

Status: development UX/version hygiene fix.

- The visible ComfyUI node title now includes the current R-label, for example `Singularity R81`.
- The internal ComfyUI node key remains `Singularity`, so existing workflows keep compatibility.
- The Python display name is derived from `EVENT_HORIZON_RUNTIME_VERSION`.
- The frontend extension also applies the visible title on node creation and workflow load, so older saved nodes are easier to visually verify after an update.

## 0.1.1-r80 Dev - Compact Strategy Transform

Status: development fix for prompt-transform duplication.

- Prompt transform now compacts the user prompt before adding Strategy carrier language.
- Negated/protective wording is folded into positive admissible behavior instead of being repeated as a second prohibition layer.
- This keeps the active prompt closer to one StrategyCarrier: scene/action content plus one compact topology/continuity interpretation.
- This is intended to reduce duplicated motion dynamics after cascade stitching.
- Formula meaning: transformation should transcode the prompt into one readable event route, not stack a second independent explanation on top of the first one.

## 0.1.1-r79 Dev - Prompt Identity Continuity

Status: development fix for cascade prompt continuity.

- Fixed the Continue prompt identity route.
- The backend now compares the Continue payload against:
  - the launch raw prompt,
  - the launch active transformed prompt,
  - the current active prompt,
  - the last runtime raw prompt,
  - and the last runtime active prompt.
- If the prompt identity is the same, the next cascade reuses the current active StrategyCarrier instead of transforming the same prompt again.
- If the user genuinely edits the prompt at a pause, the next cascade still receives a new local StrategyCarrier.
- Reports now expose the match basis through fields such as `same_prompt_match_basis`, `prompt_continuity_reused`, and `prompt_continuity_policy`.
- Formula meaning: a pause/continue boundary must not create a new `Strategy(t)` when the user intent did not change.

## 0.1.1-r78 Dev - Wan Native Source Anchor

Status: development fix for source-image topology.

- Added `image_crop = wan_native`.
- New default: Singularity passes the source image directly into official `WanImageToVideo`.
- This avoids an extra external `ImageScale` resize/crop before Wan performs its own target-grid normalization.
- Kept old explicit modes:
  - `disabled` = pre-scale in Singularity without center crop.
  - `center` = pre-scale and center-crop in Singularity.
- Formula meaning: `width` / `height` remain the Wan latent/video grid, while `wan_native` preserves the source image as a SourceAnchor until Wan encodes it.

## 0.1.1-r62 Public Alpha Desktop Cascade UI Fix

Status: public alpha update for ComfyUI Desktop / modern frontend cascade continuation.

The main r62 goal is to fix a public r60 regression where the backend could pause correctly but the detached Source / Tail / Result panel might not appear in modern ComfyUI Desktop.

### Emergency Hotfix

- Added a status-polling fallback through `/singularity/cascade/status/{node_id}`.
- The pause UI no longer depends only on one websocket event.
- If ComfyUI Desktop drops or misses the pause event, the frontend can recover the paused state, show the tail frames, and enable `Resume Cascade / Continue`.
- Restored the always-on media panel behavior: Source / Tail / Result is rendered immediately for the node and no longer waits for the first cascade pause before appearing.
- Restored the stable high overlay layer so the panel is not hidden behind the modern ComfyUI canvas.
- Kept the public manual green tail selection route. Formula recommendation is visible again as an experimental proposal toggle, but remains off by default.

The broader r62/r60 goal remains to make the manual cascade frame-selection workflow safer for public users: cleaner defaults, less UI overlap, preserved native image upload, and research controls that are clearly off by default.

### Short Summary

- Updated runtime/package version to `0.1.1-r62`.
- Improved the detached Source / Tail 1 / Tail 2 / Tail 3 / Result pause panel.
- The detached media panel is now visible before generation starts, so users can verify that the UI extension loaded before waiting through a long cascade.
- The panel uses the stable high overlay layer again; this is intentionally prioritized over experimental occlusion hiding for the emergency public hotfix.
- Kept the native ComfyUI `source_image_file` upload button.
- Restored the public UI for `use_formula_recommendation` after the Desktop pause regression investigation; it stays `false` by default and manual green tail selection remains primary.
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
  - `image_crop = wan_native`
- Rewrote the README and homepage/CVTI description for normal users, with plain explanations of cascade continuation, drift, math modes, reports, and public-alpha limits.

### What Changed For Users

The node now opens as a clean two-cascade frame-selection test instead of a local development preset. A new user can add the node, connect the model/CLIP/VAE route, choose or upload a source image, write a prompt, generate, choose a tail frame, continue, and inspect one final stitched video.

The math layer is still present, but the public default is observer-first:

```text
math_control_mode = OBSERVE_ONLY
high_delta_strength = 1.0
low_delta_strength = 1.0
```

This means r62 starts as a safer baseline. Users who want to test delta behavior can intentionally switch to `LATENT_DELTA_SCALE`.

### UI Notes

r62 specifically targets the ComfyUI Desktop behavior seen after the modern frontend update. The pause panel is still detached under the node, but it is now rendered as an always-on media surface with high overlay priority. This makes the UI visible before the first cascade and prevents a missed pause panel from trapping the workflow.

### Compatibility Notes

Old workflows that already contain `use_formula_recommendation` should keep loading. The field remains accepted by the Python signature and visible in the node, but new public nodes default to manual green tail selection.

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

Legacy active research mode.

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

### `STRATEGY_PRESSURE_WINDOW`

Unified active research mode.

What it does:

- measures local high/low pressure as sampler-route evidence;
- keeps the model-native sampler path;
- keeps CFG native;
- reads `high_delta_strength` and `low_delta_strength` as pressure intent;
- compresses that intent into a small bounded window around `1.0`;
- applies the bounded value through the same latent transition formula.

Practical meaning:

- use this when raw `LATENT_DELTA_SCALE` is too rough;
- big test values are safe enough to study because they are compressed;
- reports show the exact effective strength that was applied.

Formula meaning:

```text
local sampler pressure
-> Strategy return to S_global_event_route
-> bounded pressure window
-> latent transition control
```

This is the preferred next test surface for high/low visible-motion coupling.

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






