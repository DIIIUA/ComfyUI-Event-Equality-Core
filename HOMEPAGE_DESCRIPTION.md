# Singularity for ComfyUI

Singularity is an experimental ComfyUI node and workflow helper for Wan image-to-video continuation. It is built for people who want to make longer Wan I2V / V2V videos without manually rebuilding the continuation chain after every short clip.

In simple terms: generate a clip, pause at the cascade boundary, inspect the last candidate frames, choose the frame you want to continue from, press Continue, and get one stitched final video.

## What This Is

This package is two things at once:

1. A single ComfyUI custom node: `Singularity`.
2. A practical continuation workflow concept for Wan2.2-style High/Low I2V generation.

It is designed around the same idea many Wan users already use manually:

```text
make a strong first clip
choose a good last frame
re-inject that frame
extend again
stitch everything together
```

Singularity moves that loop into the node itself.

## What It Does For The User

- Generates Wan I2V clips.
- Extends the video through multiple cascade stages.
- Supports up to five cascade segments in this public alpha.
- Pauses between cascade stages when you ask it to.
- Shows Source / Tail 1 / Tail 2 / Tail 3 / Result under the node.
- Lets you manually choose the tail frame used for the next segment.
- Continues the same running workflow instead of starting a separate unrelated clip.
- Saves one final stitched video.
- Saves a markdown report with runtime and math diagnostics.

## Why This Matters

ComfyUI long-video workflows often turn into spaghetti:

- many repeated sampler groups;
- manual last-frame extraction;
- reloading frames into new nodes;
- separate MP4 outputs per stage;
- stitching later;
- unclear drift between segments.

Singularity is meant to make that process easier to inspect and control. You still decide what looks best, but the node gives you the pause point, the tail choices, the stitched output, and the report evidence.

## How To Use It In A Workflow

A typical Wan2.2 I2V / V2V extend workflow looks like this:

```text
Wan High/Low models -> Singularity -> final video + report
```

Recommended first settings:

```text
cascade_count = 2
pause_after_cascade_1 = true
frames_per_cascade = 49
fps = 16
math_control_mode = LATENT_DELTA_SCALE
high_delta_strength = 1.0
low_delta_strength = 1.0
save_video = true
save_report = true
```

When the pause UI appears:

```text
1. Look at Source and Tail 1/2/3.
2. Click the tail frame that best continues the motion.
3. Press Resume Cascade / Continue.
4. Wait for the final stitched video.
```

## Wan2.2 High/Low Context

The node is especially useful with Wan2.2-style HighNoise / LowNoise model setups:

- High stage: establishes motion, structure, and the next scene direction.
- Low stage: refines details and stabilizes the visible result.

It can work with GGUF quant High/Low routes when your ComfyUI setup already supports them. Models are not included.

## Fast Iteration Advice

For fast testing, use short segments first:

```text
49 frames at 16 fps = about 3 seconds
```

Shorter clips make it easier to control the prompt and inspect motion. Once the route works, increase frame counts or cascade count.

If you use low-step / LightX2V-style workflows, you can iterate faster. If you disable the speed setup and use heavier custom LoRAs or higher steps, render time will increase heavily. That is normal for Wan.

## Prompting Advice

For cascade extension, prompt precision matters.

Write:

- what the subject is doing;
- where the camera is moving;
- what should remain consistent;
- what the ending pose or motion should feel like;
- repeated identity/environment anchors for every segment.

Weak prompts drift more easily. Strong prompts give the continuation frame a better route to follow.

## Diagnostics And Math

Singularity records generation as an event:

```text
Outcome(t-1) + ObservedBehavior(t-1)
= Strategy(t)
= ObservedBehavior(t+1) + Outcome(t+1)
```

For normal users, this means:

- the node watches how the video changes;
- it records how much motion happened;
- it checks whether the route structurally completed;
- it records cascade continuity;
- it helps compare fixed-seed runs.

The math is not magic. It does not guarantee better video. It gives you evidence and controlled comparison tools.

## Drift: What To Watch For

Drift means the video starts moving away from what you intended.

Examples:

- Source drift: it stops respecting the source image.
- Prompt drift: it ignores the prompt.
- Identity drift: the subject changes too much.
- Motion drift: movement becomes unstable or chaotic.
- Cascade drift: the next segment does not continue cleanly.
- Math drift: research settings start breaking the generation instead of guiding it.

Singularity helps expose drift. It does not automatically solve every drift case.

## Public Alpha Notice

This is a public alpha.

Use it for:

- Wan I2V continuation tests;
- manual tail-frame selection;
- comparing prompts, seeds, and delta settings;
- collecting reports;
- studying cascade continuity.

Do not treat it as:

- a final production-stable video system;
- universal all-model support;
- guaranteed quality improvement;
- infinite cascade support.

`CompletionGate = PASS` means the structural route completed and a final video exists. It does not mean the video is visually perfect. Always inspect the video.

## Short Version

Singularity is a Wan I2V cascade continuation node for ComfyUI. It pauses between cascade stages, lets you choose the tail frame for continuation, resumes the same run, stitches the final video, and writes detailed diagnostics so you can understand what happened instead of guessing.
