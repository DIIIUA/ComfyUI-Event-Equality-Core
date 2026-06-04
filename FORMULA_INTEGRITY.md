# Formula Integrity Rules

These rules protect Singularity from degrading into a normal pile of unrelated ComfyUI nodes.

## Core rules

1. Every signal must have a technical identity.
2. Every signal must have a formula role.
3. Every output must be readable as an EventSignal or explicitly ignored.
4. Every relation must have a formula reason.
5. SState is relation-based, not signal-based.
6. Every transition must record ObservedBehavior where possible.
7. RouteMemory is required, not optional.
8. No hidden transforms.
9. Report first, correction later.
10. Zero-strength passthrough for any future modifying node.
11. raw_ref must never be printed in reports.
12. Adapters label first, modify never in v0.1.

## Technical identity is not formula identity

Examples:

```text
LATENT can be StrategyCarrier.
LATENT can be OutcomePrevious.
LATENT can be OutcomeNext.
TEXT can be StrategyCurrent.
NOISE can be StrategyCandidate.
DELTA can be ObservedBehaviorCurrent.
```

The technical type tells what the object is.

The formula role tells where it stands in the equality.

## SState is not a signal

Do not reduce SState to prompt text, latent, image, or noise.

SState is built from:

```text
active signals
active projections
active relations
local strategies
route memory
conflicts
```

## No hidden transforms

If a node changes data, it must record:

```text
ObservedBehavior
Relation
RouteMemory stage record
optional conflict/correction record
```

## Report-first correction-later

Any new intelligence must first appear as:

```text
read
report
trace
soft correction
hard correction only much later
```

## Adapter discipline

Adapters do not own the core.

Generic Event Core reads signals and builds formula objects.

Adapters label and interpret domain-specific routes.

```text
generic core first
adapter labels second
correction never in v0.1
```

