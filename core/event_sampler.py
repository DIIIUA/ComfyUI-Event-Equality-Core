"""
EventSamplerCore v0.1.1-r6

This is the first replacement layer for KSamplerAdvanced.

It is not yet a full step-loop rewrite. It is a boundary replacement wrapper:
- owns the sampler window semantics,
- records EventSampler boundary metadata,
- forbids silent success,
- can still use ComfyUI's sampler operation as the low-level operation function.

Next target:
- replace/extend the low-level operation with an Event-native step loop.
"""

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Optional


@dataclass
class EventSamplerWindow:
    branch_name: str
    branch_role: str
    seed: int
    steps: int
    cfg: float
    sampler_name: str
    scheduler: str
    start_at_step: int
    end_at_step: int
    add_noise: str
    return_with_leftover_noise: str
    sd3_shift: float = 0.0


@dataclass
class EventSamplerResult:
    ok: bool
    latent_before: Any
    latent_after: Any
    window: EventSamplerWindow
    event_records: list
    error: Optional[str] = None

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "window": asdict(self.window),
            "event_records": self.event_records,
            "error": self.error,
        }


class EventSamplerCore:
    """
    Boundary replacement wrapper around a low-level sampler operation.

    It is designed so WanEventWorkflowCore can stop thinking in terms of
    "call KSamplerAdvanced node" and start thinking in terms of:
        EventSamplerHigh / EventSamplerLow
    with before/after state, branch role, route memory and future step trace.
    """

    def __init__(self, operation_fn: Callable[..., Any]):
        self.operation_fn = operation_fn

    def sample_window(
        self,
        *,
        model,
        positive,
        negative,
        latent,
        window: EventSamplerWindow,
    ) -> EventSamplerResult:
        records = [{
            "stage": "event_sampler_begin",
            "branch_name": window.branch_name,
            "branch_role": window.branch_role,
            "start_at_step": window.start_at_step,
            "end_at_step": window.end_at_step,
            "replacement_layer": "boundary_replacement",
            "step_loop": "not_yet_native",
        }]

        try:
            latent_after = self.operation_fn(
                model=model,
                positive=positive,
                negative=negative,
                latent=latent,
                seed=window.seed,
                steps=window.steps,
                cfg=window.cfg,
                sampler_name=window.sampler_name,
                scheduler=window.scheduler,
                start_at_step=window.start_at_step,
                end_at_step=window.end_at_step,
                add_noise=window.add_noise,
                return_leftover_noise=window.return_with_leftover_noise,
            )
            records.append({
                "stage": "event_sampler_end",
                "status": "ok",
                "branch_name": window.branch_name,
                "branch_role": window.branch_role,
            })
            return EventSamplerResult(True, latent, latent_after, window, records)
        except Exception as e:
            records.append({
                "stage": "event_sampler_failed",
                "status": "failed",
                "branch_name": window.branch_name,
                "branch_role": window.branch_role,
                "error": str(e),
            })
            return EventSamplerResult(False, latent, None, window, records, error=str(e))
