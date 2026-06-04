_EVENT_HORIZON_CASCADE_CACHE = {
    "latent": None,
    "frames": None,
    "segment_index": -1
}
import copy
import csv
import json
import math
import os
import re
import inspect
import importlib
import time
import hashlib
from pathlib import Path
from datetime import datetime
from .enums import (
    DEBUG_MODES,
    DEBUG_BASIC,
    TECH_TEXT,
    TECH_LATENT,
    TECH_IMAGE,
    TECH_NOISE,
    TECH_CONDITIONING,
    TECH_DELTA,
    FORMULA_ROLES_CODE_PASS_6,
    RELATION_TYPES_CODE_PASS_6,
    EQ_UNKNOWN,
    SPACE_TEXT,
    SPACE_LATENT,
    SPACE_IMAGE,
    SPACE_NOISE,
    SPACE_CONDITIONING,
    SPACE_DELTA,
    ROLE_UNKNOWN,
    ROLE_OUTCOME_PREVIOUS,
    ROLE_OUTCOME_NEXT,
    ROLE_OBSERVED_BEHAVIOR,
    SEV_LOW,
    SEV_INFO,
    SEV_MEDIUM,
    CONFLICT_ROLE_UNKNOWN,
    CONFLICT_NO_READER_FOUND,
    CONFLICT_EMPTY_RELATION,
    CONFLICT_INVALID_SIGNAL_ID,
    CONFLICT_DELTA_UNAVAILABLE,
    CONFLICT_BOUNDARY_SHAPE_MISMATCH,
    CONFLICT_SAMPLER_BOUNDARY_SHAPE_MISMATCH,
    CONFLICT_SAMPLER_DELTA_UNAVAILABLE,
    CONFLICT_SAMPLER_METADATA_MISSING,
    CONFLICT_NO_STABLE_ANCHORS,
    CONFLICT_NO_ACTIVE_CHANGES,
    CONFLICT_NO_CONTACT_RULE,
    CONFLICT_NO_ENDPOINT,
    CONFLICT_NO_FORBIDDEN_DRIFT,
    CONFLICT_NOISE_MISSING,
    CONFLICT_NOISE_SHAPE_UNKNOWN,
    CONFLICT_NOISE_READER_FAILED,
    CONFLICT_NOISE_LATENT_SHAPE_MISMATCH,
    CONFLICT_ALPHA_ROUTE_MISSING,
    CONFLICT_ALPHA_PARTIAL_INPUT,
    CONFLICT_FROZEN_ROUTE_MISSING,
    CONFLICT_FROZEN_PARTIAL_OBSERVABILITY,
    CONFLICT_PASSTHROUGH_OUTPUT_MISSING,
    CONFLICT_PACKET_BRANCH_CREATED,
    CONFLICT_REPORT_SAVE_FAILED,
    REL_GUIDES,
    REL_CONSTRAINS,
    REL_EXPANDS_TO,
    REL_TRANSFORMS_INTO,
)
from .packet import (
    make_event_packet,
    ensure_packet,
    record_stage,
    add_signal,
    add_projection,
    add_relation,
    add_conflict,
    get_latest_signals,
    get_projection_ids_for_signal,
    signal_exists,
)
from .signal import make_event_signal
from .relation import make_event_relation
from .conflict import make_conflict
from .event_sampler import EventSamplerCore, EventSamplerWindow, EventSamplerResult
import os
import sys

# Ensure parent directory is in sys.path so relative imports work
# when loaded by ComfyUI's custom node loader
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from ..reports.markdown_report import build_markdown_report
from ..resolvers.role_resolver import resolve_role
from ..resolvers.operator_registry import OperatorRegistry
from ..resolvers.s_resolver import build_sstate_from_packet
from ..utils.tensor_stats import compute_tensor_delta, extract_latent_samples, safe_shape
from ..utils.frozen_helpers import build_input_signatures, build_passthrough_status, score_observability, collect_shared_targets, now_run_id
from ..adapters.wan.wan_adapter import apply_wan_adapter

EVENT_HORIZON_RUNTIME_VERSION = "0.1.1-r59"
EVENT_HORIZON_RUNTIME_NAME = "Singularity R59 Strategy Math Native Loop"
EVENT_HORIZON_BODY_VERSION = "0.1-r59"


def _event_json_safe(value, depth=0):
    if depth > 8:
        return str(type(value).__name__)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _event_json_safe(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_event_json_safe(v, depth + 1) for v in value]
    return str(value)



def _read_signal(packet, technical_type, representation_space, raw_ref, source_stage, manual_role, route_id, reader_operator, metadata=None, route_position=""):
    packet = ensure_packet(packet)
    role = resolve_role(
        technical_type,
        position_context={"stage_name": source_stage, "route_position": route_position},
        manual_role=manual_role,
    )

    conflict_ids = []
    signal = make_event_signal(
        technical_type=technical_type,
        formula_role=role,
        representation_space=representation_space,
        raw_ref=raw_ref,
        source_stage=source_stage,
        created_by=f"EventRead{technical_type.title()}",
        reader_operator=reader_operator,
        route_id=route_id,
        position_context={"route_position": route_position},
        metadata=metadata or {},
    )

    projection = OperatorRegistry().read(signal)

    if role == ROLE_UNKNOWN:
        conflict = make_conflict(
            CONFLICT_ROLE_UNKNOWN,
            severity=SEV_LOW,
            involved_signal_ids=[signal["id"]],
            stage_position=source_stage,
            suspected_cause="RoleResolver could not assign a formula role.",
            observed_symptom=f"{technical_type} resolved to UnknownRole.",
            suggested_response="Provide a manual role or improve position_context.",
        )
        packet = add_conflict(packet, conflict)
        conflict_ids.append(conflict["id"])

    if projection.get("metadata", {}).get("warning") == CONFLICT_NO_READER_FOUND or projection.get("operator_name") == "UnknownReader":
        conflict = make_conflict(
            CONFLICT_NO_READER_FOUND,
            severity=SEV_MEDIUM,
            involved_signal_ids=[signal["id"]],
            involved_projection_ids=[projection["id"]],
            stage_position=source_stage,
            suspected_cause="OperatorRegistry found no reader for this signal.",
            observed_symptom=f"No reader for technical type {technical_type}.",
            suggested_response="Add/register a ReaderOperator for this technical type.",
        )
        packet = add_conflict(packet, conflict)
        conflict_ids.append(conflict["id"])

    packet = add_signal(packet, signal)
    packet = add_projection(packet, projection)
    packet = record_stage(
        packet,
        stage_name=f"EventRead{technical_type.title()}",
        action="READ_SIGNAL",
        observed_behavior=f"{technical_type} read as {role}",
        output_signal_ids=[signal["id"]],
        projection_ids=[projection["id"]],
        conflict_ids=conflict_ids,
        formula_note=f"{technical_type} projected into EVENT_SPACE",
    )
    return packet, signal, projection, conflict_ids


def _parse_id_list(value):
    text = str(value or "").strip()
    if not text:
        return []
    if text.lower() in ("latest", "auto"):
        return []
    return [x.strip() for x in text.split(",") if x.strip()]



def _event_core_list_input_images():
    try:
        import folder_paths
        input_dir = folder_paths.get_input_directory()
        if not os.path.isdir(input_dir):
            return ["none"]
        valid_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
        files = []
        for name in os.listdir(input_dir):
            path = os.path.join(input_dir, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in valid_ext:
                files.append(name)
        files = sorted(files)
        return ["none"] + files if files else ["none"]
    except Exception:
        return ["none"]


def _event_core_list_output_folders():
    try:
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        entries = ["default"]
        if os.path.isdir(output_dir):
            for root, dirs, files in os.walk(output_dir):
                rel = os.path.relpath(root, output_dir)
                if rel != ".":
                    entries.append(rel.replace("\\", "/"))
                if len(entries) >= 100:
                    break
        return sorted(set(entries))
    except Exception:
        return ["default"]



def _event_core_preferred_media_dirs(media_type="video"):
    media_type = str(media_type or "video").lower()
    if media_type == "video":
        folder_names = ["VID"]
    elif media_type == "image":
        folder_names = ["PIC"]
    elif media_type == "report":
        folder_names = ["reports"]
    else:
        folder_names = [media_type]

    # Old D:\AI NSFW paths removed (updated 2026-06)
    roots = []
    candidates = []
    for root in roots:
        for name in folder_names:
            candidates.append(root / name)
    return candidates


class SingularityCascadeMixin:
    def _math_tensor_summary(self, obj, records, stage, reference=None, strict=False):
        """
        r34 measurement-only math report.
        Does not modify tensors. Records finite status, norm, mean/std, optional delta and relative delta.
        """
        try:
            import torch
            t = self._tensor_from_latent_like(obj)
            if t is None:
                rec = {"stage": stage, "status": "unavailable", "reason": "no_tensor_found"}
                records.append(rec)
                return rec

            tf = t.detach().float()
            finite_mask = torch.isfinite(tf)
            finite_ok = bool(finite_mask.all().item())
            finite_ratio = float(finite_mask.float().mean().item()) if tf.numel() else 0.0

            safe = torch.nan_to_num(tf, nan=0.0, posinf=0.0, neginf=0.0)
            rec = {
                "stage": stage,
                "status": "ok" if finite_ok else "nonfinite",
                "shape": list(tf.shape),
                "dtype": str(t.dtype) if hasattr(t, "dtype") else str(type(t)),
                "device": str(t.device) if hasattr(t, "device") else "unknown",
                "finite_ok": finite_ok,
                "finite_ratio": finite_ratio,
                "mean": float(safe.mean().item()) if safe.numel() else 0.0,
                "std": float(safe.std().item()) if safe.numel() > 1 else 0.0,
                "min": float(safe.min().item()) if safe.numel() else 0.0,
                "max": float(safe.max().item()) if safe.numel() else 0.0,
                "norm": float(torch.linalg.vector_norm(safe).item()) if safe.numel() else 0.0,
            }

            if reference is not None:
                rt = self._tensor_from_latent_like(reference)
                if rt is not None:
                    rf = rt.detach().float()
                    rsafe = torch.nan_to_num(rf, nan=0.0, posinf=0.0, neginf=0.0)
                    if rsafe.shape == safe.shape:
                        delta = safe - rsafe
                        ref_norm = torch.linalg.vector_norm(rsafe).item() if rsafe.numel() else 0.0
                        delta_norm = torch.linalg.vector_norm(delta).item() if delta.numel() else 0.0
                        rec.update({
                            "delta_norm": float(delta_norm),
                            "reference_norm": float(ref_norm),
                            "relative_delta": float(delta_norm / (ref_norm + 1e-12)),
                            "delta_mean": float(delta.mean().item()) if delta.numel() else 0.0,
                            "delta_std": float(delta.std().item()) if delta.numel() > 1 else 0.0,
                            "delta_min": float(delta.min().item()) if delta.numel() else 0.0,
                            "delta_max": float(delta.max().item()) if delta.numel() else 0.0,
                        })
                    else:
                        rec.update({
                            "delta_status": "shape_mismatch",
                            "reference_shape": list(rf.shape),
                        })

            records.append(rec)
            if strict and not finite_ok:
                raise RuntimeError(f"{stage} contains NaN/Inf; finite_ratio={finite_ratio}")
            return rec
        except Exception as e:
            rec = {"stage": stage, "status": "failed", "error": str(e)}
            records.append(rec)
            if strict:
                raise
            return rec

    def _frame_motion_math(self, frames, records, stage):
        """
        r34: measure temporal frame-to-frame motion after decode.
        Works on decoded frame tensors [T,H,W,C] or similar without modifying frames.
        """
        try:
            import torch
            t = self._tensor_from_latent_like(frames)
            if t is None:
                records.append({"stage": stage, "status": "unavailable", "reason": "no_tensor_found"})
                return None

            tf = torch.nan_to_num(t.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
            if tf.dim() < 2 or tf.shape[0] < 2:
                records.append({"stage": stage, "status": "unavailable", "reason": "not_enough_frames", "shape": list(tf.shape)})
                return None

            delta = tf[1:] - tf[:-1]
            flat = delta.reshape(delta.shape[0], -1)
            norms = torch.linalg.vector_norm(flat, dim=1)
            mean_abs = delta.abs().reshape(delta.shape[0], -1).mean(dim=1)

            avg = float(norms.mean().item())
            mx = float(norms.max().item())
            mn = float(norms.min().item())
            std = float(norms.std().item()) if norms.numel() > 1 else 0.0
            spike_ratio = float(mx / (avg + 1e-12))
            cv_ratio = float(std / (avg + 1e-12))

            p25 = float(torch.quantile(norms, 0.25).item()) if norms.numel() else 0.0
            p50 = float(torch.quantile(norms, 0.50).item()) if norms.numel() else 0.0
            p75 = float(torch.quantile(norms, 0.75).item()) if norms.numel() else 0.0
            p90 = float(torch.quantile(norms, 0.90).item()) if norms.numel() else 0.0
            p95 = float(torch.quantile(norms, 0.95).item()) if norms.numel() else 0.0
            iqr = float(p75 - p25)
            p95_to_p50_ratio = float(p95 / (p50 + 1e-12))

            # Consecutive delta direction stability.
            cosine_mean = None
            reversal_ratio = None
            if flat.shape[0] >= 2:
                a = flat[1:]
                b = flat[:-1]
                dot = (a * b).sum(dim=1)
                denom = torch.linalg.vector_norm(a, dim=1) * torch.linalg.vector_norm(b, dim=1) + 1e-12
                cos = dot / denom
                cosine_mean = float(torch.nan_to_num(cos, nan=0.0).mean().item())
                reversal_ratio = float((cos < -0.05).float().mean().item())

            # Frame-to-frame motion acceleration/jerk proxy on norm trajectory.
            jerk_abs_mean = 0.0
            jerk_ratio = 0.0
            if norms.numel() >= 2:
                d1 = norms[1:] - norms[:-1]
                jerk_abs_mean = float(d1.abs().mean().item())
                jerk_ratio = float(jerk_abs_mean / (avg + 1e-12))

            # Heuristic stability score for run-to-run comparisons (observer-only).
            rev_component = max(0.0, 1.0 - float(reversal_ratio if reversal_ratio is not None else 0.5))
            spike_component = 1.0 / (1.0 + max(0.0, spike_ratio - 1.0))
            cv_component = 1.0 / (1.0 + cv_ratio)
            jerk_component = 1.0 / (1.0 + jerk_ratio)
            stability_score = float(
                0.40 * rev_component
                + 0.25 * spike_component
                + 0.20 * cv_component
                + 0.15 * jerk_component
            )
            if stability_score >= 0.72:
                motion_profile = "stable"
            elif stability_score >= 0.55:
                motion_profile = "mixed"
            else:
                motion_profile = "volatile"

            rec = {
                "stage": stage,
                "status": "ok",
                "shape": list(tf.shape),
                "frame_delta_count": int(delta.shape[0]),
                "frame_delta_norm_mean": avg,
                "frame_delta_norm_std": std,
                "frame_delta_norm_min": mn,
                "frame_delta_norm_max": mx,
                "frame_delta_norm_p25": p25,
                "frame_delta_norm_p50": p50,
                "frame_delta_norm_p75": p75,
                "frame_delta_norm_p90": p90,
                "frame_delta_norm_p95": p95,
                "frame_delta_norm_iqr": iqr,
                "frame_delta_norm_cv_ratio": cv_ratio,
                "frame_delta_p95_to_p50_ratio": p95_to_p50_ratio,
                "frame_delta_spike_ratio": spike_ratio,
                "frame_delta_abs_mean": float(mean_abs.mean().item()) if mean_abs.numel() else 0.0,
                "frame_delta_abs_max": float(mean_abs.max().item()) if mean_abs.numel() else 0.0,
                "frame_delta_cosine_mean": cosine_mean,
                "frame_delta_reversal_ratio": reversal_ratio,
                "frame_delta_jerk_abs_mean": jerk_abs_mean,
                "frame_delta_jerk_ratio": jerk_ratio,
                "frame_motion_stability_score": stability_score,
                "frame_motion_profile": motion_profile,
                "frame_motion_score_formula": "0.40*(1-reversal)+0.25*(1/(1+max(0,spike-1)))+0.20*(1/(1+cv))+0.15*(1/(1+jerk_ratio))",
                "frame_motion_score_note": "heuristic observer-only ranking metric for run-to-run comparison",
            }
            records.append(rec)
            return rec
        except Exception as e:
            rec = {"stage": stage, "status": "failed", "error": str(e)}
            records.append(rec)
            return rec

    # _dual_branch_delta_coupling_math fully excised (physical cut #21): removed the observer-only smart alignment
    # and energy scoring layer on top of the raw high/low deltas. The dual-branch now interacts without this interpretive comfort math.
        except Exception as e:
            rec.update({"status": "failed", "error": str(e)})
            records.append(rec)
            return rec

    def _cascade_boundary_math(self, previous_frames, next_frames, records, segment_index):
        try:
            import torch
            a = self._tensor_from_latent_like(previous_frames)
            b = self._tensor_from_latent_like(next_frames)
            if a is None or b is None:
                records.append({"stage": "EventMathCascadeBoundary", "status": "unavailable", "segment_index": int(segment_index)})
                return None
            af = torch.nan_to_num(a.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
            bf = torch.nan_to_num(b.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
            if af.shape[0] < 1 or bf.shape[0] < 1:
                records.append({"stage": "EventMathCascadeBoundary", "status": "unavailable", "segment_index": int(segment_index), "reason": "empty_frames"})
                return None
            last_prev = af[-1]
            first_next = bf[0]
            if last_prev.shape != first_next.shape:
                records.append({
                    "stage": "EventMathCascadeBoundary",
                    "status": "shape_mismatch",
                    "segment_index": int(segment_index),
                    "prev_shape": list(last_prev.shape),
                    "next_shape": list(first_next.shape),
                })
                return None
            d = first_next - last_prev
            rec = {
                "stage": "EventMathCascadeBoundary",
                "status": "ok",
                "segment_index": int(segment_index),
                "boundary_delta_norm": float(torch.linalg.vector_norm(d).item()),
                "boundary_delta_abs_mean": float(d.abs().mean().item()),
                "boundary_delta_min": float(d.min().item()),
                "boundary_delta_max": float(d.max().item()),
            }
            records.append(rec)
            self._event_universal_boundary_math(
                records,
                f"EventCascadeBoundary_{int(segment_index)}",
                before_state=a[-1:] if hasattr(a, "__getitem__") else previous_frames,
                after_state=b[:1] if hasattr(b, "__getitem__") else next_frames,
                observed_behavior="previous cascade last frame connects to next cascade first frame",
                route_id=f"route_cascade_boundary_{int(segment_index)}",
                control_mode="REPORT_ONLY",
                metadata=rec,
            )
            return rec
        except Exception as e:
            rec = {"stage": "EventMathCascadeBoundary", "status": "failed", "segment_index": int(segment_index), "error": str(e)}
            records.append(rec)
            return rec

    def _event_core_body_init(self, packet, execution_records, run_id, route_name="wan_terminal_one_node"):
        """
        r44 internal Event Core Body.
        This is not a separate visual ComfyUI node. It is the internal body of the single Singularity node.
        """
        packet = ensure_packet(packet)
        body = {
            "body_version": EVENT_HORIZON_BODY_VERSION,
            "body_name": "One Node Event Core Body + Runtime Monitor Body",
            "external_node": "Singularity",
            "visual_node_policy": "single_external_node_internal_event_body",
            "route_name": str(route_name),
            "run_id": str(run_id),
            "formula": "NodeInputState + NodeObservedBehavior = EventSingularity = NodeOutputState",
            "body_layers": [
                "EventPacket",
                "FormulaReader",
                "RoleResolver",
                "RouteMemory",
                "SState",
                "InternalWanPipeline",
                "SamplerBoundaries",
                "DecodeBoundary",
                "OutputBoundary",
                "ReportBuilder",
            ],
            "technical_wires": ["MODEL", "CLIP", "VAE", "IMAGE", "CONDITIONING", "LATENT", "DELTA", "FRAMES", "VIDEO"],
            "s_wire": [],
            "stage_math_records": [],
            "boundary_records": [],
            "live_route_timeline": [],
            "runtime_monitor_records": [],
            "local_sstates": [],
            "event_conflicts": [],
            "conflict_policy": "report_first_no_generation_abort_unless_technical_failure",
            "runtime_layer_policy": "observer-only until equivalence or safety proof",
        }
        self._event_live_body = {
            "run_id": str(run_id),
            "route_name": str(route_name),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "start_perf_counter": float(time.perf_counter()),
            "route_timeline": [],
            "s_wire_live": [],
            "runtime_monitor": [],
            "local_sstates": [],
            "conflicts": [],
            "policy": "observer_only_no_generation_side_effects",
        }
        packet.setdefault("metadata", {})["event_core_body"] = body
        execution_records.append({
            "stage": "EventCoreBodyInit",
            "status": "recorded",
            "body_version": body["body_version"],
            "body_name": body["body_name"],
            "visual_node_policy": body["visual_node_policy"],
            "formula": body["formula"],
            "route_name": body["route_name"],
            "message": "Event Core Body is internal to this one Singularity node; no manual Event graph required.",
        })
        packet = record_stage(
            packet,
            stage_name="EventCoreBody",
            action="INIT_INTERNAL_BODY",
            observed_behavior="Single Singularity node created internal EventPacket, S-Wire, RouteMemory, and Formula route body.",
            metadata=body,
            formula_note=body["formula"],
        )
        return packet

    def _event_universal_stage_math(self, records, stage_name, input_state=None, output_state=None,
                                    observed_behavior="", formula_role="", route_id="", next_requirement="",
                                    control_mode="REPORT_ONLY", metadata=None):
        """
        r40 universal stage math record.
        Common wrapper for every internal stage:
            NodeInputState + NodeObservedBehavior = NodeSState = NodeOutputState
        Report-first. Does not modify generation.
        """
        metadata = metadata or {}
        rec = {
            "stage": f"EventUniversalMath_{stage_name}",
            "status": "recorded",
            "formula": "NodeInputState + NodeObservedBehavior = NodeSState = NodeOutputState",
            "stage_name": str(stage_name),
            "formula_role": str(formula_role or ""),
            "route_id": str(route_id or ""),
            "observed_behavior": str(observed_behavior or ""),
            "next_requirement": str(next_requirement or ""),
            "control_mode": str(control_mode or "REPORT_ONLY"),
            "metadata": metadata,
        }
        try:
            if input_state is not None:
                t = self._tensor_from_latent_like(input_state)
                if t is not None:
                    rec["input_state_shape"] = list(t.shape)
                    rec["input_state_dtype"] = str(getattr(t, "dtype", "unknown"))
                else:
                    rec["input_state_type"] = type(input_state).__name__
            if output_state is not None:
                t = self._tensor_from_latent_like(output_state)
                if t is not None:
                    rec["output_state_shape"] = list(t.shape)
                    rec["output_state_dtype"] = str(getattr(t, "dtype", "unknown"))
                else:
                    rec["output_state_type"] = type(output_state).__name__
        except Exception as e:
            rec["summary_error"] = str(e)
        records.append(rec)
        self._event_core_live_record(
            stage_name=stage_name,
            record_type="stage_math",
            status=rec.get("status", ""),
            formula_role=rec.get("formula_role", ""),
            route_id=rec.get("route_id", ""),
            observed_behavior=rec.get("observed_behavior", ""),
            metadata={"execution_record_index": len(records) - 1, "control_mode": rec.get("control_mode", "")},
        )
        return rec

    def _safe_event_universal_stage_math(self, records, stage_name, **kwargs):
        try:
            return self._event_universal_stage_math(records, stage_name, **kwargs)
        except Exception as e:
            try:
                records.append({
                    "stage": f"EventUniversalMath_{stage_name}",
                    "status": "failed_nonfatal",
                    "error": str(e),
                    "formula": "NodeInputState + NodeObservedBehavior = NodeSState = NodeOutputState",
                })
            except Exception:
                pass
            return None

    def _event_universal_boundary_math(self, records, boundary_name, before_state=None, after_state=None,
                                       observed_behavior="", route_id="", control_mode="REPORT_ONLY", metadata=None):
        """
        r41 boundary record for transitions between stages.
        It records:
            StageBefore.OutputState + BoundaryObservedBehavior = StageAfter.InputState
        """
        metadata = metadata or {}
        rec = {
            "stage": f"EventUniversalBoundary_{boundary_name}",
            "status": "recorded",
            "formula": "StageBefore.OutputState + BoundaryObservedBehavior = StageAfter.InputState",
            "boundary_name": str(boundary_name),
            "route_id": str(route_id or ""),
            "observed_behavior": str(observed_behavior or ""),
            "control_mode": str(control_mode or "REPORT_ONLY"),
            "metadata": metadata,
        }
        try:
            if before_state is not None:
                t = self._tensor_from_latent_like(before_state)
                if t is not None:
                    rec["before_shape"] = list(t.shape)
                    rec["before_dtype"] = str(getattr(t, "dtype", "unknown"))
                else:
                    rec["before_type"] = type(before_state).__name__
            if after_state is not None:
                t = self._tensor_from_latent_like(after_state)
                if t is not None:
                    rec["after_shape"] = list(t.shape)
                    rec["after_dtype"] = str(getattr(t, "dtype", "unknown"))
                else:
                    rec["after_type"] = type(after_state).__name__
        except Exception as e:
            rec["summary_error"] = str(e)
        records.append(rec)
        self._event_core_live_record(
            stage_name=boundary_name,
            record_type="boundary_math",
            status=rec.get("status", ""),
            route_id=rec.get("route_id", ""),
            observed_behavior=rec.get("observed_behavior", ""),
            metadata={"execution_record_index": len(records) - 1, "control_mode": rec.get("control_mode", "")},
        )
        return rec

    def _event_control_warning(self, records, mode, high_delta_strength, low_delta_strength):
        try:
            high = float(high_delta_strength)
            low = float(low_delta_strength)
            mode = str(mode or "OBSERVE_ONLY")
            if mode == "OBSERVE_ONLY" and (abs(high - 1.0) > 1e-9 or abs(low - 1.0) > 1e-9):
                records.append({
                    "stage": "EventMathControlWarning",
                    "status": "strength_ignored_in_observe_only",
                    "mode": mode,
                    "high_delta_strength": high,
                    "low_delta_strength": low,
                    "message": "high_delta_strength / low_delta_strength are recorded but not applied while math_control_mode is OBSERVE_ONLY",
                })
        except Exception as e:
            records.append({
                "stage": "EventMathControlWarning",
                "status": "failed",
                "error": str(e),
            })

    def _apply_latent_delta_control(
        self,
        latent_before,
        latent_after,
        branch_name,
        records,
        *,
        strength_override=None,
        step_index=None,
        window_steps=None,
    ):
        """
        Active math control used directly in the sampler transition path.
        Formula:
            controlled_after = latent_before + (latent_after - latent_before) * strength_runtime
        where strength_runtime may include:
          - requested branch strength
          - StrategyCarrier high->low coupling multiplier
          - per-step schedule factor inside the denoise window
        """
        try:
            mode = str(getattr(self, "_event_math_control_mode", "OBSERVE_ONLY") or "OBSERVE_ONLY")
            strengths = getattr(self, "_event_delta_strengths", {}) or {}
            branch_lower = str(branch_name or "").lower()
            if strength_override is not None:
                base_strength = float(strength_override)
            elif branch_lower == "high" or branch_lower.endswith("_high") or "_high_" in branch_lower:
                base_strength = float(strengths.get("high", 1.0))
            elif branch_lower == "low" or branch_lower.endswith("_low") or "_low_" in branch_lower:
                base_strength = float(strengths.get("low", 1.0))
            else:
                base_strength = float(strengths.get(str(branch_name), 1.0))
        except Exception:
            mode = "OBSERVE_ONLY"
            base_strength = 1.0

        branch_lower = str(branch_name or "").lower()
        coupling = getattr(self, "_event_strategy_coupling", {}) or {}
        coupling_multiplier = 1.0
        try:
            if "low" in branch_lower:
                coupling_multiplier = float(coupling.get("low_strength_multiplier", 1.0) or 1.0)
        except Exception:
            coupling_multiplier = 1.0

        scheduled_strength = float(base_strength * coupling_multiplier)
        step_schedule_factor = 1.0
        try:
            if step_index is not None and window_steps is not None and int(window_steps) > 0:
                progress = float((int(step_index) + 1) / max(1, int(window_steps)))
                if "high" in branch_lower:
                    # High branch: slightly stronger earlier, softer toward window end.
                    step_schedule_factor = 1.10 - (0.20 * progress)
                elif "low" in branch_lower:
                    # Low branch: slightly softer early, stronger toward detail refinement end.
                    step_schedule_factor = 0.90 + (0.20 * progress)
                scheduled_strength = scheduled_strength * step_schedule_factor
        except Exception:
            step_schedule_factor = 1.0

        strength_runtime = max(0.0, min(2.0, float(scheduled_strength)))

        if mode not in ("LATENT_DELTA_SCALE", "DEEP_STEP_DELTA_CONTROL") or abs(strength_runtime - 1.0) < 1e-9:
            records.append({
                "stage": f"EventMathDeltaControl_{branch_name}",
                "status": "bypass",
                "mode": mode,
                "base_strength": base_strength,
                "coupling_multiplier": coupling_multiplier,
                "step_schedule_factor": step_schedule_factor,
                "strength_runtime": strength_runtime,
                "step_index": int(step_index) if step_index is not None else None,
                "window_steps": int(window_steps) if window_steps is not None else None,
                "formula": "latent_after unchanged",
            })
            return latent_after

        try:
            import torch
            before_t = self._tensor_from_latent_like(latent_before)
            after_t = self._tensor_from_latent_like(latent_after)
            if before_t is None or after_t is None:
                records.append({
                    "stage": f"EventMathDeltaControl_{branch_name}",
                    "status": "unavailable",
                    "mode": mode,
                    "base_strength": base_strength,
                    "coupling_multiplier": coupling_multiplier,
                    "step_schedule_factor": step_schedule_factor,
                    "strength_runtime": strength_runtime,
                    "reason": "missing before/after tensor",
                })
                return latent_after
            if before_t.shape != after_t.shape:
                records.append({
                    "stage": f"EventMathDeltaControl_{branch_name}",
                    "status": "shape_mismatch",
                    "mode": mode,
                    "base_strength": base_strength,
                    "coupling_multiplier": coupling_multiplier,
                    "step_schedule_factor": step_schedule_factor,
                    "strength_runtime": strength_runtime,
                    "before_shape": list(before_t.shape),
                    "after_shape": list(after_t.shape),
                })
                return latent_after

            before_f = before_t.detach().float()
            after_f = after_t.detach().float()
            controlled = before_f + (after_f - before_f) * float(strength_runtime)
            controlled = controlled.to(dtype=after_t.dtype, device=after_t.device)

            if isinstance(latent_after, dict) and "samples" in latent_after:
                out = dict(latent_after)
                out["samples"] = controlled
            else:
                out = controlled

            records.append({
                "stage": f"EventMathDeltaControl_{branch_name}",
                "status": "applied",
                "mode": mode,
                "base_strength": base_strength,
                "coupling_multiplier": coupling_multiplier,
                "step_schedule_factor": step_schedule_factor,
                "strength_runtime": strength_runtime,
                "step_index": int(step_index) if step_index is not None else None,
                "window_steps": int(window_steps) if window_steps is not None else None,
                "formula": "controlled_after = latent_before + (latent_after - latent_before) * strength_runtime",
            })
            self._math_tensor_summary(
                out,
                records,
                f"EventMath_{branch_name}_latent_after_controlled",
                reference=latent_before,
                strict=False,
            )
            return out
        except Exception as e:
            records.append({
                "stage": f"EventMathDeltaControl_{branch_name}",
                "status": "failed_passthrough",
                "mode": mode,
                "base_strength": base_strength,
                "coupling_multiplier": coupling_multiplier,
                "step_schedule_factor": step_schedule_factor,
                "strength_runtime": strength_runtime,
                "error": str(e),
            })
            return latent_after

    def _update_strategy_coupling_from_high(self, latent_before, latent_after, branch_name, records):
        """
        High->low StrategyCarrier coupling:
        derive low-branch strength multiplier from high branch delta energy ratio.
        """
        try:
            import torch
            before_t = self._tensor_from_latent_like(latent_before)
            after_t = self._tensor_from_latent_like(latent_after)
            if before_t is None or after_t is None:
                records.append({
                    "stage": "EventMathStrategyCarrierCoupling",
                    "status": "unavailable",
                    "branch_name": str(branch_name or ""),
                    "reason": "missing_high_branch_tensor",
                })
                self._event_strategy_coupling = {"low_strength_multiplier": 1.0}
                return
            if before_t.shape != after_t.shape:
                records.append({
                    "stage": "EventMathStrategyCarrierCoupling",
                    "status": "shape_mismatch",
                    "branch_name": str(branch_name or ""),
                    "before_shape": list(before_t.shape),
                    "after_shape": list(after_t.shape),
                })
                self._event_strategy_coupling = {"low_strength_multiplier": 1.0}
                return

            before_f = torch.nan_to_num(before_t.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
            after_f = torch.nan_to_num(after_t.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
            delta = after_f - before_f
            before_norm = float(torch.linalg.vector_norm(before_f).item()) if before_f.numel() else 0.0
            after_norm = float(torch.linalg.vector_norm(after_f).item()) if after_f.numel() else 0.0
            delta_norm = float(torch.linalg.vector_norm(delta).item()) if delta.numel() else 0.0
            # Important for I2V: high branch often starts from near-zero latent seed.
            # Using before_norm alone would explode relative_delta and force artificial low-branch suppression.
            baseline_norm = float(max(before_norm, after_norm, 1e-12))
            relative_delta = float(delta_norm / baseline_norm)

            if before_norm <= 1e-9 and after_norm > 0.0:
                low_multiplier = 1.0
                policy = "neutralize_low_strength_when_high_baseline_is_zero"
            elif relative_delta > 1.35:
                low_multiplier = max(0.75, min(1.0, 1.35 / (relative_delta + 1e-12)))
                policy = "reduce_low_strength_after_energetic_high_branch"
            elif relative_delta < 0.85:
                low_multiplier = min(1.25, max(1.0, 0.85 / (relative_delta + 1e-12)))
                policy = "raise_low_strength_after_soft_high_branch"
            else:
                low_multiplier = 1.0
                policy = "keep_low_strength_neutral"

            self._event_strategy_coupling = {
                "low_strength_multiplier": float(low_multiplier),
                "source_branch": str(branch_name or ""),
                "relative_delta": float(relative_delta),
                "delta_norm": float(delta_norm),
                "before_norm": float(before_norm),
                "after_norm": float(after_norm),
                "baseline_norm": float(baseline_norm),
                "policy": policy,
            }
            records.append({
                "stage": "EventMathStrategyCarrierCoupling",
                "status": "applied",
                "source_branch": str(branch_name or ""),
                "relative_delta": float(relative_delta),
                "low_strength_multiplier": float(low_multiplier),
                "before_norm": float(before_norm),
                "after_norm": float(after_norm),
                "baseline_norm": float(baseline_norm),
                "policy": policy,
                "formula": "StrategyCarrier(high) statistics modulate low branch control strength before refinement.",
            })
        except Exception as e:
            self._event_strategy_coupling = {"low_strength_multiplier": 1.0}
            records.append({
                "stage": "EventMathStrategyCarrierCoupling",
                "status": "failed",
                "branch_name": str(branch_name or ""),
                "error": str(e),
            })

    def _event_sampler_trace_config(self):
        cfg = getattr(self, "_event_sampler_trace", {}) or {}
        mode = str(cfg.get("mode", "OFF") or "OFF").upper()
        try:
            max_steps = int(cfg.get("max_steps", 64) or 64)
        except Exception:
            max_steps = 64
        max_steps = max(1, min(65535, max_steps))
        return mode, max_steps

    def _event_sampler_step_trace_shadow(self, model, positive, negative, latent_before, latent_after_main, window, records):
        """
        Optional sampler step-level shadow trace (observer-only).
        It does not replace the active sampler path and does not mutate generation outputs.
        """
        mode, max_steps = self._event_sampler_trace_config()
        branch = str(getattr(window, "branch_name", "branch") or "branch")

        if mode != "SHADOW_STEP_TRACE":
            records.append({
                "stage": f"EventSamplerStepTraceSummary_{branch}",
                "status": "disabled",
                "trace_mode": mode,
                "max_steps": max_steps,
            })
            return None

        try:
            start = int(getattr(window, "start_at_step", 0) or 0)
            end = int(getattr(window, "end_at_step", 0) or 0)
            requested_steps = max(0, end - start)
            if requested_steps <= 0:
                records.append({
                    "stage": f"EventSamplerStepTraceSummary_{branch}",
                    "status": "unavailable",
                    "trace_mode": mode,
                    "reason": "empty_window",
                    "start_at_step": start,
                    "end_at_step": end,
                })
                return None

            traced_end = min(end, start + max_steps)
            shadow_latent = latent_before
            traced = 0

            records.append({
                "stage": f"EventSamplerStepTraceBegin_{branch}",
                "status": "recorded",
                "trace_mode": mode,
                "requested_steps": requested_steps,
                "traced_steps_limit": max_steps,
                "traced_step_range": [start, traced_end],
                "formula": "step-level shadow trace: X(k+1) = sampler_step(X(k), constraints)",
            })

            for step_index in range(start, traced_end):
                step_add_noise = str(window.add_noise) if step_index == start else "disable"
                step_return_leftover = "enable"
                if step_index == traced_end - 1 and traced_end == end:
                    step_return_leftover = str(getattr(window, "return_with_leftover_noise", "disable"))

                self._math_tensor_summary(
                    shadow_latent,
                    records,
                    f"EventMath_{branch}_step_{step_index}_latent_before",
                    strict=False,
                )

                step_after = self._low_level_sampler_operation(
                    model=model,
                    positive=positive,
                    negative=negative,
                    latent=shadow_latent,
                    seed=int(window.seed),
                    steps=int(window.steps),
                    cfg=float(window.cfg),
                    sampler_name=str(window.sampler_name),
                    scheduler=str(window.scheduler),
                    start_at_step=int(step_index),
                    end_at_step=int(step_index + 1),
                    add_noise=step_add_noise,
                    return_leftover_noise=step_return_leftover,
                )

                self._math_tensor_summary(
                    step_after,
                    records,
                    f"EventMath_{branch}_step_{step_index}_latent_after",
                    reference=shadow_latent,
                    strict=False,
                )

                self._safe_event_universal_stage_math(
                    records,
                    f"EventSamplerStepTrace_{branch}_k{step_index}",
                    input_state=shadow_latent,
                    output_state=step_after,
                    observed_behavior=f"{branch} sampler step trace transformed latent at step {step_index}->{step_index + 1}",
                    formula_role="LATENT OutcomePrevious(step) + sampler step delta = LATENT OutcomeNext(step)",
                    route_id=f"route_sampler_step_trace_{branch}_{step_index}",
                    next_requirement="next denoise step requires latent output from current step",
                    control_mode="REPORT_ONLY",
                    metadata={
                        "branch": branch,
                        "step_index": int(step_index),
                        "window_start": start,
                        "window_end": end,
                        "shadow_trace_mode": mode,
                        "add_noise": step_add_noise,
                        "return_with_leftover_noise": step_return_leftover,
                    },
                )
                shadow_latent = step_after
                traced += 1

            compare_rec = self._math_tensor_summary(
                latent_after_main,
                records,
                f"EventMath_{branch}_step_trace_vs_window_output",
                reference=shadow_latent,
                strict=False,
            )
            records.append({
                "stage": f"EventSamplerStepTraceSummary_{branch}",
                "status": "recorded",
                "trace_mode": mode,
                "requested_steps": requested_steps,
                "traced_steps": traced,
                "truncated": bool(traced_end < end),
                "max_steps": max_steps,
                "trace_vs_window_relative_delta": compare_rec.get("relative_delta") if isinstance(compare_rec, dict) else None,
                "formula": "shadow step trace is observer-only and compared against main window output",
            })
            return True
        except Exception as e:
            records.append({
                "stage": f"EventSamplerStepTraceSummary_{branch}",
                "status": "failed_nonfatal",
                "trace_mode": mode,
                "max_steps": max_steps,
                "error": str(e),
            })
            return None

    def _event_sample_window_math_native(self, model, positive, negative, latent, window, records):
        """
        Event-native active math sampler loop.
        Runs one denoise step at a time and applies delta control per step.
        """
        branch = str(getattr(window, "branch_name", "branch") or "branch")
        start = int(getattr(window, "start_at_step", 0) or 0)
        end = int(getattr(window, "end_at_step", 0) or 0)
        window_steps = max(0, end - start)

        event_records = [{
            "stage": "event_sampler_begin",
            "status": "begin",
            "branch_name": branch,
            "branch_role": getattr(window, "branch_role", ""),
            "start_at_step": start,
            "end_at_step": end,
            "replacement_layer": "event_native_math_loop",
            "step_loop": "native_math_active",
            "math_control_mode": str(getattr(self, "_event_math_control_mode", "OBSERVE_ONLY")),
        }]

        if window_steps <= 0:
            event_records.append({
                "stage": "event_sampler_failed",
                "status": "failed",
                "branch_name": branch,
                "branch_role": getattr(window, "branch_role", ""),
                "error": "empty sampler window",
            })
            return EventSamplerResult(False, latent, None, window, event_records, error="empty sampler window")

        current = latent
        try:
            for local_index, step_index in enumerate(range(start, end)):
                step_add_noise = str(window.add_noise) if local_index == 0 else "disable"
                step_return_leftover = str(getattr(window, "return_with_leftover_noise", "disable")) if local_index == (window_steps - 1) else "enable"

                self._math_tensor_summary(
                    current,
                    records,
                    f"EventMath_{branch}_native_step_{step_index}_latent_before",
                    strict=False,
                )

                raw_after = self._low_level_sampler_operation(
                    model=model,
                    positive=positive,
                    negative=negative,
                    latent=current,
                    seed=int(window.seed),
                    steps=int(window.steps),
                    cfg=float(window.cfg),
                    sampler_name=str(window.sampler_name),
                    scheduler=str(window.scheduler),
                    start_at_step=int(step_index),
                    end_at_step=int(step_index + 1),
                    add_noise=step_add_noise,
                    return_leftover_noise=step_return_leftover,
                )
                self._math_tensor_summary(
                    raw_after,
                    records,
                    f"EventMath_{branch}_native_step_{step_index}_latent_after_raw",
                    reference=current,
                    strict=False,
                )

                controlled_after = self._apply_latent_delta_control(
                    current,
                    raw_after,
                    branch,
                    records,
                    step_index=local_index,
                    window_steps=window_steps,
                )
                self._finite_guard(
                    controlled_after,
                    records,
                    f"EventFiniteGuard_{branch}_native_step_{step_index}_latent_after",
                    strict=True,
                )
                self._math_tensor_summary(
                    controlled_after,
                    records,
                    f"EventMath_{branch}_native_step_{step_index}_latent_after_controlled",
                    reference=current,
                    strict=False,
                )

                self._safe_event_universal_stage_math(
                    records,
                    f"EventSamplerNativeStep_{branch}_k{step_index}",
                    input_state=current,
                    output_state=controlled_after,
                    observed_behavior=f"{branch} native math sampler step {step_index}->{step_index + 1} applied delta control in-loop",
                    formula_role="LATENT OutcomePrevious(step) + controlled sampler delta = LATENT OutcomeNext(step)",
                    route_id=f"route_sampler_native_step_{branch}_{step_index}",
                    next_requirement="next denoise step consumes controlled latent",
                    control_mode=str(getattr(self, "_event_math_control_mode", "OBSERVE_ONLY")),
                    metadata={
                        "branch": branch,
                        "step_index": int(step_index),
                        "local_index": int(local_index),
                        "window_start": int(start),
                        "window_end": int(end),
                        "add_noise": step_add_noise,
                        "return_with_leftover_noise": step_return_leftover,
                    },
                )

                current = controlled_after

            event_records.append({
                "stage": "event_sampler_end",
                "status": "ok",
                "branch_name": branch,
                "branch_role": getattr(window, "branch_role", ""),
                "step_loop": "native_math_active",
                "executed_steps": int(window_steps),
            })
            return EventSamplerResult(True, latent, current, window, event_records)
        except Exception as e:
            event_records.append({
                "stage": "event_sampler_failed",
                "status": "failed",
                "branch_name": branch,
                "branch_role": getattr(window, "branch_role", ""),
                "step_loop": "native_math_active",
                "error": str(e),
            })
            return EventSamplerResult(False, latent, None, window, event_records, error=str(e))

    def _event_sample_window(self, model, positive, negative, latent, window, records):
        self._math_tensor_summary(latent, records, f"EventMath_{window.branch_name}_latent_before", strict=False)
        mode = str(getattr(self, "_event_math_control_mode", "OBSERVE_ONLY") or "OBSERVE_ONLY")
        # Preserve model-native generation path by default.
        # Math in LATENT_DELTA_SCALE should guide/measure behavior, not replace denoising physics step-by-step.
        use_native_math_loop = (mode == "DEEP_STEP_DELTA_CONTROL")

        if mode == "LATENT_DELTA_SCALE":
            records.append({
                "stage": "EventMathSamplerPathPolicy",
                "status": "native_sampler_preserved",
                "math_control_mode": mode,
                "native_step_loop_replacement": False,
                "formula": "Math acts as semantic overlay/observer and post-window control, while sampler core stays model-native.",
            })
        elif mode == "DEEP_STEP_DELTA_CONTROL":
            records.append({
                "stage": "EventMathSamplerPathPolicy",
                "status": "native_step_loop_active",
                "math_control_mode": mode,
                "native_step_loop_replacement": True,
                "formula": "WARNING: Experimental deep-step delta control is active. High risk of noise.",
            })

        if use_native_math_loop:
            result = self._event_sample_window_math_native(model, positive, negative, latent, window, records)
        else:
            core = EventSamplerCore(self._low_level_sampler_operation)
            result = core.sample_window(model=model, positive=positive, negative=negative, latent=latent, window=window)

        records.extend(result.event_records)
        if not result.ok:
            raise RuntimeError(result.error or f"EventSampler {window.branch_name} failed")

        self._math_tensor_summary(result.latent_after, records, f"EventMath_{window.branch_name}_latent_after", reference=latent, strict=False)

        if not use_native_math_loop:
            self._event_sampler_step_trace_shadow(
                model=model,
                positive=positive,
                negative=negative,
                latent_before=latent,
                latent_after_main=result.latent_after,
                window=window,
                records=records,
            )
            controlled_latent_after = self._apply_latent_delta_control(latent, result.latent_after, window.branch_name, records)
        else:
            records.append({
                "stage": f"EventSamplerStepTraceSummary_{window.branch_name}",
                "status": "skipped_for_native_math_loop",
                "trace_mode": str(getattr(self, "_event_sampler_trace", {}).get("mode", "OFF")),
                "reason": "native_math_loop_already_generated_step_level_records",
            })
            controlled_latent_after = result.latent_after

        branch_name_lower = str(getattr(window, "branch_name", "") or "").lower()
        if "high" in branch_name_lower and mode == "LATENT_DELTA_SCALE":
            self._update_strategy_coupling_from_high(latent, controlled_latent_after, window.branch_name, records)

        self._finite_guard(controlled_latent_after, records, f"EventFiniteGuard_{window.branch_name}_latent_after", strict=True)
        self._event_universal_stage_math(
            records,
            f"EventSampler{str(window.branch_name).capitalize()}",
            input_state=latent,
            output_state=controlled_latent_after,
            observed_behavior=f"{window.branch_name} sampler transformed latent through step window {window.start_at_step}->{window.end_at_step}",
            formula_role="LATENT OutcomePrevious + sampler update = LATENT OutcomeNext",
            route_id=f"route_sampler_{window.branch_name}",
            next_requirement="next sampler stage or VAE decode requires event-compatible latent",
            control_mode=mode,
            metadata={
                "branch": str(window.branch_name),
                "start_at_step": int(window.start_at_step),
                "end_at_step": int(window.end_at_step),
                "add_noise": str(window.add_noise),
                "return_with_leftover_noise": str(getattr(window, "return_with_leftover_noise", "")),
                "sampler_execution_path": "native_math_loop" if use_native_math_loop else "boundary_replacement",
            },
        )
        
        # First boundary puncture: math returns only the raw impact weight.
        delta_tensor, _ = compute_tensor_delta(latent, controlled_latent_after)
        
        raw_delta_norm = 0.0
        if delta_tensor is not None:
            import torch
            raw_delta_norm = float(torch.linalg.vector_norm(delta_tensor).item())

        print(f"[RAW] {window.branch_name} raw_delta_norm={raw_delta_norm}")

        records.append({
            "stage": f"EventMathStrategyProposal_{window.branch_name}",
            "status": "recorded",
            "raw_delta_norm": raw_delta_norm,
            "formula": "Outcome(t-1) + ObservedBehavior(t-1) -> Raw Norm (Strategy)"
        })

        return controlled_latent_after, raw_delta_norm, delta_tensor

    def _decode_tiled(self, vae, latent, tile_size, overlap, temporal_size, temporal_overlap, records):
        try:
            result = self._call_node_method(
                "VAEDecodeTiled",
                ["decode"],
                samples=latent,
                vae=vae,
                tile_size=int(tile_size),
                overlap=int(overlap),
                temporal_size=int(temporal_size),
                temporal_overlap=int(temporal_overlap),
            )
            image = self._first_output(result, preferred_names=["image", "images", "IMAGE"], stage="EventVAEDecodeTiled", records=records)
            self._finite_guard(image, records, "EventFiniteGuard_decoded_frames", strict=True)
            records.append({"stage": "EventVAEDecodeTiled", "status": "ok", "formula": "final latent + tiled decode behavior = EventSingularity_decode = visible frames"})
            self._event_universal_stage_math(
                records,
                "EventVAEDecodeTiled",
                input_state=latent,
                output_state=image,
                observed_behavior="sampled latent decoded into visible image/frame tensor",
                formula_role="LATENT OutcomeNext -> IMAGE/FRAMES VisibleOutcome",
                route_id="route_decode",
                next_requirement="Output/VHS requires visible frames to package final media",
                control_mode="REPORT_ONLY",
                metadata={"tile_size": int(tile_size), "overlap": int(overlap), "temporal_size": int(temporal_size), "temporal_overlap": int(temporal_overlap)},
            )
            return image
        except Exception as e:
            records.append({"stage": "EventVAEDecodeTiled", "status": "failed", "error": str(e)})
            raise



