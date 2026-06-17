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

EVENT_HORIZON_RUNTIME_VERSION = "0.1.1-r113"
EVENT_HORIZON_RUNTIME_NAME = "Singularity R113 Widget Order Hotfix"
EVENT_HORIZON_BODY_VERSION = "0.1-r113"


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

    def _cascade_seam_motion_review(self, segment_batches, concatenated_frames, records, frames_per_cascade=None):
        """
        r74 observer-only cascade seam review.
        It distinguishes the boundary jump from post-seam acceleration inside the next segment.
        """
        try:
            import torch

            def safe_float(value, default=None):
                try:
                    out = float(value)
                except Exception:
                    return default
                return out if math.isfinite(out) else default

            def clamp01(value):
                value = safe_float(value, 0.0)
                return max(0.0, min(1.0, value))

            def tensor_frames(value):
                t = self._tensor_from_latent_like(value)
                if t is None:
                    return None
                t = torch.nan_to_num(t.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
                if t.dim() < 2 or t.shape[0] < 1:
                    return None
                return t

            def delta_abs_values(frames):
                if frames is None or frames.shape[0] < 2:
                    return []
                delta = frames[1:] - frames[:-1]
                values = delta.abs().reshape(delta.shape[0], -1).mean(dim=1)
                return [float(x) for x in values.detach().cpu().tolist()]

            def mean(values):
                values = [safe_float(v, None) for v in (values or [])]
                values = [v for v in values if v is not None]
                return float(sum(values) / len(values)) if values else None

            def std(values):
                values = [safe_float(v, None) for v in (values or [])]
                values = [v for v in values if v is not None]
                if len(values) < 2:
                    return 0.0 if values else None
                mu = sum(values) / len(values)
                return float((sum((v - mu) ** 2 for v in values) / (len(values) - 1)) ** 0.5)

            def window_mean(values, start=None, end=None):
                values = list(values or [])
                subset = values[slice(start, end)]
                return mean(subset)

            def ratio(numerator, denominator):
                numerator = safe_float(numerator, None)
                denominator = safe_float(denominator, None)
                if numerator is None or denominator is None or denominator <= 0:
                    return None
                return float(numerator / (denominator + 1e-12))

            segments = [tensor_frames(item) for item in (segment_batches or [])]
            segments = [item for item in segments if item is not None and item.shape[0] >= 1]
            if len(segments) < 2:
                rec = {
                    "stage": "EventCascadeSeamMotionReview",
                    "status": "not_applicable",
                    "reason": "requires_at_least_two_segments",
                    "observed_segments": len(segments),
                    "control_mode": "REPORT_ONLY",
                }
                records.append(rec)
                return rec

            concat = tensor_frames(concatenated_frames)
            global_values = delta_abs_values(concat) if concat is not None else []
            global_mean = mean(global_values)
            global_std = std(global_values)

            per_segment = []
            segment_offsets = []
            offset = 0
            for idx, segment in enumerate(segments, start=1):
                values = delta_abs_values(segment)
                seg_mean = mean(values)
                seg_std = std(values)
                segment_offsets.append(offset)
                offset += int(segment.shape[0])
                early_n = min(8, len(values))
                late_n = min(8, len(values))
                per_segment.append({
                    "segment_index": int(idx),
                    "frame_count": int(segment.shape[0]),
                    "transition_count": len(values),
                    "mean_abs_delta": seg_mean,
                    "std_abs_delta": seg_std,
                    "max_abs_delta": max(values) if values else None,
                    "early_mean_abs_delta": window_mean(values, 0, early_n) if early_n else None,
                    "late_mean_abs_delta": window_mean(values, len(values) - late_n, len(values)) if late_n else None,
                    "max_to_segment_mean_ratio": ratio(max(values) if values else None, seg_mean),
                    "segment_to_global_mean_ratio": ratio(seg_mean, global_mean),
                })

            boundary_reviews = []
            pressure_terms = {}
            for idx in range(1, len(segments)):
                prev = segments[idx - 1]
                nxt = segments[idx]
                boundary_delta = nxt[0] - prev[-1]
                boundary_abs = float(boundary_delta.abs().reshape(-1).mean().item())
                prev_summary = per_segment[idx - 1]
                next_summary = per_segment[idx]
                prev_mean = prev_summary.get("mean_abs_delta")
                next_mean = next_summary.get("mean_abs_delta")
                prev_late = prev_summary.get("late_mean_abs_delta")
                next_early = next_summary.get("early_mean_abs_delta")
                boundary_to_prev = ratio(boundary_abs, prev_mean)
                boundary_to_next = ratio(boundary_abs, next_mean)
                next_to_prev = ratio(next_mean, prev_mean)
                next_early_to_prev_late = ratio(next_early, prev_late)
                next_max_to_prev_mean = ratio(next_summary.get("max_abs_delta"), prev_mean)

                seam_pressure = clamp01(((boundary_to_prev if boundary_to_prev is not None else 1.0) - 1.0) / 1.5)
                post_pressure = clamp01(((next_to_prev if next_to_prev is not None else 1.0) - 1.0) / 1.5)
                early_pressure = clamp01(((next_early_to_prev_late if next_early_to_prev_late is not None else 1.0) - 1.0) / 1.5)
                late_spike_pressure = clamp01(((next_max_to_prev_mean if next_max_to_prev_mean is not None else 1.0) - 1.35) / 1.5)
                pressure_key = f"boundary_{idx}_to_{idx + 1}"
                pressure_terms[pressure_key] = {
                    "seam_pressure": seam_pressure,
                    "post_segment_pressure": post_pressure,
                    "early_post_seam_pressure": early_pressure,
                    "late_segment_spike_pressure": late_spike_pressure,
                }
                boundary_reviews.append({
                    "boundary_index": int(idx),
                    "previous_segment": int(idx),
                    "next_segment": int(idx + 1),
                    "previous_segment_frame_count": int(prev.shape[0]),
                    "next_segment_frame_count": int(nxt.shape[0]),
                    "boundary_abs_delta": boundary_abs,
                    "boundary_to_previous_segment_mean_ratio": boundary_to_prev,
                    "boundary_to_next_segment_mean_ratio": boundary_to_next,
                    "next_segment_to_previous_segment_mean_ratio": next_to_prev,
                    "next_early_to_previous_late_ratio": next_early_to_prev_late,
                    "next_max_to_previous_mean_ratio": next_max_to_prev_mean,
                    "seam_pressure": seam_pressure,
                    "post_segment_pressure": post_pressure,
                    "early_post_seam_pressure": early_pressure,
                    "late_segment_spike_pressure": late_spike_pressure,
                })

            top_transitions = []
            if global_values:
                boundaries = []
                running = 0
                for segment in segments[:-1]:
                    running += int(segment.shape[0])
                    boundaries.append(running - 1)
                top_indices = sorted(
                    range(len(global_values)),
                    key=lambda i: global_values[i],
                    reverse=True,
                )[:12]
                for idx in top_indices:
                    boundary_distance = min((abs(idx - b) for b in boundaries), default=None)
                    segment_id = 1
                    for boundary_pos in boundaries:
                        if idx >= boundary_pos:
                            segment_id += 1
                    top_transitions.append({
                        "transition": f"{idx}->{idx + 1}",
                        "transition_index": int(idx),
                        "approx_segment": int(segment_id),
                        "abs_delta": float(global_values[idx]),
                        "to_global_mean_ratio": ratio(global_values[idx], global_mean),
                        "distance_to_nearest_seam_transition": int(boundary_distance) if boundary_distance is not None else None,
                    })

            max_boundary_to_previous = max(
                [v for v in (item.get("boundary_to_previous_segment_mean_ratio") for item in boundary_reviews) if v is not None],
                default=None,
            )
            max_post_ratio = max(
                [v for v in (item.get("next_segment_to_previous_segment_mean_ratio") for item in boundary_reviews) if v is not None],
                default=None,
            )
            max_early_ratio = max(
                [v for v in (item.get("next_early_to_previous_late_ratio") for item in boundary_reviews) if v is not None],
                default=None,
            )
            max_late_spike_ratio = max(
                [v for v in (item.get("next_max_to_previous_mean_ratio") for item in boundary_reviews) if v is not None],
                default=None,
            )
            max_seam_pressure = max(
                [item.get("seam_pressure", 0.0) for item in boundary_reviews],
                default=0.0,
            )
            max_post_pressure = max(
                [item.get("post_segment_pressure", 0.0) for item in boundary_reviews],
                default=0.0,
            )
            max_early_pressure = max(
                [item.get("early_post_seam_pressure", 0.0) for item in boundary_reviews],
                default=0.0,
            )
            max_late_spike_pressure = max(
                [item.get("late_segment_spike_pressure", 0.0) for item in boundary_reviews],
                default=0.0,
            )
            post_seam_acceleration_score = clamp01(
                0.25 * max_seam_pressure
                + 0.35 * max_post_pressure
                + 0.20 * max_early_pressure
                + 0.20 * max_late_spike_pressure
            )
            attribution_map = {
                "cascade_boundary_jump": max_seam_pressure,
                "next_segment_motion_pressure": max_post_pressure,
                "early_post_seam_acceleration": max_early_pressure,
                "late_segment_spike": max_late_spike_pressure,
            }
            attribution = max(attribution_map, key=attribution_map.get)
            if attribution_map.get(attribution, 0.0) < 0.12:
                attribution = "no_single_dominant_pressure"

            rec = {
                "stage": "EventCascadeSeamMotionReview",
                "status": "reviewed",
                "review_version": "cascade_seam_motion_review_v1_report_only",
                "formula": "VisibleOutcome(previous segment) + post-seam ObservedBehavior = Strategy(next segment) = continued visible Outcome.",
                "control_mode": "REPORT_ONLY",
                "active_control_allowed": False,
                "observed_segments": len(segments),
                "frames_per_cascade_requested": int(frames_per_cascade) if frames_per_cascade is not None else None,
                "segment_frame_counts": [int(item.shape[0]) for item in segments],
                "global_transition_count": len(global_values),
                "global_mean_abs_delta": global_mean,
                "global_std_abs_delta": global_std,
                "per_segment": per_segment,
                "boundary_reviews": boundary_reviews,
                "top_transitions": top_transitions,
                "max_boundary_to_previous_segment_mean_ratio": max_boundary_to_previous,
                "max_post_segment_to_previous_segment_mean_ratio": max_post_ratio,
                "max_early_post_seam_to_previous_late_ratio": max_early_ratio,
                "max_late_spike_to_previous_mean_ratio": max_late_spike_ratio,
                "post_seam_acceleration_score": post_seam_acceleration_score,
                "attribution": attribution,
                "pressure_terms": {
                    "max_seam_pressure": max_seam_pressure,
                    "max_post_segment_pressure": max_post_pressure,
                    "max_early_post_seam_pressure": max_early_pressure,
                    "max_late_segment_spike_pressure": max_late_spike_pressure,
                    "by_boundary": pressure_terms,
                },
                "interpretation": (
                    "post-seam acceleration dominates; review repeated prompt-conditioning pressure before adding damping"
                    if attribution in ("next_segment_motion_pressure", "early_post_seam_acceleration", "late_segment_spike")
                    else "seam boundary jump dominates; review MirrorCut frame choice and boundary continuity"
                    if attribution == "cascade_boundary_jump"
                    else "no dominant cascade seam pressure detected"
                ),
                "next_action": "Compare fixed-seed 2-cascade tests; if post_seam_acceleration_score remains high, test prompt-per-segment route before active motion damping.",
            }
            records.append(rec)
            return rec
        except Exception as e:
            rec = {
                "stage": "EventCascadeSeamMotionReview",
                "status": "failed",
                "error": str(e),
                "control_mode": "REPORT_ONLY",
                "active_control_allowed": False,
            }
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
            mode = str(mode or "OBSERVE_ONLY").upper()
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

    def _event_strategy_control_surface_plan(self, records=None):
        try:
            mode = str(getattr(self, "_event_math_control_mode", "OBSERVE_ONLY") or "OBSERVE_ONLY").upper()
            field_mode = str(getattr(self, "_event_strategy_field_mode", "OFF") or "OFF").upper()
            strengths = getattr(self, "_event_delta_strengths", {}) or {}
            bridge_controls = getattr(self, "_event_latent_memory_bridge_controls", {}) or {}
            high = float(strengths.get("high", 1.0) or 1.0)
            low = float(strengths.get("low", 1.0) or 1.0)
        except Exception:
            mode = "OBSERVE_ONLY"
            field_mode = "OFF"
            bridge_controls = {}
            high = 1.0
            low = 1.0

        active_mode = mode in ("LATENT_DELTA_SCALE", "STRATEGY_PRESSURE_WINDOW", "LATENT_MEMORY_BRIDGE", "DEEP_STEP_DELTA_CONTROL")
        strategy_field_delta_active_mode = mode in ("LATENT_DELTA_SCALE", "STRATEGY_PRESSURE_WINDOW", "DEEP_STEP_DELTA_CONTROL")
        strategy_field_reportable = field_mode in ("REPORT_ONLY", "HIGH_NOISE_FIELD", "LOW_REFINEMENT_FIELD", "DUAL_FIELD")
        strategy_field_active = strategy_field_delta_active_mode and field_mode in ("HIGH_NOISE_FIELD", "LOW_REFINEMENT_FIELD", "DUAL_FIELD")
        if mode == "STRATEGY_PRESSURE_WINDOW":
            path = "unified_strategy_pressure_window"
            policy = "bounded_pressure_intent"
            active_control_allowed = True
            model_native_sampler_preserved = True
            cfg_preserved = True
        elif mode == "LATENT_MEMORY_BRIDGE":
            path = "segment_entry_latent_memory_bridge"
            policy = "bounded_first_slice_memory_return"
            active_control_allowed = True
            model_native_sampler_preserved = True
            cfg_preserved = True
        elif mode == "LATENT_DELTA_SCALE":
            path = "legacy_latent_delta_scale"
            policy = "raw_branch_delta_scale"
            active_control_allowed = True
            model_native_sampler_preserved = True
            cfg_preserved = True
        elif mode == "DEEP_STEP_DELTA_CONTROL":
            path = "deep_step_delta_control"
            policy = "research_native_step_loop"
            active_control_allowed = True
            model_native_sampler_preserved = False
            cfg_preserved = False
        else:
            path = "observe_only"
            policy = "record_without_tensor_mutation"
            active_control_allowed = False
            model_native_sampler_preserved = True
            cfg_preserved = True

        branch_policies = {
            "high": {
                "formula_role": "ObservedBehavior(high) -> StrategyCarrier(low)",
                "pressure_window_max": 0.020,
                "pressure_compression": 20.0,
                "coupling_allowed": False,
                "requested_strength": float(high),
                "strategy_field_role": "HighStrategyNoiseField",
                "strategy_field_window_max": 0.080,
                "strategy_field_compression": 12.0,
                "strategy_field_policy": "event_birth_direction_with_source_anchor_preservation",
                "pressure_field_return_weight": 0.50,
                "mirror_residual_weight": 0.12,
                "trajectory_weight": 0.02,
                "recursive_relation_depth": 7,
                "recursive_relation_decay": 0.45,
                "recursive_relation_feedback_weight": 0.04,
                "source_anchor_return_weight": 0.35,
                "source_anchor_guard_role": "preserve source identity while birthing motion direction",
                "source_anchor_guard_max_intent_compression": 0.18,
                "source_anchor_guard_max_window_compression": 0.12,
            },
            "low": {
                "formula_role": "ObservedBehavior(low) -> decode-ready Outcome",
                "pressure_window_max": 0.012,
                "pressure_compression": 20.0,
                "coupling_allowed": True,
                "requested_strength": float(low),
                "strategy_field_role": "LowStrategyRefinementField",
                "strategy_field_window_max": 0.006,
                "strategy_field_compression": 90.0,
                "strategy_field_policy": "detail_refinement_with_background_blur_guard",
                "pressure_field_return_weight": 0.20,
                "mirror_residual_weight": 0.10,
                "trajectory_weight": 0.02,
                "recursive_relation_depth": 7,
                "recursive_relation_decay": 0.45,
                "recursive_relation_feedback_weight": 0.03,
                "source_anchor_return_weight": 0.25,
                "source_anchor_guard_role": "refine detail without letting background/global scene become motion carrier",
                "source_anchor_guard_max_intent_compression": 0.34,
                "source_anchor_guard_max_window_compression": 0.24,
            },
            "default": {
                "formula_role": "latent transition",
                "pressure_window_max": 0.016,
                "pressure_compression": 20.0,
                "coupling_allowed": False,
                "requested_strength": 1.0,
                "strategy_field_role": "GenericStrategyField",
                "strategy_field_window_max": 0.016,
                "strategy_field_compression": 20.0,
                "strategy_field_policy": "report_only",
                "pressure_field_return_weight": 0.30,
                "mirror_residual_weight": 0.10,
                "trajectory_weight": 0.02,
                "recursive_relation_depth": 7,
                "recursive_relation_decay": 0.45,
                "recursive_relation_feedback_weight": 0.03,
                "source_anchor_return_weight": 0.25,
                "source_anchor_guard_role": "generic source-anchor return",
                "source_anchor_guard_max_intent_compression": 0.20,
                "source_anchor_guard_max_window_compression": 0.15,
            },
        }

        plan = {
            "stage": "EventStrategyControlSurfacePlan",
            "status": "active" if active_mode else "observe_only",
            "version": "strategy_control_surface_v7_noise_field_strategy_bridge",
            "mode": mode,
            "strategy_field_mode": field_mode,
            "strategy_field_reportable": bool(strategy_field_reportable),
            "strategy_field_active": bool(strategy_field_active),
            "parent_strategy": "S_global_event_route",
            "active_generation_math_path": path,
            "policy": policy,
            "active_control_allowed": bool(active_control_allowed),
            "model_native_sampler_preserved": bool(model_native_sampler_preserved),
            "cfg_preserved": bool(cfg_preserved),
            "prompt_text_injection_allowed": False,
            "semantic_math_in_prompt_allowed": False,
            "source_anchor_return_window": {
                "status": "available" if mode == "STRATEGY_PRESSURE_WINDOW" else "inactive",
                "version": "source_anchor_return_window_v2_spatial_gate",
                "policy": "bounded_spatial_source_anchor_gate",
                "prompt_text_injection_allowed": False,
                "semantic_math_in_prompt_allowed": False,
                "formula": "Visible/source anchor and selected tail pressure may narrow low/region-proven pressure windows before latent delta is applied. High branch is not damped by tail/restart pressure alone.",
            },
            "spatial_carrier_preservation_map": {
                "status": "available" if mode == "STRATEGY_PRESSURE_WINDOW" else "inactive",
                "version": "spatial_carrier_preservation_map_v3_denoise_phase_map",
                "policy": "cached_scene_spatial_carriers_report_first_post_window_guard",
                "prompt_text_injection_allowed": False,
                "semantic_math_in_prompt_allowed": False,
                "active_requires": "same-scene AUTO_APPLY background_anchor_preservation evidence plus non-endpoint step-level denoise-safe window",
                "formula": "Spatial carriers are read as local Strategy points. In post-window or endpoint denoise phases they remain report-only, because reducing background delta can preserve raw noise instead of source identity.",
            },
            "denoise_phase_map": {
                "status": "available",
                "version": "denoise_phase_map_v1_report_guard",
                "policy": "classify high/low/post-window/endpoint before any local math may touch delta",
                "active_control_allowed": False,
                "future_active_candidate": "low mid-window step-level refinement only",
                "prompt_text_injection_allowed": False,
                "semantic_math_in_prompt_allowed": False,
                "formula": "Strategy must know the denoise phase before local math acts. High birth, post-window, and endpoint phases are report-only; only low mid-window can become an active collision surface later.",
            },
            "noise_source_field_map": {
                "status": "available",
                "version": "noise_source_field_map_v1_report_only",
                "policy": "read latent source/noise pressure without tensor mutation",
                "active_control_allowed": False,
                "future_active_candidate": "pre-high source/noise shaping after enough report evidence",
                "prompt_text_injection_allowed": False,
                "semantic_math_in_prompt_allowed": False,
                "formula": "The source/noise field is read as Outcome(t-1)+ObservedBehavior(t-1) evidence before deciding whether any Strategy pressure should become active math.",
            },
            "noise_field_strategy_bridge": {
                "status": "available",
                "version": "noise_field_strategy_bridge_v1_model_attractor",
                "policy": "route source/noise evidence to the next denoise-safe Strategy surface",
                "active_control_allowed": False,
                "future_active_candidate": "pre-high source/noise shaping or low mid-window refinement only",
                "prompt_text_injection_allowed": False,
                "semantic_math_in_prompt_allowed": False,
                "formula": "NoiseFieldStrategyBridge reads the source/noise field as a bridge from Outcome+ObservedBehavior back to StrategyAttractor(model). Empty fields become no-pressure; non-empty fields name the safe future surface instead of mutating the current post-window delta.",
            },
            "segment_entry_latent_memory_bridge": {
                "status": "available" if mode == "LATENT_MEMORY_BRIDGE" else "inactive",
                "version": "segment_entry_latent_memory_bridge_v2_explicit_controls",
                "policy": "previous latent tail returns to next segment entry before high sampler",
                "active_control_allowed": bool(mode == "LATENT_MEMORY_BRIDGE"),
                "branch_delta_overlay_allowed": False,
                "controls": {
                    "wan_alpha": float((bridge_controls or {}).get("wan_alpha", 0.10)),
                    "concat_alpha": float((bridge_controls or {}).get("concat_alpha", 0.06)),
                    "wan_max_step": float((bridge_controls or {}).get("wan_max_step", 0.45)),
                    "concat_max_step": float((bridge_controls or {}).get("concat_max_step", 0.28)),
                },
                "prompt_text_injection_allowed": False,
                "semantic_math_in_prompt_allowed": False,
                "formula": "previous latent OutcomePrevious + explicit bounded memory delta = next Wan latent StrategyCarrier before high sampler",
            },
            "branch_policies": branch_policies,
            "requested_strengths": {
                "high": float(high),
                "low": float(low),
            },
            "formula": "One Strategy control surface chooses how local high/low pressure may become latent transition control after returning to S_global_event_route. StrategyField is a bounded semantic field at high/low vector collision points; SourceAnchorReturnWindow narrows pressure when the continuation route risks turning the whole scene into motion. No prompt text injection.",
        }
        self._event_strategy_control_surface_plan_state = plan

        signature = json.dumps(_event_json_safe({
            "mode": mode,
            "field_mode": field_mode,
            "high": high,
            "low": low,
            "bridge": bridge_controls,
            "path": path,
        }), sort_keys=True, ensure_ascii=True)
        if records is not None and getattr(self, "_event_strategy_control_surface_plan_signature", "") != signature:
            records.append(plan)
            self._event_strategy_control_surface_plan_signature = signature
        return plan

    def _event_strategy_control_surface_apply(
        self,
        branch_name,
        scheduled_strength,
        base_strength,
        coupling_multiplier,
        step_schedule_factor,
        records,
        *,
        step_index=None,
        window_steps=None,
    ):
        plan = self._event_strategy_control_surface_plan(records)
        mode = str(plan.get("mode", "OBSERVE_ONLY") or "OBSERVE_ONLY").upper()
        branch_lower = str(branch_name or "").lower()
        if "high" in branch_lower:
            branch_key = "high"
        elif "low" in branch_lower:
            branch_key = "low"
        else:
            branch_key = "default"
        branch_policy = (plan.get("branch_policies", {}) or {}).get(branch_key, {}) or {}

        try:
            requested_strength = float(scheduled_strength)
        except Exception:
            requested_strength = 1.0

        pressure_intent = float(requested_strength - 1.0)
        max_window = float(branch_policy.get("pressure_window_max", 0.016) or 0.016)
        compression = float(branch_policy.get("pressure_compression", 20.0) or 20.0)
        background_preservation = getattr(self, "_event_background_anchor_preservation_control", {}) or {}
        background_intent_multiplier = 1.0
        background_window_multiplier = 1.0
        background_temporal_multiplier = 1.0
        background_preservation_status = str(background_preservation.get("status", "inactive") or "inactive")
        if (
            mode == "STRATEGY_PRESSURE_WINDOW"
            and branch_key == "low"
            and pressure_intent > 0.0
            and background_preservation_status == "active"
        ):
            try:
                background_intent_multiplier = float(background_preservation.get("low_positive_intent_multiplier", 1.0) or 1.0)
            except Exception:
                background_intent_multiplier = 1.0
            background_intent_multiplier = max(0.25, min(1.0, background_intent_multiplier))
            try:
                background_window_multiplier = float(background_preservation.get("max_delta_window_multiplier", 1.0) or 1.0)
            except Exception:
                background_window_multiplier = 1.0
            background_window_multiplier = max(0.25, min(1.0, background_window_multiplier))
            try:
                background_temporal_multiplier = float(background_preservation.get("temporal_stability_multiplier", 1.0) or 1.0)
            except Exception:
                background_temporal_multiplier = 1.0
            background_temporal_multiplier = max(0.25, min(1.0, background_temporal_multiplier))
        pressure_intent_after_background_anchor = pressure_intent * background_intent_multiplier
        max_window_after_background_anchor = max_window * background_window_multiplier
        compressed_intent = math.tanh(pressure_intent_after_background_anchor * compression) if abs(pressure_intent_after_background_anchor) > 1e-12 else 0.0

        field_mode = str(plan.get("strategy_field_mode", "OFF") or "OFF").upper()
        field_branch_allowed = (
            field_mode == "DUAL_FIELD"
            or (field_mode == "HIGH_NOISE_FIELD" and branch_key == "high")
            or (field_mode == "LOW_REFINEMENT_FIELD" and branch_key == "low")
        )
        field_reportable = field_mode in ("REPORT_ONLY", "HIGH_NOISE_FIELD", "LOW_REFINEMENT_FIELD", "DUAL_FIELD")
        field_active = bool(field_branch_allowed and mode in ("LATENT_DELTA_SCALE", "STRATEGY_PRESSURE_WINDOW", "DEEP_STEP_DELTA_CONTROL"))
        field_window = float(branch_policy.get("strategy_field_window_max", max_window) or max_window)
        field_compression = float(branch_policy.get("strategy_field_compression", compression) or compression)
        field_intent = pressure_intent_after_background_anchor
        if branch_key == "low" and field_intent > 0.0:
            field_intent = field_intent * background_temporal_multiplier
        remaining_strategy = getattr(self, "_event_cascade_remaining_strategy", {}) or {}
        remaining_strategy_applied = False
        remaining_strategy_multipliers = {}
        remaining_strategy_segment = None
        try:
            match = re.search(r"cascade[_\s-]*(\d+)", branch_lower)
            remaining_strategy_segment = int(match.group(1)) if match else None
        except Exception:
            remaining_strategy_segment = None
        try:
            remaining_target_segment = int(remaining_strategy.get("applies_to_segment", 0) or 0)
        except Exception:
            remaining_target_segment = 0
        if (
            isinstance(remaining_strategy, dict)
            and remaining_target_segment > 0
            and remaining_strategy_segment == remaining_target_segment
            and field_active
        ):
            branch_multipliers = remaining_strategy.get("branch_multipliers", {}) or {}
            try:
                if branch_key == "high":
                    intent_multiplier = float(branch_multipliers.get("high_field_intent_multiplier", 1.0) or 1.0)
                    window_multiplier = float(branch_multipliers.get("high_field_window_multiplier", 1.0) or 1.0)
                elif branch_key == "low":
                    intent_multiplier = float(branch_multipliers.get("low_field_intent_multiplier", 1.0) or 1.0)
                    window_multiplier = float(branch_multipliers.get("low_field_window_multiplier", 1.0) or 1.0)
                else:
                    intent_multiplier = 1.0
                    window_multiplier = 1.0
            except Exception:
                intent_multiplier = 1.0
                window_multiplier = 1.0
            intent_multiplier = max(0.10, min(1.0, intent_multiplier))
            window_multiplier = max(0.10, min(1.0, window_multiplier))
            field_intent = field_intent * intent_multiplier
            field_window = field_window * window_multiplier
            remaining_strategy_applied = True
            remaining_strategy_multipliers = {
                "field_intent_multiplier": float(intent_multiplier),
                "field_window_multiplier": float(window_multiplier),
            }

        def clamp01(value):
            try:
                out = float(value)
            except Exception:
                out = 0.0
            if not math.isfinite(out):
                out = 0.0
            return max(0.0, min(1.0, out))

        source_anchor_return_window = {
            "status": "inactive",
            "version": "source_anchor_return_window_v2_spatial_gate",
            "source": "current_strategy_control_surface",
            "branch_key": branch_key,
            "prompt_text_injection_allowed": False,
            "semantic_math_in_prompt_allowed": False,
            "formula_role": str(branch_policy.get("source_anchor_guard_role", "source-anchor return") or "source-anchor return"),
            "reason": "mode_or_branch_not_active",
            "input_pressures": {},
            "multipliers": {
                "pressure_intent_multiplier": 1.0,
                "max_delta_window_multiplier": 1.0,
                "field_intent_multiplier": 1.0,
                "field_window_multiplier": 1.0,
                "temporal_stability_multiplier": 1.0,
            },
            "formula": (
                "SourceAnchorReturnWindow reads spatial SourceAnchor pressure, tail route pressure, "
                "selected cut pressure, and positive branch pressure as local Strategy points. "
                "High branch is not damped by tail/restart pressure alone; low branch can be guarded "
                "as the decode-ready refinement carrier."
            ),
        }
        if mode == "STRATEGY_PRESSURE_WINDOW" and field_active:
            try:
                restart_risk_pressure = clamp01(remaining_strategy.get("restart_risk", 0.0) if isinstance(remaining_strategy, dict) else 0.0)
            except Exception:
                restart_risk_pressure = 0.0
            try:
                motion_memory_pressure = clamp01(remaining_strategy.get("motion_memory_pressure", 0.0) if isinstance(remaining_strategy, dict) else 0.0)
            except Exception:
                motion_memory_pressure = 0.0
            try:
                late_cut_pressure = clamp01(remaining_strategy.get("late_cut_pressure", 0.0) if isinstance(remaining_strategy, dict) else 0.0)
            except Exception:
                late_cut_pressure = 0.0
            try:
                route_pressure = clamp01(remaining_strategy.get("route_pressure", 0.0) if isinstance(remaining_strategy, dict) else 0.0)
            except Exception:
                route_pressure = 0.0
            try:
                background_anchor_pressure = clamp01(background_preservation.get("background_anchor_pressure", 0.0) if isinstance(background_preservation, dict) else 0.0)
            except Exception:
                background_anchor_pressure = 0.0
            try:
                cached_late_pressure = clamp01(background_preservation.get("late_segment_pressure", 0.0) if isinstance(background_preservation, dict) else 0.0)
            except Exception:
                cached_late_pressure = 0.0
            try:
                top_band_pressure = clamp01(background_preservation.get("top_band_pressure", 0.0) if isinstance(background_preservation, dict) else 0.0)
            except Exception:
                top_band_pressure = 0.0
            try:
                cached_spatial_anchor_pressure = clamp01(background_preservation.get("spatial_anchor_pressure", 0.0) if isinstance(background_preservation, dict) else 0.0)
            except Exception:
                cached_spatial_anchor_pressure = 0.0
            try:
                cached_background_region_pressure = clamp01(background_preservation.get("background_region_pressure", 0.0) if isinstance(background_preservation, dict) else 0.0)
            except Exception:
                cached_background_region_pressure = 0.0
            dominant_background_region = ""
            if isinstance(background_preservation, dict):
                dominant_background_region = str(background_preservation.get("dominant_background_region", "") or "")
            try:
                branch_pressure = clamp01(abs(float(pressure_intent)) / max(float(max_window), 1e-9))
            except Exception:
                branch_pressure = 0.0

            tail_continuation_pressure = max(restart_risk_pressure, motion_memory_pressure, late_cut_pressure * route_pressure)
            background_source_pressure = max(
                background_anchor_pressure,
                top_band_pressure,
                cached_spatial_anchor_pressure,
                cached_background_region_pressure,
            )
            branch_tail_pressure = tail_continuation_pressure if branch_key == "low" else 0.0
            source_anchor_pressure = max(background_source_pressure, branch_tail_pressure)
            late_spike_pressure = max(
                cached_late_pressure,
                (motion_memory_pressure if branch_key == "low" else 0.0),
                (restart_risk_pressure * late_cut_pressure if branch_key == "low" else 0.0),
            )
            should_guard = (
                background_source_pressure > 0.0
                or late_spike_pressure > 0.0
                or (branch_key == "low" and pressure_intent > 0.0 and branch_pressure > 0.0)
                or (branch_key == "low" and branch_tail_pressure > 0.0)
            )
            if should_guard:
                try:
                    max_intent_compression = float(branch_policy.get("source_anchor_guard_max_intent_compression", 0.20) or 0.20)
                except Exception:
                    max_intent_compression = 0.20
                try:
                    max_window_compression = float(branch_policy.get("source_anchor_guard_max_window_compression", 0.15) or 0.15)
                except Exception:
                    max_window_compression = 0.15
                max_intent_compression = max(0.0, min(0.60, max_intent_compression))
                max_window_compression = max(0.0, min(0.45, max_window_compression))

                if branch_key == "low" and pressure_intent > 0.0:
                    intent_loss = min(max_intent_compression, 0.16 * source_anchor_pressure + 0.12 * late_spike_pressure + 0.06 * branch_pressure)
                    window_loss = min(max_window_compression, 0.12 * source_anchor_pressure + 0.10 * late_spike_pressure + 0.04 * branch_pressure)
                    temporal_loss = min(0.22, 0.10 * late_spike_pressure + 0.08 * top_band_pressure + 0.04 * source_anchor_pressure)
                elif branch_key == "high":
                    intent_loss = min(max_intent_compression, 0.10 * background_source_pressure + 0.06 * cached_late_pressure)
                    window_loss = min(max_window_compression, 0.06 * background_source_pressure + 0.04 * cached_late_pressure)
                    temporal_loss = min(0.16, 0.08 * cached_late_pressure + 0.04 * background_source_pressure)
                else:
                    intent_loss = min(max_intent_compression, 0.10 * source_anchor_pressure)
                    window_loss = min(max_window_compression, 0.08 * source_anchor_pressure)
                    temporal_loss = min(0.14, 0.06 * late_spike_pressure)

                source_pressure_intent_multiplier = max(0.40, 1.0 - intent_loss)
                source_window_multiplier = max(0.50, 1.0 - window_loss)
                source_temporal_multiplier = max(0.60, 1.0 - temporal_loss)

                pressure_intent_after_background_anchor *= source_pressure_intent_multiplier
                max_window_after_background_anchor *= source_window_multiplier
                field_intent *= source_pressure_intent_multiplier
                field_window *= source_window_multiplier
                background_temporal_multiplier *= source_temporal_multiplier
                compressed_intent = math.tanh(pressure_intent_after_background_anchor * compression) if abs(pressure_intent_after_background_anchor) > 1e-12 else 0.0

                source_anchor_return_window.update({
                    "status": "active",
                    "reason": "spatial_source_anchor_or_low_tail_pressure",
                    "input_pressures": {
                        "restart_risk_pressure": float(restart_risk_pressure),
                        "motion_memory_pressure": float(motion_memory_pressure),
                        "late_cut_pressure": float(late_cut_pressure),
                        "route_pressure": float(route_pressure),
                        "background_anchor_pressure": float(background_anchor_pressure),
                        "cached_late_segment_pressure": float(cached_late_pressure),
                        "top_band_pressure": float(top_band_pressure),
                        "cached_spatial_anchor_pressure": float(cached_spatial_anchor_pressure),
                        "cached_background_region_pressure": float(cached_background_region_pressure),
                        "background_source_pressure": float(background_source_pressure),
                        "tail_continuation_pressure": float(tail_continuation_pressure),
                        "branch_tail_pressure": float(branch_tail_pressure),
                        "branch_pressure": float(branch_pressure),
                        "source_anchor_pressure": float(source_anchor_pressure),
                        "late_spike_pressure": float(late_spike_pressure),
                    },
                    "dominant_background_region": dominant_background_region,
                    "branch_guard_scope": "background_spatial_or_low_refinement",
                    "multipliers": {
                        "pressure_intent_multiplier": float(source_pressure_intent_multiplier),
                        "max_delta_window_multiplier": float(source_window_multiplier),
                        "field_intent_multiplier": float(source_pressure_intent_multiplier),
                        "field_window_multiplier": float(source_window_multiplier),
                        "temporal_stability_multiplier": float(source_temporal_multiplier),
                    },
                    "losses": {
                        "intent_loss": float(intent_loss),
                        "window_loss": float(window_loss),
                        "temporal_loss": float(temporal_loss),
                    },
                    "policy": "current_run_spatial_source_anchor_and_low_refinement_return",
                })
        field_compressed_intent = math.tanh(field_intent * field_compression) if abs(field_intent) > 1e-12 else 0.0
        field_effective_strength = 1.0 + (field_compressed_intent * field_window)
        field_effective_strength = max(1.0 - field_window, min(1.0 + field_window, field_effective_strength))

        if mode == "STRATEGY_PRESSURE_WINDOW":
            effective_strength = 1.0 + (compressed_intent * max_window_after_background_anchor * background_temporal_multiplier)
            effective_strength = max(
                1.0 - max_window_after_background_anchor,
                min(1.0 + max_window_after_background_anchor, effective_strength),
            )
            apply_policy = (
                "bounded_pressure_window_background_late_top_return"
                if background_intent_multiplier < 0.999999 or background_window_multiplier < 0.999999 or background_temporal_multiplier < 0.999999
                else "bounded_pressure_window"
            )
        elif mode in ("LATENT_DELTA_SCALE", "DEEP_STEP_DELTA_CONTROL"):
            effective_strength = max(0.0, min(2.0, float(requested_strength)))
            apply_policy = "raw_delta_scale" if mode == "LATENT_DELTA_SCALE" else "deep_step_delta_scale"
        else:
            effective_strength = 1.0
            apply_policy = "observe_only_no_mutation"

        raw_effective_strength = float(effective_strength)
        strategy_pressure_unfold = {
            "status": "not_used",
            "reason": "mode is not STRATEGY_PRESSURE_WINDOW or StrategyField is inactive",
        }
        if field_active:
            if mode == "STRATEGY_PRESSURE_WINDOW":
                global_strategy_delta = float(raw_effective_strength - 1.0)
                local_field_delta = float(field_effective_strength - 1.0)
                mirror_residual_delta = float(global_strategy_delta - local_field_delta)
                previous_pressure_state = getattr(self, "_event_strategy_pressure_window_branch_state", {}) or {}
                previous_branch_state = previous_pressure_state.get(branch_key, {}) if isinstance(previous_pressure_state, dict) else {}
                try:
                    previous_local_delta = float(previous_branch_state.get("local_field_delta", local_field_delta))
                except Exception:
                    previous_local_delta = local_field_delta
                trajectory_delta = float(local_field_delta - previous_local_delta)

                try:
                    field_return_weight = float(branch_policy.get("pressure_field_return_weight", 0.40) or 0.40)
                except Exception:
                    field_return_weight = 0.40
                try:
                    mirror_residual_weight = float(branch_policy.get("mirror_residual_weight", 0.20) or 0.20)
                except Exception:
                    mirror_residual_weight = 0.20
                try:
                    trajectory_weight = float(branch_policy.get("trajectory_weight", 0.08) or 0.08)
                except Exception:
                    trajectory_weight = 0.08
                try:
                    relation_depth = int(branch_policy.get("recursive_relation_depth", 9) or 9)
                except Exception:
                    relation_depth = 9
                try:
                    relation_decay = float(branch_policy.get("recursive_relation_decay", 0.55) or 0.55)
                except Exception:
                    relation_decay = 0.55
                try:
                    relation_feedback_weight = float(branch_policy.get("recursive_relation_feedback_weight", 0.10) or 0.10)
                except Exception:
                    relation_feedback_weight = 0.10
                try:
                    source_anchor_return_weight = float(branch_policy.get("source_anchor_return_weight", 0.25) or 0.25)
                except Exception:
                    source_anchor_return_weight = 0.25
                field_return_weight = max(0.0, min(1.0, field_return_weight))
                mirror_residual_weight = max(0.0, min(1.0, mirror_residual_weight))
                trajectory_weight = max(0.0, min(1.0, trajectory_weight))
                relation_depth = max(1, min(99, relation_depth))
                relation_decay = max(0.05, min(0.95, relation_decay))
                relation_feedback_weight = max(0.0, min(0.25, relation_feedback_weight))
                source_anchor_return_weight = max(0.0, min(1.0, source_anchor_return_weight))

                relation_seed_delta = (
                    global_strategy_delta
                    + (local_field_delta * field_return_weight)
                    + (mirror_residual_delta * mirror_residual_weight)
                    + (trajectory_delta * trajectory_weight)
                )
                model_attractor_delta = (
                    relation_seed_delta * (1.0 - source_anchor_return_weight)
                    + global_strategy_delta * source_anchor_return_weight
                )
                relation_accumulator = 0.0
                relation_current = float(model_attractor_delta)
                for relation_index in range(1, relation_depth + 1):
                    relation_current = math.tanh(relation_current * (1.0 + (0.01 * relation_index)))
                    relation_accumulator += relation_current * (relation_decay ** relation_index)
                recursive_relation_delta = float(model_attractor_delta + (relation_feedback_weight * relation_accumulator))
                combined_limit = max_window_after_background_anchor + (field_window * field_return_weight)
                combined_limit = max(
                    1e-9,
                    min(float(field_window), float(combined_limit)),
                )
                combined_delta = max(-combined_limit, min(combined_limit, recursive_relation_delta))
                effective_strength = float(1.0 + combined_delta)
                if branch_key == "high":
                    apply_policy = "strategy_pressure_window_model_attractor_high_return"
                elif branch_key == "low":
                    apply_policy = "strategy_pressure_window_model_attractor_low_return"
                else:
                    apply_policy = "strategy_pressure_window_model_attractor_return"
                strategy_pressure_unfold = {
                    "status": "active",
                    "version": "model_attractor_v1",
                    "compressed_source_formula": "H_66_99(M_t, B_t, O_t)",
                    "expand_compress_strategy_delta": float(global_strategy_delta),
                    "observed_behavior_field_delta": float(local_field_delta),
                    "mirror_residual_delta": float(mirror_residual_delta),
                    "trajectory_delta": float(trajectory_delta),
                    "field_return_weight": float(field_return_weight),
                    "mirror_residual_weight": float(mirror_residual_weight),
                    "trajectory_weight": float(trajectory_weight),
                    "recursive_relation_depth": int(relation_depth),
                    "recursive_relation_equivalent_depth": 99,
                    "recursive_relation_decay": float(relation_decay),
                    "recursive_relation_feedback_weight": float(relation_feedback_weight),
                    "relation_seed_delta": float(relation_seed_delta),
                    "model_attractor_delta": float(model_attractor_delta),
                    "model_attractor_return_weight": float(source_anchor_return_weight),
                    "relation_accumulator": float(relation_accumulator),
                    "recursive_relation_delta": float(recursive_relation_delta),
                    "combined_limit": float(combined_limit),
                    "combined_delta": float(combined_delta),
                    "combined_effective_strength": float(effective_strength),
                    "strategy_return": "local StrategyField is unfolded through mirror residual and trajectory, then returned to the model as StrategyAttractor and closed back to S_global_event_route",
                    "prompt_text_injection": False,
                }
            else:
                effective_strength = float(field_effective_strength)
                if branch_key == "high":
                    apply_policy = "high_strategy_noise_field_bounded_delta"
                elif branch_key == "low":
                    apply_policy = "low_strategy_refinement_field_blur_guard"
                else:
                    apply_policy = "strategy_field_bounded_delta"
        elif field_reportable and mode == "OBSERVE_ONLY":
            apply_policy = f"{apply_policy}_strategy_field_report_only"
        if str(source_anchor_return_window.get("status", "") or "") == "active":
            apply_policy = f"{apply_policy}_source_anchor_return"

        coupling = getattr(self, "_event_strategy_coupling", {}) or {}
        try:
            high_relative_delta = float(coupling.get("relative_delta", 0.0) or 0.0)
        except Exception:
            high_relative_delta = 0.0

        active = mode in ("LATENT_DELTA_SCALE", "STRATEGY_PRESSURE_WINDOW", "LATENT_MEMORY_BRIDGE", "DEEP_STEP_DELTA_CONTROL") and abs(effective_strength - 1.0) >= 1e-9
        rec = {
            "stage": f"EventStrategyControlSurfaceApply_{branch_name}",
            "status": "active" if active else "neutral",
            "mode": mode,
            "parent_strategy": "S_global_event_route",
            "branch_name": str(branch_name or ""),
            "branch_key": branch_key,
            "formula_role": branch_policy.get("formula_role", "latent transition"),
            "apply_policy": apply_policy,
            "base_strength": float(base_strength),
            "coupling_multiplier": float(coupling_multiplier),
            "step_schedule_factor": float(step_schedule_factor),
            "scheduled_strength": float(requested_strength),
            "pressure_intent": float(pressure_intent),
            "pressure_intent_after_background_anchor": float(pressure_intent_after_background_anchor),
            "compressed_pressure_intent": float(compressed_intent),
            "max_delta_window": float(max_window),
            "max_delta_window_after_background_anchor": float(max_window_after_background_anchor),
            "raw_effective_strength_before_strategy_field": float(raw_effective_strength),
            "effective_strength": float(effective_strength),
            "strategy_field": {
                "mode": field_mode,
                "status": "active" if field_active else ("report_only" if field_reportable else "off"),
                "branch_allowed": bool(field_branch_allowed),
                "role": str(branch_policy.get("strategy_field_role", "GenericStrategyField") or "GenericStrategyField"),
                "policy": str(branch_policy.get("strategy_field_policy", "report_only") or "report_only"),
                "field_intent": float(field_intent),
                "field_compressed_intent": float(field_compressed_intent),
                "field_window": float(field_window),
                "field_effective_strength": float(field_effective_strength),
                "strategy_return": "local field must return to S_global_event_route before the next node/stage",
                "prompt_text_injection": False,
            },
            "cascade_remaining_strategy": {
                "status": str(remaining_strategy.get("status", "off") if isinstance(remaining_strategy, dict) else "off"),
                "applied": bool(remaining_strategy_applied),
                "target_segment": int(remaining_target_segment),
                "current_segment": int(remaining_strategy_segment) if remaining_strategy_segment is not None else None,
                "resume_frame_index": int(remaining_strategy.get("resume_frame_index", 0) or 0) if isinstance(remaining_strategy, dict) else 0,
                "progress_ratio": float(remaining_strategy.get("progress_ratio", 0.0) or 0.0) if isinstance(remaining_strategy, dict) else 0.0,
                "restart_risk": float(remaining_strategy.get("restart_risk", 0.0) or 0.0) if isinstance(remaining_strategy, dict) else 0.0,
                "motion_memory_pressure": float(remaining_strategy.get("motion_memory_pressure", 0.0) or 0.0) if isinstance(remaining_strategy, dict) else 0.0,
                "tail_observed_behavior": remaining_strategy.get("tail_observed_behavior", {}) if isinstance(remaining_strategy, dict) else {},
                "multipliers": remaining_strategy_multipliers,
                "prompt_text_injection": False,
                "formula": "RemainingStrategy binds selected tail Outcome(t-1) plus tail ObservedBehavior(t-1) before next-segment field pressure returns to S_global_event_route.",
            },
            "high_relative_delta_evidence": float(high_relative_delta),
            "background_anchor_preservation": {
                "status": background_preservation_status,
                "applied": bool(
                    background_intent_multiplier < 0.999999
                    or background_window_multiplier < 0.999999
                    or background_temporal_multiplier < 0.999999
                ),
                "low_positive_intent_multiplier": float(background_intent_multiplier),
                "max_delta_window_multiplier": float(background_window_multiplier),
                "temporal_stability_multiplier": float(background_temporal_multiplier),
                "background_anchor_status": str(background_preservation.get("background_anchor_status", "") or ""),
                "background_anchor_pressure": float(background_preservation.get("background_anchor_pressure", 0.0) or 0.0),
                "late_segment_pressure": float(background_preservation.get("late_segment_pressure", 0.0) or 0.0),
                "top_band_pressure": float(background_preservation.get("top_band_pressure", 0.0) or 0.0),
                "scene_key": str(background_preservation.get("scene_key", "") or ""),
                "policy": str(background_preservation.get("policy", "") or ""),
                "source": str(background_preservation.get("source", "") or ""),
            },
            "source_anchor_return_window": source_anchor_return_window,
            "strategy_pressure_unfold": strategy_pressure_unfold,
            "step_index": int(step_index) if step_index is not None else None,
            "window_steps": int(window_steps) if window_steps is not None else None,
            "formula": "StrategyControlSurface(mode, branch, pressure) returns one effective_strength for latent transition control. StrategyField reads high/low as collision points: high may birth motion direction, low may refine only inside a narrow blur-safe window, both returning to S_global_event_route without prompt text injection.",
        }
        records.append(rec)

        if mode == "STRATEGY_PRESSURE_WINDOW":
            try:
                branch_state = getattr(self, "_event_strategy_pressure_window_branch_state", {}) or {}
                if not isinstance(branch_state, dict):
                    branch_state = {}
                branch_state[branch_key] = {
                    "branch_name": str(branch_name or ""),
                    "global_strategy_delta": float(raw_effective_strength - 1.0),
                    "local_field_delta": float(field_effective_strength - 1.0),
                    "effective_strength": float(effective_strength),
                    "combined_delta": float(effective_strength - 1.0),
                    "apply_policy": str(apply_policy),
                }
                self._event_strategy_pressure_window_branch_state = branch_state
            except Exception:
                pass
            alias_rec = dict(rec)
            alias_rec["stage"] = f"EventStrategyPressureWindow_{branch_name}"
            alias_rec["status"] = "active_window" if active else "neutral_window"
            alias_rec["policy"] = str(apply_policy)
            alias_rec["formula"] = "Local sampler pressure is expanded from compressed Strategy through mirror residual, trajectory, and recursive relation passes, then returned to S_global_event_route before latent delta is applied."
            records.append(alias_rec)
            self._event_strategy_pressure_window_last = alias_rec

        self._event_strategy_control_surface_last = rec
        return rec

    def _event_denoise_phase_map(
        self,
        branch_name,
        records=None,
        *,
        step_index=None,
        window_steps=None,
        surface_rec=None,
    ):
        branch_lower = str(branch_name or "").lower()
        if "high" in branch_lower:
            branch_key = "high"
        elif "low" in branch_lower:
            branch_key = "low"
        else:
            branch_key = "default"

        step_progress = None
        try:
            if step_index is not None and window_steps is not None and int(window_steps) > 0:
                step_progress = float((int(step_index) + 1) / max(1, int(window_steps)))
        except Exception:
            step_progress = None
        post_window_delta = step_progress is None
        endpoint_band = bool(step_progress is not None and (step_progress <= 0.12 or step_progress >= 0.88))
        if post_window_delta:
            phase_scope = "post_window_delta"
            progress_bucket = "post_window"
        elif step_progress <= 0.12:
            phase_scope = "early_endpoint_step"
            progress_bucket = "early_endpoint"
        elif step_progress >= 0.88:
            phase_scope = "late_endpoint_step"
            progress_bucket = "late_endpoint"
        else:
            phase_scope = "mid_window_step"
            progress_bucket = "mid_window"

        rec = {
            "stage": f"EventDenoisePhaseMap_{branch_name}",
            "status": "guarded_report_only",
            "version": "denoise_phase_map_v1_report_guard",
            "branch_name": str(branch_name or ""),
            "branch_key": branch_key,
            "phase_scope": phase_scope,
            "progress_bucket": progress_bucket,
            "step_index": int(step_index) if step_index is not None else None,
            "window_steps": int(window_steps) if window_steps is not None else None,
            "step_progress": float(step_progress) if step_progress is not None else None,
            "post_window_delta": bool(post_window_delta),
            "endpoint_band": bool(endpoint_band),
            "strategy_control_surface": (surface_rec or {}).get("stage", "") if isinstance(surface_rec, dict) else "",
            "strategy_control_apply_policy": (surface_rec or {}).get("apply_policy", "") if isinstance(surface_rec, dict) else "",
            "active_control_allowed": False,
            "prompt_text_injection_allowed": False,
            "semantic_math_in_prompt_allowed": False,
            "formula_role": "phase placement for ObservedBehavior(delta) before Strategy pressure may become active control",
            "formula": "Local math may act only after it knows whether this delta is event birth, endpoint/source return, post-window summary, or mid-window refinement.",
        }

        if branch_key == "high":
            rec.update({
                "placement": "high_birth_source_noise_report_only",
                "reason": "high_branch_birth_denoise_phase",
                "guard": "High branch is where the event direction is born and denoised; late/background damping here can preserve unfinished noise.",
                "next_safe_surface": "pre_high_source_noise_field_research",
            })
        elif post_window_delta:
            rec.update({
                "placement": "post_window_report_only",
                "reason": "post_window_endpoint_denoise_safety",
                "guard": "Post-window delta is an integral summary without internal denoise phase evidence.",
                "next_safe_surface": "step_level_mid_window_trace",
            })
        elif endpoint_band:
            rec.update({
                "placement": "endpoint_report_only",
                "reason": "endpoint_step_denoise_safety",
                "guard": "Start/end denoise steps carry source return and final cleanup; local carrier math remains report-only there.",
                "next_safe_surface": "mid_window_low_step_refinement",
            })
        elif branch_key == "low":
            rec.update({
                "status": "candidate",
                "placement": "low_mid_window_refinement_candidate",
                "reason": "low_mid_window_has_denoise_phase_context",
                "guard": "This is the first safe candidate zone for future active spatial/relation math; r103 still records it before changing behavior.",
                "active_control_allowed": True,
                "next_safe_surface": "mid_window_low_step_refinement",
            })
        else:
            rec.update({
                "placement": "generic_mid_window_report_only",
                "reason": "unknown_branch_report_only",
                "guard": "Unknown branch does not receive active local math.",
                "next_safe_surface": "branch_role_resolution",
            })

        self._event_denoise_phase_map_last = rec
        if records is not None:
            try:
                seen = getattr(self, "_event_denoise_phase_map_signatures", set())
                if not isinstance(seen, set):
                    seen = set()
                signature = json.dumps(_event_json_safe({
                    "branch_key": branch_key,
                    "branch_name": str(branch_name or ""),
                    "phase_scope": phase_scope,
                    "progress_bucket": progress_bucket,
                    "placement": rec.get("placement", ""),
                }), sort_keys=True, ensure_ascii=True)
                if signature not in seen:
                    records.append(rec)
                    seen.add(signature)
                    self._event_denoise_phase_map_signatures = seen
            except Exception:
                records.append(rec)
        return rec

    def _event_noise_source_field_map(
        self,
        latent_state,
        branch_name,
        records=None,
        *,
        phase_map=None,
    ):
        branch_lower = str(branch_name or "").lower()
        if "high" in branch_lower:
            branch_key = "high"
        elif "low" in branch_lower:
            branch_key = "low"
        else:
            branch_key = "default"

        phase_map = phase_map if isinstance(phase_map, dict) else {}
        rec = {
            "stage": f"EventNoiseSourceFieldMap_{branch_name}",
            "status": "unavailable",
            "version": "noise_source_field_map_v1_report_only",
            "branch_name": str(branch_name or ""),
            "branch_key": branch_key,
            "phase_scope": str(phase_map.get("phase_scope", "") or ""),
            "phase_placement": str(phase_map.get("placement", "") or ""),
            "denoise_phase_active_candidate": bool(phase_map.get("active_control_allowed", False)),
            "active_control_allowed": False,
            "future_active_candidate": (
                "pre_high_source_noise_shaping" if branch_key == "high"
                else "low_mid_window_refinement" if branch_key == "low"
                else "report_only"
            ),
            "prompt_text_injection_allowed": False,
            "semantic_math_in_prompt_allowed": False,
            "formula_role": "read source/noise field pressure before Strategy becomes local control",
            "formula": "NoiseSourceField reads latent region energy as evidence only. It does not change tensors in r103; it tells later math where the model is already carrying source, background, and motion pressure.",
        }
        try:
            import torch
            t = self._tensor_from_latent_like(latent_state)
            if t is None:
                rec["reason"] = "latent_tensor_unavailable"
                return rec
            tf = torch.nan_to_num(t.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
            rec["latent_shape"] = list(tf.shape)
            if tf.dim() < 2:
                rec["reason"] = "latent_rank_too_low"
                return rec

            spatial = tf.abs()
            if spatial.dim() > 2:
                reduce_dims = tuple(range(0, spatial.dim() - 2))
                energy = spatial.mean(dim=reduce_dims)
            else:
                energy = spatial
            if energy.dim() != 2:
                energy = energy.reshape(-1, energy.shape[-1])
            h = int(energy.shape[-2])
            w = int(energy.shape[-1])
            if h <= 1 or w <= 1:
                rec["reason"] = "latent_spatial_shape_too_small"
                rec["spatial_shape"] = [h, w]
                return rec

            def band_mean(y0, y1, x0, x1):
                y0i = max(0, min(h - 1, int(round(float(y0) * h))))
                y1i = max(y0i + 1, min(h, int(round(float(y1) * h))))
                x0i = max(0, min(w - 1, int(round(float(x0) * w))))
                x1i = max(x0i + 1, min(w, int(round(float(x1) * w))))
                return float(energy[y0i:y1i, x0i:x1i].mean().detach().cpu().item())

            eps = 1e-9
            top = band_mean(0.00, 0.25, 0.00, 1.00)
            center = band_mean(0.25, 0.75, 0.25, 0.75)
            bottom = band_mean(0.75, 1.00, 0.00, 1.00)
            left = band_mean(0.00, 1.00, 0.00, 0.22)
            right = band_mean(0.00, 1.00, 0.78, 1.00)
            outer = float((top + bottom + left + right) / 4.0)
            global_mean = float(energy.mean().detach().cpu().item())
            global_std = float(energy.std(unbiased=False).detach().cpu().item())
            flat = energy.flatten()
            total_raw = float(flat.sum().detach().cpu().item())
            if total_raw <= eps or global_mean <= eps:
                rec.update({
                    "status": "recorded_zero_field",
                    "reason": "zero_or_flat_source_noise_field",
                    "spatial_shape": [h, w],
                    "region_energy": {
                        "top": round(top, 6),
                        "center": round(center, 6),
                        "bottom": round(bottom, 6),
                        "left": round(left, 6),
                        "right": round(right, 6),
                        "outer_mean": round(outer, 6),
                        "global_mean": round(global_mean, 6),
                        "global_std": round(global_std, 6),
                    },
                    "region_ratios": {
                        "center_outer_ratio": 1.0,
                        "top_center_ratio": 1.0,
                        "field_cv": 0.0,
                        "spatial_entropy_norm": 1.0,
                    },
                    "source_field_pressure": 0.0,
                    "recommended_next_surface": "latent_field_unavailable_until_nonzero_carrier",
                })
                return rec
            total = flat.sum().clamp_min(eps)
            prob = flat / total
            entropy_norm = float((-(prob * (prob + eps).log()).sum() / math.log(max(2, int(prob.numel())))).detach().cpu().item())
            center_outer_ratio = float(center / max(outer, eps))
            top_center_ratio = float(top / max(center, eps))
            field_cv = float(global_std / max(global_mean, eps))
            source_field_pressure = max(
                0.0,
                min(
                    1.0,
                    (min(field_cv, 2.0) / 2.0) * 0.45
                    + min(abs(math.log(max(center_outer_ratio, eps))), 2.0) * 0.20
                    + min(abs(math.log(max(top_center_ratio, eps))), 2.0) * 0.20
                    + (1.0 - max(0.0, min(1.0, entropy_norm))) * 0.15,
                ),
            )
            rec.update({
                "status": "recorded",
                "spatial_shape": [h, w],
                "region_energy": {
                    "top": round(top, 6),
                    "center": round(center, 6),
                    "bottom": round(bottom, 6),
                    "left": round(left, 6),
                    "right": round(right, 6),
                    "outer_mean": round(outer, 6),
                    "global_mean": round(global_mean, 6),
                    "global_std": round(global_std, 6),
                },
                "region_ratios": {
                    "center_outer_ratio": round(center_outer_ratio, 6),
                    "top_center_ratio": round(top_center_ratio, 6),
                    "field_cv": round(field_cv, 6),
                    "spatial_entropy_norm": round(entropy_norm, 6),
                },
                "source_field_pressure": round(source_field_pressure, 6),
                "recommended_next_surface": (
                    "pre_high_source_noise_shaping_research"
                    if branch_key == "high"
                    else "mid_window_low_refinement_research"
                    if bool(phase_map.get("active_control_allowed", False))
                    else "report_only_until_step_phase_available"
                ),
            })
            return rec
        except Exception as e:
            rec["status"] = "failed"
            rec["error"] = str(e)
            return rec
        finally:
            if records is not None:
                try:
                    seen = getattr(self, "_event_noise_source_field_map_signatures", set())
                    if not isinstance(seen, set):
                        seen = set()
                    signature = json.dumps(_event_json_safe({
                        "branch_key": branch_key,
                        "branch_name": str(branch_name or ""),
                        "phase_scope": rec.get("phase_scope", ""),
                        "phase_placement": rec.get("phase_placement", ""),
                        "spatial_shape": rec.get("spatial_shape", []),
                    }), sort_keys=True, ensure_ascii=True)
                    if signature not in seen:
                        records.append(rec)
                        seen.add(signature)
                        self._event_noise_source_field_map_signatures = seen
                except Exception:
                    records.append(rec)

    def _event_noise_field_strategy_bridge(
        self,
        branch_name,
        records=None,
        *,
        phase_map=None,
        noise_field_map=None,
        surface_rec=None,
    ):
        branch_lower = str(branch_name or "").lower()
        if "high" in branch_lower:
            branch_key = "high"
        elif "low" in branch_lower:
            branch_key = "low"
        else:
            branch_key = "default"

        def clamp01(value):
            try:
                out = float(value)
            except Exception:
                out = 0.0
            if not math.isfinite(out):
                out = 0.0
            return max(0.0, min(1.0, out))

        phase_map = phase_map if isinstance(phase_map, dict) else {}
        noise_field_map = noise_field_map if isinstance(noise_field_map, dict) else {}
        surface_rec = surface_rec if isinstance(surface_rec, dict) else {}

        ratios = noise_field_map.get("region_ratios", {}) if isinstance(noise_field_map.get("region_ratios", {}), dict) else {}
        try:
            center_outer_ratio = float(ratios.get("center_outer_ratio", 1.0) or 1.0)
        except Exception:
            center_outer_ratio = 1.0
        try:
            top_center_ratio = float(ratios.get("top_center_ratio", 1.0) or 1.0)
        except Exception:
            top_center_ratio = 1.0
        field_cv = clamp01(float(ratios.get("field_cv", 0.0) or 0.0) / 2.0)
        entropy_norm = clamp01(ratios.get("spatial_entropy_norm", 1.0))
        source_pressure = clamp01(noise_field_map.get("source_field_pressure", 0.0))
        center_outer_pressure = clamp01(abs(math.log(max(center_outer_ratio, 1e-9))) / 2.0)
        top_center_pressure = clamp01(abs(math.log(max(top_center_ratio, 1e-9))) / 2.0)
        entropy_pressure = clamp01(1.0 - entropy_norm)
        bridge_pressure = clamp01(
            (source_pressure * 0.45)
            + (field_cv * 0.20)
            + (center_outer_pressure * 0.15)
            + (top_center_pressure * 0.10)
            + (entropy_pressure * 0.10)
        )

        phase_active = bool(phase_map.get("active_control_allowed", False))
        noise_status = str(noise_field_map.get("status", "unavailable") or "unavailable")
        phase_scope = str(phase_map.get("phase_scope", "") or "")
        phase_placement = str(phase_map.get("placement", "") or "")
        next_safe_surface = str(phase_map.get("next_safe_surface", "") or "")

        rec = {
            "stage": f"EventNoiseFieldStrategyBridge_{branch_name}",
            "status": "report_only",
            "version": "noise_field_strategy_bridge_v1_model_attractor",
            "branch_name": str(branch_name or ""),
            "branch_key": branch_key,
            "phase_scope": phase_scope,
            "phase_placement": phase_placement,
            "noise_source_field_map": {
                "stage": noise_field_map.get("stage", ""),
                "status": noise_status,
                "version": noise_field_map.get("version", ""),
                "source_field_pressure": float(source_pressure),
                "recommended_next_surface": noise_field_map.get("recommended_next_surface", ""),
            },
            "denoise_phase_map": {
                "stage": phase_map.get("stage", ""),
                "status": phase_map.get("status", ""),
                "active_control_allowed": bool(phase_active),
                "next_safe_surface": next_safe_surface,
            },
            "strategy_control_surface": {
                "stage": surface_rec.get("stage", ""),
                "apply_policy": surface_rec.get("apply_policy", ""),
                "effective_strength": surface_rec.get("effective_strength", None),
            },
            "pressures": {
                "source_field_pressure": float(source_pressure),
                "field_cv_pressure": float(field_cv),
                "center_outer_pressure": float(center_outer_pressure),
                "top_center_pressure": float(top_center_pressure),
                "entropy_pressure": float(entropy_pressure),
                "bridge_pressure": float(bridge_pressure),
            },
            "active_control_allowed": False,
            "prompt_text_injection_allowed": False,
            "semantic_math_in_prompt_allowed": False,
            "model_attractor_route": (
                "Outcome(t-1)+ObservedBehavior(t-1) source/noise evidence "
                "-> StrategyAttractor(model) -> denoise-safe next surface "
                "-> Strategy Return"
            ),
            "formula_role": "noise/source evidence bridge into model-centered Strategy placement",
            "formula": "This bridge does not mutate tensors in this build. It decides whether a source/noise field is real evidence and where that evidence may safely become future math.",
        }

        if noise_status in ("recorded_zero_field", "unavailable", "failed"):
            rec.update({
                "status": "blocked_zero_or_unavailable_field" if noise_status == "recorded_zero_field" else "unavailable",
                "reason": "zero_or_unavailable_noise_field_is_not_strategy_pressure",
                "recommended_current_action": "ignore_as_active_pressure",
                "recommended_next_surface": "wait_for_nonzero_source_noise_carrier",
            })
        elif branch_key == "high":
            rec.update({
                "status": "pre_high_candidate_report_only",
                "reason": "high_post_window_delta_is_too_late_for_active_noise_shaping",
                "recommended_current_action": "report_only_on_current_post_window_delta",
                "recommended_next_surface": "pre_high_source_noise_shaping",
                "current_surface_safe": False,
            })
        elif branch_key == "low" and phase_active:
            rec.update({
                "status": "low_mid_window_candidate_report_only",
                "reason": "low_mid_window_has_phase_context_but_current_build_keeps_bridge_report_only",
                "recommended_current_action": "record_candidate_before_active_step_control",
                "recommended_next_surface": "mid_window_low_refinement",
                "current_surface_safe": True,
            })
        elif branch_key == "low":
            rec.update({
                "status": "post_window_report_only",
                "reason": "low_post_window_delta_is_integral_summary_not_step_control",
                "recommended_current_action": "do_not_mutate_from_noise_bridge",
                "recommended_next_surface": "step_level_mid_window_trace",
                "current_surface_safe": False,
            })
        else:
            rec.update({
                "status": "generic_report_only",
                "reason": "branch_role_unresolved",
                "recommended_current_action": "resolve_branch_before_control",
                "recommended_next_surface": "branch_role_resolution",
                "current_surface_safe": False,
            })

        if records is not None:
            try:
                seen = getattr(self, "_event_noise_field_strategy_bridge_signatures", set())
                if not isinstance(seen, set):
                    seen = set()
                signature = json.dumps(_event_json_safe({
                    "branch_name": str(branch_name or ""),
                    "phase_scope": phase_scope,
                    "noise_status": noise_status,
                    "status": rec.get("status", ""),
                    "spatial_shape": noise_field_map.get("spatial_shape", []),
                }), sort_keys=True, ensure_ascii=True)
                if signature not in seen:
                    records.append(rec)
                    seen.add(signature)
                    self._event_noise_field_strategy_bridge_signatures = seen
            except Exception:
                records.append(rec)
        self._event_noise_field_strategy_bridge_last = rec
        return rec

    def _event_spatial_carrier_preservation_map(
        self,
        branch_name,
        delta_t,
        records,
        *,
        step_index=None,
        window_steps=None,
        surface_rec=None,
        denoise_phase_map=None,
    ):
        branch_lower = str(branch_name or "").lower()
        if "high" in branch_lower:
            branch_key = "high"
        elif "low" in branch_lower:
            branch_key = "low"
        else:
            branch_key = "default"

        def clamp01(value):
            try:
                out = float(value)
            except Exception:
                out = 0.0
            if not math.isfinite(out):
                out = 0.0
            return max(0.0, min(1.0, out))

        mode = str(getattr(self, "_event_math_control_mode", "OBSERVE_ONLY") or "OBSERVE_ONLY").upper()
        background_preservation = getattr(self, "_event_background_anchor_preservation_control", {}) or {}
        if not isinstance(background_preservation, dict):
            background_preservation = {}
        status = str(background_preservation.get("status", "inactive") or "inactive")
        spatial_pressure = clamp01(background_preservation.get("spatial_anchor_pressure", 0.0))
        background_region_pressure = clamp01(background_preservation.get("background_region_pressure", 0.0))
        background_anchor_pressure = clamp01(background_preservation.get("background_anchor_pressure", 0.0))
        top_band_pressure = clamp01(background_preservation.get("top_band_pressure", 0.0))
        late_segment_pressure = clamp01(background_preservation.get("late_segment_pressure", 0.0))
        dominant_region = str(background_preservation.get("dominant_background_region", "") or "")
        carrier_pressure = max(spatial_pressure, background_region_pressure, background_anchor_pressure, top_band_pressure)
        rec = {
            "stage": f"EventSpatialCarrierPreservationMap_{branch_name}",
            "status": "inactive",
            "version": "spatial_carrier_preservation_map_v3_denoise_phase_map",
            "mode": mode,
            "branch_name": str(branch_name or ""),
            "branch_key": branch_key,
            "source": "background_anchor_preservation_control",
            "background_anchor_preservation_status": status,
            "dominant_background_region": dominant_region,
            "input_pressures": {
                "spatial_anchor_pressure": float(spatial_pressure),
                "background_region_pressure": float(background_region_pressure),
                "background_anchor_pressure": float(background_anchor_pressure),
                "top_band_pressure": float(top_band_pressure),
                "late_segment_pressure": float(late_segment_pressure),
                "carrier_pressure": float(carrier_pressure),
            },
            "prompt_text_injection_allowed": False,
            "semantic_math_in_prompt_allowed": False,
            "formula_role": "ObservedBehavior(delta) spatial carrier gate before Outcome(after)",
            "formula": "delta_after = delta_after * spatial_carrier_gain; Strategy remains the model-centered attractor and prompt text stays untouched.",
        }
        if isinstance(denoise_phase_map, dict):
            rec["denoise_phase_map"] = {
                "stage": denoise_phase_map.get("stage", ""),
                "status": denoise_phase_map.get("status", ""),
                "version": denoise_phase_map.get("version", ""),
                "phase_scope": denoise_phase_map.get("phase_scope", ""),
                "placement": denoise_phase_map.get("placement", ""),
                "reason": denoise_phase_map.get("reason", ""),
                "active_control_allowed": bool(denoise_phase_map.get("active_control_allowed", False)),
                "next_safe_surface": denoise_phase_map.get("next_safe_surface", ""),
            }
        gain = None
        try:
            step_progress = None
            try:
                if step_index is not None and window_steps is not None and int(window_steps) > 0:
                    step_progress = float((int(step_index) + 1) / max(1, int(window_steps)))
            except Exception:
                step_progress = None
            post_window_delta = step_progress is None
            rec["step_index"] = int(step_index) if step_index is not None else None
            rec["window_steps"] = int(window_steps) if window_steps is not None else None
            rec["step_progress"] = float(step_progress) if step_progress is not None else None
            rec["post_window_delta"] = bool(post_window_delta)

            if mode != "STRATEGY_PRESSURE_WINDOW":
                rec["reason"] = "requires_strategy_pressure_window"
                return rec, None
            if status != "active":
                rec["reason"] = "no_active_cached_spatial_background_evidence"
                return rec, None
            if carrier_pressure < 0.20:
                rec["reason"] = "carrier_pressure_below_threshold"
                return rec, None
            if isinstance(denoise_phase_map, dict) and not bool(denoise_phase_map.get("active_control_allowed", False)):
                rec["status"] = "guarded_report_only"
                rec["reason"] = str(denoise_phase_map.get("reason", "") or "denoise_phase_report_only")
                rec["active_control_allowed"] = False
                rec["guard"] = str(denoise_phase_map.get("guard", "") or "Denoise phase map does not allow active spatial carrier math here.")
                rec["next_safe_surface"] = str(denoise_phase_map.get("next_safe_surface", "") or "")
                return rec, None
            if branch_key == "high":
                rec["status"] = "guarded_report_only"
                rec["reason"] = "high_branch_denoise_safety"
                rec["active_control_allowed"] = False
                rec["guard"] = "High branch still births and denoises the event; reducing background delta here can preserve raw noise."
                return rec, None
            if post_window_delta:
                rec["status"] = "guarded_report_only"
                rec["reason"] = "post_window_endpoint_denoise_safety"
                rec["active_control_allowed"] = False
                rec["guard"] = "Post-window delta has no internal denoise phase information; preserving background by lowering delta can freeze late/start endpoint noise."
                return rec, None
            if step_progress <= 0.12 or step_progress >= 0.88:
                rec["status"] = "guarded_report_only"
                rec["reason"] = "endpoint_step_denoise_safety"
                rec["active_control_allowed"] = False
                rec["guard"] = "Start/end denoise steps are endpoint/source-return phases; spatial carrier control is report-only there."
                return rec, None
            if delta_t is None or not hasattr(delta_t, "shape") or len(delta_t.shape) < 2:
                rec["reason"] = "delta_tensor_unavailable"
                return rec, None

            import torch
            h = int(delta_t.shape[-2])
            w = int(delta_t.shape[-1])
            if h <= 1 or w <= 1:
                rec["reason"] = "latent_spatial_shape_too_small"
                rec["latent_shape"] = list(delta_t.shape)
                return rec, None

            gain2d = torch.ones((h, w), dtype=torch.float32, device=delta_t.device)
            roi_gains = []

            if branch_key == "high":
                max_attenuation = min(0.12, 0.08 * carrier_pressure + 0.04 * late_segment_pressure)
                min_gain = 0.88
            elif branch_key == "low":
                max_attenuation = min(0.28, 0.18 * carrier_pressure + 0.06 * late_segment_pressure + 0.04 * top_band_pressure)
                min_gain = 0.72
            else:
                max_attenuation = min(0.18, 0.12 * carrier_pressure)
                min_gain = 0.82
            if max_attenuation <= 1e-9:
                rec["reason"] = "zero_attenuation"
                return rec, None

            def apply_roi(name, y0, y1, x0, x1, pressure, attenuation_scale=1.0):
                pressure_f = clamp01(pressure)
                if pressure_f <= 0.0:
                    return
                y0i = max(0, min(h - 1, int(round(float(y0) * h))))
                y1i = max(y0i + 1, min(h, int(round(float(y1) * h))))
                x0i = max(0, min(w - 1, int(round(float(x0) * w))))
                x1i = max(x0i + 1, min(w, int(round(float(x1) * w))))
                attenuation = max(0.0, min(max_attenuation, max_attenuation * pressure_f * float(attenuation_scale)))
                region_gain = max(min_gain, 1.0 - attenuation)
                current = gain2d[y0i:y1i, x0i:x1i]
                gain2d[y0i:y1i, x0i:x1i] = torch.minimum(current, torch.full_like(current, float(region_gain)))
                roi_gains.append({
                    "name": name,
                    "pressure": float(pressure_f),
                    "gain": float(region_gain),
                    "y0": int(y0i),
                    "y1": int(y1i),
                    "x0": int(x0i),
                    "x1": int(x1i),
                })

            if top_band_pressure > 0.0:
                apply_roi("top_band_background", 0.00, 0.22, 0.00, 1.00, top_band_pressure, 0.95)

            dominant = dominant_region.lower()
            if "lower" in dominant and "side" in dominant:
                apply_roi("lower_left_side_floor", 0.58, 1.00, 0.00, 0.30, background_region_pressure or carrier_pressure, 1.00)
                apply_roi("lower_right_side_floor", 0.58, 1.00, 0.70, 1.00, background_region_pressure or carrier_pressure, 1.00)
            elif "left" in dominant:
                apply_roi("left_background_carrier", 0.30, 1.00, 0.00, 0.34, background_region_pressure or carrier_pressure, 1.00)
            elif "right" in dominant:
                apply_roi("right_background_carrier", 0.30, 1.00, 0.66, 1.00, background_region_pressure or carrier_pressure, 1.00)
            elif "top" in dominant:
                apply_roi("top_background_carrier", 0.00, 0.34, 0.00, 1.00, background_region_pressure or carrier_pressure, 0.90)
            elif background_region_pressure > 0.0:
                apply_roi("side_background_left", 0.25, 1.00, 0.00, 0.20, background_region_pressure, 0.70)
                apply_roi("side_background_right", 0.25, 1.00, 0.80, 1.00, background_region_pressure, 0.70)

            if background_anchor_pressure > 0.0:
                apply_roi("outer_source_anchor_left", 0.00, 1.00, 0.00, 0.12, background_anchor_pressure, 0.45)
                apply_roi("outer_source_anchor_right", 0.00, 1.00, 0.88, 1.00, background_anchor_pressure, 0.45)

            if not roi_gains:
                rec["reason"] = "no_roi_selected"
                return rec, None

            view_shape = [1 for _ in range(len(delta_t.shape))]
            view_shape[-2] = h
            view_shape[-1] = w
            gain = gain2d.view(*view_shape)
            rec.update({
                "status": "active",
                "reason": "cached_spatial_carriers_bound_background_delta",
                "policy": "soft_region_gain_multiplies_latent_delta",
                "active_control_allowed": True,
                "latent_shape": list(delta_t.shape),
                "gain_shape": list(gain.shape),
                "min_gain": float(gain2d.min().detach().cpu().item()),
                "max_gain": float(gain2d.max().detach().cpu().item()),
                "mean_gain": float(gain2d.mean().detach().cpu().item()),
                "max_attenuation": float(max_attenuation),
                "roi_gains": roi_gains,
            })
            return rec, gain
        except Exception as e:
            rec["status"] = "failed_passthrough"
            rec["error"] = str(e)
            return rec, None
        finally:
            if records is not None:
                records.append(rec)

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
            mode = str(getattr(self, "_event_math_control_mode", "OBSERVE_ONLY") or "OBSERVE_ONLY").upper()
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

        surface_rec = self._event_strategy_control_surface_apply(
            branch_name,
            scheduled_strength,
            base_strength,
            coupling_multiplier,
            step_schedule_factor,
            records,
            step_index=step_index,
            window_steps=window_steps,
        )
        denoise_phase_rec = self._event_denoise_phase_map(
            branch_name,
            records,
            step_index=step_index,
            window_steps=window_steps,
            surface_rec=surface_rec,
        )
        noise_source_field_rec = self._event_noise_source_field_map(
            latent_before,
            branch_name,
            records,
            phase_map=denoise_phase_rec,
        )
        noise_field_strategy_bridge_rec = self._event_noise_field_strategy_bridge(
            branch_name,
            records,
            phase_map=denoise_phase_rec,
            noise_field_map=noise_source_field_rec,
            surface_rec=surface_rec,
        )
        noise_field_strategy_bridge_summary = {
            "stage": noise_field_strategy_bridge_rec.get("stage", ""),
            "status": noise_field_strategy_bridge_rec.get("status", ""),
            "version": noise_field_strategy_bridge_rec.get("version", ""),
            "bridge_pressure": (noise_field_strategy_bridge_rec.get("pressures", {}) or {}).get("bridge_pressure", None),
            "recommended_next_surface": noise_field_strategy_bridge_rec.get("recommended_next_surface", ""),
            "active_control_allowed": bool(noise_field_strategy_bridge_rec.get("active_control_allowed", False)),
        }
        try:
            strength_runtime = float(surface_rec.get("effective_strength", 1.0))
        except Exception:
            strength_runtime = 1.0

        if mode not in ("LATENT_DELTA_SCALE", "DEEP_STEP_DELTA_CONTROL", "STRATEGY_PRESSURE_WINDOW", "LATENT_MEMORY_BRIDGE") or abs(strength_runtime - 1.0) < 1e-9:
            records.append({
                "stage": f"EventMathDeltaControl_{branch_name}",
                "status": "bypass",
                "mode": mode,
                "strategy_control_surface": surface_rec.get("stage", ""),
                "strategy_control_apply_policy": surface_rec.get("apply_policy", ""),
                "base_strength": base_strength,
                "coupling_multiplier": coupling_multiplier,
                "step_schedule_factor": step_schedule_factor,
                "strength_runtime": strength_runtime,
                "denoise_phase_map": {
                    "stage": denoise_phase_rec.get("stage", ""),
                    "status": denoise_phase_rec.get("status", ""),
                    "phase_scope": denoise_phase_rec.get("phase_scope", ""),
                    "placement": denoise_phase_rec.get("placement", ""),
                    "active_control_allowed": bool(denoise_phase_rec.get("active_control_allowed", False)),
                },
                "noise_field_strategy_bridge": noise_field_strategy_bridge_summary,
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
                    "strategy_control_surface": surface_rec.get("stage", ""),
                    "strategy_control_apply_policy": surface_rec.get("apply_policy", ""),
                    "base_strength": base_strength,
                    "coupling_multiplier": coupling_multiplier,
                    "step_schedule_factor": step_schedule_factor,
                    "strength_runtime": strength_runtime,
                    "noise_field_strategy_bridge": noise_field_strategy_bridge_summary,
                    "reason": "missing before/after tensor",
                })
                return latent_after
            if before_t.shape != after_t.shape:
                records.append({
                    "stage": f"EventMathDeltaControl_{branch_name}",
                    "status": "shape_mismatch",
                    "mode": mode,
                    "strategy_control_surface": surface_rec.get("stage", ""),
                    "strategy_control_apply_policy": surface_rec.get("apply_policy", ""),
                    "base_strength": base_strength,
                    "coupling_multiplier": coupling_multiplier,
                    "step_schedule_factor": step_schedule_factor,
                    "strength_runtime": strength_runtime,
                    "before_shape": list(before_t.shape),
                    "after_shape": list(after_t.shape),
                    "noise_field_strategy_bridge": noise_field_strategy_bridge_summary,
                })
                return latent_after

            before_f = before_t.detach().float()
            after_f = after_t.detach().float()
            delta_f = after_f - before_f
            spatial_carrier_rec, spatial_carrier_gain = self._event_spatial_carrier_preservation_map(
                branch_name,
                delta_f,
                records,
                step_index=step_index,
                window_steps=window_steps,
                surface_rec=surface_rec,
                denoise_phase_map=denoise_phase_rec,
            )
            if spatial_carrier_gain is not None:
                controlled = before_f + delta_f * float(strength_runtime) * spatial_carrier_gain
                spatial_formula = "controlled_after = latent_before + (latent_after - latent_before) * strength_runtime * spatial_carrier_gain"
            else:
                controlled = before_f + delta_f * float(strength_runtime)
                spatial_formula = "controlled_after = latent_before + (latent_after - latent_before) * strength_runtime"
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
                "strategy_control_surface": surface_rec.get("stage", ""),
                "strategy_control_apply_policy": surface_rec.get("apply_policy", ""),
                "base_strength": base_strength,
                "coupling_multiplier": coupling_multiplier,
                "step_schedule_factor": step_schedule_factor,
                "strength_runtime": strength_runtime,
                "spatial_carrier_preservation_map": {
                    "stage": spatial_carrier_rec.get("stage", ""),
                    "status": spatial_carrier_rec.get("status", "not_recorded"),
                    "version": spatial_carrier_rec.get("version", ""),
                    "policy": spatial_carrier_rec.get("policy", ""),
                    "dominant_background_region": spatial_carrier_rec.get("dominant_background_region", ""),
                    "input_pressures": spatial_carrier_rec.get("input_pressures", {}),
                    "min_gain": spatial_carrier_rec.get("min_gain", None),
                    "mean_gain": spatial_carrier_rec.get("mean_gain", None),
                    "max_gain": spatial_carrier_rec.get("max_gain", None),
                },
                "denoise_phase_map": {
                    "stage": denoise_phase_rec.get("stage", ""),
                    "status": denoise_phase_rec.get("status", ""),
                    "phase_scope": denoise_phase_rec.get("phase_scope", ""),
                    "placement": denoise_phase_rec.get("placement", ""),
                    "active_control_allowed": bool(denoise_phase_rec.get("active_control_allowed", False)),
                },
                "noise_field_strategy_bridge": noise_field_strategy_bridge_summary,
                "step_index": int(step_index) if step_index is not None else None,
                "window_steps": int(window_steps) if window_steps is not None else None,
                "formula": spatial_formula,
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
                "strategy_control_surface": surface_rec.get("stage", ""),
                "strategy_control_apply_policy": surface_rec.get("apply_policy", ""),
                "base_strength": base_strength,
                "coupling_multiplier": coupling_multiplier,
                "step_schedule_factor": step_schedule_factor,
                "strength_runtime": strength_runtime,
                "noise_field_strategy_bridge": noise_field_strategy_bridge_summary,
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
        mode = str(getattr(self, "_event_math_control_mode", "OBSERVE_ONLY") or "OBSERVE_ONLY").upper()
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
        elif mode == "STRATEGY_PRESSURE_WINDOW":
            records.append({
                "stage": "EventMathSamplerPathPolicy",
                "status": "native_sampler_preserved_unified_pressure_window",
                "math_control_mode": mode,
                "native_step_loop_replacement": False,
                "formula": "Math acts through one bounded Strategy pressure window after the native sampler window, then returns to S_global_event_route.",
            })
        elif mode == "LATENT_MEMORY_BRIDGE":
            records.append({
                "stage": "EventMathSamplerPathPolicy",
                "status": "native_sampler_preserved_segment_entry_latent_memory_bridge",
                "math_control_mode": mode,
                "native_step_loop_replacement": False,
                "formula": "Bounded latent memory is applied before high sampler entry; high/low sampler physics stay model-native.",
            })
        elif mode == "DEEP_STEP_DELTA_CONTROL":
            records.append({
                "stage": "EventMathSamplerPathPolicy",
                "status": "native_step_loop_active",
                "math_control_mode": mode,
                "native_step_loop_replacement": True,
                "formula": "WARNING: Experimental deep-step delta control is active. High risk of noise.",
            })

        self._record_sampler_route_parity_probe(
            records,
            f"EventRawVsSingularityParity_SamplerWindowBegin_{window.branch_name}",
            branch_name=str(window.branch_name),
            route_variant="native_math_loop" if use_native_math_loop else "event_sampler_core_boundary",
            latent_before=latent,
            latent_after=None,
            model=model,
            seed=getattr(window, "seed", None),
            steps=getattr(window, "steps", None),
            cfg=getattr(window, "cfg", None),
            sampler_name=getattr(window, "sampler_name", None),
            scheduler=getattr(window, "scheduler", None),
            start_at_step=getattr(window, "start_at_step", None),
            end_at_step=getattr(window, "end_at_step", None),
            add_noise=getattr(window, "add_noise", None),
            return_leftover_noise=getattr(window, "return_with_leftover_noise", None),
            sd3_shift=getattr(window, "sd3_shift", None),
            extra={
                "branch_role": str(getattr(window, "branch_role", "")),
                "math_control_mode": mode,
                "probe_scope": "sampler_input_before_native_operation",
            },
        )

        if use_native_math_loop:
            result = self._event_sample_window_math_native(model, positive, negative, latent, window, records)
        else:
            core = EventSamplerCore(self._low_level_sampler_operation)
            result = core.sample_window(model=model, positive=positive, negative=negative, latent=latent, window=window)

        records.extend(result.event_records)
        if not result.ok:
            raise RuntimeError(result.error or f"EventSampler {window.branch_name} failed")

        self._record_sampler_route_parity_probe(
            records,
            f"EventRawVsSingularityParity_SamplerWindowRawAfter_{window.branch_name}",
            branch_name=str(window.branch_name),
            route_variant="native_math_loop" if use_native_math_loop else "event_sampler_core_boundary",
            latent_before=latent,
            latent_after=result.latent_after,
            model=model,
            seed=getattr(window, "seed", None),
            steps=getattr(window, "steps", None),
            cfg=getattr(window, "cfg", None),
            sampler_name=getattr(window, "sampler_name", None),
            scheduler=getattr(window, "scheduler", None),
            start_at_step=getattr(window, "start_at_step", None),
            end_at_step=getattr(window, "end_at_step", None),
            add_noise=getattr(window, "add_noise", None),
            return_leftover_noise=getattr(window, "return_with_leftover_noise", None),
            sd3_shift=getattr(window, "sd3_shift", None),
            extra={
                "branch_role": str(getattr(window, "branch_role", "")),
                "math_control_mode": mode,
                "probe_scope": "sampler_output_before_delta_overlay",
            },
        )

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
        if "high" in branch_name_lower and mode in ("LATENT_DELTA_SCALE", "STRATEGY_PRESSURE_WINDOW"):
            self._update_strategy_coupling_from_high(latent, controlled_latent_after, window.branch_name, records)

        self._record_sampler_route_parity_probe(
            records,
            f"EventRawVsSingularityParity_SamplerWindowControlledAfter_{window.branch_name}",
            branch_name=str(window.branch_name),
            route_variant="native_math_loop" if use_native_math_loop else "event_sampler_core_boundary",
            latent_before=latent,
            latent_after=controlled_latent_after,
            model=model,
            seed=getattr(window, "seed", None),
            steps=getattr(window, "steps", None),
            cfg=getattr(window, "cfg", None),
            sampler_name=getattr(window, "sampler_name", None),
            scheduler=getattr(window, "scheduler", None),
            start_at_step=getattr(window, "start_at_step", None),
            end_at_step=getattr(window, "end_at_step", None),
            add_noise=getattr(window, "add_noise", None),
            return_leftover_noise=getattr(window, "return_with_leftover_noise", None),
            sd3_shift=getattr(window, "sd3_shift", None),
            extra={
                "branch_role": str(getattr(window, "branch_role", "")),
                "math_control_mode": mode,
                "probe_scope": "sampler_output_after_delta_overlay",
            },
        )

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

    def _decode_tiled(self, vae, latent, tile_size, overlap, temporal_size, temporal_overlap, records, segment_index=None, route_label=""):
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
            self._record_raw_vs_singularity_parity_probe(
                records,
                "EventRawVsSingularityParity_Decode",
                route_kind="vae_tiled_decode",
                input_state=latent,
                output_state=image,
                metadata={
                    "segment_index": int(segment_index) if segment_index is not None else None,
                    "route_label": str(route_label or ""),
                    "tile_size": int(tile_size),
                    "overlap": int(overlap),
                    "temporal_size": int(temporal_size),
                    "temporal_overlap": int(temporal_overlap),
                    "vae_route_signature": self._object_route_cache_signature(vae),
                },
            )
            return image
        except Exception as e:
            records.append({"stage": "EventVAEDecodeTiled", "status": "failed", "error": str(e)})
            raise




