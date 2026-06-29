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

EVENT_HORIZON_RUNTIME_VERSION = "0.1.1-r178"
EVENT_HORIZON_RUNTIME_NAME = "Singularity R178 Tail 5 Continuation Gate"
EVENT_HORIZON_BODY_VERSION = "0.1-R178"


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
            pixel_region_rec = self._pixel_region_motion_map_record(tf, stage)
            if pixel_region_rec:
                records.append(pixel_region_rec)
            return rec
        except Exception as e:
            rec = {"stage": stage, "status": "failed", "error": str(e)}
            records.append(rec)
            return rec

    def _pixel_region_motion_map_record(self, frames_tensor, source_stage):
        """
        R144 observer-only pixel region motion map.

        This is deliberately lightweight: it reads decoded frame deltas and
        separates center/action motion from edge/top/bottom/background motion.
        It never changes frames or tensors.
        """
        try:
            import torch

            t = torch.nan_to_num(frames_tensor.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
            if t.dim() == 5:
                # Common video layouts: [B,T,H,W,C] or [B,C,T,H,W].
                if t.shape[-1] in (1, 3, 4):
                    t = t[0]
                elif t.shape[1] in (1, 3, 4):
                    t = t[0].permute(1, 2, 3, 0)
                else:
                    return {
                        "stage": "EventPixelRegionMotionMap",
                        "status": "unavailable",
                        "source_stage": str(source_stage),
                        "reason": "unsupported_5d_shape",
                        "shape": list(t.shape),
                    }
            elif t.dim() == 4:
                if t.shape[-1] in (1, 3, 4):
                    pass
                elif t.shape[1] in (1, 3, 4):
                    t = t.permute(0, 2, 3, 1)
                else:
                    return {
                        "stage": "EventPixelRegionMotionMap",
                        "status": "unavailable",
                        "source_stage": str(source_stage),
                        "reason": "unsupported_4d_shape",
                        "shape": list(t.shape),
                    }
            else:
                return {
                    "stage": "EventPixelRegionMotionMap",
                    "status": "unavailable",
                    "source_stage": str(source_stage),
                    "reason": "unsupported_rank",
                    "shape": list(t.shape),
                }

            if t.shape[0] < 2:
                return {
                    "stage": "EventPixelRegionMotionMap",
                    "status": "unavailable",
                    "source_stage": str(source_stage),
                    "reason": "not_enough_frames",
                    "shape": list(t.shape),
                }

            delta = (t[1:] - t[:-1]).abs()
            motion = delta.mean(dim=-1)
            h = int(motion.shape[1])
            w = int(motion.shape[2])
            if h <= 4 or w <= 4:
                return {
                    "stage": "EventPixelRegionMotionMap",
                    "status": "unavailable",
                    "source_stage": str(source_stage),
                    "reason": "too_small",
                    "shape": list(t.shape),
                }

            def roi_mean(y, x, hh, ww):
                y0 = max(0, min(h - 1, int(round(float(y) * h))))
                x0 = max(0, min(w - 1, int(round(float(x) * w))))
                y1 = max(y0 + 1, min(h, int(round(float(y + hh) * h))))
                x1 = max(x0 + 1, min(w, int(round(float(x + ww) * w))))
                region = motion[:, y0:y1, x0:x1]
                return float(region.mean().item()) if region.numel() else 0.0

            center_y, center_x, center_h, center_w = 0.18, 0.18, 0.64, 0.64
            center = roi_mean(center_y, center_x, center_h, center_w)
            object_contact = roi_mean(0.38, 0.18, 0.54, 0.64)
            top = roi_mean(0.00, 0.00, 0.22, 1.00)
            bottom = roi_mean(0.80, 0.00, 0.20, 1.00)
            left = roi_mean(0.00, 0.00, 1.00, 0.18)
            right = roi_mean(0.00, 0.82, 1.00, 0.18)
            whole = float(motion.mean().item()) if motion.numel() else 0.0

            mask = torch.ones((h, w), dtype=torch.bool, device=motion.device)
            y0 = max(0, min(h - 1, int(round(center_y * h))))
            x0 = max(0, min(w - 1, int(round(center_x * w))))
            y1 = max(y0 + 1, min(h, int(round((center_y + center_h) * h))))
            x1 = max(x0 + 1, min(w, int(round((center_x + center_w) * w))))
            mask[y0:y1, x0:x1] = False
            edge_region = motion[:, mask]
            edge = float(edge_region.mean().item()) if edge_region.numel() else 0.0

            per_delta = motion.mean(dim=(1, 2))
            seam_index = int(per_delta.numel() // 2) if per_delta.numel() else 0
            seam_motion = float(per_delta[seam_index].item()) if per_delta.numel() else 0.0
            max_motion = float(per_delta.max().item()) if per_delta.numel() else 0.0
            mean_motion = float(per_delta.mean().item()) if per_delta.numel() else 0.0

            def safe_ratio(a, b):
                return float(a / (b + 1e-12))

            center_edge_ratio = safe_ratio(center, edge)
            edge_center_ratio = safe_ratio(edge, center)
            top_center_ratio = safe_ratio(top, center)
            bottom_center_ratio = safe_ratio(bottom, center)
            object_center_ratio = safe_ratio(object_contact, center)
            seam_ratio = safe_ratio(seam_motion, mean_motion)
            background_pixel_leakage_score = max(0.0, min(1.0, edge_center_ratio))
            center_action_pixel_dominance = max(0.0, min(1.0, center_edge_ratio / 1.5))

            if center_edge_ratio >= 1.15 and seam_ratio < 1.85:
                status = "pixel_region_motion_center_action_dominant"
                severity = "INFO"
                next_surface = "same_sampler_strategy_ring_report_only"
            elif seam_ratio >= 1.85:
                status = "pixel_region_motion_seam_watch"
                severity = "WARNING"
                next_surface = "seam_local_action_background_review_report_only"
            elif edge_center_ratio >= 0.95:
                status = "pixel_region_motion_background_leakage_watch"
                severity = "WARNING"
                next_surface = "separate_center_action_background_tiles_report_only"
            else:
                status = "pixel_region_motion_mixed_report_only"
                severity = "INFO"
                next_surface = "collect_pixel_region_motion_evidence"

            return {
                "stage": "EventPixelRegionMotionMap",
                "status": status,
                "severity": severity,
                "map_version": "pixel_region_motion_map_v1_report_only",
                "source_stage": str(source_stage),
                "control_mode": "REPORT_ONLY",
                "active_control_allowed": False,
                "same_run_control_allowed": False,
                "prompt_text_injection_allowed": False,
                "semantic_math_in_prompt_allowed": False,
                "shape": list(t.shape),
                "frame_delta_count": int(per_delta.numel()),
                "whole_motion_mean": mean_motion,
                "whole_motion_max": max_motion,
                "estimated_seam_delta_index": seam_index,
                "estimated_seam_motion": seam_motion,
                "estimated_seam_ratio": seam_ratio,
                "center_action_pixel_motion": center,
                "object_contact_pixel_motion": object_contact,
                "edge_background_pixel_motion": edge,
                "top_background_pixel_motion": top,
                "bottom_background_pixel_motion": bottom,
                "left_edge_pixel_motion": left,
                "right_edge_pixel_motion": right,
                "center_edge_pixel_ratio": center_edge_ratio,
                "edge_center_pixel_ratio": edge_center_ratio,
                "top_center_pixel_ratio": top_center_ratio,
                "bottom_center_pixel_ratio": bottom_center_ratio,
                "object_center_pixel_ratio": object_center_ratio,
                "background_pixel_leakage_score": background_pixel_leakage_score,
                "center_action_pixel_dominance": center_action_pixel_dominance,
                "normalized_regions": {
                    "center_action": [center_y, center_x, center_h, center_w],
                    "object_contact": [0.38, 0.18, 0.54, 0.64],
                    "top_background": [0.00, 0.00, 0.22, 1.00],
                    "bottom_background": [0.80, 0.00, 0.20, 1.00],
                    "edge_background": "outside center_action",
                },
                "next_control_surface": next_surface,
                "formula": "Decoded pixel regions separate visible center action from edge/background motion before pressure can become a future route.",
                "next_action": "Compare this pixel map against R143 pressure separation; disagreement means pressure is scalar/report artifact, not visual proof.",
            }
        except Exception as e:
            return {
                "stage": "EventPixelRegionMotionMap",
                "status": "failed",
                "source_stage": str(source_stage),
                "error": str(e),
                "control_mode": "REPORT_ONLY",
                "active_control_allowed": False,
            }

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

    def _cascade_seam_impulse_review(self, segment_batches, concatenated_frames, records, frames_per_cascade=None):
        """
        R153 observer-only seam impulse review.

        This reads the cascade boundary as a Strategy collision:
        tail Outcome(previous segment) -> boundary jump -> post-continue
        ObservedBehavior(next segment). R153 also binds that route-vector
        reading to the visible concatenated-frame transition, because a seam
        can be visible even when the tail/entry motion vector looks nominal.
        It does not mutate frames or tensors.
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

            def ratio(numerator, denominator):
                numerator = safe_float(numerator, None)
                denominator = safe_float(denominator, None)
                if numerator is None or denominator is None or denominator <= 0:
                    return None
                return float(numerator / (denominator + 1e-12))

            def mean(values):
                values = [safe_float(v, None) for v in (values or [])]
                values = [v for v in values if v is not None]
                return float(sum(values) / len(values)) if values else None

            def median(values):
                values = [safe_float(v, None) for v in (values or [])]
                values = sorted(v for v in values if v is not None)
                if not values:
                    return None
                mid = len(values) // 2
                if len(values) % 2:
                    return float(values[mid])
                return float((values[mid - 1] + values[mid]) * 0.5)

            def bounded_slice(values, center, before=5, after=7):
                if center is None:
                    return []
                try:
                    center = int(center)
                except Exception:
                    return []
                if center < 0 or center >= len(values):
                    return []
                start = max(0, center - int(before))
                end = min(len(values), center + int(after) + 1)
                return values[start:end]

            def transition_rank(values, transition_index):
                if transition_index is None:
                    return None
                try:
                    transition_index = int(transition_index)
                except Exception:
                    return None
                values = [safe_float(v, None) for v in (values or [])]
                if transition_index < 0 or transition_index >= len(values) or values[transition_index] is None:
                    return None
                target = values[transition_index]
                return int(1 + sum(1 for v in values if v is not None and v > target))

            def to_frames(value):
                t = self._tensor_from_latent_like(value)
                if t is None:
                    return None
                t = torch.nan_to_num(t.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
                if t.dim() == 5:
                    if t.shape[-1] in (1, 3, 4):
                        t = t[0]
                    elif t.shape[1] in (1, 3, 4):
                        t = t[0].permute(1, 2, 3, 0)
                    else:
                        return None
                elif t.dim() == 4:
                    if t.shape[-1] in (1, 3, 4):
                        pass
                    elif t.shape[1] in (1, 3, 4):
                        t = t.permute(0, 2, 3, 1)
                    else:
                        return None
                else:
                    return None
                return t if t.shape[0] >= 1 else None

            def transition_maps(frames):
                if frames is None or frames.shape[0] < 2:
                    return None
                delta = (frames[1:] - frames[:-1]).abs()
                return delta.mean(dim=-1)

            def transition_magnitudes(motion):
                if motion is None or motion.numel() == 0:
                    return []
                values = motion.reshape(motion.shape[0], -1).mean(dim=1)
                return [float(x) for x in values.detach().cpu().tolist()]

            def motion_centroids(motion):
                if motion is None or motion.dim() != 3 or motion.shape[0] < 1:
                    return []
                h = int(motion.shape[1])
                w = int(motion.shape[2])
                yy = torch.linspace(0.0, 1.0, h, device=motion.device).view(1, h, 1)
                xx = torch.linspace(0.0, 1.0, w, device=motion.device).view(1, 1, w)
                weight = motion.clamp_min(0.0)
                denom = weight.sum(dim=(1, 2)).clamp_min(1e-12)
                cy = (weight * yy).sum(dim=(1, 2)) / denom
                cx = (weight * xx).sum(dim=(1, 2)) / denom
                return [(float(y), float(x)) for y, x in zip(cy.detach().cpu().tolist(), cx.detach().cpu().tolist())]

            def centroid_mean(points):
                points = [p for p in (points or []) if isinstance(p, (list, tuple)) and len(p) == 2]
                if not points:
                    return None
                return (
                    float(sum(float(p[0]) for p in points) / len(points)),
                    float(sum(float(p[1]) for p in points) / len(points)),
                )

            def centroid_distance(a, b):
                if a is None or b is None:
                    return None
                return float(((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2) ** 0.5)

            def vector_between(first, last):
                if first is None or last is None:
                    return None
                return (float(last[0]) - float(first[0]), float(last[1]) - float(first[1]))

            def vector_norm(v):
                if v is None:
                    return None
                return float((float(v[0]) ** 2 + float(v[1]) ** 2) ** 0.5)

            def cosine(a, b):
                if a is None or b is None:
                    return None
                an = vector_norm(a)
                bn = vector_norm(b)
                if an is None or bn is None or an <= 1e-8 or bn <= 1e-8:
                    return None
                return float((float(a[0]) * float(b[0]) + float(a[1]) * float(b[1])) / (an * bn + 1e-12))

            segments = [to_frames(item) for item in (segment_batches or [])]
            segments = [item for item in segments if item is not None and item.shape[0] >= 1]
            if len(segments) < 2:
                rec = {
                    "stage": "EventCascadeSeamImpulseReview",
                    "status": "not_applicable",
                    "severity": "INFO",
                    "reason": "requires_at_least_two_segments",
                    "observed_segments": len(segments),
                    "control_mode": "REPORT_ONLY",
                    "active_control_allowed": False,
                }
                records.append(rec)
                return rec

            concat = to_frames(concatenated_frames)
            concat_motion = transition_maps(concat)
            global_values = transition_magnitudes(concat_motion)
            global_mean = mean(global_values)
            global_median = median(global_values)
            boundary_transition_indices = []
            running_frames = 0
            for segment in segments[:-1]:
                running_frames += int(segment.shape[0])
                boundary_transition_indices.append(max(0, running_frames - 1))

            boundary_reviews = []
            for idx in range(1, len(segments)):
                prev = segments[idx - 1]
                nxt = segments[idx]
                prev_motion = transition_maps(prev)
                next_motion = transition_maps(nxt)
                prev_values = transition_magnitudes(prev_motion)
                next_values = transition_magnitudes(next_motion)
                tail_n = min(6, len(prev_values))
                entry_n = min(6, len(next_values))
                prev_tail_values = prev_values[-tail_n:] if tail_n else []
                next_entry_values = next_values[:entry_n] if entry_n else []
                prev_tail_mean = mean(prev_tail_values)
                next_entry_mean = mean(next_entry_values)
                prev_segment_mean = mean(prev_values)
                next_segment_mean = mean(next_values)

                boundary_delta = nxt[0] - prev[-1]
                boundary_abs = float(boundary_delta.abs().reshape(-1).mean().item())

                prev_centroids = motion_centroids(prev_motion)
                next_centroids = motion_centroids(next_motion)
                prev_tail_centroids = prev_centroids[-tail_n:] if tail_n else []
                next_entry_centroids = next_centroids[:entry_n] if entry_n else []
                prev_tail_center = centroid_mean(prev_tail_centroids)
                next_entry_center = centroid_mean(next_entry_centroids)
                centroid_jump = centroid_distance(prev_tail_center, next_entry_center)
                prev_vector = vector_between(prev_tail_centroids[0], prev_tail_centroids[-1]) if len(prev_tail_centroids) >= 2 else None
                next_vector = vector_between(next_entry_centroids[0], next_entry_centroids[-1]) if len(next_entry_centroids) >= 2 else None
                direction_cos = cosine(prev_vector, next_vector)

                boundary_to_tail = ratio(boundary_abs, prev_tail_mean)
                entry_to_tail = ratio(next_entry_mean, prev_tail_mean)
                next_to_prev = ratio(next_segment_mean, prev_segment_mean)
                boundary_to_global = ratio(boundary_abs, global_mean)
                visible_transition_index = (
                    boundary_transition_indices[idx - 1]
                    if idx - 1 < len(boundary_transition_indices)
                    else None
                )
                visible_transition_delta = (
                    safe_float(global_values[int(visible_transition_index)], None)
                    if visible_transition_index is not None and int(visible_transition_index) < len(global_values)
                    else None
                )
                visible_window_values = bounded_slice(global_values, visible_transition_index)
                visible_window_max = max(visible_window_values) if visible_window_values else None
                visible_window_mean = mean(visible_window_values)
                visible_seam_over_median = ratio(visible_window_max, global_median)
                visible_seam_over_mean = ratio(visible_window_max, global_mean)
                visible_boundary_over_median = ratio(visible_transition_delta, global_median)
                visible_transition_rank = transition_rank(global_values, visible_transition_index)
                visible_window_top_rank = min(
                    [rank for rank in (
                        transition_rank(global_values, transition_index)
                        for transition_index in range(
                            max(0, int(visible_transition_index or 0) - 5),
                            min(len(global_values), int(visible_transition_index or 0) + 8),
                        )
                    ) if rank is not None],
                    default=None,
                )
                centroid_jump_score = clamp01((centroid_jump or 0.0) / 0.22)
                boundary_jump_score = clamp01(((boundary_to_tail if boundary_to_tail is not None else 1.0) - 1.0) / 1.4)
                entry_acceleration_score = clamp01(((entry_to_tail if entry_to_tail is not None else 1.0) - 1.0) / 1.2)
                post_segment_pressure = clamp01(((next_to_prev if next_to_prev is not None else 1.0) - 1.0) / 1.2)
                visible_median_score = clamp01(((visible_seam_over_median if visible_seam_over_median is not None else 1.0) - 1.25) / 1.0)
                visible_boundary_score = clamp01(((visible_boundary_over_median if visible_boundary_over_median is not None else 1.0) - 1.20) / 1.0)
                visible_rank_score = 0.0
                if visible_window_top_rank is not None:
                    visible_rank_score = clamp01((10.0 - float(visible_window_top_rank)) / 9.0)
                visible_seam_delta_score = clamp01(
                    0.58 * visible_median_score
                    + 0.24 * visible_boundary_score
                    + 0.18 * visible_rank_score
                )
                direction_switch_score = 0.0
                if direction_cos is not None:
                    direction_switch_score = clamp01((0.35 - direction_cos) / 1.35)

                vector_seam_impulse_score = clamp01(
                    0.28 * boundary_jump_score
                    + 0.27 * entry_acceleration_score
                    + 0.22 * centroid_jump_score
                    + 0.15 * direction_switch_score
                    + 0.08 * post_segment_pressure
                )
                seam_impulse_score = max(vector_seam_impulse_score, visible_seam_delta_score)
                if seam_impulse_score >= 0.52:
                    status = "cascade_seam_impulse_high"
                    severity = "WARNING"
                    next_surface = "tail_next_source_strategy_continuity_report_only"
                elif seam_impulse_score >= 0.30:
                    status = "cascade_seam_impulse_watch"
                    severity = "INFO"
                    next_surface = "cascade_tail_entry_alignment_report_only"
                else:
                    status = "cascade_seam_impulse_nominal"
                    severity = "INFO"
                    next_surface = "keep_observing_cascade_seam"

                boundary_reviews.append({
                    "boundary_index": int(idx),
                    "previous_segment": int(idx),
                    "next_segment": int(idx + 1),
                    "status": status,
                    "severity": severity,
                    "tail_window_transitions": int(tail_n),
                    "entry_window_transitions": int(entry_n),
                    "boundary_abs_delta": boundary_abs,
                    "prev_tail_mean_abs_delta": prev_tail_mean,
                    "next_entry_mean_abs_delta": next_entry_mean,
                    "prev_segment_mean_abs_delta": prev_segment_mean,
                    "next_segment_mean_abs_delta": next_segment_mean,
                    "boundary_to_tail_ratio": boundary_to_tail,
                    "entry_to_tail_ratio": entry_to_tail,
                    "next_to_prev_segment_ratio": next_to_prev,
                    "boundary_to_global_ratio": boundary_to_global,
                    "global_mean_abs_delta": global_mean,
                    "global_median_abs_delta": global_median,
                    "visible_boundary_transition_index": visible_transition_index,
                    "visible_boundary_abs_delta": visible_transition_delta,
                    "visible_seam_window_max_abs_delta": visible_window_max,
                    "visible_seam_window_mean_abs_delta": visible_window_mean,
                    "visible_boundary_over_median_ratio": visible_boundary_over_median,
                    "visible_seam_over_median_ratio": visible_seam_over_median,
                    "visible_seam_over_mean_ratio": visible_seam_over_mean,
                    "visible_transition_rank": visible_transition_rank,
                    "visible_window_top_transition_rank": visible_window_top_rank,
                    "visible_boundary_score": visible_boundary_score,
                    "visible_rank_score": visible_rank_score,
                    "visible_seam_delta_score": visible_seam_delta_score,
                    "prev_tail_motion_center": list(prev_tail_center) if prev_tail_center is not None else None,
                    "next_entry_motion_center": list(next_entry_center) if next_entry_center is not None else None,
                    "tail_entry_centroid_distance": centroid_jump,
                    "tail_motion_vector": list(prev_vector) if prev_vector is not None else None,
                    "entry_motion_vector": list(next_vector) if next_vector is not None else None,
                    "tail_entry_direction_cosine": direction_cos,
                    "boundary_jump_score": boundary_jump_score,
                    "entry_acceleration_score": entry_acceleration_score,
                    "centroid_jump_score": centroid_jump_score,
                    "direction_switch_score": direction_switch_score,
                    "post_segment_pressure": post_segment_pressure,
                    "vector_seam_impulse_score": vector_seam_impulse_score,
                    "seam_impulse_score": seam_impulse_score,
                    "next_control_surface": next_surface,
                })

            worst = max(boundary_reviews, key=lambda item: float(item.get("seam_impulse_score", 0.0) or 0.0))
            max_score = float(worst.get("seam_impulse_score", 0.0) or 0.0)
            if max_score >= 0.52:
                status = "cascade_seam_impulse_high"
                severity = "WARNING"
                next_surface = "tail_next_source_strategy_continuity_report_only"
            elif max_score >= 0.30:
                status = "cascade_seam_impulse_watch"
                severity = "INFO"
                next_surface = "cascade_tail_entry_alignment_report_only"
            else:
                status = "cascade_seam_impulse_nominal"
                severity = "INFO"
                next_surface = "keep_observing_cascade_seam"

            rec = {
                "stage": "EventCascadeSeamImpulseReview",
                "status": status,
                "severity": severity,
                "review_version": "cascade_seam_impulse_review_v2_visible_delta_binder_report_only",
                "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
                "runtime_name": EVENT_HORIZON_RUNTIME_NAME,
                "control_mode": "REPORT_ONLY",
                "active_control_allowed": False,
                "same_run_control_allowed": False,
                "prompt_text_injection_allowed": False,
                "semantic_math_in_prompt_allowed": False,
                "observed_segments": len(segments),
                "frames_per_cascade_requested": int(frames_per_cascade) if frames_per_cascade is not None else None,
                "boundary_count": len(boundary_reviews),
                "worst_boundary_index": int(worst.get("boundary_index", 0) or 0),
                "seam_impulse_score": max_score,
                "vector_seam_impulse_score": worst.get("vector_seam_impulse_score", 0.0),
                "visible_seam_delta_score": worst.get("visible_seam_delta_score", 0.0),
                "visible_boundary_score": worst.get("visible_boundary_score", 0.0),
                "visible_rank_score": worst.get("visible_rank_score", 0.0),
                "visible_boundary_transition_index": worst.get("visible_boundary_transition_index", None),
                "visible_boundary_abs_delta": worst.get("visible_boundary_abs_delta", None),
                "visible_seam_window_max_abs_delta": worst.get("visible_seam_window_max_abs_delta", None),
                "visible_seam_window_mean_abs_delta": worst.get("visible_seam_window_mean_abs_delta", None),
                "visible_boundary_over_median_ratio": worst.get("visible_boundary_over_median_ratio", None),
                "visible_seam_over_median_ratio": worst.get("visible_seam_over_median_ratio", None),
                "visible_seam_over_mean_ratio": worst.get("visible_seam_over_mean_ratio", None),
                "visible_transition_rank": worst.get("visible_transition_rank", None),
                "visible_window_top_transition_rank": worst.get("visible_window_top_transition_rank", None),
                "global_mean_abs_delta": worst.get("global_mean_abs_delta", None),
                "global_median_abs_delta": worst.get("global_median_abs_delta", None),
                "boundary_jump_score": worst.get("boundary_jump_score", 0.0),
                "entry_acceleration_score": worst.get("entry_acceleration_score", 0.0),
                "centroid_jump_score": worst.get("centroid_jump_score", 0.0),
                "direction_switch_score": worst.get("direction_switch_score", 0.0),
                "tail_entry_direction_cosine": worst.get("tail_entry_direction_cosine", None),
                "tail_entry_centroid_distance": worst.get("tail_entry_centroid_distance", None),
                "boundary_to_tail_ratio": worst.get("boundary_to_tail_ratio", None),
                "entry_to_tail_ratio": worst.get("entry_to_tail_ratio", None),
                "boundary_reviews": boundary_reviews,
                "next_control_surface": next_surface,
                "formula": "Tail Outcome(previous segment) + visible boundary ObservedBehavior = Strategy(next segment) = post-continue Outcome. A high score means the next segment is visibly reborn instead of continuing the selected tail route.",
                "interpretation": (
                    "sharp cascade seam impulse: visible outcome and/or route vector says the next segment re-enters motion as a fresh Strategy birth"
                    if status == "cascade_seam_impulse_high"
                    else "watch cascade seam: tail and entry vectors are not fully aligned"
                    if status == "cascade_seam_impulse_watch"
                    else "cascade seam impulse is nominal in this observer pass"
                ),
                "next_action": "Use this report to design tail-next-source Strategy continuity; do not add damping until repeated fixed-seed evidence confirms the impulse class.",
            }
            records.append(rec)
            return rec
        except Exception as e:
            rec = {
                "stage": "EventCascadeSeamImpulseReview",
                "status": "failed",
                "severity": "WARNING",
                "error": str(e),
                "control_mode": "REPORT_ONLY",
                "active_control_allowed": False,
            }
            records.append(rec)
            return rec

    def _tail_next_source_strategy_continuity_proposal(self, segment_batches, concatenated_frames, records, frames_per_cascade=None):
        """
        R154 report-only continuity proposal.

        R153 names whether the seam is visible. R154 names what the next
        cascade would have to inherit from the selected/tail source to read as
        continuation instead of a fresh Strategy birth. No tensors are mutated.
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

            def ratio(numerator, denominator):
                numerator = safe_float(numerator, None)
                denominator = safe_float(denominator, None)
                if numerator is None or denominator is None or denominator <= 0:
                    return None
                return float(numerator / (denominator + 1e-12))

            def mean(values):
                values = [safe_float(v, None) for v in (values or [])]
                values = [v for v in values if v is not None]
                return float(sum(values) / len(values)) if values else None

            def median(values):
                values = [safe_float(v, None) for v in (values or [])]
                values = sorted(v for v in values if v is not None)
                if not values:
                    return None
                mid = len(values) // 2
                if len(values) % 2:
                    return float(values[mid])
                return float((values[mid - 1] + values[mid]) * 0.5)

            def latest_stage(stage_name):
                for record in reversed(records or []):
                    if isinstance(record, dict) and str(record.get("stage", "") or "") == stage_name:
                        return record
                return {}

            def to_frames(value):
                t = self._tensor_from_latent_like(value)
                if t is None:
                    return None
                t = torch.nan_to_num(t.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
                if t.dim() == 5:
                    if t.shape[-1] in (1, 3, 4):
                        t = t[0]
                    elif t.shape[1] in (1, 3, 4):
                        t = t[0].permute(1, 2, 3, 0)
                    else:
                        return None
                elif t.dim() == 4:
                    if t.shape[-1] in (1, 3, 4):
                        pass
                    elif t.shape[1] in (1, 3, 4):
                        t = t.permute(0, 2, 3, 1)
                    else:
                        return None
                else:
                    return None
                return t if t.shape[0] >= 1 else None

            def transition_maps(frames):
                if frames is None or frames.shape[0] < 2:
                    return None
                return (frames[1:] - frames[:-1]).abs().mean(dim=-1)

            def transition_magnitudes(motion):
                if motion is None or motion.numel() == 0:
                    return []
                values = motion.reshape(motion.shape[0], -1).mean(dim=1)
                return [float(x) for x in values.detach().cpu().tolist()]

            def frame_abs_delta(a, b):
                if a is None or b is None:
                    return None
                try:
                    return float((a.detach().float() - b.detach().float()).abs().reshape(-1).mean().item())
                except Exception:
                    return None

            segments = [to_frames(item) for item in (segment_batches or [])]
            segments = [item for item in segments if item is not None and item.shape[0] >= 1]
            if len(segments) < 2:
                rec = {
                    "stage": "EventTailNextSourceStrategyContinuityProposal",
                    "status": "not_applicable",
                    "severity": "INFO",
                    "reason": "requires_at_least_two_segments",
                    "observed_segments": len(segments),
                    "control_mode": "REPORT_ONLY",
                    "active_control_allowed": False,
                }
                records.append(rec)
                return rec

            seam = latest_stage("EventCascadeSeamImpulseReview")
            worst_boundary_index = int(seam.get("worst_boundary_index", 1) or 1) if isinstance(seam, dict) else 1
            worst_boundary_index = max(1, min(worst_boundary_index, len(segments) - 1))
            prev = segments[worst_boundary_index - 1]
            nxt = segments[worst_boundary_index]

            concat = to_frames(concatenated_frames)
            concat_values = transition_magnitudes(transition_maps(concat))
            global_median = median(concat_values)
            global_mean = mean(concat_values)

            prev_motion_values = transition_magnitudes(transition_maps(prev))
            next_motion_values = transition_magnitudes(transition_maps(nxt))
            tail_window_values = prev_motion_values[-min(6, len(prev_motion_values)):] if prev_motion_values else []
            entry_window_values = next_motion_values[:min(6, len(next_motion_values))] if next_motion_values else []
            tail_motion_mean = mean(tail_window_values)
            entry_motion_mean = mean(entry_window_values)
            entry_to_tail_ratio = ratio(entry_motion_mean, tail_motion_mean)

            source_frame_delta = frame_abs_delta(prev[-1], nxt[0])
            source_delta_over_median = ratio(source_frame_delta, global_median)
            source_delta_over_mean = ratio(source_frame_delta, global_mean)

            seam_score = clamp01(seam.get("seam_impulse_score", 0.0) if isinstance(seam, dict) else 0.0)
            vector_score = clamp01(seam.get("vector_seam_impulse_score", 0.0) if isinstance(seam, dict) else 0.0)
            visible_score = clamp01(seam.get("visible_seam_delta_score", 0.0) if isinstance(seam, dict) else 0.0)
            visible_ratio = safe_float(seam.get("visible_seam_over_median_ratio", None), None) if isinstance(seam, dict) else None
            direction_switch = clamp01(seam.get("direction_switch_score", 0.0) if isinstance(seam, dict) else 0.0)
            entry_accel = clamp01(seam.get("entry_acceleration_score", 0.0) if isinstance(seam, dict) else 0.0)

            source_gap_score = clamp01(((source_delta_over_median if source_delta_over_median is not None else 1.0) - 1.10) / 1.4)
            entry_ratio_score = clamp01(((entry_to_tail_ratio if entry_to_tail_ratio is not None else 1.0) - 1.0) / 1.2)
            continuity_pressure_score = clamp01(max(
                seam_score,
                visible_score,
                0.34 * source_gap_score + 0.26 * entry_ratio_score + 0.22 * direction_switch + 0.18 * entry_accel,
            ))

            if continuity_pressure_score >= 0.52:
                status = "tail_next_source_continuity_bridge_required"
                severity = "WARNING"
                next_surface = "tail_next_source_strategy_continuity_bridge_report_only"
            elif continuity_pressure_score >= 0.30:
                status = "tail_next_source_continuity_watch"
                severity = "INFO"
                next_surface = "tail_next_source_strategy_continuity_probe"
            else:
                status = "tail_next_source_continuity_nominal"
                severity = "INFO"
                next_surface = "keep_observing_tail_next_source"

            rec = {
                "stage": "EventTailNextSourceStrategyContinuityProposal",
                "status": status,
                "severity": severity,
                "proposal_version": "tail_next_source_strategy_continuity_v2_entry_window_bridge_route",
                "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
                "runtime_name": EVENT_HORIZON_RUNTIME_NAME,
                "control_mode": "REPORT_ONLY",
                "active_control_allowed": False,
                "same_run_control_allowed": False,
                "prompt_text_injection_allowed": False,
                "semantic_math_in_prompt_allowed": False,
                "observed_segments": len(segments),
                "frames_per_cascade_requested": int(frames_per_cascade) if frames_per_cascade is not None else None,
                "target_boundary_index": int(worst_boundary_index),
                "previous_segment": int(worst_boundary_index),
                "next_segment": int(worst_boundary_index + 1),
                "tail_source_frame_index": int(prev.shape[0] - 1),
                "next_entry_frame_index": 0,
                "tail_motion_mean_abs_delta": tail_motion_mean,
                "entry_motion_mean_abs_delta": entry_motion_mean,
                "entry_to_tail_motion_ratio": entry_to_tail_ratio,
                "source_frame_abs_delta": source_frame_delta,
                "source_delta_over_global_median": source_delta_over_median,
                "source_delta_over_global_mean": source_delta_over_mean,
                "global_median_abs_delta": global_median,
                "global_mean_abs_delta": global_mean,
                "cascade_seam_impulse_status": seam.get("status", "") if isinstance(seam, dict) else "",
                "cascade_seam_impulse_score": seam_score,
                "vector_seam_impulse_score": vector_score,
                "visible_seam_delta_score": visible_score,
                "visible_seam_over_median_ratio": visible_ratio,
                "source_gap_score": source_gap_score,
                "entry_ratio_score": entry_ratio_score,
                "direction_switch_score": direction_switch,
                "entry_acceleration_score": entry_accel,
                "continuity_pressure_score": continuity_pressure_score,
                "required_carriers": [
                    "selected_tail_frame_as_OutcomePrevious",
                    "tail_motion_window_as_ObservedBehaviorPrevious",
                    "first_next_source_frame_as_StrategyCarrier",
                    "visible_boundary_delta_as_ObservedBehaviorCurrent",
                    "post_continue_motion_window_as_OutcomeNext",
                    "background_identity_anchor_as_non_action_carrier",
                    "short_decayed_entry_latent_window_as_future_bridge_candidate",
                ],
                "bridge_policy": "report_only_prepare_tail_next_source_inheritance; active route should use bounded decayed entry-window latent bridge, no prompt rewrite",
                "next_control_surface": next_surface,
                "formula": "Selected tail Outcome(t-1) + tail motion ObservedBehavior(t-1) = Strategy(next source) = first next motion ObservedBehavior(t+1) + post-continue Outcome(t+1).",
                "interpretation": (
                    "next cascade should inherit selected tail source and tail motion before any active seam bridge is attempted"
                    if status == "tail_next_source_continuity_bridge_required"
                    else "tail-next-source continuity should be watched; evidence is not strong enough for a bridge route"
                    if status == "tail_next_source_continuity_watch"
                    else "tail-next-source continuity is nominal in this observer pass"
                ),
                "next_action": "If repeated live runs keep this high, test the SAFE selected-tail package first, then MAX_RISK_STRATEGY_RING only as an explicit A/B run; compare seam window rank and direction switch before promoting active defaults.",
            }
            records.append(rec)
            return rec
        except Exception as e:
            rec = {
                "stage": "EventTailNextSourceStrategyContinuityProposal",
                "status": "failed",
                "severity": "WARNING",
                "error": str(e),
                "control_mode": "REPORT_ONLY",
                "active_control_allowed": False,
            }
            records.append(rec)
            return rec

    def _cascade_seam_phase_classifier(self, segment_batches, concatenated_frames, records, frames_per_cascade=None):
        """
        R172 report-only seam phase classifier.

        R153/R154 say that the seam is high and what must be inherited.
        R172 asks why the seam is high before any new active mutation:
        prompt phase re-entry, latent carrier mismatch, background anchor
        conflict, center-action overdrive, or sampler handoff reset.
        """
        try:
            import torch

            def safe_float(value, default=0.0):
                try:
                    out = float(value)
                except Exception:
                    return default
                return out if math.isfinite(out) else default

            def clamp01(value):
                return max(0.0, min(1.0, safe_float(value, 0.0)))

            def ratio(numerator, denominator):
                numerator = safe_float(numerator, None)
                denominator = safe_float(denominator, None)
                if numerator is None or denominator is None or denominator <= 0:
                    return None
                return float(numerator / (denominator + 1e-12))

            def mean(values):
                values = [safe_float(v, None) for v in (values or [])]
                values = [v for v in values if v is not None]
                return float(sum(values) / len(values)) if values else None

            def latest_stage(stage_name):
                for record in reversed(records or []):
                    if isinstance(record, dict) and str(record.get("stage", "") or "") == stage_name:
                        return record
                return {}

            def latest_prefix(prefix):
                for record in reversed(records or []):
                    if isinstance(record, dict) and str(record.get("stage", "") or "").startswith(prefix):
                        return record
                return {}

            def to_frames(value):
                t = self._tensor_from_latent_like(value)
                if t is None:
                    return None
                t = torch.nan_to_num(t.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
                if t.dim() == 5:
                    if t.shape[-1] in (1, 3, 4):
                        t = t[0]
                    elif t.shape[1] in (1, 3, 4):
                        t = t[0].permute(1, 2, 3, 0)
                    else:
                        return None
                elif t.dim() == 4:
                    if t.shape[-1] in (1, 3, 4):
                        pass
                    elif t.shape[1] in (1, 3, 4):
                        t = t.permute(0, 2, 3, 1)
                    else:
                        return None
                else:
                    return None
                return t if t.shape[0] >= 1 else None

            def transition_maps(frames):
                if frames is None or frames.shape[0] < 2:
                    return None
                return (frames[1:] - frames[:-1]).abs().mean(dim=-1)

            def transition_magnitudes(motion):
                if motion is None or motion.numel() == 0:
                    return []
                values = motion.reshape(motion.shape[0], -1).mean(dim=1)
                return [float(x) for x in values.detach().cpu().tolist()]

            def slope(values):
                values = [safe_float(v, None) for v in (values or [])]
                values = [v for v in values if v is not None]
                if len(values) < 2:
                    return 0.0
                n = len(values)
                xs = list(range(n))
                mx = float(sum(xs) / n)
                my = float(sum(values) / n)
                denom = sum((x - mx) ** 2 for x in xs)
                if denom <= 1e-12:
                    return 0.0
                return float(sum((x - mx) * (y - my) for x, y in zip(xs, values)) / denom)

            def score_ratio(value, neutral=1.0, span=1.0):
                if value is None:
                    return 0.0
                return clamp01((safe_float(value, neutral) - neutral) / max(span, 1e-12))

            def nested(d, *keys, default=None):
                cur = d if isinstance(d, dict) else {}
                for key in keys:
                    if not isinstance(cur, dict):
                        return default
                    cur = cur.get(key, default)
                return cur

            segments = [to_frames(item) for item in (segment_batches or [])]
            segments = [item for item in segments if item is not None and item.shape[0] >= 1]
            if len(segments) < 2:
                rec = {
                    "stage": "EventCascadeSeamPhaseClassifier",
                    "status": "not_applicable",
                    "severity": "INFO",
                    "reason": "requires_at_least_two_segments",
                    "observed_segments": len(segments),
                    "control_mode": "REPORT_ONLY",
                    "active_control_allowed": False,
                }
                records.append(rec)
                return rec

            seam = latest_stage("EventCascadeSeamImpulseReview")
            tail = latest_stage("EventTailNextSourceStrategyContinuityProposal")
            pixel = latest_stage("EventPixelRegionMotionMapSelection") or latest_stage("EventPixelRegionMotionMap")
            action_background = latest_stage("EventActionBackgroundSeparationEvidence")
            pressure_pixel = latest_stage("EventPressurePixelReweightingProposal")
            bridge = latest_stage("EventSegmentEntryLatentMemoryBridge")
            prompt_phase = latest_stage("EventCascadePhasePromptTransform")
            prompt_runtime = latest_stage("EventCascadePromptRuntimeUpdate")

            boundary_index = int(safe_float(seam.get("worst_boundary_index", tail.get("target_boundary_index", 1)), 1))
            boundary_index = max(1, min(boundary_index, len(segments) - 1))
            prev = segments[boundary_index - 1]
            nxt = segments[boundary_index]
            prev_motion_values = transition_magnitudes(transition_maps(prev))
            next_motion_values = transition_magnitudes(transition_maps(nxt))
            tail_window = prev_motion_values[-min(8, len(prev_motion_values)):] if prev_motion_values else []
            entry_window = next_motion_values[:min(8, len(next_motion_values))] if next_motion_values else []
            tail_mean = mean(tail_window)
            entry_mean = mean(entry_window)
            entry_to_tail = ratio(entry_mean, tail_mean)
            tail_slope = slope(tail_window)
            entry_slope = slope(entry_window)
            entry_slope_to_tail_mean = ratio(abs(entry_slope), tail_mean)
            tail_slope_to_tail_mean = ratio(abs(tail_slope), tail_mean)
            entry_slope_pressure = score_ratio(entry_slope_to_tail_mean, neutral=0.03, span=0.18)
            tail_slope_pressure = score_ratio(tail_slope_to_tail_mean, neutral=0.03, span=0.18)

            seam_score = clamp01(seam.get("seam_impulse_score", 0.0))
            visible_score = clamp01(seam.get("visible_seam_delta_score", 0.0))
            vector_score = clamp01(seam.get("vector_seam_impulse_score", 0.0))
            visible_ratio = safe_float(seam.get("visible_seam_over_median_ratio", 1.0), 1.0)
            visible_rank = safe_float(seam.get("visible_window_top_transition_rank", 99.0), 99.0)
            visible_rank_pressure = clamp01((10.0 - visible_rank) / 9.0)
            boundary_jump = clamp01(seam.get("boundary_jump_score", 0.0))
            entry_accel = clamp01(seam.get("entry_acceleration_score", 0.0))
            direction_switch = clamp01(seam.get("direction_switch_score", 0.0))
            direction_cos = safe_float(seam.get("tail_entry_direction_cosine", 1.0), 1.0)
            source_gap = clamp01(tail.get("source_gap_score", 0.0))
            entry_ratio_score = clamp01(tail.get("entry_ratio_score", score_ratio(entry_to_tail, neutral=1.0, span=1.2)))
            continuity_pressure = clamp01(tail.get("continuity_pressure_score", seam_score))
            source_over_median = safe_float(tail.get("source_delta_over_global_median", 1.0), 1.0)

            bridge_controls = bridge.get("bridge_controls", {}) if isinstance(bridge, dict) else {}
            if not isinstance(bridge_controls, dict):
                bridge_controls = {}
            admissibility_guard = bridge_controls.get("admissibility_guard", {})
            if not isinstance(admissibility_guard, dict):
                admissibility_guard = {}
            concat_bridge = bridge.get("concat_latent_bridge", {}) if isinstance(bridge, dict) else {}
            if not isinstance(concat_bridge, dict):
                concat_bridge = {}
            raw_delta_abs_mean = safe_float(concat_bridge.get("raw_delta_abs_mean", 0.0), 0.0)
            bounded_delta_abs_mean = safe_float(concat_bridge.get("bounded_delta_abs_mean", 0.0), 0.0)
            bridge_guard_status = str(admissibility_guard.get("status", bridge.get("status", "")) if isinstance(bridge, dict) else "")
            hard_guard_pressure = 1.0 if "hard_guard" in bridge_guard_status or "strong" in bridge_guard_status else 0.0
            latent_delta_pressure = clamp01(raw_delta_abs_mean / 1.8) if raw_delta_abs_mean else 0.0
            bounded_delta_pressure = clamp01(bounded_delta_abs_mean / 0.035) if bounded_delta_abs_mean else 0.0

            pixel_center_edge = safe_float(pixel.get("center_edge_pixel_ratio", 1.0) if isinstance(pixel, dict) else 1.0, 1.0)
            pixel_edge_center = safe_float(pixel.get("edge_center_pixel_ratio", 0.0) if isinstance(pixel, dict) else 0.0, 0.0)
            pixel_background_leakage = clamp01(pixel.get("background_pixel_leakage_score", 0.0) if isinstance(pixel, dict) else 0.0)
            action_bg_ratio = safe_float(action_background.get("action_to_background_ratio", pixel_center_edge) if isinstance(action_background, dict) else pixel_center_edge, pixel_center_edge)
            background_leakage = clamp01(action_background.get("background_leakage_score", pixel_background_leakage) if isinstance(action_background, dict) else pixel_background_leakage)
            corrected_leakage = clamp01(pressure_pixel.get("corrected_background_leakage_score", background_leakage) if isinstance(pressure_pixel, dict) else background_leakage)
            background_factor = safe_float(pressure_pixel.get("bounded_background_pressure_factor", 1.0) if isinstance(pressure_pixel, dict) else 1.0, 1.0)
            seam_guard = clamp01(pressure_pixel.get("seam_protection_weight", 0.0) if isinstance(pressure_pixel, dict) else 0.0)

            prompt_phase_status = str(prompt_phase.get("status", "") if isinstance(prompt_phase, dict) else "")
            prompt_runtime_status = str(prompt_runtime.get("status", "") if isinstance(prompt_runtime, dict) else "")
            same_prompt_preserved = "skipped_same_prompt" in prompt_phase_status or "same_prompt" in prompt_phase_status
            prompt_runtime_reused_active = prompt_runtime_status == "reused_active_strategy"
            prompt_text_changed = bool(
                (prompt_phase.get("positive_prompt_changed", False) if isinstance(prompt_phase, dict) else False)
                or (prompt_runtime.get("positive_prompt_changed_from_previous_active", False) if isinstance(prompt_runtime, dict) else False)
                or (prompt_runtime.get("negative_prompt_changed_from_previous_active", False) if isinstance(prompt_runtime, dict) else False)
                or prompt_runtime_status == "applied"
            )
            prompt_change_score = clamp01(
                (0.70 if prompt_text_changed else 0.0)
                + (0.12 if prompt_runtime_status == "applied" else 0.0)
                + (0.08 if prompt_phase_status in ("applied", "already_current_phase") else 0.0)
            )
            semantic_reentry_base = 0.18 if same_prompt_preserved else 0.08
            if prompt_runtime_reused_active:
                semantic_reentry_base += 0.06
            if prompt_text_changed:
                semantic_reentry_base += 0.04
            prompt_reentry_base = semantic_reentry_base

            semantic_phase_reentry_score = clamp01(
                semantic_reentry_base
                + 0.30 * visible_score
                + 0.20 * entry_accel
                + 0.16 * entry_ratio_score
                + 0.14 * visible_rank_pressure
                + 0.10 * score_ratio(visible_ratio, neutral=1.35, span=1.2)
            )
            prompt_phase_reentry_score = clamp01(
                prompt_reentry_base
                + 0.30 * visible_score
                + 0.20 * entry_accel
                + 0.16 * entry_ratio_score
                + 0.14 * visible_rank_pressure
                + 0.10 * score_ratio(visible_ratio, neutral=1.35, span=1.2)
            )
            latent_carrier_mismatch_score = clamp01(
                0.28 * source_gap
                + 0.22 * hard_guard_pressure
                + 0.18 * boundary_jump
                + 0.16 * latent_delta_pressure
                + 0.10 * bounded_delta_pressure
                + 0.06 * score_ratio(source_over_median, neutral=1.25, span=1.5)
            )
            background_anchor_conflict_score = clamp01(
                0.30 * background_leakage
                + 0.22 * pixel_background_leakage
                + 0.18 * corrected_leakage
                + 0.12 * seam_guard
                + 0.10 * clamp01(1.0 - background_factor)
                + 0.08 * score_ratio(pixel_edge_center, neutral=0.20, span=0.70)
            )
            center_action_overdrive_score = clamp01(
                0.25 * entry_accel
                + 0.20 * entry_ratio_score
                + 0.18 * score_ratio(pixel_center_edge, neutral=1.4, span=3.0)
                + 0.15 * score_ratio(action_bg_ratio, neutral=2.0, span=20.0)
                + 0.12 * entry_slope_pressure
                + 0.10 * visible_score
            )
            sampler_handoff_reset_score = clamp01(
                0.25 * direction_switch
                + 0.22 * boundary_jump
                + 0.18 * vector_score
                + 0.15 * clamp01((0.60 - direction_cos) / 1.2)
                + 0.12 * source_gap
                + 0.08 * tail_slope_pressure
            )

            axis_scores = {
                "semantic_phase_reentry": semantic_phase_reentry_score,
                "prompt_text_change": prompt_change_score,
                "prompt_phase_reentry": prompt_phase_reentry_score,
                "latent_carrier_mismatch": latent_carrier_mismatch_score,
                "background_anchor_conflict": background_anchor_conflict_score,
                "center_action_overdrive": center_action_overdrive_score,
                "sampler_handoff_reset": sampler_handoff_reset_score,
            }
            ordered_axes = sorted(axis_scores.items(), key=lambda item: item[1], reverse=True)
            dominant_axis, dominant_score = ordered_axes[0]

            next_routes = {
                "semantic_phase_reentry": "cascade_semantic_phase_schedule_report_only",
                "prompt_text_change": "cascade_prompt_payload_change_review",
                "prompt_phase_reentry": "cascade_prompt_phase_schedule_report_only",
                "latent_carrier_mismatch": "selected_tail_latent_admissibility_report_only",
                "background_anchor_conflict": "boundary_background_anchor_report_only",
                "center_action_overdrive": "center_action_pressure_window_report_only",
                "sampler_handoff_reset": "same_sampler_strategy_handoff_probe",
            }
            if dominant_score >= 0.52:
                status = f"seam_phase_{dominant_axis}_high"
                severity = "WARNING"
            elif dominant_score >= 0.30:
                status = f"seam_phase_{dominant_axis}_watch"
                severity = "INFO"
            else:
                status = "seam_phase_nominal"
                severity = "INFO"

            rec = {
                "stage": "EventCascadeSeamPhaseClassifier",
                "status": status,
                "severity": severity,
                "classifier_version": "cascade_seam_phase_classifier_v1_report_only",
                "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
                "runtime_name": EVENT_HORIZON_RUNTIME_NAME,
                "control_mode": "REPORT_ONLY",
                "active_control_allowed": False,
                "same_run_control_allowed": False,
                "prompt_text_injection_allowed": False,
                "semantic_math_in_prompt_allowed": False,
                "observed_segments": len(segments),
                "frames_per_cascade_requested": int(frames_per_cascade) if frames_per_cascade is not None else None,
                "target_boundary_index": int(boundary_index),
                "dominant_axis": dominant_axis,
                "dominant_score": dominant_score,
                "axis_scores": axis_scores,
                "axis_order": [axis for axis, _score in ordered_axes],
                "axis_score_order": [{"axis": axis, "score": score} for axis, score in ordered_axes],
                "semantic_phase_reentry_score": semantic_phase_reentry_score,
                "same_prompt_global_phase_reentry_score": semantic_phase_reentry_score,
                "prompt_text_change_score": prompt_change_score,
                "seam_impulse_score": seam_score,
                "continuity_pressure_score": continuity_pressure,
                "visible_seam_delta_score": visible_score,
                "visible_seam_over_median_ratio": visible_ratio,
                "visible_window_top_transition_rank": visible_rank,
                "vector_seam_impulse_score": vector_score,
                "boundary_jump_score": boundary_jump,
                "entry_acceleration_score": entry_accel,
                "direction_switch_score": direction_switch,
                "tail_entry_direction_cosine": direction_cos,
                "source_gap_score": source_gap,
                "entry_ratio_score": entry_ratio_score,
                "source_delta_over_global_median": source_over_median,
                "tail_motion_mean_abs_delta": tail_mean,
                "entry_motion_mean_abs_delta": entry_mean,
                "entry_to_tail_motion_ratio": entry_to_tail,
                "tail_window_slope": tail_slope,
                "entry_window_slope": entry_slope,
                "tail_slope_pressure": tail_slope_pressure,
                "entry_slope_pressure": entry_slope_pressure,
                "prompt_phase_status": prompt_phase_status,
                "prompt_runtime_status": prompt_runtime_status,
                "same_prompt_preserved": bool(same_prompt_preserved),
                "prompt_runtime_reused_active_strategy": bool(prompt_runtime_reused_active),
                "prompt_text_changed_for_next_segment": bool(prompt_text_changed),
                "prompt_changed_for_next_segment": bool(prompt_text_changed),
                "bridge_status": bridge.get("status", "") if isinstance(bridge, dict) else "",
                "bridge_guard_status": bridge_guard_status,
                "bridge_raw_delta_abs_mean": raw_delta_abs_mean,
                "bridge_bounded_delta_abs_mean": bounded_delta_abs_mean,
                "pixel_center_edge_ratio": pixel_center_edge,
                "pixel_edge_center_ratio": pixel_edge_center,
                "pixel_background_leakage_score": pixel_background_leakage,
                "action_to_background_ratio": action_bg_ratio,
                "background_leakage_score": background_leakage,
                "corrected_background_leakage_score": corrected_leakage,
                "background_pressure_factor": background_factor,
                "seam_protection_weight": seam_guard,
                "candidate_next_route": next_routes.get(dominant_axis, "keep_observing_seam_phase"),
                "next_control_surface": next_routes.get(dominant_axis, "keep_observing_seam_phase"),
                "diagnostic_only": True,
                "formula": "Seam phase = prompt phase carrier + latent carrier + background anchor + center action + sampler handoff, all returning to one model-attractor Strategy before any active control.",
                "interpretation": (
                    "The prompt text is clean, but the same global event Strategy re-enters the next cascade; test semantic phase scheduling before latent strengthening."
                    if dominant_axis == "semantic_phase_reentry"
                    else "Runtime prompt text changed at Continue; review prompt payload identity before any latent or sampler control."
                    if dominant_axis == "prompt_text_change"
                    else
                    "Second cascade most likely re-enters a global prompt phase; test phase scheduling before stronger latent memory."
                    if dominant_axis == "prompt_phase_reentry"
                    else "Latent/source carriers are still too far apart; active bridge needs admissibility proof before more alpha."
                    if dominant_axis == "latent_carrier_mismatch"
                    else "Background/source anchor competes with action; use background-aware continuity evidence before motion control."
                    if dominant_axis == "background_anchor_conflict"
                    else "Center action pressure dominates the seam; future control should window action instead of damping the whole frame."
                    if dominant_axis == "center_action_overdrive"
                    else "Sampler handoff behaves like a reset; investigate same-sampler or handoff-state continuity before prompt changes."
                    if dominant_axis == "sampler_handoff_reset"
                    else "No strong seam phase class was proven."
                ),
                "next_action": "Use repeated fixed-seed R172 reports to pick one R173 active surface; do not strengthen R171 selected-tail echo until the dominant seam phase is stable.",
            }
            records.append(rec)
            return rec
        except Exception as e:
            rec = {
                "stage": "EventCascadeSeamPhaseClassifier",
                "status": "failed",
                "severity": "WARNING",
                "error": str(e),
                "control_mode": "REPORT_ONLY",
                "active_control_allowed": False,
            }
            records.append(rec)
            return rec

    def _cascade_semantic_phase_schedule_proposal(self, records, requested_segments=None):
        """
        R173 report-only semantic phase scheduler.

        R172 proved that a high seam can come from the same clean global
        Strategy being re-entered on the next cascade. This record does not
        rewrite prompt text; it names the next control surface.
        """
        if records is None:
            records = []

        def latest_stage(stage_name):
            for record in reversed(records or []):
                if isinstance(record, dict) and str(record.get("stage", "") or "") == str(stage_name):
                    return record
            return {}

        def safe_float(value, default=0.0):
            try:
                out = float(value)
            except Exception:
                return float(default)
            return out if math.isfinite(out) else float(default)

        def clamp01(value):
            return max(0.0, min(1.0, safe_float(value, 0.0)))

        phase = latest_stage("EventCascadeSeamPhaseClassifier")
        prompt_phase = latest_stage("EventCascadePhasePromptTransform")
        prompt_runtime = latest_stage("EventCascadePromptRuntimeUpdate")
        prompt_card = latest_stage("EventPromptCarrierContinuityCard")
        compiler = latest_stage("EventPromptStrategyCompiler")
        entry_bridge = latest_stage("EventSegmentEntryLatentMemoryBridge")

        if not isinstance(phase, dict) or not phase:
            rec = {
                "stage": "EventCascadeSemanticPhaseScheduleProposal",
                "status": "not_recorded",
                "severity": "INFO",
                "reason": "missing_seam_phase_classifier",
                "control_mode": "REPORT_ONLY",
                "active_control_allowed": False,
            }
            records.append(rec)
            return rec

        axis_scores = phase.get("axis_scores", {})
        if not isinstance(axis_scores, dict):
            axis_scores = {}
        semantic_score = clamp01(
            phase.get(
                "semantic_phase_reentry_score",
                axis_scores.get("semantic_phase_reentry", axis_scores.get("prompt_phase_reentry", 0.0)),
            )
        )
        prompt_text_change_score = clamp01(
            phase.get("prompt_text_change_score", axis_scores.get("prompt_text_change", 0.0))
        )
        latent_score = clamp01(axis_scores.get("latent_carrier_mismatch", 0.0))
        dominant_axis = str(phase.get("dominant_axis", "") or "")
        same_prompt_preserved = bool(phase.get("same_prompt_preserved", False))
        prompt_text_changed = bool(phase.get("prompt_text_changed_for_next_segment", False))
        prompt_clean = bool(
            same_prompt_preserved
            and not prompt_text_changed
            and str(prompt_card.get("status", "") if isinstance(prompt_card, dict) else "") in (
                "clean_same_strategy",
                "clean_strategy",
                "",
            )
        )
        try:
            segment_count = int(requested_segments or phase.get("observed_segments") or 1)
        except Exception:
            segment_count = 1
        segment_count = max(1, segment_count)

        schedule_required = bool(
            semantic_score >= 0.52
            and prompt_clean
            and dominant_axis in ("semantic_phase_reentry", "prompt_phase_reentry")
        )
        if schedule_required:
            status = "semantic_phase_schedule_required_report_only"
            severity = "WARNING"
            next_surface = "cascade_local_strategy_phase_windows"
        elif semantic_score >= 0.30:
            status = "semantic_phase_schedule_watch_report_only"
            severity = "INFO"
            next_surface = "collect_semantic_phase_evidence"
        else:
            status = "semantic_phase_schedule_not_needed"
            severity = "INFO"
            next_surface = "keep_current_prompt_route"

        phase_windows = []
        for index in range(1, segment_count + 1):
            if index == 1:
                role = "source_state_birth_and_initial_action"
            elif index == segment_count:
                role = "selected_tail_continuation_and_endpoint_return"
            else:
                role = "selected_tail_continuation_and_mid_action"
            phase_windows.append({
                "segment": int(index),
                "role": role,
                "model_facing_prompt_text": "unchanged",
                "strategy_contract": "inherit selected tail state, advance only the remaining event phase, and return to global Strategy",
            })

        graph = compiler.get("strategy_graph", {}) if isinstance(compiler, dict) else {}
        if not isinstance(graph, dict):
            graph = {}
        bridge_controls = entry_bridge.get("bridge_controls", {}) if isinstance(entry_bridge, dict) else {}
        if not isinstance(bridge_controls, dict):
            bridge_controls = {}
        bridge_phase_carrier = bridge_controls.get("semantic_phase_window_carrier", {})
        if not isinstance(bridge_phase_carrier, dict):
            bridge_phase_carrier = {}
        phase_alpha_policy = str(bridge_phase_carrier.get("alpha_policy", "") or "")
        active_phase_window_carrier = bool(
            bridge_phase_carrier.get("matches_segment", False)
            and phase_alpha_policy not in ("", "inactive")
            and not phase_alpha_policy.startswith("report_only")
            and bool(bridge_phase_carrier.get("tensor_control_allowed", True))
        )

        rec = {
            "stage": "EventCascadeSemanticPhaseScheduleProposal",
            "status": status,
            "severity": severity,
            "proposal_version": "semantic_phase_schedule_v5_r177_region_guard_baseline_restore",
            "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
            "runtime_name": EVENT_HORIZON_RUNTIME_NAME,
            "control_mode": "REPORT_ONLY",
            "active_control_allowed": bool(active_phase_window_carrier),
            "same_run_control_allowed": False,
            "prompt_text_injection_allowed": False,
            "semantic_math_in_prompt_allowed": False,
            "prompt_text_changed_for_next_segment": bool(prompt_text_changed),
            "prompt_runtime_status": str(prompt_runtime.get("status", "") if isinstance(prompt_runtime, dict) else ""),
            "prompt_phase_status": str(prompt_phase.get("status", "") if isinstance(prompt_phase, dict) else ""),
            "prompt_carrier_clean": bool(prompt_clean),
            "same_prompt_preserved": bool(same_prompt_preserved),
            "dominant_phase_axis": dominant_axis,
            "semantic_phase_reentry_score": semantic_score,
            "prompt_text_change_score": prompt_text_change_score,
            "latent_carrier_mismatch_score": latent_score,
            "requested_segments": int(segment_count),
            "phase_windows": phase_windows,
            "relation_complexity_score": graph.get("relation_complexity_score"),
            "action_pressure_score": graph.get("action_pressure_score"),
            "anchor_pressure_score": graph.get("anchor_pressure_score"),
            "free_model_reasoning_need_score": graph.get("free_model_reasoning_need_score"),
            "active_phase_window_carrier": bool(active_phase_window_carrier),
            "phase_window_carrier": bridge_phase_carrier,
            "entry_bridge_status": str(entry_bridge.get("status", "") if isinstance(entry_bridge, dict) else ""),
            "next_control_surface": next_surface,
            "candidate_active_surface_after_repeated_evidence": (
                "runtime_segment_strategy_window_without_prompt_text_injection"
                if schedule_required
                else "none"
            ),
            "formula": (
                "Same clean StrategyCandidate text can still re-enter as a whole global event. "
                "R173 separates text identity from semantic phase identity; R174 proved the non-text carrier route; "
                "R175 proved simple alpha dampening still touches the wrong surface; R177 restores the R173 region guard baseline and keeps the carrier as report evidence only."
            ),
            "interpretation": (
                "Do not treat the seam as prompt text corruption. Treat it as global event phase re-entry: "
                "the next cascade needs a local Strategy window while keeping the model-facing prompt clean."
                if schedule_required
                else "No strong semantic phase schedule requirement was proven in this run."
            ),
            "next_action": (
                "Unexpected active carrier in R177; review phase_window_carrier.tensor_control_allowed before trusting this run."
                if active_phase_window_carrier
                else "R177 keeps semantic phase carrier report-only and restores the R173 regional guard baseline; next active surface should be sampler-entry Strategy pressure or non-text per-segment phase scheduling."
                if schedule_required
                else "Keep observing; do not add active phase control from this single report."
            ),
        }
        records.append(rec)
        return rec

    def _event_dual_math_package_summary(self, records):
        """
        R158 package layer.

        SAFE package: reconstruct what the next source should inherit from the
        selected/tail frame, but never mutate tensors.

        MAX_RISK package: explicitly marks a route where the segment-entry
        latent bridge may override the hard guard with a tiny Strategy ring.
        """
        if records is None:
            records = []

        def latest_stage(stage_name):
            for record in reversed(records or []):
                if isinstance(record, dict) and str(record.get("stage", "") or "") == str(stage_name):
                    return record
            return {}

        def safe_float(value, default=0.0):
            try:
                out = float(value)
            except Exception:
                return float(default)
            return out if math.isfinite(out) else float(default)

        def clamp01(value):
            return max(0.0, min(1.0, safe_float(value, 0.0)))

        mode = str(getattr(self, "_event_math_control_mode", "OBSERVE_ONLY") or "OBSERVE_ONLY").upper()
        seam = latest_stage("EventCascadeSeamImpulseReview")
        tail = latest_stage("EventTailNextSourceStrategyContinuityProposal")

        continuity_pressure = clamp01(tail.get("continuity_pressure_score", 0.0) if isinstance(tail, dict) else 0.0)
        seam_score = clamp01(seam.get("seam_impulse_score", 0.0) if isinstance(seam, dict) else 0.0)
        visible_score = clamp01(tail.get("visible_seam_delta_score", seam.get("visible_seam_delta_score", 0.0) if isinstance(seam, dict) else 0.0) if isinstance(tail, dict) else 0.0)
        source_gap_score = clamp01(tail.get("source_gap_score", 0.0) if isinstance(tail, dict) else 0.0)
        entry_ratio_score = clamp01(tail.get("entry_ratio_score", 0.0) if isinstance(tail, dict) else 0.0)
        direction_switch = clamp01(tail.get("direction_switch_score", seam.get("direction_switch_score", 0.0) if isinstance(seam, dict) else 0.0) if isinstance(tail, dict) else 0.0)
        entry_accel = clamp01(tail.get("entry_acceleration_score", seam.get("entry_acceleration_score", 0.0) if isinstance(seam, dict) else 0.0) if isinstance(tail, dict) else 0.0)

        rebirth_risk = clamp01(max(
            continuity_pressure,
            seam_score,
            0.32 * source_gap_score + 0.26 * entry_ratio_score + 0.22 * direction_switch + 0.20 * entry_accel,
        ))
        source_inheritance_score = clamp01(1.0 - (0.65 * source_gap_score + 0.35 * visible_score))
        strategy_ring_pressure = clamp01(max(rebirth_risk, 0.55 * continuity_pressure + 0.45 * seam_score))

        dominant_axis = "source_tail_inheritance"
        if direction_switch >= max(source_gap_score, entry_ratio_score, visible_score):
            dominant_axis = "direction_switch"
        elif entry_ratio_score >= max(source_gap_score, visible_score):
            dominant_axis = "entry_motion_ratio"
        elif source_gap_score >= visible_score:
            dominant_axis = "source_identity_gap"
        elif visible_score > 0.0:
            dominant_axis = "visible_boundary_delta"

        safe_active = mode == "SELECTED_TAIL_SOURCE_RECONSTRUCTION"
        safe_status = (
            "safe_package_active_report_only"
            if safe_active
            else "safe_package_available_inactive"
        )
        safe_rec = {
            "stage": "EventSelectedTailSourceReconstructionPackage",
            "status": safe_status,
            "severity": "INFO" if rebirth_risk < 0.52 else "WARNING",
            "package": "SAFE",
            "package_version": "selected_tail_source_reconstruction_safe_v1",
            "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
            "runtime_name": EVENT_HORIZON_RUNTIME_NAME,
            "mode": mode,
            "control_mode": "REPORT_ONLY",
            "active_control_allowed": False,
            "active_tensor_mutation_applied": False,
            "prompt_text_injection_allowed": False,
            "semantic_math_in_prompt_allowed": False,
            "observed_tail_recorded": bool(tail),
            "observed_seam_recorded": bool(seam),
            "dominant_axis": dominant_axis,
            "rebirth_risk_score": float(rebirth_risk),
            "source_inheritance_score": float(source_inheritance_score),
            "continuity_pressure_score": float(continuity_pressure),
            "cascade_seam_impulse_score": float(seam_score),
            "visible_seam_delta_score": float(visible_score),
            "source_gap_score": float(source_gap_score),
            "entry_ratio_score": float(entry_ratio_score),
            "direction_switch_score": float(direction_switch),
            "entry_acceleration_score": float(entry_accel),
            "strategy_package_role": (
                "tell the next segment what source/tail state should be preserved before active control is attempted"
            ),
            "next_control_surface": (
                "max_risk_strategy_ring"
                if rebirth_risk >= 0.52
                else "keep_selected_tail_source_reconstruction_report_only"
            ),
            "formula": (
                "SAFE: selected tail Outcome(t-1) + tail motion ObservedBehavior(t-1) is reconstructed as a "
                "next-source StrategyCarrier report. No sampler, latent, prompt, CFG, or tensor route is changed."
            ),
        }
        records.append(safe_rec)

        max_risk_active = mode == "MAX_RISK_STRATEGY_RING"
        max_risk_rec = {
            "stage": "EventMaxRiskStrategyRingPackage",
            "status": "max_risk_package_active" if max_risk_active else "max_risk_package_available_inactive",
            "severity": "DANGER" if max_risk_active else "WARNING",
            "package": "MAX_RISK",
            "package_version": "max_risk_strategy_ring_v1",
            "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
            "runtime_name": EVENT_HORIZON_RUNTIME_NAME,
            "mode": mode,
            "control_mode": "ACTIVE_RESEARCH" if max_risk_active else "REPORT_ONLY",
            "active_control_allowed": bool(max_risk_active),
            "active_tensor_mutation_possible": bool(max_risk_active),
            "hard_guard_override_allowed": bool(max_risk_active),
            "prompt_text_injection_allowed": False,
            "semantic_math_in_prompt_allowed": False,
            "dominant_axis": dominant_axis,
            "strategy_ring_pressure_score": float(strategy_ring_pressure),
            "rebirth_risk_score": float(rebirth_risk),
            "source_inheritance_score": float(source_inheritance_score),
            "continuity_pressure_score": float(continuity_pressure),
            "cascade_seam_impulse_score": float(seam_score),
            "visible_seam_delta_score": float(visible_score),
            "source_gap_score": float(source_gap_score),
            "entry_ratio_score": float(entry_ratio_score),
            "direction_switch_score": float(direction_switch),
            "entry_acceleration_score": float(entry_accel),
            "active_mutation_surface": "EventSegmentEntryLatentMemoryBridge",
            "override_policy": (
                "MAX_RISK_STRATEGY_RING may apply a tiny decayed entry-window latent bridge even when the hard guard "
                "would normally make the bridge report-only. Use only for deliberate A/B research."
            ),
            "expected_failure_modes": [
                "green/color cast",
                "background noise bloom",
                "identity shimmer",
                "direction reversal",
                "anatomy or object-state overcorrection",
            ],
            "formula": (
                "MAX_RISK: previous latent OutcomePrevious is forced back into next Wan latent entry as a tiny Strategy ring. "
                "This can help continuity but may directly perturb the model's denoise route."
            ),
        }
        records.append(max_risk_rec)
        return {"safe": safe_rec, "max_risk": max_risk_rec}

    # _dual_branch_delta_coupling_math fully excised (physical cut #21): removed the observer-only smart alignment
    # and energy scoring layer on top of the raw high/low deltas. The dual-branch now interacts without this interpretive comfort math.

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

    def _event_semantic_relation_pressure_router(self, records=None, mode="OBSERVE_ONLY", field_mode="OFF"):
        def clamp01(value):
            try:
                out = float(value)
            except Exception:
                out = 0.0
            if not math.isfinite(out):
                out = 0.0
            return max(0.0, min(1.0, out))

        def clamp(value, low, high):
            try:
                out = float(value)
            except Exception:
                out = 0.0
            if not math.isfinite(out):
                out = 0.0
            return max(float(low), min(float(high), out))

        def latest_record(stage_name):
            for item in reversed(records or []):
                if isinstance(item, dict) and str(item.get("stage", "") or "") == stage_name:
                    return item
            return {}

        compiler = latest_record("EventPromptStrategyCompiler")
        prompt_apply = latest_record("EventPromptStrategyTranscodeApply")
        prompt_lock = latest_record("EventPromptPurityLock")
        runtime_update = latest_record("EventCascadePromptRuntimeUpdate")

        runtime_density_map = runtime_update.get("semantic_density_context_map", {}) if isinstance(runtime_update, dict) else {}
        prompt_apply_density_map = prompt_apply.get("semantic_density_context_map", {}) if isinstance(prompt_apply, dict) else {}
        compiler_density_map = compiler.get("semantic_density_context_map", {}) if isinstance(compiler, dict) else {}

        if (
            isinstance(runtime_density_map, dict)
            and runtime_density_map
            and any(k in runtime_density_map for k in ("meaning_density_score", "context_density_score", "density_context_balance_score"))
        ):
            density_map = runtime_density_map or {}
            active_prompt_source = "runtime_prompt_update"
        elif (
            isinstance(prompt_apply_density_map, dict)
            and prompt_apply_density_map
            and any(k in prompt_apply_density_map for k in ("meaning_density_score", "context_density_score", "density_context_balance_score"))
        ):
            density_map = prompt_apply_density_map or {}
            active_prompt_source = "prompt_transcode_apply"
        elif isinstance(compiler_density_map, dict):
            density_map = compiler_density_map or {}
            active_prompt_source = "prompt_strategy_compiler"
        else:
            density_map = {}
            active_prompt_source = "not_available"

        strategy_graph = compiler.get("strategy_graph", {}) if isinstance(compiler, dict) else {}
        if not isinstance(strategy_graph, dict):
            strategy_graph = {}
        topology_map = compiler.get("object_topology_map", {}) if isinstance(compiler, dict) else {}
        if not isinstance(topology_map, dict):
            topology_map = {}
        relation_ontology = compiler.get("object_relation_ontology", {}) if isinstance(compiler, dict) else {}
        if not isinstance(relation_ontology, dict):
            relation_ontology = {}

        relation_complexity = clamp01(strategy_graph.get("relation_complexity_score", 0.0))
        action_pressure = clamp01(strategy_graph.get("action_pressure_score", 0.0))
        anchor_pressure = clamp01(strategy_graph.get("anchor_pressure_score", 0.0))
        semantic_conflict = clamp01(strategy_graph.get("semantic_conflict_score", 0.0))
        collapse_risk = clamp01(strategy_graph.get("collapse_risk_score", 0.0))
        topology_pressure = clamp01(
            runtime_update.get("object_topology_pressure_score", topology_map.get("topology_pressure_score", density_map.get("object_topology_pressure_score", 0.0)))
            if isinstance(runtime_update, dict)
            else topology_map.get("topology_pressure_score", density_map.get("object_topology_pressure_score", 0.0))
        )
        contact_pressure = clamp01(topology_map.get("contact_pressure_score", 0.0))
        rigidity_pressure = clamp01(topology_map.get("rigidity_confidence_score", 0.0))
        flexibility_pressure = clamp01(topology_map.get("flexibility_pressure_score", 0.0))
        meaning_density = clamp01(density_map.get("meaning_density_score", 0.0))
        context_density = clamp01(density_map.get("context_density_score", 0.0))
        balance_score = clamp01(density_map.get("density_context_balance_score", 0.0))
        semantic_gap = clamp01(1.0 - balance_score)

        relation_status = str(
            (runtime_update.get("object_relation_ontology_status", "") if isinstance(runtime_update, dict) else "")
            or (prompt_apply.get("object_relation_ontology_status", "") if isinstance(prompt_apply, dict) else "")
            or relation_ontology.get("status", "")
            or "absent"
        )
        relation_active = relation_status == "active"
        rigidity_lock = bool(
            (runtime_update.get("rigidity_lock_recommended", False) if isinstance(runtime_update, dict) else False)
            or topology_map.get("rigidity_lock_recommended", False)
        )
        contact_depth_axis = bool(
            (runtime_update.get("contact_depth_axis_recommended", False) if isinstance(runtime_update, dict) else False)
            or topology_map.get("contact_depth_axis_recommended", False)
        )

        prompt_clean = bool(
            (not prompt_lock or prompt_lock.get("prompt_purity_lock", True))
            and not bool(prompt_lock.get("prompt_text_injection_allowed", False))
            and not bool(prompt_lock.get("semantic_math_in_prompt_allowed", False))
            and not bool(prompt_apply.get("prompt_text_injection_allowed", False))
            and not bool(prompt_apply.get("semantic_math_in_prompt_allowed", False))
        )
        mode = str(mode or "OBSERVE_ONLY").upper()
        field_mode = str(field_mode or "OFF").upper()
        active_mode = mode == "STRATEGY_PRESSURE_WINDOW"
        field_high_allowed = field_mode in ("HIGH_NOISE_FIELD", "DUAL_FIELD")
        field_low_allowed = field_mode in ("LOW_REFINEMENT_FIELD", "DUAL_FIELD")

        relation_pressure = clamp01(
            0.22 * relation_complexity
            + 0.22 * action_pressure
            + 0.18 * topology_pressure
            + 0.14 * (1.0 if relation_active else 0.0)
            + 0.12 * semantic_gap
            + 0.12 * anchor_pressure
        )
        stability_guard = clamp01(max(collapse_risk, semantic_conflict, flexibility_pressure * 0.70, semantic_gap * 0.50))
        high_base_intent = clamp(
            (0.0022 * relation_pressure + 0.0008 * action_pressure + 0.0004 * topology_pressure)
            * (1.0 - 0.35 * stability_guard),
            0.0,
            0.0030,
        )
        low_base_intent = clamp(
            (0.0016 * topology_pressure + 0.0007 * contact_pressure + 0.0006 * relation_pressure + 0.0004 * rigidity_pressure)
            * (1.0 - 0.45 * collapse_risk),
            0.0,
            0.0025,
        )
        high_intent_multiplier = clamp(1.0 + 0.035 * relation_pressure - 0.070 * stability_guard, 0.92, 1.04)
        low_intent_multiplier = clamp(1.0 + 0.030 * topology_pressure - 0.090 * max(collapse_risk, semantic_gap), 0.90, 1.035)
        high_window_multiplier = clamp(1.0 - 0.030 * stability_guard + 0.015 * action_pressure, 0.92, 1.02)
        low_window_multiplier = clamp(1.0 - 0.050 * max(collapse_risk, semantic_gap) + 0.010 * rigidity_pressure, 0.88, 1.01)

        def branch_route(branch_key, allowed, base_intent, intent_multiplier, window_multiplier):
            branch_active = bool(active_mode and allowed and prompt_clean and relation_pressure > 0.0)
            return {
                "branch_key": branch_key,
                "status": "active_candidate" if branch_active else "report_only",
                "active_control_allowed": branch_active,
                "base_pressure_intent": float(base_intent if branch_active else 0.0),
                "intent_multiplier": float(intent_multiplier if branch_active else 1.0),
                "window_multiplier": float(window_multiplier if branch_active else 1.0),
                "formula_role": (
                    "relation pressure births model-readable direction before low refinement"
                    if branch_key == "high"
                    else "relation pressure preserves carrier continuity during low refinement"
                ),
            }

        high_route = branch_route("high", field_high_allowed, high_base_intent, high_intent_multiplier, high_window_multiplier)
        low_route = branch_route("low", field_low_allowed, low_base_intent, low_intent_multiplier, low_window_multiplier)

        status = "active_candidate" if (high_route["active_control_allowed"] or low_route["active_control_allowed"]) else "report_only"
        return {
            "stage": "EventSemanticRelationPressureRouter",
            "status": status,
            "version": "semantic_relation_pressure_router_v1_prompt_purity_control_surface",
            "mode": mode,
            "strategy_field_mode": field_mode,
            "active_prompt_source": active_prompt_source,
            "parent_strategy": "S_global_event_route",
            "prompt_purity_lock": bool(prompt_clean),
            "prompt_text_injection_allowed": False,
            "semantic_math_in_prompt_allowed": False,
            "input_pressure": {
                "relation_complexity": float(relation_complexity),
                "action_pressure": float(action_pressure),
                "anchor_pressure": float(anchor_pressure),
                "semantic_conflict": float(semantic_conflict),
                "collapse_risk": float(collapse_risk),
                "object_topology_pressure": float(topology_pressure),
                "contact_pressure": float(contact_pressure),
                "rigidity_pressure": float(rigidity_pressure),
                "flexibility_pressure": float(flexibility_pressure),
                "meaning_density": float(meaning_density),
                "context_density": float(context_density),
                "density_context_balance": float(balance_score),
                "semantic_gap": float(semantic_gap),
                "relation_pressure": float(relation_pressure),
                "stability_guard": float(stability_guard),
            },
            "relation_evidence": {
                "object_relation_ontology_status": relation_status,
                "object_relation_active": bool(relation_active),
                "rigidity_lock_recommended": bool(rigidity_lock),
                "contact_depth_axis_recommended": bool(contact_depth_axis),
                "balance_axis": str(density_map.get("balance_axis", "") or ""),
            },
            "branch_routes": {
                "high": high_route,
                "low": low_route,
                "default": {
                    "branch_key": "default",
                    "status": "report_only",
                    "active_control_allowed": False,
                    "base_pressure_intent": 0.0,
                    "intent_multiplier": 1.0,
                    "window_multiplier": 1.0,
                    "formula_role": "generic route remains report-only",
                },
            },
            "activation_rule": "Only STRATEGY_PRESSURE_WINDOW plus a matching StrategyField mode can turn semantic relation pressure into a tiny bounded numeric pressure. OBSERVE_ONLY and LATENT_DELTA_SCALE stay report-only here.",
            "formula": "Prompt/topology relation pressure is read as local Strategy evidence, routed into high/low control surfaces, and returned to S_global_event_route without adding words to the model-facing prompt.",
        }

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

        safe_report_package_mode = mode == "SELECTED_TAIL_SOURCE_RECONSTRUCTION"
        active_mode = mode in ("LATENT_DELTA_SCALE", "STRATEGY_PRESSURE_WINDOW", "LATENT_MEMORY_BRIDGE", "PRESSURE_PIXEL_REWEIGHTING", "SOURCE_NOISE_FIELD_SHAPING", "MAX_RISK_STRATEGY_RING", "DEEP_STEP_DELTA_CONTROL")
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
        elif mode == "SELECTED_TAIL_SOURCE_RECONSTRUCTION":
            path = "selected_tail_source_reconstruction_safe"
            policy = "safe_report_only_tail_source_reconstruction"
            active_control_allowed = False
            model_native_sampler_preserved = True
            cfg_preserved = True
        elif mode == "SOURCE_NOISE_FIELD_SHAPING":
            path = "source_noise_birth_shaping"
            policy = "tiny_pre_high_source_noise_spatial_gain"
            active_control_allowed = True
            model_native_sampler_preserved = True
            cfg_preserved = True
        elif mode == "MAX_RISK_STRATEGY_RING":
            path = "max_risk_strategy_ring"
            policy = "explicit_max_risk_latent_entry_guard_override"
            active_control_allowed = True
            model_native_sampler_preserved = True
            cfg_preserved = True
        elif mode == "PRESSURE_PIXEL_REWEIGHTING":
            path = "r151_pressure_pixel_feathered_identity_guard_candidate"
            policy = "r151_pressure_pixel_feathered_identity_guard_low_branch_candidate"
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
        semantic_relation_router = self._event_semantic_relation_pressure_router(
            records,
            mode=mode,
            field_mode=field_mode,
        )

        plan = {
            "stage": "EventStrategyControlSurfacePlan",
            "status": "active" if active_mode else ("safe_report_package" if safe_report_package_mode else "observe_only"),
            "version": "strategy_control_surface_v10_dual_math_packages",
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
            "spatial_strategy_return": {
                "status": "available" if mode == "STRATEGY_PRESSURE_WINDOW" else "inactive",
                "version": "spatial_strategy_return_v1",
                "policy": "region_role_pressure_returns_to_model_attractor",
                "prompt_text_injection_allowed": False,
                "semantic_math_in_prompt_allowed": False,
                "active_requires": "STRATEGY_PRESSURE_WINDOW plus active StrategyField and source/background/tail/semantic pressure evidence",
                "formula": "Each incoming carrier is read as a local Strategy intersection: center/action, edge/background anchor, and lower/temporal tail. Their pressure is folded back into one model-attractor route instead of spreading as scene-wide motion.",
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
                "version": "noise_source_field_map_v10_R171_regional_tail_guard",
                "policy": "read latent source/noise pressure; active only in SOURCE_NOISE_FIELD_SHAPING before high sampler",
                "active_control_allowed": bool(mode == "SOURCE_NOISE_FIELD_SHAPING"),
                "future_active_candidate": "pre-high source/noise shaping after enough report evidence",
                "prompt_text_injection_allowed": False,
                "semantic_math_in_prompt_allowed": False,
                "formula": "The source/noise field is read as Outcome(t-1)+ObservedBehavior(t-1) evidence before deciding whether any Strategy pressure should become active math.",
            },
            "noise_field_strategy_bridge": {
                "status": "available",
                "version": "noise_field_strategy_bridge_v10_R171_model_attractor_regional_tail_guard",
                "policy": "route source/noise evidence to the next denoise-safe Strategy surface; R171 may apply tiny gain everywhere, additive source-image carrier only on anchor/periphery zones, and a tiny two-slice post-drop seam-entry echo with selected-tail regional/background guard before high sampler",
                "active_control_allowed": bool(mode == "SOURCE_NOISE_FIELD_SHAPING"),
                "future_active_candidate": "pre-high source/noise shaping or low mid-window refinement only",
                "prompt_text_injection_allowed": False,
                "semantic_math_in_prompt_allowed": False,
                "formula": "NoiseFieldStrategyBridge reads the source/noise field as a bridge from Outcome+ObservedBehavior back to StrategyAttractor(model). Empty fields become no-pressure; non-empty fields name the safe future surface instead of mutating the current post-window delta.",
            },
            "source_noise_birth_shaping": {
                "status": "available" if mode == "SOURCE_NOISE_FIELD_SHAPING" else "inactive",
                "version": "source_noise_birth_shaping_v6_anchor_only_additive_carrier",
                "policy": "tiny low-frequency source/noise spatial gain with anchor-only additive source carrier on Wan latent seed and Wan positive concat conditioning before high sampler",
                "active_control_allowed": bool(mode == "SOURCE_NOISE_FIELD_SHAPING"),
                "branch_delta_overlay_allowed": False,
                "prompt_text_injection_allowed": False,
                "semantic_math_in_prompt_allowed": False,
                "formula": (
                    "Wan latent seed and Wan positive concat conditioning are read as source/noise StrategyCarriers before high sampler. "
                    "Only tiny feathered spatial gain and anchor-only additive source carrier pressure may be applied; central microdetail remains model-native, and sampler, CFG, prompt text, and hard tail bridge stay model-native."
                ),
            },
            "selected_tail_source_reconstruction_safe": {
                "status": "available" if mode == "SELECTED_TAIL_SOURCE_RECONSTRUCTION" else "inactive",
                "version": "selected_tail_source_reconstruction_safe_v1_report_only",
                "package": "SAFE",
                "policy": "reconstruct selected/tail source inheritance as evidence without tensor mutation",
                "active_control_allowed": False,
                "branch_delta_overlay_allowed": False,
                "prompt_text_injection_allowed": False,
                "semantic_math_in_prompt_allowed": False,
                "formula": (
                    "selected tail OutcomePrevious + tail motion ObservedBehaviorPrevious = next-source StrategyCarrier. "
                    "This package reports whether the next segment should inherit source/tail state before any active bridge exists."
                ),
            },
            "max_risk_strategy_ring": {
                "status": "available" if mode == "MAX_RISK_STRATEGY_RING" else "inactive",
                "version": "max_risk_strategy_ring_v1_explicit_research",
                "package": "MAX_RISK",
                "policy": "explicitly allow a tiny hard-guard override in segment-entry latent memory bridge",
                "active_control_allowed": bool(mode == "MAX_RISK_STRATEGY_RING"),
                "branch_delta_overlay_allowed": False,
                "hard_guard_override_allowed": bool(mode == "MAX_RISK_STRATEGY_RING"),
                "prompt_text_injection_allowed": False,
                "semantic_math_in_prompt_allowed": False,
                "formula": (
                    "The previous latent tail is forced back into the next segment entry as a tiny Strategy ring. "
                    "This is intentionally maximum-risk research: it may improve continuity or create color/noise/identity artifacts."
                ),
            },
            "pressure_pixel_reweighting_active_candidate": {
                "status": "available" if mode == "PRESSURE_PIXEL_REWEIGHTING" else "inactive",
                "version": "pressure_pixel_reweighting_active_candidate_v4_feathered_identity_guard",
                "policy": "R151 pressure-pixel candidate keeps the R149/R150 quality guard, protects identity carriers, and feathers local spatial gain",
                "active_control_allowed": bool(mode == "PRESSURE_PIXEL_REWEIGHTING"),
                "high_branch_passthrough": True,
                "low_branch_only": True,
                "same_run_oracle": False,
                "prompt_text_injection_allowed": False,
                "semantic_math_in_prompt_allowed": False,
                "source_evidence": "R148 deterministic A/B plus R149 human visual grain verdict plus R150 top/face pixel artifact",
                "quality_guard": {
                    "status": "active",
                    "quality_guard_factor": 0.50,
                    "candidate_cap": 0.0020,
                    "policy": "reduce whole-low pressure before any future spatial expansion",
                },
                "local_spatial_pressure_guard": {
                    "status": "available" if mode == "PRESSURE_PIXEL_REWEIGHTING" else "inactive",
                    "source": "R149 mega-check visible Outcome + R150 live pixel artifact; no same-run oracle",
                    "policy": "preserve upper identity/action pressure while feathering edge/background delta",
                    "active_control_allowed": bool(mode == "PRESSURE_PIXEL_REWEIGHTING"),
                },
                "formula": "Observed R148/R149/R150 pixel Outcome and pressure disagreement become Outcome(t-1)+ObservedBehavior(t-1) evidence for a smaller, feathered low refinement Strategy. The model remains the attractor; the branch gets only a tiny quality-guarded strength delta with protected identity carriers.",
            },
            "segment_entry_latent_memory_bridge": {
                "status": "available" if mode in ("LATENT_MEMORY_BRIDGE", "MAX_RISK_STRATEGY_RING", "SOURCE_NOISE_FIELD_SHAPING") else "inactive",
                "version": "segment_entry_latent_memory_bridge_v12_regional_tail_guard",
                "policy": "previous latent tail returns to next segment entry before high sampler; max-risk mode can override the hard guard with tiny alpha; SOURCE_NOISE_FIELD_SHAPING uses a tiny SOURCE_NOISE_MICRO_CONCAT_ONLY two-slice guard override with selected-tail source decay floor",
                "active_control_allowed": bool(mode in ("LATENT_MEMORY_BRIDGE", "MAX_RISK_STRATEGY_RING", "SOURCE_NOISE_FIELD_SHAPING")),
                "max_risk_guard_override_allowed": bool(mode == "MAX_RISK_STRATEGY_RING"),
                "source_noise_micro_concat_only_guard_override_allowed": bool(mode == "SOURCE_NOISE_FIELD_SHAPING"),
                "branch_delta_overlay_allowed": False,
                "controls": {
                    "wan_alpha": float((bridge_controls or {}).get("wan_alpha", 0.10)),
                    "concat_alpha": float((bridge_controls or {}).get("concat_alpha", 0.06)),
                    "wan_max_step": float((bridge_controls or {}).get("wan_max_step", 0.45)),
                    "concat_max_step": float((bridge_controls or {}).get("concat_max_step", 0.28)),
                },
                "prompt_text_injection_allowed": False,
                "semantic_math_in_prompt_allowed": False,
                "formula": "previous latent OutcomePrevious + explicit bounded memory delta = next StrategyCarrier before high sampler; SOURCE_NOISE_FIELD_SHAPING uses concat_latent_image only, while LATENT_MEMORY_BRIDGE/MAX_RISK can still test Wan latent entry.",
            },
            "semantic_relation_pressure_router": semantic_relation_router,
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
                "safe_report_package_mode": safe_report_package_mode,
                "semantic_relation_router": {
                "status": semantic_relation_router.get("status", ""),
                "source": semantic_relation_router.get("active_prompt_source", ""),
                "input_pressure": semantic_relation_router.get("input_pressure", {}),
                "branch_routes": semantic_relation_router.get("branch_routes", {}),
            },
        }), sort_keys=True, ensure_ascii=True)
        if records is not None and getattr(self, "_event_strategy_control_surface_plan_signature", "") != signature:
            records.append(semantic_relation_router)
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

        field_mode = str(plan.get("strategy_field_mode", "OFF") or "OFF").upper()
        field_branch_allowed = (
            field_mode == "DUAL_FIELD"
            or (field_mode == "HIGH_NOISE_FIELD" and branch_key == "high")
            or (field_mode == "LOW_REFINEMENT_FIELD" and branch_key == "low")
        )
        field_reportable = field_mode in ("REPORT_ONLY", "HIGH_NOISE_FIELD", "LOW_REFINEMENT_FIELD", "DUAL_FIELD")
        field_active = bool(field_branch_allowed and mode in ("LATENT_DELTA_SCALE", "STRATEGY_PRESSURE_WINDOW", "DEEP_STEP_DELTA_CONTROL"))

        semantic_router = plan.get("semantic_relation_pressure_router", {}) if isinstance(plan.get("semantic_relation_pressure_router", {}), dict) else {}
        semantic_branch_route = (
            (semantic_router.get("branch_routes", {}) or {}).get(branch_key, {})
            if isinstance(semantic_router.get("branch_routes", {}), dict)
            else {}
        )
        semantic_router_applied = False
        semantic_base_pressure_intent = 0.0
        semantic_intent_multiplier = 1.0
        semantic_window_multiplier = 1.0
        if (
            mode == "STRATEGY_PRESSURE_WINDOW"
            and field_active
            and isinstance(semantic_branch_route, dict)
            and bool(semantic_branch_route.get("active_control_allowed", False))
        ):
            try:
                semantic_base_pressure_intent = float(semantic_branch_route.get("base_pressure_intent", 0.0) or 0.0)
            except Exception:
                semantic_base_pressure_intent = 0.0
            try:
                semantic_intent_multiplier = float(semantic_branch_route.get("intent_multiplier", 1.0) or 1.0)
            except Exception:
                semantic_intent_multiplier = 1.0
            try:
                semantic_window_multiplier = float(semantic_branch_route.get("window_multiplier", 1.0) or 1.0)
            except Exception:
                semantic_window_multiplier = 1.0
            semantic_base_pressure_intent = max(-0.0030, min(0.0030, semantic_base_pressure_intent))
            semantic_intent_multiplier = max(0.85, min(1.08, semantic_intent_multiplier))
            semantic_window_multiplier = max(0.80, min(1.05, semantic_window_multiplier))
            pressure_intent_after_background_anchor = (
                pressure_intent_after_background_anchor * semantic_intent_multiplier
                + semantic_base_pressure_intent
            )
            max_window_after_background_anchor = max_window_after_background_anchor * semantic_window_multiplier
            semantic_router_applied = True
        compressed_intent = math.tanh(pressure_intent_after_background_anchor * compression) if abs(pressure_intent_after_background_anchor) > 1e-12 else 0.0

        field_window = float(branch_policy.get("strategy_field_window_max", max_window) or max_window)
        if semantic_router_applied:
            field_window = field_window * semantic_window_multiplier
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

        pressure_pixel_active_candidate = {
            "stage": f"EventPressurePixelReweightingActiveCandidate_{branch_name}",
            "status": "inactive",
            "severity": "INFO",
            "version": "pressure_pixel_reweighting_active_candidate_v4_feathered_identity_guard",
            "mode": mode,
            "branch_name": str(branch_name or ""),
            "branch_key": branch_key,
            "source": "R147 fixed-seed PASS evidence + R148 deterministic A/B grain verdict + R149 quality guard + R150 artifact correction",
            "source_runtime_version": "0.1.1-r149-dev",
            "human_visual_quality_verdict": "R148 active candidate introduced visible extra grain",
            "same_run_oracle": False,
            "prompt_text_injection_allowed": False,
            "semantic_math_in_prompt_allowed": False,
            "high_branch_passthrough": True,
            "low_branch_only": True,
            "candidate_delta": 0.0,
            "candidate_effective_strength": 1.0,
            "low_strength_multiplier": 1.0,
            "quality_guard": {
                "status": "active",
                "grain_artifact_reported": True,
                "quality_guard_factor": 0.50,
                "candidate_cap": 0.0020,
            },
            "local_spatial_pressure_guard": {
                "status": "inactive",
                "version": "pressure_pixel_feathered_identity_guard_v2",
                "source": "R149 quality guard + R150 top/face artifact correction",
                "same_run_oracle": False,
                "active_control_allowed": False,
                "policy": "inactive until low branch candidate is active",
            },
            "formula": (
                "R147/R149 pixel Outcome and scalar pressure disagreement are read as previous evidence. "
                "Only the next low branch may receive a quality-guarded bounded delta; high stays native. "
                "R151 adds a feathered identity-protected local/spatial guard so pressure returns through regions without treating face/top identity as background."
            ),
        }

        def build_pressure_pixel_active_candidate():
            pixel_trust = 0.7615070273368614
            pressure_trust = 0.5768564286193962
            background_factor = 0.8358471363710388
            action_factor = 1.0910896522124691
            seam_guard = 0.5510783527760094
            center_action_priority = 0.5382811908679574
            r149_center_edge_ratio = 1.355514816829044
            r149_edge_center_ratio = 0.7377270890131548
            r149_background_leakage = 0.7377270890131548
            r149_estimated_seam_ratio = 1.7181074972860457
            r149_action_center_confidence = 0.5240003162482261
            r149_last10_laplacian_fraction_toward_r148 = 0.029806809024513797
            try:
                user_multiplier = float(base_strength)
            except Exception:
                user_multiplier = 1.0
            if not math.isfinite(user_multiplier):
                user_multiplier = 1.0
            user_multiplier = max(0.0, min(2.0, user_multiplier))

            trust_gap = max(0.0, float(pixel_trust - pressure_trust))
            raw_delta = float(center_action_priority * trust_gap * 0.035)
            seam_damping = float(1.0 - min(0.35, max(0.0, seam_guard) * 0.18))
            background_damping = float(0.92 + 0.08 * max(0.0, min(1.0, background_factor)))
            action_priority_boost = float(0.96 + 0.04 * max(0.0, min(1.20, action_factor)))
            pre_cap_delta = raw_delta * seam_damping * background_damping * action_priority_boost
            quality_guard_factor = 0.50
            quality_guard_cap = 0.0020
            r148_reference_delta = 0.0031038281594852626
            pre_quality_guard_delta = pre_cap_delta * user_multiplier
            quality_guarded_delta = pre_quality_guard_delta * quality_guard_factor
            bounded_delta = max(0.0, min(quality_guard_cap, quality_guarded_delta))
            branch_delta = float(bounded_delta if branch_key == "low" else 0.0)
            status = "active_candidate" if branch_delta > 0.0 else ("passthrough" if branch_key == "high" else "neutral")
            policy = (
                "r151_pressure_pixel_feathered_identity_low_refinement_candidate"
                if branch_delta > 0.0
                else "r151_pressure_pixel_feathered_identity_branch_passthrough"
            )
            action_focus_pressure = clamp01((r149_center_edge_ratio - 1.0) / 0.75)
            seam_spatial_pressure = clamp01((r149_estimated_seam_ratio - 1.35) / 0.75)
            detail_recovery_guard = clamp01(1.0 - r149_last10_laplacian_fraction_toward_r148)
            spatial_background_pressure = clamp01(
                0.42 * r149_background_leakage
                + 0.24 * r149_edge_center_ratio
                + 0.20 * seam_spatial_pressure
                + 0.14 * max(0.0, 1.0 - r149_action_center_confidence)
            )
            spatial_action_preservation = clamp01(
                0.46 * action_focus_pressure
                + 0.34 * r149_action_center_confidence
                + 0.20 * detail_recovery_guard
            )
            spatial_max_attenuation = max(0.0, min(0.022, 0.014 * spatial_background_pressure + 0.006 * seam_spatial_pressure))
            spatial_min_gain = max(0.978, 1.0 - spatial_max_attenuation)
            local_spatial_guard = {
                "status": "active" if branch_delta > 0.0 and branch_key == "low" else ("passthrough" if branch_key == "high" else "inactive"),
                "version": "pressure_pixel_feathered_identity_guard_v2",
                "source": "R149 mega-check visible Outcome + R150 live pixel artifact; no same-run oracle",
                "same_run_oracle": False,
                "active_control_allowed": bool(branch_delta > 0.0 and branch_key == "low"),
                "policy": "preserve upper identity/action delta while feathering edge/background delta",
                "normalized_action_roi": [0.12, 0.18, 0.72, 0.64],
                "identity_protection_rois": [
                    {"name": "upper_center_identity_face_carrier", "rect": [0.00, 0.22, 0.42, 0.56], "protection": 1.00},
                    {"name": "center_action_identity_carrier", "rect": [0.12, 0.18, 0.70, 0.64], "protection": 0.72},
                ],
                "background_rois": [
                    {"name": "top_left_background_band", "rect": [0.00, 0.00, 0.24, 0.22], "pressure": float(spatial_background_pressure), "attenuation_scale": 0.40},
                    {"name": "top_right_background_band", "rect": [0.00, 0.78, 0.24, 0.22], "pressure": float(spatial_background_pressure), "attenuation_scale": 0.40},
                    {"name": "left_edge_background", "rect": [0.00, 0.00, 1.00, 0.14], "pressure": float(spatial_background_pressure), "attenuation_scale": 0.46},
                    {"name": "right_edge_background", "rect": [0.00, 0.86, 1.00, 0.14], "pressure": float(spatial_background_pressure), "attenuation_scale": 0.46},
                    {"name": "bottom_background_band", "rect": [0.84, 0.00, 0.16, 1.00], "pressure": float(spatial_background_pressure), "attenuation_scale": 0.34},
                ],
                "input_pressures": {
                    "r149_center_edge_ratio": float(r149_center_edge_ratio),
                    "r149_edge_center_ratio": float(r149_edge_center_ratio),
                    "r149_background_leakage": float(r149_background_leakage),
                    "r149_estimated_seam_ratio": float(r149_estimated_seam_ratio),
                    "r149_action_center_confidence": float(r149_action_center_confidence),
                    "action_focus_pressure": float(action_focus_pressure),
                    "seam_spatial_pressure": float(seam_spatial_pressure),
                    "detail_recovery_guard": float(detail_recovery_guard),
                    "spatial_background_pressure": float(spatial_background_pressure),
                    "spatial_action_preservation": float(spatial_action_preservation),
                },
                "gain_policy": {
                    "center_action_gain": 1.0,
                    "max_background_attenuation": float(spatial_max_attenuation),
                    "min_background_gain": float(spatial_min_gain),
                    "feather_kernel": 5,
                    "feather_passes": 2,
                    "identity_protection_enabled": True,
                    "whole_low_delta_unchanged": True,
                },
                "formula": (
                    "The R149/R150 visible Outcome returns as a corrected local Strategy map: upper identity/action remains the active carrier, "
                    "edge/background remains an anchor. The same tiny low delta is multiplied by a feathered spatial gain instead of a hard rectangular map."
                ),
            }
            return {
                "stage": f"EventPressurePixelReweightingActiveCandidate_{branch_name}",
                "status": status,
                "severity": "INFO",
                "version": "pressure_pixel_reweighting_active_candidate_v4_feathered_identity_guard",
                "mode": mode,
                "branch_name": str(branch_name or ""),
                "branch_key": branch_key,
                "source": "R147 fixed-seed PASS evidence + R148 deterministic A/B grain verdict + R149 quality guard + R150 artifact correction",
                "source_runtime_version": "0.1.1-r149-dev",
                "source_report_label": "R147 EventPressurePixelReweightingProposal + R148 human visual review + R149 quality guard review + R150 identity/top-band artifact review",
                "human_visual_quality_verdict": "R148 active candidate introduced visible extra grain",
                "same_run_oracle": False,
                "prompt_text_injection_allowed": False,
                "semantic_math_in_prompt_allowed": False,
                "high_branch_passthrough": bool(branch_key == "high"),
                "low_branch_only": True,
                "candidate_delta": float(branch_delta),
                "candidate_effective_strength": float(1.0 + branch_delta),
                "low_strength_multiplier": float(user_multiplier),
                "candidate_cap": float(quality_guard_cap),
                "r148_reference_delta": float(r148_reference_delta),
                "r147_inputs": {
                    "pixel_outcome_trust_weight": float(pixel_trust),
                    "scalar_pressure_trust_weight": float(pressure_trust),
                    "bounded_background_pressure_factor": float(background_factor),
                    "action_preservation_factor": float(action_factor),
                    "seam_protection_weight": float(seam_guard),
                    "center_action_priority": float(center_action_priority),
                },
                "derived_weights": {
                    "trust_gap": float(trust_gap),
                    "raw_delta": float(raw_delta),
                    "seam_damping": float(seam_damping),
                    "background_damping": float(background_damping),
                    "action_priority_boost": float(action_priority_boost),
                    "pre_cap_delta": float(pre_cap_delta),
                    "pre_quality_guard_delta": float(pre_quality_guard_delta),
                    "quality_guard_factor": float(quality_guard_factor),
                    "quality_guarded_delta": float(quality_guarded_delta),
                    "spatial_background_pressure": float(spatial_background_pressure),
                    "spatial_action_preservation": float(spatial_action_preservation),
                    "spatial_max_attenuation": float(spatial_max_attenuation),
                },
                "quality_guard": {
                    "status": "active",
                    "source": "R148 deterministic A/B plus human visual grain verdict; R149 accepted as cleaner active candidate",
                    "grain_artifact_reported": True,
                    "quality_guard_factor": float(quality_guard_factor),
                    "candidate_cap": float(quality_guard_cap),
                    "r148_reference_delta": float(r148_reference_delta),
                    "pre_quality_guard_delta": float(pre_quality_guard_delta),
                    "post_quality_guard_delta": float(branch_delta),
                    "policy": "reduce whole-low pressure before any spatial expansion",
                },
                "local_spatial_pressure_guard": local_spatial_guard,
                "apply_policy": policy,
                "next_evidence": "compare fixed-seed R151 against R149/R150; accept only if R150 top/face pixel artifact disappears while useful action continuation remains",
                "formula": (
                    "Outcome_pixel(t-1)+PressureDisagreement(t-1) returns to Strategy(t) as a "
                    "quality-guarded low-refinement strength delta plus a feathered local/spatial background gain. "
                    "R148 proved the route can move Outcome(t+1); R149 proved the cleaner scalar guard; "
                    "R150 showed that hard top/background routing can mistake identity for background; R151 protects identity carriers and feathers the map. "
                    "This is not current-run oracle control and not prompt text."
                ),
            }

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

        cascade_return_normalizer = {
            "status": "inactive",
            "version": "cascade_return_normalizer_v1",
            "reason": "mode_or_remaining_strategy_not_active",
            "formula_role": "return local continuation Strategy pressure to S_global_event_route",
            "input_pressures": {},
            "multipliers": {
                "pressure_intent_multiplier": 1.0,
                "max_delta_window_multiplier": 1.0,
                "field_intent_multiplier": 1.0,
                "field_window_multiplier": 1.0,
                "temporal_stability_multiplier": 1.0,
            },
            "losses": {
                "intent_loss": 0.0,
                "window_loss": 0.0,
                "temporal_loss": 0.0,
            },
            "prompt_text_injection": False,
            "semantic_math_in_prompt": False,
            "formula": (
                "When a pause-selected Outcome(t-1) becomes the next segment carrier, "
                "local relation pressure must return to the global Strategy instead of "
                "expanding as a second independent scene-wide push."
            ),
        }
        if mode == "STRATEGY_PRESSURE_WINDOW" and field_active and isinstance(remaining_strategy, dict):
            remaining_status = str(remaining_strategy.get("status", "") or "")
            remaining_active = bool(remaining_status == "active" or remaining_target_segment > 0)
            if remaining_active:
                semantic_pressures = (
                    semantic_router.get("input_pressure", {})
                    if isinstance(semantic_router, dict) and isinstance(semantic_router.get("input_pressure", {}), dict)
                    else {}
                )
                restart_risk_pressure = clamp01(remaining_strategy.get("restart_risk", 0.0))
                motion_memory_pressure = clamp01(remaining_strategy.get("motion_memory_pressure", 0.0))
                late_cut_pressure = clamp01(remaining_strategy.get("late_cut_pressure", 0.0))
                route_pressure = clamp01(remaining_strategy.get("route_pressure", 0.0))
                progress_ratio = clamp01(remaining_strategy.get("progress_ratio", 0.0))
                semantic_gap_pressure = clamp01(semantic_pressures.get("semantic_gap", 0.0))
                relation_pressure_value = clamp01(semantic_pressures.get("relation_pressure", 0.0))
                stability_guard_pressure = clamp01(semantic_pressures.get("stability_guard", 0.0))
                topology_pressure_value = clamp01(semantic_pressures.get("object_topology_pressure", 0.0))

                continuation_pressure = max(
                    restart_risk_pressure,
                    motion_memory_pressure,
                    late_cut_pressure * max(route_pressure, 0.35),
                    max(0.0, progress_ratio - 0.85) * 0.50,
                )
                broad_strategy_pressure = max(
                    semantic_gap_pressure,
                    stability_guard_pressure,
                    relation_pressure_value * 0.35,
                    topology_pressure_value * 0.25,
                )
                if continuation_pressure > 0.0 or broad_strategy_pressure > 0.0:
                    if branch_key == "high":
                        intent_loss = min(
                            0.10,
                            0.040 * continuation_pressure
                            + 0.025 * semantic_gap_pressure
                            + 0.010 * relation_pressure_value,
                        )
                        window_loss = min(
                            0.08,
                            0.030 * continuation_pressure
                            + 0.020 * broad_strategy_pressure,
                        )
                        temporal_loss = min(
                            0.08,
                            0.025 * continuation_pressure
                            + 0.015 * broad_strategy_pressure,
                        )
                    elif branch_key == "low":
                        intent_loss = min(
                            0.09,
                            0.030 * continuation_pressure
                            + 0.020 * semantic_gap_pressure
                            + 0.015 * relation_pressure_value,
                        )
                        window_loss = min(
                            0.08,
                            0.025 * continuation_pressure
                            + 0.018 * broad_strategy_pressure
                            + 0.010 * topology_pressure_value,
                        )
                        temporal_loss = min(
                            0.10,
                            0.025 * continuation_pressure
                            + 0.020 * broad_strategy_pressure
                            + 0.010 * stability_guard_pressure,
                        )
                    else:
                        intent_loss = min(0.08, 0.030 * continuation_pressure + 0.020 * broad_strategy_pressure)
                        window_loss = min(0.06, 0.020 * continuation_pressure + 0.015 * broad_strategy_pressure)
                        temporal_loss = min(0.08, 0.020 * continuation_pressure + 0.015 * broad_strategy_pressure)

                    cascade_pressure_multiplier = max(0.70, 1.0 - intent_loss)
                    cascade_window_multiplier = max(0.75, 1.0 - window_loss)
                    cascade_temporal_multiplier = max(0.78, 1.0 - temporal_loss)

                    pressure_intent_after_background_anchor *= cascade_pressure_multiplier
                    max_window_after_background_anchor *= cascade_window_multiplier
                    field_intent *= cascade_pressure_multiplier
                    field_window *= cascade_window_multiplier
                    background_temporal_multiplier *= cascade_temporal_multiplier
                    compressed_intent = math.tanh(pressure_intent_after_background_anchor * compression) if abs(pressure_intent_after_background_anchor) > 1e-12 else 0.0

                    cascade_return_normalizer.update({
                        "status": "active",
                        "reason": "continuation_strategy_return_clamp",
                        "input_pressures": {
                            "remaining_status": remaining_status,
                            "target_segment": int(remaining_target_segment),
                            "current_segment": int(remaining_strategy_segment) if remaining_strategy_segment is not None else None,
                            "resume_frame_index": int(remaining_strategy.get("resume_frame_index", 0) or 0),
                            "restart_risk_pressure": float(restart_risk_pressure),
                            "motion_memory_pressure": float(motion_memory_pressure),
                            "late_cut_pressure": float(late_cut_pressure),
                            "route_pressure": float(route_pressure),
                            "progress_ratio": float(progress_ratio),
                            "semantic_gap_pressure": float(semantic_gap_pressure),
                            "relation_pressure": float(relation_pressure_value),
                            "stability_guard_pressure": float(stability_guard_pressure),
                            "topology_pressure": float(topology_pressure_value),
                            "continuation_pressure": float(continuation_pressure),
                            "broad_strategy_pressure": float(broad_strategy_pressure),
                        },
                        "multipliers": {
                            "pressure_intent_multiplier": float(cascade_pressure_multiplier),
                            "max_delta_window_multiplier": float(cascade_window_multiplier),
                            "field_intent_multiplier": float(cascade_pressure_multiplier),
                            "field_window_multiplier": float(cascade_window_multiplier),
                            "temporal_stability_multiplier": float(cascade_temporal_multiplier),
                        },
                        "losses": {
                            "intent_loss": float(intent_loss),
                            "window_loss": float(window_loss),
                            "temporal_loss": float(temporal_loss),
                        },
                        "policy": "continuation_substrategy_returns_to_global_strategy",
                    })

        spatial_strategy_return = {
            "status": "inactive",
            "version": "spatial_strategy_return_v1",
            "reason": "mode_or_evidence_not_active",
            "formula_role": "region-role Strategy intersections return to the model attractor",
            "region_roles": {
                "center_action_carrier": 0.0,
                "edge_background_anchor": 0.0,
                "lower_temporal_tail": 0.0,
                "region_conflict_pressure": 0.0,
            },
            "input_pressures": {},
            "multipliers": {
                "pressure_intent_multiplier": 1.0,
                "max_delta_window_multiplier": 1.0,
                "field_intent_multiplier": 1.0,
                "field_window_multiplier": 1.0,
                "temporal_stability_multiplier": 1.0,
            },
            "losses": {
                "intent_loss": 0.0,
                "window_loss": 0.0,
                "temporal_loss": 0.0,
            },
            "prompt_text_injection": False,
            "semantic_math_in_prompt": False,
            "policy": "inactive",
            "formula": (
                "SpatialStrategyReturn does not add words to the prompt. It reads already observed "
                "source/background/tail/semantic pressures as separate local Strategy points and "
                "returns them as one bounded model-attractor route."
            ),
        }
        action_background_separation_gate = {
            "stage": f"EventActionBackgroundSeparationGate_{branch_name}",
            "status": "inactive",
            "version": "action_background_separation_gate_v1",
            "branch_name": str(branch_name or ""),
            "branch_key": branch_key,
            "parent_strategy": "S_global_event_route",
            "formula_role": (
                "center action carrier and edge background anchor are local sub-strategies; "
                "they must return to one model-readable Strategy without letting background "
                "become the same motion carrier as the action center"
            ),
            "input_roles": {
                "center_action_carrier": 0.0,
                "edge_background_anchor": 0.0,
                "lower_temporal_tail": 0.0,
                "region_conflict_pressure": 0.0,
            },
            "derived_pressures": {
                "background_dominance": 0.0,
                "action_dominance": 0.0,
                "motion_leakage_pressure": 0.0,
                "action_focus_pressure": 0.0,
                "separation_pressure": 0.0,
            },
            "multipliers": {
                "pressure_intent_multiplier": 1.0,
                "max_delta_window_multiplier": 1.0,
                "field_intent_multiplier": 1.0,
                "field_window_multiplier": 1.0,
                "temporal_stability_multiplier": 1.0,
            },
            "losses": {
                "intent_loss": 0.0,
                "window_loss": 0.0,
                "temporal_loss": 0.0,
            },
            "policy": "inactive",
            "prompt_text_injection": False,
            "semantic_math_in_prompt": False,
            "does_not_add_prompt_words": True,
            "formula": (
                "If background/edge pressure equals or exceeds center-action pressure, "
                "this gate compresses the global field window so the model keeps the "
                "background as OutcomePrevious/anchor instead of turning it into "
                "ObservedBehavior motion."
            ),
        }
        if mode == "STRATEGY_PRESSURE_WINDOW" and field_active:
            semantic_pressures_for_spatial = (
                semantic_router.get("input_pressure", {})
                if isinstance(semantic_router, dict) and isinstance(semantic_router.get("input_pressure", {}), dict)
                else {}
            )
            source_pressures_for_spatial = (
                source_anchor_return_window.get("input_pressures", {})
                if isinstance(source_anchor_return_window, dict) and isinstance(source_anchor_return_window.get("input_pressures", {}), dict)
                else {}
            )
            cascade_pressures_for_spatial = (
                cascade_return_normalizer.get("input_pressures", {})
                if isinstance(cascade_return_normalizer, dict) and isinstance(cascade_return_normalizer.get("input_pressures", {}), dict)
                else {}
            )

            action_pressure = clamp01(semantic_pressures_for_spatial.get("action_pressure", 0.0))
            anchor_pressure = clamp01(semantic_pressures_for_spatial.get("anchor_pressure", 0.0))
            relation_pressure_value = clamp01(semantic_pressures_for_spatial.get("relation_pressure", 0.0))
            topology_pressure_value = clamp01(semantic_pressures_for_spatial.get("object_topology_pressure", 0.0))
            stability_guard_pressure = clamp01(semantic_pressures_for_spatial.get("stability_guard", 0.0))
            semantic_gap_pressure = clamp01(semantic_pressures_for_spatial.get("semantic_gap", 0.0))

            source_or_tail_pressure = clamp01(source_pressures_for_spatial.get("source_anchor_pressure", 0.0))
            background_source_pressure = max(
                clamp01(source_pressures_for_spatial.get("background_source_pressure", 0.0)),
                clamp01(background_preservation.get("background_anchor_pressure", 0.0) if isinstance(background_preservation, dict) else 0.0),
                clamp01(background_preservation.get("spatial_anchor_pressure", 0.0) if isinstance(background_preservation, dict) else 0.0),
                clamp01(background_preservation.get("background_region_pressure", 0.0) if isinstance(background_preservation, dict) else 0.0),
                clamp01(background_preservation.get("top_band_pressure", 0.0) if isinstance(background_preservation, dict) else 0.0),
            )
            lower_tail_pressure = max(
                clamp01(source_pressures_for_spatial.get("tail_continuation_pressure", 0.0)),
                clamp01(source_pressures_for_spatial.get("late_spike_pressure", 0.0)),
                clamp01(cascade_pressures_for_spatial.get("continuation_pressure", 0.0)),
                clamp01(cascade_pressures_for_spatial.get("motion_memory_pressure", 0.0)),
                clamp01(cascade_pressures_for_spatial.get("late_cut_pressure", 0.0)),
                clamp01(remaining_strategy.get("motion_memory_pressure", 0.0) if isinstance(remaining_strategy, dict) else 0.0),
                clamp01(remaining_strategy.get("late_cut_pressure", 0.0) if isinstance(remaining_strategy, dict) else 0.0),
            )
            try:
                branch_pressure_for_spatial = clamp01(abs(float(pressure_intent)) / max(float(max_window), 1e-9))
            except Exception:
                branch_pressure_for_spatial = 0.0

            semantic_confidence_pressure = max(
                action_pressure,
                anchor_pressure,
                relation_pressure_value,
                topology_pressure_value,
                stability_guard_pressure,
                semantic_gap_pressure,
            )
            center_action_carrier = clamp01(
                0.50 * action_pressure
                + 0.25 * relation_pressure_value
                + 0.15 * topology_pressure_value
                + 0.10 * semantic_confidence_pressure * max(0.0, 1.0 - semantic_gap_pressure)
            )
            edge_background_anchor = clamp01(max(
                background_source_pressure,
                0.40 * anchor_pressure + 0.30 * stability_guard_pressure + 0.20 * semantic_gap_pressure,
                0.25 * branch_pressure_for_spatial if branch_key == "low" else 0.0,
            ))
            lower_temporal_tail = clamp01(max(
                lower_tail_pressure,
                0.50 * clamp01(cascade_pressures_for_spatial.get("route_pressure", 0.0)),
                0.35 * clamp01(remaining_strategy.get("route_pressure", 0.0) if isinstance(remaining_strategy, dict) else 0.0),
            ))
            region_conflict_pressure = clamp01(
                max(0.0, edge_background_anchor - (center_action_carrier * 0.55))
                + 0.35 * lower_temporal_tail
                + 0.15 * semantic_gap_pressure
            )
            spatial_evidence_pressure = max(
                center_action_carrier,
                edge_background_anchor,
                lower_temporal_tail,
                region_conflict_pressure,
            )

            if spatial_evidence_pressure > 0.0:
                if branch_key == "high":
                    intent_loss = min(
                        0.08,
                        max(0.0, 0.035 * edge_background_anchor + 0.020 * region_conflict_pressure + 0.012 * lower_temporal_tail - 0.015 * center_action_carrier),
                    )
                    window_loss = min(
                        0.08,
                        max(0.0, 0.030 * edge_background_anchor + 0.018 * region_conflict_pressure + 0.010 * semantic_gap_pressure - 0.012 * center_action_carrier),
                    )
                    temporal_loss = min(
                        0.07,
                        max(0.0, 0.018 * lower_temporal_tail + 0.015 * edge_background_anchor + 0.010 * region_conflict_pressure),
                    )
                    min_pressure_multiplier = 0.82
                    min_window_multiplier = 0.82
                    min_temporal_multiplier = 0.84
                elif branch_key == "low":
                    intent_loss = min(
                        0.11,
                        max(0.0, 0.050 * edge_background_anchor + 0.030 * lower_temporal_tail + 0.015 * region_conflict_pressure - 0.008 * center_action_carrier),
                    )
                    window_loss = min(
                        0.10,
                        max(0.0, 0.045 * edge_background_anchor + 0.035 * lower_temporal_tail + 0.010 * semantic_gap_pressure - 0.006 * center_action_carrier),
                    )
                    temporal_loss = min(
                        0.12,
                        max(0.0, 0.050 * lower_temporal_tail + 0.030 * edge_background_anchor + 0.015 * stability_guard_pressure),
                    )
                    min_pressure_multiplier = 0.75
                    min_window_multiplier = 0.76
                    min_temporal_multiplier = 0.78
                else:
                    intent_loss = min(0.08, 0.040 * edge_background_anchor + 0.025 * lower_temporal_tail)
                    window_loss = min(0.08, 0.035 * edge_background_anchor + 0.020 * lower_temporal_tail)
                    temporal_loss = min(0.08, 0.030 * lower_temporal_tail + 0.020 * edge_background_anchor)
                    min_pressure_multiplier = 0.78
                    min_window_multiplier = 0.78
                    min_temporal_multiplier = 0.80

                spatial_pressure_multiplier = max(min_pressure_multiplier, 1.0 - intent_loss)
                spatial_window_multiplier = max(min_window_multiplier, 1.0 - window_loss)
                spatial_temporal_multiplier = max(min_temporal_multiplier, 1.0 - temporal_loss)

                pressure_intent_after_background_anchor *= spatial_pressure_multiplier
                max_window_after_background_anchor *= spatial_window_multiplier
                field_intent *= spatial_pressure_multiplier
                field_window *= spatial_window_multiplier
                background_temporal_multiplier *= spatial_temporal_multiplier
                compressed_intent = math.tanh(pressure_intent_after_background_anchor * compression) if abs(pressure_intent_after_background_anchor) > 1e-12 else 0.0

                spatial_strategy_return.update({
                    "status": "active",
                    "reason": "region_role_strategy_pressure_detected",
                    "region_roles": {
                        "center_action_carrier": float(center_action_carrier),
                        "edge_background_anchor": float(edge_background_anchor),
                        "lower_temporal_tail": float(lower_temporal_tail),
                        "region_conflict_pressure": float(region_conflict_pressure),
                    },
                    "input_pressures": {
                        "action_pressure": float(action_pressure),
                        "anchor_pressure": float(anchor_pressure),
                        "relation_pressure": float(relation_pressure_value),
                        "topology_pressure": float(topology_pressure_value),
                        "stability_guard_pressure": float(stability_guard_pressure),
                        "semantic_gap_pressure": float(semantic_gap_pressure),
                        "semantic_confidence_pressure": float(semantic_confidence_pressure),
                        "source_or_tail_pressure": float(source_or_tail_pressure),
                        "source_or_tail_pressure_background_excluded": True,
                        "background_source_pressure": float(background_source_pressure),
                        "lower_tail_pressure": float(lower_tail_pressure),
                        "branch_pressure": float(branch_pressure_for_spatial),
                        "spatial_evidence_pressure": float(spatial_evidence_pressure),
                        "source_anchor_return_status": str(source_anchor_return_window.get("status", "") or ""),
                        "cascade_return_status": str(cascade_return_normalizer.get("status", "") or ""),
                    },
                    "multipliers": {
                        "pressure_intent_multiplier": float(spatial_pressure_multiplier),
                        "max_delta_window_multiplier": float(spatial_window_multiplier),
                        "field_intent_multiplier": float(spatial_pressure_multiplier),
                        "field_window_multiplier": float(spatial_window_multiplier),
                        "temporal_stability_multiplier": float(spatial_temporal_multiplier),
                    },
                    "losses": {
                        "intent_loss": float(intent_loss),
                        "window_loss": float(window_loss),
                        "temporal_loss": float(temporal_loss),
                    },
                    "policy": "region_aware_strategy_return_without_prompt_injection",
                })
        if mode == "STRATEGY_PRESSURE_WINDOW" and field_active:
            spatial_roles = (
                spatial_strategy_return.get("region_roles", {})
                if isinstance(spatial_strategy_return, dict) and isinstance(spatial_strategy_return.get("region_roles", {}), dict)
                else {}
            )
            spatial_inputs = (
                spatial_strategy_return.get("input_pressures", {})
                if isinstance(spatial_strategy_return, dict) and isinstance(spatial_strategy_return.get("input_pressures", {}), dict)
                else {}
            )
            center_action_carrier_gate = clamp01(spatial_roles.get("center_action_carrier", 0.0))
            edge_background_anchor_gate = clamp01(spatial_roles.get("edge_background_anchor", 0.0))
            lower_temporal_tail_gate = clamp01(spatial_roles.get("lower_temporal_tail", 0.0))
            region_conflict_gate = clamp01(spatial_roles.get("region_conflict_pressure", 0.0))
            relation_pressure_gate = clamp01(spatial_inputs.get("relation_pressure", 0.0))
            semantic_gap_gate = clamp01(spatial_inputs.get("semantic_gap_pressure", 0.0))
            background_dominance = clamp01(max(0.0, edge_background_anchor_gate - (0.72 * center_action_carrier_gate)))
            action_dominance = clamp01(max(0.0, center_action_carrier_gate - edge_background_anchor_gate))
            motion_leakage_pressure = clamp01(
                0.58 * background_dominance
                + 0.20 * region_conflict_gate
                + 0.14 * lower_temporal_tail_gate
                + 0.08 * semantic_gap_gate
            )
            action_focus_pressure = clamp01(
                0.64 * center_action_carrier_gate
                + 0.24 * relation_pressure_gate
                + 0.12 * action_dominance
            )
            separation_pressure = clamp01(
                motion_leakage_pressure
                * (1.0 - min(0.45, 0.30 * action_focus_pressure))
            )
            should_separate = bool(
                separation_pressure > 0.015
                or (
                    edge_background_anchor_gate > 0.40
                    and edge_background_anchor_gate >= (0.90 * max(center_action_carrier_gate, 1e-9))
                )
            )
            continuation_endpoint_guard = bool(
                str(branch_name or "").startswith("cascade_")
                and edge_background_anchor_gate >= 0.95
                and lower_temporal_tail_gate >= 0.75
                and region_conflict_gate >= 0.75
                and (
                    lower_temporal_tail_gate >= 0.95
                    or clamp01(spatial_inputs.get("background_source_pressure", 0.0)) >= 0.95
                    or clamp01(source_pressures_for_spatial.get("background_source_pressure", 0.0)) >= 0.95
                    or clamp01(cascade_pressures_for_spatial.get("continuation_pressure", 0.0)) >= 0.95
                    or str(source_anchor_return_window.get("status", "") or "") == "active"
                    or str(cascade_return_normalizer.get("status", "") or "") == "active"
                )
            )
            if should_separate and continuation_endpoint_guard:
                action_background_separation_gate.update({
                    "status": "guarded_report_only",
                    "reason": "continuation_endpoint_pressure_requires_step_level_spatial_control",
                    "active_control_allowed": False,
                    "input_roles": {
                        "center_action_carrier": float(center_action_carrier_gate),
                        "edge_background_anchor": float(edge_background_anchor_gate),
                        "lower_temporal_tail": float(lower_temporal_tail_gate),
                        "region_conflict_pressure": float(region_conflict_gate),
                    },
                    "derived_pressures": {
                        "background_dominance": float(background_dominance),
                        "action_dominance": float(action_dominance),
                        "motion_leakage_pressure": float(motion_leakage_pressure),
                        "action_focus_pressure": float(action_focus_pressure),
                        "separation_pressure": float(separation_pressure),
                    },
                    "multipliers": {
                        "pressure_intent_multiplier": 1.0,
                        "max_delta_window_multiplier": 1.0,
                        "field_intent_multiplier": 1.0,
                        "field_window_multiplier": 1.0,
                        "temporal_stability_multiplier": 1.0,
                    },
                    "losses": {
                        "intent_loss": 0.0,
                        "window_loss": 0.0,
                        "temporal_loss": 0.0,
                    },
                    "guard": {
                        "branch_name": str(branch_name or ""),
                        "branch_key": str(branch_key or ""),
                        "source_anchor_return_status": str(source_anchor_return_window.get("status", "") or ""),
                        "cascade_return_status": str(cascade_return_normalizer.get("status", "") or ""),
                        "background_source_pressure": float(
                            max(
                                clamp01(spatial_inputs.get("background_source_pressure", 0.0)),
                                clamp01(source_pressures_for_spatial.get("background_source_pressure", 0.0)),
                            )
                        ),
                        "continuation_pressure": float(cascade_pressures_for_spatial.get("continuation_pressure", 0.0) or 0.0),
                        "why": (
                            "Cascade continuation has maxed tail/background/conflict pressure. "
                            "Global field compression here can preserve endpoint/top-band artifact; "
                            "active control is held until step-level spatial evidence exists."
                        ),
                    },
                    "policy": "report_only_until_step_level_spatial_control",
                })
            elif should_separate:
                if branch_key == "high":
                    intent_loss = min(
                        0.055,
                        max(
                            0.0,
                            0.032 * separation_pressure
                            + 0.018 * edge_background_anchor_gate
                            + 0.010 * region_conflict_gate
                            - 0.020 * action_focus_pressure,
                        ),
                    )
                    window_loss = min(
                        0.070,
                        max(
                            0.0,
                            0.045 * separation_pressure
                            + 0.020 * lower_temporal_tail_gate
                            + 0.012 * semantic_gap_gate
                            - 0.018 * action_focus_pressure,
                        ),
                    )
                    temporal_loss = min(
                        0.060,
                        max(0.0, 0.028 * motion_leakage_pressure + 0.016 * region_conflict_gate),
                    )
                    min_pressure_multiplier = 0.88
                    min_window_multiplier = 0.86
                    min_temporal_multiplier = 0.88
                elif branch_key == "low":
                    intent_loss = min(
                        0.085,
                        max(
                            0.0,
                            0.050 * separation_pressure
                            + 0.024 * lower_temporal_tail_gate
                            + 0.014 * region_conflict_gate
                            - 0.018 * action_focus_pressure,
                        ),
                    )
                    window_loss = min(
                        0.095,
                        max(
                            0.0,
                            0.060 * separation_pressure
                            + 0.030 * lower_temporal_tail_gate
                            + 0.018 * semantic_gap_gate
                            - 0.014 * action_focus_pressure,
                        ),
                    )
                    temporal_loss = min(
                        0.090,
                        max(0.0, 0.040 * motion_leakage_pressure + 0.026 * lower_temporal_tail_gate + 0.012 * region_conflict_gate),
                    )
                    min_pressure_multiplier = 0.82
                    min_window_multiplier = 0.80
                    min_temporal_multiplier = 0.82
                else:
                    intent_loss = min(0.070, max(0.0, 0.042 * separation_pressure + 0.018 * region_conflict_gate))
                    window_loss = min(0.075, max(0.0, 0.046 * separation_pressure + 0.018 * lower_temporal_tail_gate))
                    temporal_loss = min(0.070, max(0.0, 0.032 * motion_leakage_pressure + 0.018 * lower_temporal_tail_gate))
                    min_pressure_multiplier = 0.84
                    min_window_multiplier = 0.82
                    min_temporal_multiplier = 0.84

                action_pressure_multiplier = max(min_pressure_multiplier, 1.0 - intent_loss)
                action_window_multiplier = max(min_window_multiplier, 1.0 - window_loss)
                action_temporal_multiplier = max(min_temporal_multiplier, 1.0 - temporal_loss)

                pressure_intent_after_background_anchor *= action_pressure_multiplier
                max_window_after_background_anchor *= action_window_multiplier
                field_intent *= action_pressure_multiplier
                field_window *= action_window_multiplier
                background_temporal_multiplier *= action_temporal_multiplier
                compressed_intent = math.tanh(pressure_intent_after_background_anchor * compression) if abs(pressure_intent_after_background_anchor) > 1e-12 else 0.0

                action_background_separation_gate.update({
                    "status": "active",
                    "reason": "background_edge_pressure_competes_with_center_action_carrier",
                    "input_roles": {
                        "center_action_carrier": float(center_action_carrier_gate),
                        "edge_background_anchor": float(edge_background_anchor_gate),
                        "lower_temporal_tail": float(lower_temporal_tail_gate),
                        "region_conflict_pressure": float(region_conflict_gate),
                    },
                    "derived_pressures": {
                        "background_dominance": float(background_dominance),
                        "action_dominance": float(action_dominance),
                        "motion_leakage_pressure": float(motion_leakage_pressure),
                        "action_focus_pressure": float(action_focus_pressure),
                        "separation_pressure": float(separation_pressure),
                    },
                    "multipliers": {
                        "pressure_intent_multiplier": float(action_pressure_multiplier),
                        "max_delta_window_multiplier": float(action_window_multiplier),
                        "field_intent_multiplier": float(action_pressure_multiplier),
                        "field_window_multiplier": float(action_window_multiplier),
                        "temporal_stability_multiplier": float(action_temporal_multiplier),
                    },
                    "losses": {
                        "intent_loss": float(intent_loss),
                        "window_loss": float(window_loss),
                        "temporal_loss": float(temporal_loss),
                    },
                    "policy": "center_action_preserved_edge_background_compressed_without_prompt_injection",
                })
            elif max(center_action_carrier_gate, edge_background_anchor_gate, lower_temporal_tail_gate, region_conflict_gate) > 0.0:
                action_background_separation_gate.update({
                    "status": "balanced_report",
                    "reason": "center_action_and_background_anchor_already_separable",
                    "input_roles": {
                        "center_action_carrier": float(center_action_carrier_gate),
                        "edge_background_anchor": float(edge_background_anchor_gate),
                        "lower_temporal_tail": float(lower_temporal_tail_gate),
                        "region_conflict_pressure": float(region_conflict_gate),
                    },
                    "derived_pressures": {
                        "background_dominance": float(background_dominance),
                        "action_dominance": float(action_dominance),
                        "motion_leakage_pressure": float(motion_leakage_pressure),
                        "action_focus_pressure": float(action_focus_pressure),
                        "separation_pressure": float(separation_pressure),
                    },
                    "policy": "report_only_no_extra_compression",
                })
            if str(action_background_separation_gate.get("status", "") or "") != "inactive":
                records.append(action_background_separation_gate)
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
        elif mode == "PRESSURE_PIXEL_REWEIGHTING":
            pressure_pixel_active_candidate = build_pressure_pixel_active_candidate()
            effective_strength = float(pressure_pixel_active_candidate.get("candidate_effective_strength", 1.0) or 1.0)
            apply_policy = str(pressure_pixel_active_candidate.get("apply_policy", "r151_pressure_pixel_feathered_identity_branch_passthrough") or "r151_pressure_pixel_feathered_identity_branch_passthrough")
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
        if str(cascade_return_normalizer.get("status", "") or "") == "active":
            apply_policy = f"{apply_policy}_cascade_return"
        if str(spatial_strategy_return.get("status", "") or "") == "active":
            apply_policy = f"{apply_policy}_spatial_return"
        if str(action_background_separation_gate.get("status", "") or "") == "active":
            apply_policy = f"{apply_policy}_action_background_separation"

        coupling = getattr(self, "_event_strategy_coupling", {}) or {}
        try:
            high_relative_delta = float(coupling.get("relative_delta", 0.0) or 0.0)
        except Exception:
            high_relative_delta = 0.0

        if mode == "PRESSURE_PIXEL_REWEIGHTING":
            records.append(pressure_pixel_active_candidate)
            self._event_pressure_pixel_reweighting_active_candidate_last = pressure_pixel_active_candidate

        active = mode in ("LATENT_DELTA_SCALE", "STRATEGY_PRESSURE_WINDOW", "LATENT_MEMORY_BRIDGE", "PRESSURE_PIXEL_REWEIGHTING", "DEEP_STEP_DELTA_CONTROL") and abs(effective_strength - 1.0) >= 1e-9
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
            "semantic_relation_pressure_router": {
                "stage": semantic_router.get("stage", "") if isinstance(semantic_router, dict) else "",
                "status": semantic_router.get("status", "") if isinstance(semantic_router, dict) else "",
                "version": semantic_router.get("version", "") if isinstance(semantic_router, dict) else "",
                "active_prompt_source": semantic_router.get("active_prompt_source", "") if isinstance(semantic_router, dict) else "",
                "applied": bool(semantic_router_applied),
                "branch_route": semantic_branch_route if isinstance(semantic_branch_route, dict) else {},
                "base_pressure_intent_added": float(semantic_base_pressure_intent),
                "intent_multiplier": float(semantic_intent_multiplier),
                "window_multiplier": float(semantic_window_multiplier),
                "input_pressure": semantic_router.get("input_pressure", {}) if isinstance(semantic_router, dict) else {},
                "prompt_text_injection": False,
                "semantic_math_in_prompt": False,
                "formula": "Semantic relation pressure is a non-text local Strategy router. If active, it adds only a tiny bounded pressure before the branch returns to S_global_event_route.",
            },
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
            "cascade_return_normalizer": cascade_return_normalizer,
            "spatial_strategy_return": spatial_strategy_return,
            "action_background_separation_gate": action_background_separation_gate,
            "pressure_pixel_reweighting_active_candidate": pressure_pixel_active_candidate,
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

        mode = str(getattr(self, "_event_math_control_mode", "OBSERVE_ONLY") or "OBSERVE_ONLY").upper()
        phase_map = phase_map if isinstance(phase_map, dict) else {}
        rec = {
            "stage": f"EventNoiseSourceFieldMap_{branch_name}",
            "status": "unavailable",
            "version": "noise_source_field_map_v10_R171_regional_tail_guard",
            "branch_name": str(branch_name or ""),
            "branch_key": branch_key,
            "mode": mode,
            "phase_scope": str(phase_map.get("phase_scope", "") or ""),
            "phase_placement": str(phase_map.get("placement", "") or ""),
            "denoise_phase_active_candidate": bool(phase_map.get("active_control_allowed", False)),
            "active_control_allowed": bool(mode == "SOURCE_NOISE_FIELD_SHAPING" and branch_key == "high" and str(phase_map.get("phase_scope", "") or "") == "pre_high_seed"),
            "future_active_candidate": (
                "pre_high_source_noise_shaping" if branch_key == "high"
                else "low_mid_window_refinement" if branch_key == "low"
                else "report_only"
            ),
            "prompt_text_injection_allowed": False,
            "semantic_math_in_prompt_allowed": False,
            "formula_role": "read source/noise field pressure before Strategy becomes local control",
            "formula": "NoiseSourceField reads latent region energy. It remains report-only except in R171 SOURCE_NOISE_FIELD_SHAPING, where the pre-high seed can receive tiny spatial gain, additive source-image carrier is limited to anchors, and segment entry may receive a tiny two-slice post-drop latent echo with selected-tail regional/background guard before high sampler.",
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
        mode = str(getattr(self, "_event_math_control_mode", "OBSERVE_ONLY") or "OBSERVE_ONLY").upper()

        rec = {
            "stage": f"EventNoiseFieldStrategyBridge_{branch_name}",
            "status": "report_only",
            "version": "noise_field_strategy_bridge_v1_model_attractor",
            "branch_name": str(branch_name or ""),
            "branch_key": branch_key,
            "mode": mode,
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
            "formula": "This bridge routes source/noise evidence to the model-centered Strategy surface. It mutates tensors only when R171 SOURCE_NOISE_FIELD_SHAPING acts on a pre-high Wan latent seed, Wan positive concat source-image carrier, or bounded post-drop seam-entry micro echo with selected-tail regional/background guard and anchor-only additive protection.",
        }

        if noise_status in ("recorded_zero_field", "unavailable", "failed"):
            rec.update({
                "status": "blocked_zero_or_unavailable_field" if noise_status == "recorded_zero_field" else "unavailable",
                "reason": "zero_or_unavailable_noise_field_is_not_strategy_pressure",
                "recommended_current_action": "ignore_as_active_pressure",
                "recommended_next_surface": "wait_for_nonzero_source_noise_carrier",
            })
        elif branch_key == "high" and mode == "SOURCE_NOISE_FIELD_SHAPING" and phase_scope == "pre_high_seed":
            rec.update({
                "status": "pre_high_seed_active_candidate",
                "reason": "current_surface_is_pre_high_seed_not_post_window_delta",
                "recommended_current_action": "apply_tiny_feathered_source_noise_seed_gain",
                "recommended_next_surface": "source_noise_birth_shaping",
                "current_surface_safe": True,
                "active_control_allowed": True,
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

    def _event_source_noise_birth_shaping(
        self,
        latent_seed,
        records,
        *,
        segment_index=None,
        route_label="",
        source_image=None,
        wan_positive=None,
    ):
        return_pair = wan_positive is not None

        def _return(positive_value, latent_value):
            return (positive_value, latent_value) if return_pair else latent_value

        mode = str(getattr(self, "_event_math_control_mode", "OBSERVE_ONLY") or "OBSERVE_ONLY").upper()
        if mode != "SOURCE_NOISE_FIELD_SHAPING":
            return _return(wan_positive, latent_seed)

        branch_name = f"{str(route_label or 'segment').strip() or 'segment'}_high_birth_seed"
        strengths = getattr(self, "_event_delta_strengths", {}) or {}
        try:
            intensity_multiplier = float(strengths.get("high", 1.0) or 1.0)
        except Exception:
            intensity_multiplier = 1.0
        if not math.isfinite(intensity_multiplier):
            intensity_multiplier = 1.0
        intensity_multiplier = max(0.0, min(2.0, intensity_multiplier))

        phase_map = {
            "stage": f"EventDenoisePhaseMap_{branch_name}",
            "status": "active_candidate",
            "version": "denoise_phase_map_v10_R171_regional_tail_guard",
            "branch_name": branch_name,
            "branch_key": "high",
            "phase_scope": "pre_high_seed",
            "progress_bucket": "pre_high_birth",
            "placement": "pre_high_source_noise_field",
            "reason": "R171 acts before high sampler with anchor-only additive protection and a tiny selected-tail post-drop seam-entry latent echo protected by regional/background weighting, not on the post-window high delta.",
            "active_control_allowed": True,
            "next_safe_surface": "source_noise_birth_shaping",
            "prompt_text_injection_allowed": False,
            "semantic_math_in_prompt_allowed": False,
            "formula": "Wan latent seed is the source/noise StrategyCarrier before high denoise birth.",
        }
        records.append(phase_map)

        noise_field_map = self._event_noise_source_field_map(
            latent_seed,
            branch_name,
            records,
            phase_map=phase_map,
        )
        bridge_rec = self._event_noise_field_strategy_bridge(
            branch_name,
            records,
            phase_map=phase_map,
            noise_field_map=noise_field_map,
            surface_rec={
                "stage": f"EventSourceNoiseBirthShaping_{route_label or 'segment'}",
                "apply_policy": "tiny_pre_high_source_noise_spatial_gain",
                "effective_strength": intensity_multiplier,
            },
        )

        rec = {
            "stage": f"EventSourceNoiseBirthShaping_{route_label or 'segment'}",
            "status": "bypass",
            "version": "source_noise_birth_shaping_v5_spatial_microdetail_protection_carrier",
            "mode": mode,
            "segment_index": int(segment_index) if segment_index is not None else None,
            "route_label": str(route_label or ""),
            "branch_name": branch_name,
            "active_control_allowed": True,
            "active_tensor_mutation_applied": False,
            "intensity_source": "high_delta_strength_as_shaping_intensity_multiplier",
            "intensity_multiplier": float(intensity_multiplier),
            "prompt_text_injection_allowed": False,
            "semantic_math_in_prompt_allowed": False,
            "noise_source_field_map": {
                "stage": noise_field_map.get("stage", "") if isinstance(noise_field_map, dict) else "",
                "status": noise_field_map.get("status", "") if isinstance(noise_field_map, dict) else "",
                "version": noise_field_map.get("version", "") if isinstance(noise_field_map, dict) else "",
                "source_field_pressure": noise_field_map.get("source_field_pressure", None) if isinstance(noise_field_map, dict) else None,
                "region_ratios": noise_field_map.get("region_ratios", {}) if isinstance(noise_field_map, dict) else {},
            },
            "noise_field_strategy_bridge": {
                "stage": bridge_rec.get("stage", "") if isinstance(bridge_rec, dict) else "",
                "status": bridge_rec.get("status", "") if isinstance(bridge_rec, dict) else "",
                "version": bridge_rec.get("version", "") if isinstance(bridge_rec, dict) else "",
                "bridge_pressure": (bridge_rec.get("pressures", {}) or {}).get("bridge_pressure", None) if isinstance(bridge_rec, dict) else None,
                "recommended_next_surface": bridge_rec.get("recommended_next_surface", "") if isinstance(bridge_rec, dict) else "",
            },
            "source_image_birth_carrier": {
                "status": "not_evaluated",
                "active": False,
                "pressure": 0.0,
            },
            "formula": (
                "source/noise field + source image carrier -> tiny feathered pre-high latent/conditioning birth -> high sampler. "
                "Prompt, CFG, sampler route, and hard tail bridge are not changed."
            ),
        }

        try:
            import torch
            import torch.nn.functional as F

            def _find_named_tensor(obj, key_name):
                try:
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if str(k) == str(key_name) and hasattr(v, "detach") and hasattr(v, "shape"):
                                return v
                        for v in obj.values():
                            found = _find_named_tensor(v, key_name)
                            if found is not None:
                                return found
                    if isinstance(obj, (list, tuple)):
                        for v in obj:
                            found = _find_named_tensor(v, key_name)
                            if found is not None:
                                return found
                except Exception:
                    return None
                return None

            def _replace_named_tensor(obj, key_name, replacement):
                try:
                    if isinstance(obj, dict):
                        changed = False
                        out_obj = {}
                        for k, v in obj.items():
                            if str(k) == str(key_name) and hasattr(v, "detach") and hasattr(v, "shape"):
                                out_obj[k] = replacement
                                changed = True
                            else:
                                new_v, child_changed = _replace_named_tensor(v, key_name, replacement)
                                out_obj[k] = new_v
                                changed = changed or child_changed
                        return (out_obj if changed else obj), changed
                    if isinstance(obj, list):
                        changed = False
                        out_obj = []
                        for v in obj:
                            new_v, child_changed = _replace_named_tensor(v, key_name, replacement)
                            out_obj.append(new_v)
                            changed = changed or child_changed
                        return (out_obj if changed else obj), changed
                    if isinstance(obj, tuple):
                        changed = False
                        out_obj = []
                        for v in obj:
                            new_v, child_changed = _replace_named_tensor(v, key_name, replacement)
                            out_obj.append(new_v)
                            changed = changed or child_changed
                        return (tuple(out_obj) if changed else obj), changed
                except Exception:
                    return obj, False
                return obj, False

            seed_t = self._tensor_from_latent_like(latent_seed)
            if seed_t is None or not hasattr(seed_t, "shape"):
                rec["status"] = "unavailable"
                rec["reason"] = "latent_seed_tensor_unavailable"
                records.append(rec)
                return _return(wan_positive, latent_seed)
            if len(seed_t.shape) < 2:
                rec["status"] = "unavailable"
                rec["reason"] = "latent_seed_rank_too_low"
                rec["latent_shape"] = list(seed_t.shape)
                records.append(rec)
                return _return(wan_positive, latent_seed)

            ratios = noise_field_map.get("region_ratios", {}) if isinstance(noise_field_map, dict) and isinstance(noise_field_map.get("region_ratios", {}), dict) else {}

            def clamp01(value):
                try:
                    out = float(value)
                except Exception:
                    out = 0.0
                if not math.isfinite(out):
                    out = 0.0
                return max(0.0, min(1.0, out))

            def ratio_pressure(value):
                try:
                    return clamp01(abs(math.log(max(float(value), 1e-9))) / 2.0)
                except Exception:
                    return 0.0

            source_bias2d = None
            source_bias2d_guarded = None
            microdetail_guard = {
                "status": "not_evaluated",
                "policy": "low_frequency_strategy_carrier_only",
                "active": False,
            }
            source_carrier_pressure = 0.0
            source_carrier_status = "source_image_unavailable"
            try:
                src_t = self._tensor_from_latent_like(source_image)
                if src_t is not None and hasattr(src_t, "shape") and len(src_t.shape) >= 2:
                    src = torch.nan_to_num(src_t.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
                    if src.dim() >= 3 and int(src.shape[-1]) in (1, 3, 4):
                        src = src[..., :3].mean(dim=-1)
                    else:
                        src = src.abs()
                    while src.dim() > 2:
                        src = src.mean(dim=0)
                    if src.dim() == 2 and int(src.shape[-2]) > 1 and int(src.shape[-1]) > 1:
                        src = src - src.min()
                        src_max = src.max()
                        if float(src_max.detach().cpu().item()) > 1e-9:
                            src = src / src_max.clamp_min(1e-9)
                            if int(src.shape[-2]) != int(seed_t.shape[-2]) or int(src.shape[-1]) != int(seed_t.shape[-1]):
                                src = F.interpolate(
                                    src[None, None, :, :],
                                    size=(int(seed_t.shape[-2]), int(seed_t.shape[-1])),
                                    mode="bilinear",
                                    align_corners=False,
                                )[0, 0]
                            centered_src = src - src.mean()
                            source_carrier_pressure = clamp01(float(centered_src.abs().mean().detach().cpu().item()) * 2.0)
                            denom = centered_src.abs().mean().clamp_min(1e-6)
                            source_bias2d = (centered_src / denom).clamp(-1.0, 1.0)
                            source_carrier_status = "recorded"
                        else:
                            source_carrier_status = "flat_source_image"
                    else:
                        source_carrier_status = "source_image_shape_unusable"
                else:
                    source_carrier_status = "source_image_unavailable"
            except Exception as e:
                source_bias2d = None
                source_carrier_pressure = 0.0
                source_carrier_status = "failed"
                rec["source_image_birth_carrier_error"] = str(e)

            if source_bias2d is not None:
                try:
                    source_bias_float = source_bias2d.detach().float()
                    min_dim = int(min(source_bias_float.shape[-2], source_bias_float.shape[-1]))
                    if min_dim >= 12:
                        guard_kernel = 7
                    elif min_dim >= 8:
                        guard_kernel = 5
                    else:
                        guard_kernel = 3
                    guard_padding = int(guard_kernel // 2)
                    low_frequency = F.avg_pool2d(
                        source_bias_float[None, None, :, :],
                        kernel_size=guard_kernel,
                        stride=1,
                        padding=guard_padding,
                        count_include_pad=False,
                    )[0, 0]
                    low_frequency = low_frequency - low_frequency.mean()
                    low_abs = low_frequency.abs().mean()
                    source_abs = source_bias_float.abs().mean().clamp_min(1e-6)
                    high_frequency_residual = source_bias_float - low_frequency
                    if float(low_abs.detach().cpu().item()) > 1e-8:
                        # R168 keeps the low-pass source carrier weak. It is a topology hint, not a
                        # replacement drawing pass, and later masks restrict additive use to anchors.
                        source_bias2d_guarded = ((low_frequency / low_abs.clamp_min(1e-6)) * 0.62).clamp(-1.0, 1.0)
                    else:
                        source_bias2d_guarded = source_bias_float * 0.0
                    microdetail_guard = {
                        "status": "applied",
                        "policy": "source_bias_low_pass_and_anchor_only_additive_protection",
                        "active": True,
                        "kernel_size": int(guard_kernel),
                        "R168_low_frequency_strength": 0.62,
                        "original_bias_abs_mean": float(source_abs.detach().cpu().item()),
                        "low_frequency_abs_mean": float(low_abs.detach().cpu().item()),
                        "guarded_bias_abs_mean": float(source_bias2d_guarded.abs().mean().detach().cpu().item()),
                        "high_frequency_residual_ratio": float(
                            (high_frequency_residual.abs().mean() / source_abs).detach().cpu().item()
                        ),
                        "formula": (
                            "Source image remains an Outcome carrier, but only a softened coarse topology enters Strategy; "
                            "additive source pressure is later limited to anchors so microtexture and central detail stay model-native."
                        ),
                    }
                except Exception as e:
                    source_bias2d_guarded = source_bias2d
                    microdetail_guard = {
                        "status": "failed_passthrough_raw_bias",
                        "policy": "low_frequency_strategy_carrier_only",
                        "active": False,
                        "error": str(e)[:240],
                    }

            source_pressure = clamp01(noise_field_map.get("source_field_pressure", 0.0) if isinstance(noise_field_map, dict) else 0.0)
            bridge_pressure = clamp01((bridge_rec.get("pressures", {}) or {}).get("bridge_pressure", 0.0) if isinstance(bridge_rec, dict) else 0.0)
            center_outer_pressure = ratio_pressure(ratios.get("center_outer_ratio", 1.0))
            top_center_pressure = ratio_pressure(ratios.get("top_center_ratio", 1.0))
            field_cv_pressure = clamp01(float(ratios.get("field_cv", 0.0) or 0.0) / 2.0)
            entropy_pressure = clamp01(1.0 - float(ratios.get("spatial_entropy_norm", 1.0) or 1.0))
            active_pressure = clamp01(max(source_pressure, bridge_pressure, source_carrier_pressure))
            if active_pressure <= 1e-6:
                rec["status"] = "guarded_report_only"
                rec["reason"] = "source_noise_and_source_image_pressure_zero"
                rec["input_pressures"] = {
                    "source_field_pressure": float(source_pressure),
                    "bridge_pressure": float(bridge_pressure),
                    "source_carrier_pressure": float(source_carrier_pressure),
                }
                rec["source_image_birth_carrier"] = {
                    "status": source_carrier_status,
                    "active": False,
                    "pressure": float(source_carrier_pressure),
                }
                records.append(rec)
                return _return(wan_positive, latent_seed)

            h = int(seed_t.shape[-2])
            w = int(seed_t.shape[-1])
            if h <= 1 or w <= 1:
                rec["status"] = "unavailable"
                rec["reason"] = "latent_seed_spatial_shape_too_small"
                rec["latent_shape"] = list(seed_t.shape)
                records.append(rec)
                return _return(wan_positive, latent_seed)

            max_attenuation = min(
                0.012,
                (0.0015 + 0.0045 * active_pressure + 0.0015 * entropy_pressure) * intensity_multiplier,
            )
            if max_attenuation <= 1e-8:
                rec["status"] = "guarded_report_only"
                rec["reason"] = "intensity_multiplier_zero"
                records.append(rec)
                return _return(wan_positive, latent_seed)

            gain2d = torch.ones((h, w), dtype=torch.float32, device=seed_t.device)
            roi_records = []

            def apply_roi(name, y0, y1, x0, x1, pressure, scale):
                pressure_f = clamp01(pressure)
                if pressure_f <= 0.0:
                    return
                y0i = max(0, min(h - 1, int(round(float(y0) * h))))
                y1i = max(y0i + 1, min(h, int(round(float(y1) * h))))
                x0i = max(0, min(w - 1, int(round(float(x0) * w))))
                x1i = max(x0i + 1, min(w, int(round(float(x1) * w))))
                attenuation = max(0.0, min(max_attenuation, max_attenuation * pressure_f * float(scale)))
                roi_gain = max(1.0 - max_attenuation, 1.0 - attenuation)
                current = gain2d[y0i:y1i, x0i:x1i]
                gain2d[y0i:y1i, x0i:x1i] = torch.minimum(current, torch.full_like(current, float(roi_gain)))
                roi_records.append({
                    "name": name,
                    "pressure": round(float(pressure_f), 6),
                    "attenuation": round(float(attenuation), 6),
                    "gain": round(float(roi_gain), 6),
                })

            background_pressure = clamp01(0.55 * active_pressure + 0.25 * top_center_pressure + 0.20 * field_cv_pressure)
            side_pressure = clamp01(0.45 * active_pressure + 0.30 * center_outer_pressure + 0.25 * entropy_pressure)
            bottom_pressure = clamp01(0.35 * active_pressure + 0.35 * center_outer_pressure + 0.15 * field_cv_pressure)
            apply_roi("top_source_noise_band", 0.00, 0.26, 0.00, 1.00, background_pressure, 1.00)
            apply_roi("left_source_noise_edge", 0.00, 1.00, 0.00, 0.18, side_pressure, 0.75)
            apply_roi("right_source_noise_edge", 0.00, 1.00, 0.82, 1.00, side_pressure, 0.75)
            apply_roi("bottom_source_noise_band", 0.74, 1.00, 0.00, 1.00, bottom_pressure, 0.55)

            if h >= 5 and w >= 5:
                gain2d = F.avg_pool2d(
                    gain2d[None, None, :, :],
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    count_include_pad=False,
                )[0, 0]

            def build_microdetail_protection_mask(mask_h, mask_w, device):
                protection = torch.ones((mask_h, mask_w), dtype=torch.float32, device=device)
                protection_records = []

                def apply_guard(name, y0, y1, x0, x1, multiplier):
                    y0i = max(0, min(mask_h - 1, int(round(float(y0) * mask_h))))
                    y1i = max(y0i + 1, min(mask_h, int(round(float(y1) * mask_h))))
                    x0i = max(0, min(mask_w - 1, int(round(float(x0) * mask_w))))
                    x1i = max(x0i + 1, min(mask_w, int(round(float(x1) * mask_w))))
                    mult = max(0.0, min(1.0, float(multiplier)))
                    current = protection[y0i:y1i, x0i:x1i]
                    protection[y0i:y1i, x0i:x1i] = torch.minimum(current, torch.full_like(current, mult))
                    protection_records.append({
                        "name": name,
                        "multiplier": round(mult, 6),
                        "box": [round(float(y0), 4), round(float(y1), 4), round(float(x0), 4), round(float(x1), 4)],
                    })

                apply_guard("upper_identity_detail_guard", 0.08, 0.48, 0.18, 0.82, 0.45)
                apply_guard("central_action_microdetail_guard", 0.18, 0.86, 0.18, 0.82, 0.40)
                apply_guard("lower_action_microdetail_guard", 0.48, 0.94, 0.23, 0.77, 0.30)
                if mask_h >= 5 and mask_w >= 5:
                    protection = F.avg_pool2d(
                        protection[None, None, :, :],
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        count_include_pad=False,
                    )[0, 0]
                protection = protection.clamp(0.20, 1.0)
                return protection, protection_records

            def build_additive_anchor_mask(mask_h, mask_w, device):
                anchor = torch.zeros((mask_h, mask_w), dtype=torch.float32, device=device)
                anchor_records = []

                def apply_anchor(name, y0, y1, x0, x1, multiplier):
                    y0i = max(0, min(mask_h - 1, int(round(float(y0) * mask_h))))
                    y1i = max(y0i + 1, min(mask_h, int(round(float(y1) * mask_h))))
                    x0i = max(0, min(mask_w - 1, int(round(float(x0) * mask_w))))
                    x1i = max(x0i + 1, min(mask_w, int(round(float(x1) * mask_w))))
                    mult = max(0.0, min(1.0, float(multiplier)))
                    current = anchor[y0i:y1i, x0i:x1i]
                    anchor[y0i:y1i, x0i:x1i] = torch.maximum(current, torch.full_like(current, mult))
                    anchor_records.append({
                        "name": name,
                        "multiplier": round(mult, 6),
                        "box": [round(float(y0), 4), round(float(y1), 4), round(float(x0), 4), round(float(x1), 4)],
                    })

                apply_anchor("top_background_additive_anchor", 0.00, 0.24, 0.00, 1.00, 0.85)
                apply_anchor("left_silhouette_additive_anchor", 0.00, 1.00, 0.00, 0.18, 0.70)
                apply_anchor("right_silhouette_additive_anchor", 0.00, 1.00, 0.82, 1.00, 0.70)
                apply_anchor("bottom_grounding_additive_anchor", 0.76, 1.00, 0.00, 1.00, 0.55)
                if mask_h >= 5 and mask_w >= 5:
                    anchor = F.avg_pool2d(
                        anchor[None, None, :, :],
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        count_include_pad=False,
                    )[0, 0]
                return anchor.clamp(0.0, 1.0), anchor_records

            view_shape = [1] * (len(seed_t.shape) - 2) + [h, w]
            gain = gain2d.reshape(view_shape).to(dtype=seed_t.dtype, device=seed_t.device)
            latent_detail_protection2d, latent_detail_protection_records = build_microdetail_protection_mask(h, w, seed_t.device)
            latent_additive_anchor2d, latent_additive_anchor_records = build_additive_anchor_mask(h, w, seed_t.device)
            microdetail_guard["spatial_microdetail_protection"] = {
                "version": "R168_spatial_microdetail_protection_v3",
                "latent_min": float(latent_detail_protection2d.min().detach().cpu().item()),
                "latent_mean": float(latent_detail_protection2d.mean().detach().cpu().item()),
                "latent_max": float(latent_detail_protection2d.max().detach().cpu().item()),
                "records": latent_detail_protection_records,
                "formula": "Central detail zones receive less source-carrier pressure; edge/top/bottom anchors can still return coarse source topology to Strategy.",
            }
            microdetail_guard["additive_anchor_mask"] = {
                "version": "R168_anchor_only_additive_mask_v2",
                "latent_min": float(latent_additive_anchor2d.min().detach().cpu().item()),
                "latent_mean": float(latent_additive_anchor2d.mean().detach().cpu().item()),
                "latent_max": float(latent_additive_anchor2d.max().detach().cpu().item()),
                "records": latent_additive_anchor_records,
                "formula": "Additive source-image pressure is allowed only on scene/silhouette anchors; central action detail uses model-native denoise instead of an injected source carrier.",
            }
            birth_amplitude = 0.0
            if source_bias2d is not None and source_carrier_pressure > 1e-6:
                birth_amplitude = min(
                    0.0024,
                    (0.00030 + 0.00120 * source_carrier_pressure) * intensity_multiplier,
                )
            if birth_amplitude > 1e-9:
                latent_bias2d = source_bias2d_guarded if source_bias2d_guarded is not None else source_bias2d
                latent_bias2d = (
                    latent_bias2d.to(dtype=torch.float32, device=seed_t.device)
                    * latent_detail_protection2d
                    * latent_additive_anchor2d
                ).clamp(-1.0, 1.0)
                source_bias = latent_bias2d.reshape(view_shape).to(dtype=seed_t.dtype, device=seed_t.device)
                shaped = (seed_t * gain + source_bias * float(birth_amplitude)).to(dtype=seed_t.dtype, device=seed_t.device)
            else:
                shaped = (seed_t * gain).to(dtype=seed_t.dtype, device=seed_t.device)

            if isinstance(latent_seed, dict) and "samples" in latent_seed:
                out = dict(latent_seed)
                out["samples"] = shaped
            else:
                out = shaped

            conditioning_positive = wan_positive
            conditioning_rec = {
                "status": "unavailable",
                "active": False,
                "replaced": False,
                "reason": "wan_positive_or_concat_latent_image_unavailable",
            }
            concat_latent = _find_named_tensor(wan_positive, "concat_latent_image")
            if concat_latent is not None and source_bias2d is not None and birth_amplitude > 1e-9:
                try:
                    concat_t = torch.nan_to_num(concat_latent.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
                    if hasattr(concat_t, "shape") and len(concat_t.shape) >= 2:
                        ch = int(concat_t.shape[-2])
                        cw = int(concat_t.shape[-1])
                    else:
                        ch = 0
                        cw = 0
                    if ch > 1 and cw > 1:
                        cond_gain2d = gain2d.to(dtype=torch.float32, device=concat_t.device)
                        if int(cond_gain2d.shape[-2]) != ch or int(cond_gain2d.shape[-1]) != cw:
                            cond_gain2d = F.interpolate(
                                cond_gain2d[None, None, :, :],
                                size=(ch, cw),
                                mode="bilinear",
                                align_corners=False,
                            )[0, 0]
                        # Conditioning is closest to the source-image carrier. R168 keeps gain as a
                        # very weak topology hint, while additive bias is allowed only on anchors.
                        cond_gain2d = 1.0 - ((1.0 - cond_gain2d).clamp_min(0.0) * 0.18)
                        cond_bias_source = source_bias2d_guarded if source_bias2d_guarded is not None else source_bias2d
                        cond_bias2d = cond_bias_source.to(dtype=torch.float32, device=concat_t.device)
                        if int(cond_bias2d.shape[-2]) != ch or int(cond_bias2d.shape[-1]) != cw:
                            cond_bias2d = F.interpolate(
                                cond_bias2d[None, None, :, :],
                                size=(ch, cw),
                                mode="bilinear",
                                align_corners=False,
                            )[0, 0]
                        cond_detail_protection2d, cond_detail_protection_records = build_microdetail_protection_mask(ch, cw, concat_t.device)
                        cond_additive_anchor2d, cond_additive_anchor_records = build_additive_anchor_mask(ch, cw, concat_t.device)
                        cond_bias2d = (
                            cond_bias2d
                            * cond_detail_protection2d
                            * cond_additive_anchor2d
                        ).clamp(-1.0, 1.0)
                        cond_view_shape = [1] * (len(concat_t.shape) - 2) + [ch, cw]
                        cond_gain = cond_gain2d.reshape(cond_view_shape).to(dtype=concat_latent.dtype, device=concat_t.device)
                        cond_bias = cond_bias2d.reshape(cond_view_shape).to(dtype=concat_latent.dtype, device=concat_t.device)
                        conditioning_amplitude = min(
                            0.0014,
                            (0.00016 + 0.00058 * source_carrier_pressure) * intensity_multiplier,
                        )
                        shaped_concat = (
                            concat_latent * cond_gain + cond_bias * float(conditioning_amplitude)
                        ).to(dtype=concat_latent.dtype, device=concat_t.device)
                        conditioning_positive, concat_replaced = _replace_named_tensor(
                            wan_positive,
                            "concat_latent_image",
                            shaped_concat,
                        )
                        conditioning_rec = {
                            "status": "applied" if concat_replaced else "replace_failed",
                            "active": bool(concat_replaced),
                            "replaced": bool(concat_replaced),
                            "carrier": "wan_positive.concat_latent_image",
                            "amplitude": float(conditioning_amplitude),
                            "gain_policy": {
                                "min_gain": float(cond_gain2d.min().detach().cpu().item()),
                                "mean_gain": float(cond_gain2d.mean().detach().cpu().item()),
                                "max_gain": float(cond_gain2d.max().detach().cpu().item()),
                                "parent_max_attenuation": float(max_attenuation),
                                "R168_conditioning_gain_scale": 0.18,
                            },
                            "microdetail_guard": microdetail_guard,
                            "spatial_microdetail_protection": {
                                "version": "R168_spatial_microdetail_protection_v3",
                                "conditioning_min": float(cond_detail_protection2d.min().detach().cpu().item()),
                                "conditioning_mean": float(cond_detail_protection2d.mean().detach().cpu().item()),
                                "conditioning_max": float(cond_detail_protection2d.max().detach().cpu().item()),
                                "records": cond_detail_protection_records,
                            },
                            "additive_anchor_mask": {
                                "version": "R168_anchor_only_additive_mask_v2",
                                "conditioning_min": float(cond_additive_anchor2d.min().detach().cpu().item()),
                                "conditioning_mean": float(cond_additive_anchor2d.mean().detach().cpu().item()),
                                "conditioning_max": float(cond_additive_anchor2d.max().detach().cpu().item()),
                                "records": cond_additive_anchor_records,
                            },
                            "concat_shape": [int(x) for x in list(concat_latent.shape)],
                            "policy": "tiny_low_frequency_source_topology_bias_with_anchor_only_additive_mask_on_wan_positive_concat_latent_image",
                        }
                        if concat_replaced:
                            self._math_tensor_summary(
                                shaped_concat,
                                records,
                                f"EventMath_{route_label or 'segment'}_source_conditioning_birth_shaped",
                                reference=concat_latent,
                                strict=False,
                            )
                    else:
                        conditioning_rec = {
                            "status": "unavailable",
                            "active": False,
                            "replaced": False,
                            "reason": "concat_latent_spatial_shape_too_small",
                            "concat_shape": [int(x) for x in list(concat_latent.shape)],
                        }
                except Exception as e:
                    conditioning_rec = {
                        "status": "failed_passthrough",
                        "active": False,
                        "replaced": False,
                        "error": str(e)[:240],
                    }

            rec.update({
                "status": "applied",
                "active_tensor_mutation_applied": True,
                "active_conditioning_mutation_applied": bool(conditioning_rec.get("active")),
                "input_pressures": {
                    "source_field_pressure": float(source_pressure),
                    "bridge_pressure": float(bridge_pressure),
                    "active_pressure": float(active_pressure),
                    "center_outer_pressure": float(center_outer_pressure),
                    "top_center_pressure": float(top_center_pressure),
                    "field_cv_pressure": float(field_cv_pressure),
                    "entropy_pressure": float(entropy_pressure),
                    "source_carrier_pressure": float(source_carrier_pressure),
                    "background_pressure": float(background_pressure),
                    "side_pressure": float(side_pressure),
                    "bottom_pressure": float(bottom_pressure),
                },
                "gain_policy": {
                    "max_attenuation": float(max_attenuation),
                    "min_gain": float(gain2d.min().detach().cpu().item()),
                    "mean_gain": float(gain2d.mean().detach().cpu().item()),
                    "max_gain": float(gain2d.max().detach().cpu().item()),
                    "feathered": bool(h >= 5 and w >= 5),
                    "rois": roi_records,
                },
                "source_image_birth_carrier": {
                    "status": source_carrier_status,
                    "active": bool(birth_amplitude > 1e-9),
                    "pressure": float(source_carrier_pressure),
                    "amplitude": float(birth_amplitude),
                    "policy": "tiny_guarded_low_frequency_source_image_bias_with_anchor_only_additive_mask",
                },
                "microdetail_guard": microdetail_guard,
                "source_conditioning_birth_carrier": conditioning_rec,
                "latent_shape": list(seed_t.shape),
                "policy": "tiny_feathered_pre_high_seed_gain_and_anchor_only_additive_source_conditioning_birth_carrier_no_prompt_no_cfg_no_sampler_replacement",
            })
            records.append(rec)
            self._math_tensor_summary(
                out,
                records,
                f"EventMath_{route_label or 'segment'}_source_noise_birth_shaped",
                reference=latent_seed,
                strict=False,
            )
            return _return(conditioning_positive, out)
        except Exception as e:
            rec["status"] = "failed_passthrough"
            rec["error"] = str(e)
            records.append(rec)
            return _return(wan_positive, latent_seed)

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
        pressure_pixel_candidate = {}
        local_spatial_guard = {}
        if isinstance(surface_rec, dict):
            pressure_pixel_candidate = surface_rec.get("pressure_pixel_reweighting_active_candidate", {}) or {}
            if not isinstance(pressure_pixel_candidate, dict):
                pressure_pixel_candidate = {}
            local_spatial_guard = pressure_pixel_candidate.get("local_spatial_pressure_guard", {}) or {}
            if not isinstance(local_spatial_guard, dict):
                local_spatial_guard = {}
        local_guard_inputs = (
            local_spatial_guard.get("input_pressures", {})
            if isinstance(local_spatial_guard.get("input_pressures", {}), dict)
            else {}
        )
        local_spatial_active = bool(
            mode == "PRESSURE_PIXEL_REWEIGHTING"
            and branch_key == "low"
            and str(local_spatial_guard.get("status", "") or "") == "active"
            and bool(local_spatial_guard.get("active_control_allowed", False))
        )
        if local_spatial_active:
            local_background_pressure = clamp01(local_guard_inputs.get("spatial_background_pressure", 0.0))
            local_action_pressure = clamp01(local_guard_inputs.get("spatial_action_preservation", 0.0))
            local_seam_pressure = clamp01(local_guard_inputs.get("seam_spatial_pressure", 0.0))
            spatial_pressure = max(spatial_pressure, local_action_pressure)
            background_region_pressure = max(background_region_pressure, local_background_pressure)
            background_anchor_pressure = max(background_anchor_pressure, local_background_pressure * 0.65)
            top_band_pressure = max(top_band_pressure, local_background_pressure * 0.50)
            late_segment_pressure = max(late_segment_pressure, local_seam_pressure * 0.35)
            dominant_region = str(dominant_region or "r150_local_spatial_background")
            status = "active"
        carrier_pressure = max(spatial_pressure, background_region_pressure, background_anchor_pressure, top_band_pressure)
        saturated_global_pressure = bool(
            max(spatial_pressure, background_region_pressure, background_anchor_pressure, top_band_pressure, late_segment_pressure) >= 0.995
            and min(spatial_pressure, background_region_pressure, background_anchor_pressure, top_band_pressure, late_segment_pressure) >= 0.995
        )
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
            "pressure_pixel_local_spatial_guard": {
                "status": str(local_spatial_guard.get("status", "not_recorded") or "not_recorded"),
                "version": str(local_spatial_guard.get("version", "") or ""),
                "source": str(local_spatial_guard.get("source", "") or ""),
                "active_control_allowed": bool(local_spatial_active),
                "same_run_oracle": bool(local_spatial_guard.get("same_run_oracle", False)),
                "policy": str(local_spatial_guard.get("policy", "") or ""),
            },
            "input_pressures": {
                "spatial_anchor_pressure": float(spatial_pressure),
                "background_region_pressure": float(background_region_pressure),
                "background_anchor_pressure": float(background_anchor_pressure),
                "top_band_pressure": float(top_band_pressure),
                "late_segment_pressure": float(late_segment_pressure),
                "carrier_pressure": float(carrier_pressure),
            },
            "saturated_global_pressure": bool(saturated_global_pressure),
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

            if mode not in ("STRATEGY_PRESSURE_WINDOW", "PRESSURE_PIXEL_REWEIGHTING"):
                rec["reason"] = "requires_strategy_pressure_window_or_pressure_pixel_local_spatial_guard"
                return rec, None
            if mode == "PRESSURE_PIXEL_REWEIGHTING" and not local_spatial_active:
                rec["reason"] = "pressure_pixel_local_spatial_guard_not_active"
                return rec, None
            if mode == "STRATEGY_PRESSURE_WINDOW" and status != "active":
                rec["reason"] = "no_active_cached_spatial_background_evidence"
                return rec, None
            if carrier_pressure < 0.20:
                rec["reason"] = "carrier_pressure_below_threshold"
                return rec, None
            if saturated_global_pressure:
                rec["status"] = "guarded_report_only"
                rec["reason"] = "saturated_global_spatial_pressure_requires_report_only"
                rec["active_control_allowed"] = False
                rec["guard"] = (
                    "All spatial/background/tail pressures are saturated. This is not a region-specific "
                    "topology map, so spatial carrier gain would smear global pressure into the latent delta."
                )
                rec["policy"] = "report_only_when_spatial_pressure_is_saturated_global_not_topological"
                return rec, None
            if (
                mode != "PRESSURE_PIXEL_REWEIGHTING"
                and isinstance(denoise_phase_map, dict)
                and not bool(denoise_phase_map.get("active_control_allowed", False))
            ):
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
            if post_window_delta and mode != "PRESSURE_PIXEL_REWEIGHTING":
                rec["status"] = "guarded_report_only"
                rec["reason"] = "post_window_endpoint_denoise_safety"
                rec["active_control_allowed"] = False
                rec["guard"] = "Post-window delta has no internal denoise phase information; preserving background by lowering delta can freeze late/start endpoint noise."
                return rec, None
            if post_window_delta and mode == "PRESSURE_PIXEL_REWEIGHTING":
                rec["post_window_delta_allowed_by"] = "r150_tiny_low_delta_local_spatial_guard"
            if step_progress is not None and (step_progress <= 0.12 or step_progress >= 0.88):
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
            attenuation2d = torch.zeros((h, w), dtype=torch.float32, device=delta_t.device)
            protection2d = torch.zeros((h, w), dtype=torch.float32, device=delta_t.device)
            roi_gains = []
            protection_gains = []

            if local_spatial_active:
                gain_policy = (
                    local_spatial_guard.get("gain_policy", {})
                    if isinstance(local_spatial_guard.get("gain_policy", {}), dict)
                    else {}
                )
                try:
                    max_attenuation = float(gain_policy.get("max_background_attenuation", 0.0) or 0.0)
                except Exception:
                    max_attenuation = 0.0
                try:
                    min_gain = float(gain_policy.get("min_background_gain", 0.925) or 0.925)
                except Exception:
                    min_gain = 0.925
                max_attenuation = max(0.0, min(0.022, max_attenuation))
                min_gain = max(0.978, min(1.0, min_gain))
            elif branch_key == "high":
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
                current = attenuation2d[y0i:y1i, x0i:x1i]
                attenuation2d[y0i:y1i, x0i:x1i] = torch.maximum(current, torch.full_like(current, float(attenuation)))
                roi_gains.append({
                    "name": name,
                    "pressure": float(pressure_f),
                    "gain": float(region_gain),
                    "y0": int(y0i),
                    "y1": int(y1i),
                    "x0": int(x0i),
                    "x1": int(x1i),
                })

            def apply_protection_roi(name, y0, y1, x0, x1, protection):
                protection_f = clamp01(protection)
                if protection_f <= 0.0:
                    return
                y0i = max(0, min(h - 1, int(round(float(y0) * h))))
                y1i = max(y0i + 1, min(h, int(round(float(y1) * h))))
                x0i = max(0, min(w - 1, int(round(float(x0) * w))))
                x1i = max(x0i + 1, min(w, int(round(float(x1) * w))))
                current = protection2d[y0i:y1i, x0i:x1i]
                protection2d[y0i:y1i, x0i:x1i] = torch.maximum(current, torch.full_like(current, float(protection_f)))
                protection_gains.append({
                    "name": name,
                    "protection": float(protection_f),
                    "y0": int(y0i),
                    "y1": int(y1i),
                    "x0": int(x0i),
                    "x1": int(x1i),
                })

            if local_spatial_active:
                rois = local_spatial_guard.get("background_rois", [])
                if not isinstance(rois, list):
                    rois = []
                for roi in rois:
                    if not isinstance(roi, dict):
                        continue
                    rect = roi.get("rect", [])
                    if not isinstance(rect, (list, tuple)) or len(rect) != 4:
                        continue
                    y0 = float(rect[0])
                    x0 = float(rect[1])
                    hh = float(rect[2])
                    ww = float(rect[3])
                    apply_roi(
                        str(roi.get("name", "pressure_pixel_background_roi") or "pressure_pixel_background_roi"),
                        y0,
                        y0 + hh,
                        x0,
                        x0 + ww,
                        roi.get("pressure", background_region_pressure or carrier_pressure),
                        roi.get("attenuation_scale", 1.0),
                    )
                protection_rois = local_spatial_guard.get("identity_protection_rois", [])
                if not isinstance(protection_rois, list):
                    protection_rois = []
                for roi in protection_rois:
                    if not isinstance(roi, dict):
                        continue
                    rect = roi.get("rect", [])
                    if not isinstance(rect, (list, tuple)) or len(rect) != 4:
                        continue
                    y0 = float(rect[0])
                    x0 = float(rect[1])
                    hh = float(rect[2])
                    ww = float(rect[3])
                    apply_protection_roi(
                        str(roi.get("name", "identity_protection_roi") or "identity_protection_roi"),
                        y0,
                        y0 + hh,
                        x0,
                        x0 + ww,
                        roi.get("protection", 1.0),
                    )
                if background_anchor_pressure > 0.0:
                    apply_roi("r151_outer_source_anchor_left", 0.00, 1.00, 0.00, 0.08, background_anchor_pressure, 0.16)
                    apply_roi("r151_outer_source_anchor_right", 0.00, 1.00, 0.92, 1.00, background_anchor_pressure, 0.16)
            else:
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

            try:
                if local_spatial_active:
                    import torch.nn.functional as F
                    gain_policy = (
                        local_spatial_guard.get("gain_policy", {})
                        if isinstance(local_spatial_guard.get("gain_policy", {}), dict)
                        else {}
                    )
                    kernel = int(gain_policy.get("feather_kernel", 5) or 5)
                    passes = int(gain_policy.get("feather_passes", 2) or 2)
                    kernel = max(3, min(kernel, h if h % 2 == 1 else h - 1, w if w % 2 == 1 else w - 1))
                    if kernel >= 3:
                        pad = kernel // 2
                        att = attenuation2d.view(1, 1, h, w)
                        prot = protection2d.view(1, 1, h, w)
                        for _ in range(max(1, min(4, passes))):
                            att = F.avg_pool2d(att, kernel_size=kernel, stride=1, padding=pad)
                            prot = F.avg_pool2d(prot, kernel_size=kernel, stride=1, padding=pad)
                        attenuation2d = torch.clamp(att.view(h, w), min=0.0, max=float(max_attenuation))
                        protection2d = torch.clamp(prot.view(h, w), min=0.0, max=1.0)
                    attenuation2d = attenuation2d * (1.0 - protection2d)
                gain2d = torch.clamp(1.0 - attenuation2d, min=float(min_gain), max=1.0)
            except Exception as e:
                rec["feather_warning"] = str(e)
                gain2d = torch.clamp(1.0 - attenuation2d, min=float(min_gain), max=1.0)

            view_shape = [1 for _ in range(len(delta_t.shape))]
            view_shape[-2] = h
            view_shape[-1] = w
            gain = gain2d.view(*view_shape)
            rec.update({
                "status": "active",
                "reason": "r151_pressure_pixel_feathered_identity_guard" if local_spatial_active else "cached_spatial_carriers_bound_background_delta",
                "policy": "feathered_identity_pressure_pixel_gain_multiplies_latent_delta" if local_spatial_active else "soft_region_gain_multiplies_latent_delta",
                "active_control_allowed": True,
                "latent_shape": list(delta_t.shape),
                "gain_shape": list(gain.shape),
                "min_gain": float(gain2d.min().detach().cpu().item()),
                "max_gain": float(gain2d.max().detach().cpu().item()),
                "mean_gain": float(gain2d.mean().detach().cpu().item()),
                "max_attenuation": float(max_attenuation),
                "roi_gains": roi_gains,
                "identity_protection_gains": protection_gains,
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

        if mode not in ("LATENT_DELTA_SCALE", "DEEP_STEP_DELTA_CONTROL", "STRATEGY_PRESSURE_WINDOW", "LATENT_MEMORY_BRIDGE", "PRESSURE_PIXEL_REWEIGHTING") or abs(strength_runtime - 1.0) < 1e-9:
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

    def _event_low_mid_window_spatial_control_route_key(self, window, mode):
        branch = str(getattr(window, "branch_name", "branch") or "branch")
        start = int(getattr(window, "start_at_step", 0) or 0)
        end = int(getattr(window, "end_at_step", 0) or 0)
        steps = int(getattr(window, "steps", 0) or 0)
        seed = int(getattr(window, "seed", 0) or 0)
        return (
            f"{branch}|mode={str(mode or '').upper()}|"
            f"window={start}:{end}|steps={steps}|seed={seed}"
        )

    def _event_should_use_low_mid_window_spatial_control(self, window, mode, records=None):
        """
        R126 route selector.

        This is intentionally narrower than DEEP_STEP_DELTA_CONTROL:
        only STRATEGY_PRESSURE_WINDOW + low branch + active boundary/background
        evidence may enter the step-aware loop where SpatialCarrierPreservationMap
        can act in the mid-window denoise phase.
        """
        branch = str(getattr(window, "branch_name", "branch") or "branch")
        branch_lower = branch.lower()
        start = int(getattr(window, "start_at_step", 0) or 0)
        end = int(getattr(window, "end_at_step", 0) or 0)
        window_steps = max(0, end - start)
        route_key = self._event_low_mid_window_spatial_control_route_key(window, mode)
        background_preservation = getattr(self, "_event_background_anchor_preservation_control", {}) or {}
        if not isinstance(background_preservation, dict):
            background_preservation = {}

        def clamp01(value):
            try:
                out = float(value)
            except Exception:
                out = 0.0
            if not math.isfinite(out):
                out = 0.0
            return max(0.0, min(1.0, out))

        pressures = {
            "background_anchor_pressure": clamp01(background_preservation.get("background_anchor_pressure", 0.0)),
            "spatial_anchor_pressure": clamp01(background_preservation.get("spatial_anchor_pressure", 0.0)),
            "background_region_pressure": clamp01(background_preservation.get("background_region_pressure", 0.0)),
            "top_band_pressure": clamp01(background_preservation.get("top_band_pressure", 0.0)),
            "late_segment_pressure": clamp01(background_preservation.get("late_segment_pressure", 0.0)),
        }
        carrier_pressure = max(pressures.values()) if pressures else 0.0
        saturated_global_pressure = bool(
            carrier_pressure >= 0.995
            and min(pressures.values()) >= 0.995
        )
        reasons = []
        active = True
        if str(mode or "").upper() != "STRATEGY_PRESSURE_WINDOW":
            active = False
            reasons.append("requires_strategy_pressure_window")
        if "low" not in branch_lower:
            active = False
            reasons.append("low_branch_only")
        if str(background_preservation.get("status", "") or "") != "active":
            active = False
            reasons.append("no_active_boundary_background_evidence")
        if carrier_pressure < 0.20:
            active = False
            reasons.append("carrier_pressure_below_threshold")
        if saturated_global_pressure:
            active = False
            reasons.append("saturated_global_spatial_pressure_requires_report_only")
        if window_steps < 3:
            active = False
            reasons.append("window_too_short_for_mid_window_control")

        mid_window_expected_steps = []
        try:
            for local_index in range(window_steps):
                progress = float((local_index + 1) / max(1, window_steps))
                if progress > 0.12 and progress < 0.88:
                    mid_window_expected_steps.append(int(start + local_index))
        except Exception:
            mid_window_expected_steps = []

        rec = {
            "stage": f"EventR126LowMidWindowSpatialControlRoute_{branch}",
            "status": "active" if active else ("guarded_report_only" if saturated_global_pressure else "bypass"),
            "version": "low_mid_window_spatial_route_guard_v2",
            "branch_name": branch,
            "route_key": route_key,
            "math_control_mode": str(mode or ""),
            "start_at_step": int(start),
            "end_at_step": int(end),
            "window_steps": int(window_steps),
            "activation_scope": "low_branch_mid_window_only",
            "additional_sampler_calls": int(max(0, window_steps - 1)) if active else 0,
            "estimated_sampler_call_multiplier": int(max(1, window_steps)) if active else 1,
            "mid_window_expected_steps": mid_window_expected_steps,
            "mid_window_expected_step_count": int(len(mid_window_expected_steps)),
            "public_default_safe": bool(active and str(mode or "").upper() == "STRATEGY_PRESSURE_WINDOW"),
            "risk_boundary": "active only after selected-tail background evidence; high branch and endpoint steps stay guarded",
            "background_anchor_preservation_status": str(background_preservation.get("status", "") or ""),
            "dominant_background_region": str(background_preservation.get("dominant_background_region", "") or ""),
            "input_pressures": pressures,
            "carrier_pressure": float(carrier_pressure),
            "saturated_global_pressure": bool(saturated_global_pressure),
            "reasons": reasons,
            "policy": (
                "report_only_when_spatial_pressure_is_saturated_global_not_topological"
                if saturated_global_pressure
                else "use_step_aware_low_mid_window_only_when_boundary_background_evidence_is_active"
            ),
            "guard": (
                "All spatial/background/tail pressures are saturated. This is not a local topology map, "
                "so the route must not replace the low sampler with a step loop."
                if saturated_global_pressure else ""
            ),
            "prompt_text_injection_allowed": False,
            "semantic_math_in_prompt_allowed": False,
            "formula_role": "selected-tail background/source evidence returns to low mid-window ObservedBehavior control",
            "formula": "Outcome(t-1 selected tail) + ObservedBehavior(t-1 background drift) returns to Strategy(t) before low mid-window delta becomes Outcome(t+1).",
        }
        if records is not None:
            records.append(rec)
        self._event_low_mid_window_spatial_control_route_last = rec
        return bool(active), rec

    def _event_sample_window_math_native(self, model, positive, negative, latent, window, records, route_rec=None):
        """
        Event-native active math sampler loop.
        Runs one denoise step at a time and applies delta control per step.
        """
        branch = str(getattr(window, "branch_name", "branch") or "branch")
        start = int(getattr(window, "start_at_step", 0) or 0)
        end = int(getattr(window, "end_at_step", 0) or 0)
        window_steps = max(0, end - start)
        route_rec = route_rec if isinstance(route_rec, dict) else {}
        if not isinstance(route_rec, dict):
            route_rec = {}
        mode = str(getattr(self, "_event_math_control_mode", "OBSERVE_ONLY") or "OBSERVE_ONLY").upper()
        current_route_key = self._event_low_mid_window_spatial_control_route_key(window, mode)
        route_key_matches = bool(route_rec.get("route_key", "") == current_route_key)
        replacement_reason = "deep_step_delta_control"
        if mode == "STRATEGY_PRESSURE_WINDOW" and route_rec.get("status") == "active" and route_key_matches:
            replacement_reason = "r126_low_mid_window_spatial_route_guard"
        elif mode == "STRATEGY_PRESSURE_WINDOW":
            replacement_reason = "strategy_pressure_window_route_guard_mismatch"

        event_records = [{
            "stage": "event_sampler_begin",
            "status": "begin",
            "branch_name": branch,
            "branch_role": getattr(window, "branch_role", ""),
            "start_at_step": start,
            "end_at_step": end,
            "replacement_layer": "event_native_math_loop",
            "replacement_reason": replacement_reason,
            "r126_low_mid_window_spatial_control": route_rec if replacement_reason.startswith("r126") else {},
            "route_key": current_route_key,
            "route_key_matches": bool(route_key_matches),
            "activation_scope": str(route_rec.get("activation_scope", "") or ""),
            "additional_sampler_calls": int(route_rec.get("additional_sampler_calls", 0) or 0) if replacement_reason.startswith("r126") else 0,
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
        low_mid_window_spatial_control, low_mid_window_route_rec = self._event_should_use_low_mid_window_spatial_control(
            window,
            mode,
            records,
        )
        use_native_math_loop = (mode == "DEEP_STEP_DELTA_CONTROL") or bool(low_mid_window_spatial_control)

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
                "status": (
                    "native_low_mid_window_spatial_control_active"
                    if low_mid_window_spatial_control
                    else "native_sampler_preserved_unified_pressure_window"
                ),
                "math_control_mode": mode,
                "native_step_loop_replacement": bool(low_mid_window_spatial_control),
                "r126_low_mid_window_spatial_control": {
                    "status": low_mid_window_route_rec.get("status", "") if isinstance(low_mid_window_route_rec, dict) else "",
                    "version": low_mid_window_route_rec.get("version", "") if isinstance(low_mid_window_route_rec, dict) else "",
                    "route_key": low_mid_window_route_rec.get("route_key", "") if isinstance(low_mid_window_route_rec, dict) else "",
                    "activation_scope": low_mid_window_route_rec.get("activation_scope", "") if isinstance(low_mid_window_route_rec, dict) else "",
                    "carrier_pressure": low_mid_window_route_rec.get("carrier_pressure", None) if isinstance(low_mid_window_route_rec, dict) else None,
                    "input_pressures": low_mid_window_route_rec.get("input_pressures", {}) if isinstance(low_mid_window_route_rec, dict) else {},
                    "additional_sampler_calls": low_mid_window_route_rec.get("additional_sampler_calls", 0) if isinstance(low_mid_window_route_rec, dict) else 0,
                    "mid_window_expected_steps": low_mid_window_route_rec.get("mid_window_expected_steps", []) if isinstance(low_mid_window_route_rec, dict) else [],
                    "public_default_safe": low_mid_window_route_rec.get("public_default_safe", False) if isinstance(low_mid_window_route_rec, dict) else False,
                    "risk_boundary": low_mid_window_route_rec.get("risk_boundary", "") if isinstance(low_mid_window_route_rec, dict) else "",
                    "reasons": low_mid_window_route_rec.get("reasons", []) if isinstance(low_mid_window_route_rec, dict) else [],
                },
                "formula": (
                    "R126: low branch may enter a narrow step-aware mid-window spatial carrier route when selected-tail background evidence is active and the route key matches the current sampler window."
                    if low_mid_window_spatial_control
                    else "Math acts through one bounded Strategy pressure window after the native sampler window, then returns to S_global_event_route."
                ),
            })
        elif mode in ("LATENT_MEMORY_BRIDGE", "MAX_RISK_STRATEGY_RING"):
            records.append({
                "stage": "EventMathSamplerPathPolicy",
                "status": (
                    "native_sampler_preserved_max_risk_strategy_ring"
                    if mode == "MAX_RISK_STRATEGY_RING"
                    else "native_sampler_preserved_segment_entry_latent_memory_bridge"
                ),
                "math_control_mode": mode,
                "native_step_loop_replacement": False,
                "formula": (
                    "MAX_RISK_STRATEGY_RING may apply a tiny hard-guard override before high sampler entry; high/low sampler loop stays model-native."
                    if mode == "MAX_RISK_STRATEGY_RING"
                    else "Bounded latent memory is applied before high sampler entry; high/low sampler physics stay model-native."
                ),
            })
        elif mode == "SOURCE_NOISE_FIELD_SHAPING":
            records.append({
                "stage": "EventMathSamplerPathPolicy",
                "status": "native_sampler_preserved_source_noise_birth_shaping",
                "math_control_mode": mode,
                "native_step_loop_replacement": False,
                "formula": (
                    "R171 may apply tiny guarded source/noise spatial gain plus anchor-only low-frequency source-image birth carrier to the Wan latent seed and Wan positive concat conditioning, and a tiny two-slice selected-tail post-drop seam-entry latent echo with regional/background protection before high sampler. "
                    "The high/low KSampler route, CFG, and prompt text remain model-native."
                ),
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
            result = self._event_sample_window_math_native(
                model,
                positive,
                negative,
                latent,
                window,
                records,
                route_rec=low_mid_window_route_rec if low_mid_window_spatial_control else None,
            )
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




