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

EVENT_HORIZON_RUNTIME_VERSION = "0.1.1-r61"
EVENT_HORIZON_RUNTIME_NAME = "Singularity R61 Pause UI Hotfix"
EVENT_HORIZON_BODY_VERSION = "0.1-r61"


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


class SingularityTelemetryMixin:
    def _event_core_live_record(self, stage_name, record_type="stage", status="", formula_role="", route_id="", observed_behavior="", metadata=None):
        # Live recording disabled: direct cut against excessive comfort-observability.
        return None

    def _event_core_body_collect_from_records(self, packet, execution_records):
        packet = ensure_packet(packet)
        body = packet.setdefault("metadata", {}).setdefault("event_core_body", {})
        records = [r for r in (execution_records or []) if isinstance(r, dict)]
        stages = [str(r.get("stage", "") or "") for r in records]
        stage_math = [s for s in stages if s.startswith("EventUniversalMath_")]
        boundary_math = [s for s in stages if s.startswith("EventUniversalBoundary_")]
        math_tensor = [s for s in stages if s.startswith("EventMath_")]

        body["collection_disabled"] = False
        body["collection_mode"] = "runtime_record_derived_minimal"
        body["stage_math_count"] = len(stage_math)
        body["boundary_count"] = len(boundary_math)
        body["math_tensor_record_count"] = len(math_tensor)
        body["s_wire_count"] = int(body.get("s_wire_count", 0) or 0)
        body["live_route_count"] = int(body.get("live_route_count", 0) or 0)
        body["runtime_monitor_count"] = int(body.get("runtime_monitor_count", 0) or 0)
        return packet

    def _event_strategy_matrix_from_records(self, execution_records, result_status="", saved_video_path=""):
        """
        Report-only map of places where carriers collide into Strategy.
        This is deliberately observer-only: it creates evidence records, never sampler control.
        """
        records = [r for r in (execution_records or []) if isinstance(r, dict)]

        def stage_text(record):
            parts = [
                str(record.get("stage", "") or ""),
                str(record.get("status", "") or ""),
                str(record.get("formula", "") or ""),
                str(record.get("formula_role", "") or ""),
                str(record.get("observed_behavior", "") or ""),
                str(record.get("route_id", "") or ""),
                str(record.get("branch_name", "") or ""),
                str(record.get("branch", "") or ""),
            ]
            return " ".join(parts).lower()

        def hits(*needles):
            needles_l = [str(n or "").lower() for n in needles if str(n or "").strip()]
            out = []
            for rec in records:
                text = stage_text(rec)
                if any(n in text for n in needles_l):
                    out.append(str(rec.get("stage", "") or "unknown"))
            return out

        categories = {
            "text_positive": hits("eventtextencodepositive", "textencodepositive"),
            "text_negative": hits("eventtextencodenegative", "textencodenegative"),
            "text": hits("eventtextencode", "text encode", "conditioning"),
            "image": hits("eventimagescalestart", "image scale", "source image", "eventimagescale"),
            "latent_seed": hits("eventwanimagetovideoseed", "wan_i2v_latent_seed", "wan image to video"),
            "high_sampler": hits("eventsamplerhigh", "samplerhigh", "branch_name high", "branch high"),
            "low_sampler": hits("eventsamplerlow", "samplerlow", "branch_name low", "branch low"),
            "delta_control": hits("eventmathdeltacontrol", "latent_delta_scale", "delta control"),
            "cfg_policy": hits("eventstrategycfg", "eventmathsamplerpathpolicy", "cfg_policy"),
            "motion": hits("eventmath_decoded_frame_motion", "eventmath_concatenated_frame_motion", "frame_motion"),
            "cascade": hits("singularitycascadebegin", "singularitycascadesegmentend", "singularitycascadeend"),
            "pause": hits("singularitycascadepause", "singularitycascaderesume", "mirrorcut"),
            "boundary": hits("eventuniversalboundary", "cascadeboundary", "cascade boundary"),
            "tail_formula": hits("formula_tail_mirror_break", "tailframesselect", "admissible_continuation"),
            "video": hits("eventvideocombine", "videosave", "saved_video_path"),
        }
        categories["text_any"] = sorted(set(categories["text_positive"] + categories["text_negative"] + categories["text"]))

        numeric_keys = (
            "raw_delta_norm",
            "native_delta_norm",
            "effective_delta_norm",
            "low_effective_delta_norm",
            "low_native_delta_norm",
            "delta_norm",
            "relative_delta",
            "strength_runtime",
            "base_strength",
            "coupling_multiplier",
            "step_schedule_factor",
            "resume_frame_index",
            "latent_temporal_target_t",
            "actual_output_frames",
            "total_requested_frames",
            "frames",
            "fps",
            "candidate_index",
            "mirror_break",
            "admissible_continuation",
            "past_strategy_proxy",
            "observed_for_candidate",
            "bounded_signal",
            "frame_delta_norm_mean",
            "frame_delta_norm_std",
            "frame_delta_norm_cv_ratio",
            "frame_delta_spike_ratio",
            "frame_delta_reversal_ratio",
            "frame_delta_jerk_ratio",
            "frame_motion_stability_score",
        )

        def numeric_snapshots(stage_hits):
            stage_hit_set = set(stage_hits or [])
            snapshots = []
            for rec in records:
                if str(rec.get("stage", "") or "") not in stage_hit_set:
                    continue
                snap = {}
                for key in numeric_keys:
                    if key not in rec:
                        continue
                    value = rec.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        if isinstance(value, float) and not math.isfinite(value):
                            value = str(value)
                        snap[key] = value
                if snap:
                    snap["stage"] = str(rec.get("stage", "") or "unknown")
                    snapshots.append(snap)
            if len(snapshots) > 12:
                return snapshots[:4] + snapshots[-8:]
            return snapshots

        def safe_float(value, default=None):
            try:
                out = float(value)
            except Exception:
                return default
            return out if math.isfinite(out) else default

        def clamp01(value):
            value = safe_float(value, 0.0)
            return max(0.0, min(1.0, value))

        def snapshot_values(snapshots, key):
            values = []
            for snap in snapshots or []:
                value = safe_float(snap.get(key), None)
                if value is not None:
                    values.append(value)
            return values

        def latest_snapshot_value(snapshots, key, default=None):
            for snap in reversed(snapshots or []):
                value = safe_float(snap.get(key), None)
                if value is not None:
                    return value
            return default

        def compute_collision_scores(collision_id, status, missing_conflict, snapshots):
            status = str(status or "")
            conflict_score = clamp01(missing_conflict)
            drift_score = None
            basis = {
                "base_missing_evidence_score": conflict_score,
                "metrics_used": [],
                "score_version": "strategy_collision_scores_v1_report_only",
            }

            if status == "not_applicable_single_segment":
                basis["reason"] = "single segment has no tail/next-source boundary"
                return {
                    "conflict_score": 0.0,
                    "drift_score": None,
                    "basis": basis,
                    "dynamic_recommendation": "No cascade boundary exists in this run; use a pause/resume run to score tail_next_source.",
                }

            if collision_id == "previous_next_frame_motion":
                stability = latest_snapshot_value(snapshots, "frame_motion_stability_score", None)
                spike = latest_snapshot_value(snapshots, "frame_delta_spike_ratio", 1.0)
                reversal = latest_snapshot_value(snapshots, "frame_delta_reversal_ratio", 0.0)
                jerk = latest_snapshot_value(snapshots, "frame_delta_jerk_ratio", 0.0)
                if stability is not None:
                    motion_conflict = clamp01(1.0 - stability)
                    conflict_score = max(conflict_score, motion_conflict)
                    drift_score = clamp01(
                        0.45 * motion_conflict
                        + 0.25 * clamp01((float(spike) - 1.0) / 1.5)
                        + 0.20 * clamp01(float(reversal))
                        + 0.10 * clamp01(float(jerk))
                    )
                    basis["metrics_used"].extend([
                        "frame_motion_stability_score",
                        "frame_delta_spike_ratio",
                        "frame_delta_reversal_ratio",
                        "frame_delta_jerk_ratio",
                    ])
                    basis["motion"] = {
                        "stability": stability,
                        "spike": spike,
                        "reversal": reversal,
                        "jerk": jerk,
                    }

            elif collision_id == "tail_next_source":
                mirror_values = snapshot_values(snapshots, "mirror_break")
                admissible_values = snapshot_values(snapshots, "admissible_continuation")
                if mirror_values or admissible_values:
                    best_mirror = min(mirror_values) if mirror_values else None
                    best_admissible = max(admissible_values) if admissible_values else None
                    drift_score = clamp01(best_mirror if best_mirror is not None else 1.0 - best_admissible)
                    conflict_score = max(conflict_score, 0.0 if status == "observed" else drift_score)
                    best_candidate = None
                    if mirror_values:
                        for snap in snapshots or []:
                            if safe_float(snap.get("mirror_break"), None) == best_mirror:
                                best_candidate = safe_float(snap.get("candidate_index"), None)
                                break
                    basis["metrics_used"].extend(["mirror_break", "admissible_continuation"])
                    basis["tail_formula"] = {
                        "best_mirror_break": best_mirror,
                        "best_admissible_continuation": best_admissible,
                        "best_candidate_index": int(best_candidate) if best_candidate is not None else None,
                    }

            elif collision_id == "high_low_sampler_strategy":
                bounded = latest_snapshot_value(snapshots, "bounded_signal", None)
                high_raw_values = snapshot_values(snapshots, "raw_delta_norm")
                strength_values = snapshot_values(snapshots, "strength_runtime")
                if bounded is not None:
                    drift_score = clamp01(bounded)
                    basis["metrics_used"].append("bounded_signal")
                if high_raw_values:
                    basis["raw_delta_norm_max"] = max(high_raw_values)
                    basis["metrics_used"].append("raw_delta_norm")
                if strength_values:
                    max_strength_delta = max(abs(v - 1.0) for v in strength_values)
                    if max_strength_delta > 1e-9 and drift_score is not None:
                        conflict_score = max(conflict_score, clamp01(max_strength_delta * drift_score))
                    basis["strength_runtime_values"] = strength_values[:6]

            elif collision_id == "visible_video_outcome":
                actual = latest_snapshot_value(snapshots, "actual_output_frames", None)
                total = latest_snapshot_value(snapshots, "total_requested_frames", None)
                if actual is not None:
                    basis["metrics_used"].append("actual_output_frames")
                    basis["actual_output_frames"] = actual
                if total is not None:
                    basis["metrics_used"].append("total_requested_frames")
                    basis["total_requested_frames"] = total
                if str(result_status or "").upper() != "VIDEO" or not str(saved_video_path or "").strip():
                    conflict_score = 1.0
                    drift_score = 1.0
                    basis["reason"] = "final visible video Outcome is missing"

            dynamic_recommendation = ""
            if drift_score is not None and drift_score >= 0.55:
                dynamic_recommendation = "High drift evidence: keep this collision report-only and run a fixed-seed A/B before enabling any bounded control."
            elif conflict_score >= 0.5:
                dynamic_recommendation = "Missing or conflicting evidence: collect a focused run before using this collision for guidance."

            return {
                "conflict_score": clamp01(conflict_score),
                "drift_score": clamp01(drift_score) if drift_score is not None else None,
                "basis": basis,
                "dynamic_recommendation": dynamic_recommendation,
            }

        micro_formula_blueprints = {
            "prompt_image_anchor": {
                "left_outcome": ["source_image", "image_anchor", "visible_pose/layout"],
                "left_observed_behavior": ["positive_prompt_direction", "negative_prompt_boundary", "clip_encoding_behavior"],
                "strategy_point": "The model must read prompt meaning and source image as one admissible scene.",
                "right_observed_behavior": ["conditioning_pressure_on_latent_seed", "prompt-image contradiction or agreement"],
                "right_outcome": ["wan_latent_seed", "image-conditioned StrategyCarrier"],
                "collision_math": ["semantic_anchor_agreement_score", "prompt_image_contradiction_score", "conditioning_preservation_score"],
                "intervention_surface": ["prompt relation proposal", "conditioning relation weighting", "report warning before active control"],
                "public_safe_control": "proposal_only",
            },
            "positive_negative_prompt_polarity": {
                "left_outcome": ["positive_prompt_carrier"],
                "left_observed_behavior": ["negative_prompt_countervector"],
                "strategy_point": "Positive and negative conditioning must define a clean semantic corridor, not erase required scene traits.",
                "right_observed_behavior": ["cfg_boundary_pressure", "conditioning separation"],
                "right_outcome": ["bounded conditioning pair"],
                "collision_math": ["prompt_overlap_score", "negative_conflict_score", "semantic_corridor_width"],
                "intervention_surface": ["prompt warning", "negative prompt cleanup proposal", "future conditioning mask research"],
                "public_safe_control": "proposal_only",
            },
            "image_latent_noise_seed": {
                "left_outcome": ["source_image", "scaled_image"],
                "left_observed_behavior": ["wan_image_to_video_seed_projection", "noise field initialization"],
                "strategy_point": "The source image must survive conversion into the latent/noise possibility field.",
                "right_observed_behavior": ["latent_seed_anchor_drift", "seed reproducibility behavior"],
                "right_outcome": ["wan_latent_seed"],
                "collision_math": ["latent_anchor_norm", "seed_anchor_stability", "image_to_latent_preservation"],
                "intervention_surface": ["seed/report evidence", "future anchor preservation proposal", "no prompt rewriting"],
                "public_safe_control": "report_only",
            },
            "high_low_sampler_strategy": {
                "left_outcome": ["latent_after_high", "high sampler OutcomeNext"],
                "left_observed_behavior": ["high_delta", "high sampler trajectory"],
                "strategy_point": "High output becomes the StrategyCarrier that low sampler must refine without breaking.",
                "right_observed_behavior": ["low_delta", "low sampler refinement", "delta strength/coupling behavior"],
                "right_outcome": ["latent_after_low", "decode-ready latent"],
                "collision_math": ["high_low_delta_ratio", "low_refinement_pressure", "strategy_carrier_stability"],
                "intervention_surface": ["LATENT_DELTA_SCALE", "bounded coupling multiplier", "deep-step research only after evidence"],
                "public_safe_control": "bounded_latent_delta_research",
            },
            "tail_next_source": {
                "left_outcome": ["visible_tail_frame", "trimmed current segment"],
                "left_observed_behavior": ["user selected resume frame", "MirrorCut trim behavior"],
                "strategy_point": "The selected tail frame becomes the next segment's source StrategyCarrier.",
                "right_observed_behavior": ["next segment source behavior", "cascade continuation drift"],
                "right_outcome": ["next_cascade_source_image", "stitched frame batch"],
                "collision_math": ["tail_admissibility_score", "resume_cut_delta", "continuation_boundary_score"],
                "intervention_surface": ["manual frame choice", "formula recommendation proposal", "future prompt-per-segment assist"],
                "public_safe_control": "manual_or_proposal_only",
            },
            "previous_next_frame_motion": {
                "left_outcome": ["previous_visible_frame"],
                "left_observed_behavior": ["frame_motion_delta", "boundary jump", "jerk/spike behavior"],
                "strategy_point": "Adjacent frames must preserve event continuity while allowing visible motion.",
                "right_observed_behavior": ["next_frame_motion_pressure", "continuity drift"],
                "right_outcome": ["next_visible_frame"],
                "collision_math": ["motion_delta", "spike_ratio", "reversal_ratio", "boundary_jump_score"],
                "intervention_surface": ["visual diagnostics", "future continuity guidance", "do not override CompletionGate"],
                "public_safe_control": "report_only",
            },
            "visible_video_outcome": {
                "left_outcome": ["decoded_frame_batch"],
                "left_observed_behavior": ["video combine/save behavior"],
                "strategy_point": "The internal event becomes a visible saved Outcome that must be inspected by the user.",
                "right_observed_behavior": ["final playback continuity", "fps/duration/frame count"],
                "right_outcome": ["saved_video_path", "final mp4/webm Outcome"],
                "collision_math": ["frame_count_consistency", "duration_consistency", "visible_quality_review"],
                "intervention_surface": ["report evidence", "visual review", "public release gate"],
                "public_safe_control": "report_only",
            },
        }

        def make_micro_formula(collision_id, carriers, required, optional, stage_hits, status, conflict_score, drift_score, score_basis):
            blueprint = micro_formula_blueprints.get(collision_id, {})
            expansion_state = (
                "ready_for_collision_math"
                if status in ("observed", "partial_evidence")
                else "waiting_for_required_evidence"
            )
            return {
                "local_strategy_id": f"S_collision_{collision_id}",
                "scope": "collision-local",
                "canonical_formula": "Outcome(t-1) + ObservedBehavior(t-1) = Strategy(t) = ObservedBehavior(t+1) + Outcome(t+1)",
                "left_side": {
                    "outcome_previous": blueprint.get("left_outcome", []),
                    "observed_behavior_previous": blueprint.get("left_observed_behavior", []),
                    "evidence_categories_required": list(required or []),
                },
                "strategy_point": {
                    "meaning": blueprint.get("strategy_point", "Carrier intersection becomes a local Strategy point."),
                    "carriers": list(carriers or []),
                    "stage_hit_count": len(stage_hits or []),
                    "status": status,
                },
                "right_side": {
                    "observed_behavior_next": blueprint.get("right_observed_behavior", []),
                    "outcome_next": blueprint.get("right_outcome", []),
                    "evidence_categories_optional": list(optional or []),
                },
                "collision_math": {
                    "available_metric_families": blueprint.get("collision_math", []),
                    "conflict_score": conflict_score,
                    "drift_score": drift_score,
                    "measured_now": bool(stage_hits),
                    "score_basis": score_basis,
                },
                "intervention": {
                    "surface": blueprint.get("intervention_surface", []),
                    "public_safe_control": blueprint.get("public_safe_control", "report_only"),
                    "active_control_allowed": False,
                    "activation_rule": "Only after report evidence, visual review, fixed-seed comparison, and explicit research mode.",
                },
                "expansion_state": expansion_state,
            }

        def make_collision(collision_id, carriers, required, optional, intersection, formula_role, recommendation):
            stage_hits = []
            evidence = {}
            for category in list(required or []) + list(optional or []):
                cat_hits = categories.get(category, [])
                evidence[category] = {
                    "present": bool(cat_hits),
                    "hit_count": len(cat_hits),
                    "stage_hits": cat_hits[:10],
                }
                stage_hits.extend(cat_hits)
            required_missing = [category for category in (required or []) if not categories.get(category)]
            if not required_missing:
                status = "observed"
            elif stage_hits:
                status = "partial_evidence"
            else:
                status = "missing_evidence"
            if collision_id == "tail_next_source" and not categories.get("cascade"):
                status = "not_applicable_single_segment"
            required_total = max(1, len(required or []))
            missing_conflict = float(len(required_missing) / required_total)
            stage_hits = list(dict.fromkeys(stage_hits))
            metric_snapshots = numeric_snapshots(stage_hits)
            score_result = compute_collision_scores(collision_id, status, missing_conflict, metric_snapshots)
            conflict_score = score_result.get("conflict_score", missing_conflict)
            drift_score = score_result.get("drift_score")
            score_basis = score_result.get("basis", {})
            local_formula = make_micro_formula(
                collision_id,
                carriers,
                required,
                optional,
                stage_hits,
                status,
                conflict_score,
                drift_score,
                score_basis,
            )
            dynamic_recommendation = score_result.get("dynamic_recommendation") or ""
            final_recommendation = str(recommendation or "")
            if dynamic_recommendation:
                final_recommendation = f"{final_recommendation} {dynamic_recommendation}".strip()
            return {
                "stage": f"EventVectorCollisionRecord_{collision_id}",
                "status": status,
                "collision_id": collision_id,
                "formula": "Outcome + ObservedBehavior are read at carrier intersections to locate Strategy(t); report-only, no active control.",
                "formula_role": formula_role,
                "local_formula": local_formula,
                "carriers": list(carriers or []),
                "intersection": str(intersection or ""),
                "evidence": evidence,
                "stage_hit_count": len(stage_hits),
                "stage_hits": stage_hits[:18],
                "metric_snapshots": metric_snapshots,
                "conflict_score": conflict_score,
                "drift_score": drift_score,
                "score_basis": score_basis,
                "recommendation": final_recommendation,
                "active_control_allowed": False,
                "control_mode": "REPORT_ONLY",
            }

        collisions = [
            make_collision(
                "prompt_image_anchor",
                ["positive_prompt", "negative_prompt", "source_image"],
                ["text_any", "image"],
                ["latent_seed"],
                "Prompt meaning collides with SourceAnchor before Wan latent seed.",
                "StrategyCandidate carrier + OutcomePrevious / SourceAnchor",
                "Use this collision to detect prompt-image mismatch before any active math control.",
            ),
            make_collision(
                "positive_negative_prompt_polarity",
                ["positive_prompt", "negative_prompt", "clip_conditioning"],
                ["text_positive", "text_negative"],
                ["cfg_policy"],
                "Positive and negative conditioning define the semantic boundary of StrategyCandidate.",
                "StrategyCandidate polarity boundary",
                "Keep positive and negative prompts separated; future scoring can flag semantic overlap.",
            ),
            make_collision(
                "image_latent_noise_seed",
                ["source_image", "wan_latent_seed", "seed_noise_field"],
                ["image", "latent_seed"],
                ["cfg_policy"],
                "ImageSource becomes the latent/noise possibility field consumed by the sampler.",
                "OutcomePrevious + PossibilityField -> StrategyCarrier",
                "This is the safest place for seed/reproducibility evidence, not for prompt rewriting.",
            ),
            make_collision(
                "high_low_sampler_strategy",
                ["high_sampler_outcome", "low_sampler_input", "low_sampler_observed_behavior"],
                ["high_sampler", "low_sampler"],
                ["delta_control", "cfg_policy"],
                "High output becomes low StrategyCarrier; low ObservedBehavior decides refinement stability.",
                "OutcomeNext(high) = StrategyCarrier(low)",
                "Delta scaling should be interpreted here as ObservedBehavior scaling, not generic motion tuning.",
            ),
            make_collision(
                "tail_next_source",
                ["selected_tail_frame", "trimmed_batch", "next_cascade_source"],
                ["cascade", "pause"],
                ["boundary", "motion", "tail_formula"],
                "User-selected tail frame becomes the source anchor for the next cascade segment.",
                "VisibleOutcome tail -> OutcomePrevious(next segment)",
                "This is the main continuation control point for pause/resume and future prompt-per-segment work.",
            ),
            make_collision(
                "previous_next_frame_motion",
                ["previous_frame", "next_frame", "frame_motion_delta"],
                ["motion"],
                ["boundary", "cascade"],
                "Adjacent visible frames expose continuity drift after decode and concatenation.",
                "VisibleOutcome(t-1) + ObservedBehavior(frame motion) = VisibleOutcome(t+1)",
                "Use this for visual/motion review after every manual test, not as a CompletionGate substitute.",
            ),
            make_collision(
                "visible_video_outcome",
                ["decoded_frames", "video_combine", "saved_video_path"],
                ["video"],
                ["motion", "cascade"],
                "The generated frame batch becomes a saved visible video Outcome.",
                "Final VisibleOutcome",
                "Public tests must check the actual video in addition to report PASS/BLOCKED status.",
            ),
        ]

        math_mode_values = set()
        mode_stage_allowlist = (
            "EventMathDeltaControl",
            "EventMathControlSummary",
            "EventMathSamplerPathPolicy",
            "EventStrategyCfgCoupling",
            "EventUniversalMath_",
        )
        for rec in records:
            raw_mode = rec.get("math_control_mode")
            if raw_mode is None:
                stage_name = str(rec.get("stage", "") or "")
                if any(stage_name.startswith(prefix) for prefix in mode_stage_allowlist):
                    raw_mode = rec.get("mode")
            mode_value = str(raw_mode or "").strip().upper()
            if mode_value:
                math_mode_values.add(mode_value)
        modes = sorted(math_mode_values)
        carrier_coverage = {
            "TEXT": bool(categories["text_any"]),
            "IMAGE_SOURCE": bool(categories["image"]),
            "LATENT_SEED": bool(categories["latent_seed"]),
            "SAMPLER_HIGH": bool(categories["high_sampler"]),
            "SAMPLER_LOW": bool(categories["low_sampler"]),
            "DELTA_CONTROL": bool(categories["delta_control"]),
            "CASCADE_ROUTE": bool(categories["cascade"]),
            "PAUSE_RESUME": bool(categories["pause"]),
            "FRAME_MOTION": bool(categories["motion"]),
            "VISIBLE_VIDEO": bool(categories["video"]),
        }
        observed_count = sum(1 for item in collisions if item.get("status") == "observed")
        partial_count = sum(1 for item in collisions if item.get("status") == "partial_evidence")
        missing_count = sum(1 for item in collisions if item.get("status") == "missing_evidence")
        scored_collisions = [
            item for item in collisions
            if isinstance(item.get("conflict_score"), (int, float)) or isinstance(item.get("drift_score"), (int, float))
        ]
        top_conflict = max(
            collisions,
            key=lambda item: safe_float(item.get("conflict_score"), -1.0),
        ) if collisions else {}
        drift_candidates = [item for item in collisions if safe_float(item.get("drift_score"), None) is not None]
        top_drift = max(
            drift_candidates,
            key=lambda item: safe_float(item.get("drift_score"), -1.0),
        ) if drift_candidates else {}
        matrix = {
            "stage": "EventStrategyMatrix",
            "status": "recorded",
            "matrix_version": "strategy_matrix_v2_scored_report_only",
            "formula": "Strategy(t) is mapped as intersections between carriers; this record is evidence for future bounded guidance, not active control.",
            "result_status": str(result_status or ""),
            "saved_video_path": str(saved_video_path or ""),
            "collision_count": len(collisions),
            "observed_collision_count": observed_count,
            "partial_collision_count": partial_count,
            "missing_collision_count": missing_count,
            "micro_formula_count": len([item for item in collisions if item.get("local_formula")]),
            "scored_collision_count": len(scored_collisions),
            "max_conflict_score": safe_float(top_conflict.get("conflict_score"), 0.0),
            "top_conflict_collision": top_conflict.get("collision_id", ""),
            "max_drift_score": safe_float(top_drift.get("drift_score"), None),
            "top_drift_collision": top_drift.get("collision_id", ""),
            "carrier_coverage": carrier_coverage,
            "math_control_modes_seen": modes,
            "collision_ids": [item.get("collision_id") for item in collisions],
            "local_strategy_ids": [
                (item.get("local_formula", {}) or {}).get("local_strategy_id")
                for item in collisions
                if isinstance(item.get("local_formula"), dict)
            ],
            "collision_formula_policy": "Each observed Strategy point may unfold its own local formula; this pass records the expansion surface only.",
            "active_control_allowed": False,
            "control_mode": "REPORT_ONLY",
            "next_route": "Use observed collisions as evidence before enabling any Strategy-guided sampler or prompt/latent control.",
        }
        return matrix, collisions

    def _event_strategy_guidance_proposal(self, strategy_matrix, vector_collisions):
        def safe_float(value, default=None):
            try:
                out = float(value)
            except Exception:
                return default
            return out if math.isfinite(out) else default

        collisions = [c for c in (vector_collisions or []) if isinstance(c, dict)]
        proposals = []

        def add(collision_id, kind, priority, message, test_route="", control_surface="report_only", evidence=None):
            proposals.append({
                "collision_id": str(collision_id or ""),
                "kind": str(kind or "observe"),
                "priority": str(priority or "low"),
                "message": str(message or ""),
                "test_route": str(test_route or ""),
                "control_surface": str(control_surface or "report_only"),
                "active_control_allowed": False,
                "evidence": evidence or {},
            })

        for item in collisions:
            cid = str(item.get("collision_id") or "")
            status = str(item.get("status") or "")
            conflict = safe_float(item.get("conflict_score"), 0.0) or 0.0
            drift = safe_float(item.get("drift_score"), None)
            basis = item.get("score_basis", {}) if isinstance(item.get("score_basis"), dict) else {}
            proposal_count_before = len(proposals)

            if status in ("missing_evidence", "partial_evidence"):
                add(
                    cid,
                    "collect_evidence",
                    "medium",
                    "This Strategy collision does not yet have enough evidence for guidance.",
                    "Run a focused fixed-seed smoke that exposes the missing carrier before changing math controls.",
                    evidence={"status": status, "conflict_score": conflict},
                )
                continue

            if cid == "previous_next_frame_motion" and drift is not None:
                priority = "high" if drift >= 0.55 else "medium" if drift >= 0.35 else "low"
                add(
                    cid,
                    "motion_stability_review",
                    priority,
                    "Frame-motion collision is measurable; use it to decide whether a delta candidate improves motion or only adds volatility.",
                    "Compare fixed-seed baseline against one bounded low/high delta candidate and inspect the mp4, not only report metrics.",
                    evidence={"drift_score": drift, "score_basis": basis},
                )

            if cid == "high_low_sampler_strategy" and drift is not None:
                add(
                    cid,
                    "bounded_delta_candidate",
                    "medium" if drift >= 0.35 else "low",
                    "High-to-low sampler seam has measurable delta pressure; this is the safest collision for bounded LATENT_DELTA_SCALE research.",
                    "Keep public defaults at 1.0/1.0; test low=1.0013 or high=0.992 + low=1.0013 only as explicit research.",
                    control_surface="LATENT_DELTA_SCALE_RESEARCH",
                    evidence={"drift_score": drift, "score_basis": basis},
                )

            if cid == "tail_next_source":
                tail = basis.get("tail_formula", {}) if isinstance(basis.get("tail_formula", {}), dict) else {}
                best_candidate = tail.get("best_candidate_index")
                best_mirror = tail.get("best_mirror_break")
                if best_candidate is not None:
                    add(
                        cid,
                        "tail_formula_advisor",
                        "medium",
                        "Tail formula can propose a candidate, but it must remain a gold hint unless the user explicitly enables recommendation behavior.",
                        f"Compare manual Tail choices against formula candidate {best_candidate}; do not let the proposal silently override green manual choice.",
                        control_surface="manual_or_proposal_only",
                        evidence={"best_candidate_index": best_candidate, "best_mirror_break": best_mirror},
                    )

            if conflict >= 0.5 and len(proposals) == proposal_count_before:
                add(
                    cid,
                    "collision_review",
                    "high",
                    "This observed Strategy collision is measurable and high-pressure; review it before promoting any active control.",
                    "Keep the next test fixed-seed and change one variable only.",
                    evidence={"status": status, "conflict_score": conflict, "drift_score": drift, "score_basis": basis},
                )

        if not proposals:
            add(
                "strategy_matrix",
                "continue_observation",
                "low",
                "No high-risk collision was detected in the current evidence map.",
                "Continue with neutral or one-variable fixed-seed tests; active control remains gated.",
            )

        top_conflict = strategy_matrix.get("top_conflict_collision", "") if isinstance(strategy_matrix, dict) else ""
        top_drift = strategy_matrix.get("top_drift_collision", "") if isinstance(strategy_matrix, dict) else ""
        return {
            "stage": "EventStrategyGuidanceProposal",
            "status": "proposal_only",
            "proposal_version": "strategy_guidance_v1_report_only",
            "formula": "Strategy Matrix scores become test proposals; they do not modify generation.",
            "active_control_allowed": False,
            "control_mode": "REPORT_ONLY",
            "top_conflict_collision": top_conflict,
            "top_drift_collision": top_drift,
            "proposal_count": len(proposals),
            "proposals": proposals,
            "public_default_policy": "Keep public defaults high_delta_strength=1.0 and low_delta_strength=1.0 until fixed-seed visual evidence proves a bounded preset.",
        }

    def _event_core_cascade_progress(self, execution_records, result_status="", saved_video_path=""):
        records = [r for r in (execution_records or []) if isinstance(r, dict)]
        result_status_u = str(result_status or "").upper()

        plan = {}
        for rec in records:
            if str(rec.get("stage", "") or "") == "SingularityCascadePlan":
                plan = rec
                break
        if not plan:
            for rec in records:
                candidate = rec.get("cascade_execution_plan")
                if isinstance(candidate, dict):
                    plan = candidate
                    break

        requested_segments = 1
        frames_per_cascade = None
        try:
            requested_segments = max(1, int(plan.get("requested_segments", 1))) if isinstance(plan, dict) else 1
        except Exception:
            requested_segments = 1
        try:
            frames_per_cascade = int(plan.get("frames_per_cascade")) if isinstance(plan, dict) and plan.get("frames_per_cascade") is not None else None
        except Exception:
            frames_per_cascade = None

        if requested_segments <= 1:
            for rec in records:
                if str(rec.get("stage", "") or "") in ("SingularityCascadeExecutionGate", "SingularityCascadeBegin"):
                    try:
                        requested_segments = max(requested_segments, int(rec.get("cascade_count", requested_segments) or requested_segments))
                    except Exception:
                        pass
                    if frames_per_cascade is None:
                        try:
                            frames_per_cascade = int(rec.get("frames_per_cascade"))
                        except Exception:
                            pass

        segment_end_indices = []
        for rec in records:
            if str(rec.get("stage", "") or "") == "SingularityCascadeSegmentEnd":
                try:
                    segment_end_indices.append(int(rec.get("segment_index", 0)))
                except Exception:
                    pass
        completed_segments = len(set(i for i in segment_end_indices if i > 0))
        last_completed_segment = max(segment_end_indices) if segment_end_indices else 0

        cascade_end = None
        for rec in records:
            if str(rec.get("stage", "") or "") == "SingularityCascadeEnd":
                cascade_end = rec
        if isinstance(cascade_end, dict):
            try:
                completed_segments = max(completed_segments, int(cascade_end.get("segments", completed_segments) or completed_segments))
            except Exception:
                pass
            try:
                requested_segments = max(requested_segments, int(cascade_end.get("requested_segments", requested_segments) or requested_segments))
            except Exception:
                pass

        pause_wait_segments = []
        pause_continue_segments = []
        pause_cancel_segments = []
        for rec in records:
            stage = str(rec.get("stage", "") or "")
            if not stage.startswith("SingularityCascadePause_"):
                continue
            parts = stage.split("_")
            segment = None
            if len(parts) >= 2:
                try:
                    segment = int(parts[1])
                except Exception:
                    segment = None
            status = str(rec.get("status", "") or "").lower()
            if status == "waiting_for_continue" and segment is not None:
                pause_wait_segments.append(segment)
            elif status == "continue" and segment is not None:
                pause_continue_segments.append(segment)
            elif status == "cancelled" and segment is not None:
                pause_cancel_segments.append(segment)

        pause_after_segments = []
        if isinstance(plan, dict):
            try:
                pause_after_segments = [int(x) for x in (plan.get("pause_after_segments", []) or [])]
            except Exception:
                pause_after_segments = []

        final_output_ok = result_status_u == "VIDEO" and bool(str(saved_video_path or "").strip())
        route_complete = completed_segments >= requested_segments and last_completed_segment >= requested_segments
        cancelled = result_status_u == "CANCELLED" or bool(pause_cancel_segments)
        waiting_for_continue = bool(set(pause_wait_segments) - set(pause_continue_segments) - set(pause_cancel_segments))

        if final_output_ok and route_complete:
            status = "COMPLETE_VIDEO"
        elif cancelled:
            status = "CANCELLED_PARTIAL"
        elif waiting_for_continue:
            status = "PAUSED_WAITING"
        elif not route_complete:
            status = "INCOMPLETE_ROUTE"
        else:
            status = "NO_FINAL_VIDEO"

        return {
            "status": status,
            "requested_segments": int(requested_segments),
            "completed_segments": int(completed_segments),
            "last_completed_segment": int(last_completed_segment),
            "final_segment_index": int(requested_segments),
            "frames_per_cascade": frames_per_cascade,
            "pause_after_segments": pause_after_segments,
            "pause_wait_segments": sorted(set(pause_wait_segments)),
            "pause_continue_segments": sorted(set(pause_continue_segments)),
            "pause_cancel_segments": sorted(set(pause_cancel_segments)),
            "route_complete": bool(route_complete),
            "final_output_ok": bool(final_output_ok),
            "cancelled": bool(cancelled),
            "waiting_for_continue": bool(waiting_for_continue),
            "saved_video_path": str(saved_video_path or ""),
            "result_status": str(result_status or ""),
            "plan_policy": str(plan.get("policy", "") if isinstance(plan, dict) else ""),
            "ignored_pause_after_segments": list(plan.get("ignored_pause_after_segments", []) or []) if isinstance(plan, dict) else [],
        }

    def _event_core_body_consistency_audit(self, execution_records, result_status="", saved_video_path="", failure_reason=""):
        """
        r45 internal consistency audit.
        This does not modify generation. It only checks whether the one-node Event Core Body route is internally coherent.
        """
        records = list(execution_records or [])
        stages = [str(r.get("stage", "")) for r in records]

        required_exact = [
            "EventCoreBodyInit",
            "EventOneNodePolicy",
            "EventUniversalPipelineMap",
        ]

        required_prefix = [
            "EventUniversalMath_EventTextEncodePositive",
            "EventUniversalMath_EventTextEncodeNegative",
            "EventUniversalMath_EventImageScaleStart",
            "EventUniversalMath_EventWanImageToVideoSeed",
            "EventUniversalMath_EventSamplerHigh",
            "EventUniversalMath_EventSamplerLow",
            "EventUniversalMath_EventVAEDecodeTiled",
        ]

        expected_if_video = [
            "EventUniversalMath_EventVideoSaveBegin",
            "EventUniversalMath_EventVideoCombine",
        ]

        def has_stage(name):
            return name in stages

        def has_prefix(prefix):
            return any(s.startswith(prefix) for s in stages)

        missing_exact = [x for x in required_exact if not has_stage(x)]
        missing_prefix = [x for x in required_prefix if not has_prefix(x)]

        if str(result_status or "").upper() == "VIDEO":
            missing_video = [x for x in expected_if_video if not has_prefix(x)]
        else:
            missing_video = []

        stage_math = [s for s in stages if s.startswith("EventUniversalMath_")]
        boundary_math = [s for s in stages if s.startswith("EventUniversalBoundary_")]
        math_tensor = [s for s in stages if s.startswith("EventMath_")]
        cascade_progress = self._event_core_cascade_progress(
            records,
            result_status=result_status,
            saved_video_path=saved_video_path,
        )

        duplicate_counts = {}
        for s in stages:
            duplicate_counts[s] = duplicate_counts.get(s, 0) + 1
        duplicates = {k: v for k, v in duplicate_counts.items() if v > 1 and k not in ("SingularityStageDelay",)}

        passed = not missing_exact and not missing_prefix and not missing_video
        severity = "PASS" if passed else ("WARN" if str(result_status or "").upper() in ("FAILED", "ERROR") else "FAIL")

        audit = {
            "stage": "EventCoreBodyConsistencyAudit",
            "status": "pass" if passed else "issues_found",
            "severity": severity,
            "formula": "One external node must preserve one internal Event Core Body route from input strategy to output state.",
            "result_status": str(result_status or ""),
            "saved_video_path": str(saved_video_path or ""),
            "failure_reason": str(failure_reason or ""),
            "checks": {
                "one_external_node_policy_present": has_stage("EventOneNodePolicy"),
                "event_core_body_initialized": has_stage("EventCoreBodyInit"),
                "pipeline_map_present": has_stage("EventUniversalPipelineMap"),
                "stage_math_count": len(stage_math),
                "boundary_math_count": len(boundary_math),
                "math_tensor_record_count": len(math_tensor),
                "required_exact_missing": missing_exact,
                "required_prefix_missing": missing_prefix,
                "video_stage_missing": missing_video,
                "duplicate_stage_names": duplicates,
                "cascade_progress": cascade_progress,
            },
            "interpretation": (
                "Internal Event Core Body route is coherent enough for next research step."
                if passed else
                "Internal Event Core Body route has missing stage records; inspect missing lists before adding new controls."
            ),
        }
        return audit

    def _event_core_body_stage_order_audit(self, execution_records):
        """
        r48: corrected order audit.
        Boot/static records may appear before EventCoreBodyInit and must not block the gate.
        This audit checks the runtime route order only:
            Text -> Image -> Seed -> High -> Cleanup -> Low -> Decode -> Output
        """
        stages = [str(r.get("stage", "")) for r in (execution_records or [])]

        boot_required = [
            "EventCoreBodyInit",
            "EventOneNodePolicy",
            "EventUniversalPipelineMap",
        ]

        runtime_order_prefix = [
            "EventUniversalMath_EventTextEncodePositive",
            "EventUniversalMath_EventTextEncodeNegative",
            "EventUniversalMath_EventImageScaleStart",
            "EventUniversalMath_EventWanImageToVideoSeed",
            "EventUniversalMath_EventModelShift_high",
            "EventUniversalMath_EventSamplerHigh",
            "EventUniversalMath_EventCleanupBetweenSamplers",
            "EventUniversalMath_EventModelShift_low",
            "EventUniversalMath_EventSamplerLow",
            "EventUniversalMath_EventVAEDecodeTiled",
            "EventMath_decoded_frame_motion",
            "EventUniversalMath_EventVideoSaveBegin",
            "EventUniversalMath_EventVideoCombine",
        ]

        def first_index(prefix):
            for i, s in enumerate(stages):
                if s == prefix or s.startswith(prefix):
                    return i
            return None

        boot_presence = {name: (first_index(name) is not None) for name in boot_required}

        order_items = []
        last_idx = -1
        violations = []
        missing_runtime = []
        for prefix in runtime_order_prefix:
            idx = first_index(prefix)
            order_items.append({"stage": prefix, "first_index": idx})
            if idx is None:
                missing_runtime.append(prefix)
                continue
            if idx < last_idx:
                violations.append({
                    "stage": prefix,
                    "first_index": idx,
                    "previous_required_index": last_idx,
                    "reason": "runtime route stage appeared before an earlier runtime route stage",
                })
            last_idx = max(last_idx, idx)

        # Missing runtime stages are reported here, but the completion gate still uses the
        # consistency audit missing lists as the source of truth for blocking.
        # This avoids double-blocking on optional/non-video branches.
        return {
            "stage": "EventCoreBodyStageOrderAudit",
            "status": "pass" if not violations else "issues_found",
            "formula": "Runtime route order should preserve Text -> Image -> Seed -> High -> Low -> Decode -> Output.",
            "checked_stage_count": len(runtime_order_prefix),
            "present_checked_count": sum(1 for x in order_items if x["first_index"] is not None),
            "boot_presence": boot_presence,
            "order_items": order_items,
            "missing_runtime_route_stages": missing_runtime,
            "violations": violations,
            "audit_policy": "boot records are presence-checked but excluded from order violations",
        }


    def _event_core_body_completion_gate(self, audit, order_audit):
        """
        r46: one final gate before tuning.
        PASS means the internal one-node Event Core Body is coherent enough to start tuning/control work.
        """
        checks = audit.get("checks", {}) if isinstance(audit, dict) else {}
        missing_total = (
            len(checks.get("required_exact_missing", []) or []) +
            len(checks.get("required_prefix_missing", []) or []) +
            len(checks.get("video_stage_missing", []) or [])
        )
        stage_math_count = int(checks.get("stage_math_count", 0) or 0)
        one_node_ok = bool(checks.get("one_external_node_policy_present")) and bool(checks.get("event_core_body_initialized"))
        order_ok = isinstance(order_audit, dict) and order_audit.get("status") == "pass"
        cascade_progress = checks.get("cascade_progress", {}) if isinstance(checks.get("cascade_progress", {}), dict) else {}
        result_status_u = str(audit.get("result_status", "") if isinstance(audit, dict) else "").upper()
        route_complete = bool(cascade_progress.get("route_complete"))
        final_output_ok = bool(cascade_progress.get("final_output_ok"))
        cancelled = bool(cascade_progress.get("cancelled")) or result_status_u == "CANCELLED"
        waiting_for_continue = bool(cascade_progress.get("waiting_for_continue"))
        blocking_reasons = []
        if not one_node_ok:
            blocking_reasons.append("one_node_body_not_initialized")
        if missing_total != 0:
            blocking_reasons.append("missing_required_records")
        if stage_math_count < 7:
            blocking_reasons.append("stage_math_count_below_minimum")
        if not order_ok:
            blocking_reasons.append("stage_order_not_passed")
        if not route_complete:
            blocking_reasons.append("cascade_route_not_complete")
        if not final_output_ok:
            blocking_reasons.append("final_video_output_missing")
        if cancelled:
            blocking_reasons.append("run_cancelled")
        if waiting_for_continue:
            blocking_reasons.append("waiting_for_continue")

        pass_gate = (
            one_node_ok and
            missing_total == 0 and
            stage_math_count >= 7 and
            order_ok and
            route_complete and
            final_output_ok and
            not cancelled and
            not waiting_for_continue
        )
        if pass_gate:
            gate_status = "PASS"
        elif cancelled:
            gate_status = "CANCELLED"
        else:
            gate_status = "BLOCKED"

        return {
            "stage": "EventCoreBodyCompletionGate",
            "status": gate_status,
            "formula": "Tuning may begin only after the one-node Event Core Body is coherent, ordered, and complete enough.",
            "one_node_ok": one_node_ok,
            "stage_order_ok": order_ok,
            "missing_total": missing_total,
            "stage_math_count": stage_math_count,
            "boundary_math_count": int(checks.get("boundary_math_count", 0) or 0),
            "math_tensor_record_count": int(checks.get("math_tensor_record_count", 0) or 0),
            "result_status": audit.get("result_status", "") if isinstance(audit, dict) else "",
            "saved_video_path": audit.get("saved_video_path", "") if isinstance(audit, dict) else "",
            "cascade_progress": cascade_progress,
            "route_complete": route_complete,
            "final_output_ok": final_output_ok,
            "blocking_reasons": blocking_reasons,
            "next_action": (
                "Core body gate passed. Tuning/control work may begin on the next iteration."
                if pass_gate else
                "Run was cancelled before final video Outcome; partial records are diagnostic only."
                if cancelled else
                "Core body gate blocked. Finish the requested cascade plan and final video Outcome before tuning."
            ),
        }

    def _event_core_local_sstate_breakdown(self, packet, execution_records, result_status="", saved_video_path=""):
        stages = [str(r.get("stage", "")) for r in (execution_records or [])]

        def has(prefix):
            return any(s == prefix or s.startswith(prefix) for s in stages)

        def first_status(prefix):
            for rec in execution_records or []:
                stage = str(rec.get("stage", ""))
                if stage == prefix or stage.startswith(prefix):
                    return str(rec.get("status", ""))
            return "missing"

        meta = packet.get("metadata", {}) if isinstance(packet, dict) else {}
        wan_interface = meta.get("wan_workflow_interface", {}) if isinstance(meta.get("wan_workflow_interface", {}), dict) else {}
        return [
            {
                "name": "S_text",
                "formula_role": "StrategyCandidate carrier",
                "granularity": "Stage",
                "stage_present": has("EventUniversalMath_EventTextEncodePositive") or has("EventUniversalMath_EventTextEncodeNegative"),
                "status": first_status("EventUniversalMath_EventTextEncodePositive"),
                "contents": "prompt/negative prompt conditioning references; no raw prompt text stored here",
            },
            {
                "name": "S_seed",
                "formula_role": "Possibility field",
                "granularity": "Stage",
                "stage_present": has("EventUniversalMath_EventWanImageToVideoSeed"),
                "status": first_status("EventUniversalMath_EventWanImageToVideoSeed"),
                "contents": f"seed={wan_interface.get('seed', '')}, frames_per_cascade={wan_interface.get('frames', '')}",
            },
            {
                "name": "S_high",
                "formula_role": "OutcomeNext(high) = StrategyCarrier(low)",
                "granularity": "Stage",
                "stage_present": has("EventUniversalMath_EventSamplerHigh"),
                "status": first_status("EventUniversalMath_EventSamplerHigh"),
                "contents": "high latent output plus high delta/control records",
            },
            {
                "name": "S_low",
                "formula_role": "OutcomeNext(low)",
                "granularity": "Stage",
                "stage_present": has("EventUniversalMath_EventSamplerLow"),
                "status": first_status("EventUniversalMath_EventSamplerLow"),
                "contents": "low latent output plus low delta/control records",
            },
            {
                "name": "S_decode",
                "formula_role": "Translation state",
                "granularity": "Stage",
                "stage_present": has("EventUniversalMath_EventVAEDecodeTiled"),
                "status": first_status("EventUniversalMath_EventVAEDecodeTiled"),
                "contents": "VAE tile settings and decoded frame tensor reference",
            },
            {
                "name": "S_output",
                "formula_role": "Final Outcome",
                "granularity": "Pipeline",
                "stage_present": has("EventUniversalMath_EventVideoCombine"),
                "status": str(result_status or ""),
                "contents": f"saved_video_path={saved_video_path or ''}",
            },
        ]

    def _event_core_conflict_integration(self, packet, audit, order_audit, gate):
        packet = ensure_packet(packet)
        checks = audit.get("checks", {}) if isinstance(audit, dict) else {}
        conflicts = []

        def add_core_conflict(conflict_type, severity, stage_position, symptom, suggested, metadata=None):
            conflict = make_conflict(
                conflict_type,
                severity=severity,
                stage_position=stage_position,
                suspected_cause="Event Core Body structural or formula-role audit found a route inconsistency.",
                observed_symptom=symptom,
                suggested_response=suggested,
                metadata=metadata or {},
            )
            conflicts.append(conflict)
            return add_conflict(packet, conflict)

        for field, severity in (
            ("required_exact_missing", "CRITICAL"),
            ("required_prefix_missing", "BLOCKED"),
            ("video_stage_missing", "BLOCKED"),
        ):
            for missing in checks.get(field, []) or []:
                packet = add_core_conflict(
                    "EventCoreBodyMissingRecord",
                    severity,
                    str(missing),
                    f"{field}: {missing}",
                    "Restore the missing internal Event Core Body stage record before adding active controls.",
                    {"missing_field": field, "missing_stage": missing},
                )

        for violation in order_audit.get("violations", []) if isinstance(order_audit, dict) else []:
            packet = add_core_conflict(
                "EventCoreBodyStageOrderViolation",
                "BLOCKED",
                str(violation.get("stage", "EventCoreBodyStageOrderAudit")),
                str(violation.get("reason", "runtime route order violation")),
                "Fix runtime route ordering; boot/static records are presence-only and should not block order.",
                violation,
            )

        if isinstance(gate, dict) and gate.get("status") != "PASS":
            blocking_reasons = gate.get("blocking_reasons", [])
            packet = add_core_conflict(
                "EventCoreBodyCompletionGateBlocked",
                "BLOCKED",
                "EventCoreBodyCompletionGate",
                f"CompletionGate={gate.get('status')}, missing_total={gate.get('missing_total')}, stage_order_ok={gate.get('stage_order_ok')}, blocking_reasons={blocking_reasons}",
                "Do not enable active tuning/cache/compile paths until the requested cascade plan reaches final video Outcome.",
                gate,
            )

        return packet, conflicts

    def _event_core_body_summary_record(self, audit, order_audit, gate, body=None):
        checks = audit.get("checks", {}) if isinstance(audit, dict) else {}
        body = body if isinstance(body, dict) else {}
        return {
            "stage": "EventCoreBodySummary",
            "status": gate.get("status", "UNKNOWN") if isinstance(gate, dict) else "UNKNOWN",
            "body_version": EVENT_HORIZON_BODY_VERSION,
            "one_node_ok": gate.get("one_node_ok", False) if isinstance(gate, dict) else False,
            "stage_order_ok": gate.get("stage_order_ok", False) if isinstance(gate, dict) else False,
            "audit_gate": gate.get("status", "UNKNOWN") if isinstance(gate, dict) else "UNKNOWN",
            "missing_total": gate.get("missing_total", 0) if isinstance(gate, dict) else 0,
            "stage_math_count": gate.get("stage_math_count", 0) if isinstance(gate, dict) else 0,
            "boundary_math_count": gate.get("boundary_math_count", 0) if isinstance(gate, dict) else 0,
            "math_tensor_record_count": gate.get("math_tensor_record_count", 0) if isinstance(gate, dict) else 0,
            "route_complete": gate.get("route_complete", False) if isinstance(gate, dict) else False,
            "final_output_ok": gate.get("final_output_ok", False) if isinstance(gate, dict) else False,
            "cascade_progress": gate.get("cascade_progress", {}) if isinstance(gate, dict) else {},
            "blocking_reasons": gate.get("blocking_reasons", []) if isinstance(gate, dict) else [],
            "live_route_count": body.get("live_route_count", 0),
            "runtime_monitor_count": body.get("runtime_monitor_count", 0),
            "local_sstate_count": len(body.get("local_sstates", []) or []),
            "event_conflict_count": len(body.get("event_conflicts", []) or []),
            "strategy_matrix_status": (body.get("strategy_matrix", {}) or {}).get("status", "not_recorded") if isinstance(body.get("strategy_matrix", {}), dict) else "not_recorded",
            "vector_collision_count": len(body.get("vector_collision_records", []) or []),
            "vector_collision_observed_count": (body.get("strategy_matrix", {}) or {}).get("observed_collision_count", 0) if isinstance(body.get("strategy_matrix", {}), dict) else 0,
            "local_micro_formula_count": (body.get("strategy_matrix", {}) or {}).get("micro_formula_count", 0) if isinstance(body.get("strategy_matrix", {}), dict) else 0,
            "strategy_guidance_proposal_count": (body.get("strategy_guidance_proposal", {}) or {}).get("proposal_count", 0) if isinstance(body.get("strategy_guidance_proposal", {}), dict) else 0,
            "top_conflict_collision": (body.get("strategy_matrix", {}) or {}).get("top_conflict_collision", "") if isinstance(body.get("strategy_matrix", {}), dict) else "",
            "top_drift_collision": (body.get("strategy_matrix", {}) or {}).get("top_drift_collision", "") if isinstance(body.get("strategy_matrix", {}), dict) else "",
            "required_exact_missing": checks.get("required_exact_missing", []),
            "required_prefix_missing": checks.get("required_prefix_missing", []),
            "video_stage_missing": checks.get("video_stage_missing", []),
            "order_violations": order_audit.get("violations", []) if isinstance(order_audit, dict) else [],
            "next_action": gate.get("next_action", "") if isinstance(gate, dict) else "",
        }

    def _event_core_body_report_card(self, audit, gate=None):
        checks = audit.get("checks", {}) if isinstance(audit, dict) else {}
        gate = gate if isinstance(gate, dict) else {}
        gate_status = gate.get("status", audit.get("status", "unknown") if isinstance(audit, dict) else "unknown")
        return {
            "stage": "EventCoreBodyReportCard",
            "status": gate_status,
            "severity": "PASS" if gate_status == "PASS" else ("WARNING" if gate_status == "CANCELLED" else "BLOCKED"),
            "one_node_ok": bool(checks.get("one_external_node_policy_present")) and bool(checks.get("event_core_body_initialized")),
            "stage_math_count": checks.get("stage_math_count", 0),
            "boundary_math_count": checks.get("boundary_math_count", 0),
            "missing_total": (
                len(checks.get("required_exact_missing", []) or []) +
                len(checks.get("required_prefix_missing", []) or []) +
                    len(checks.get("video_stage_missing", []) or [])
            ),
            "route_complete": gate.get("route_complete", False),
            "final_output_ok": gate.get("final_output_ok", False),
            "blocking_reasons": gate.get("blocking_reasons", []),
            "next_action": (
                "Continue only if EventCoreBodyCompletionGate is PASS on normal and cascade runs."
                if gate_status == "PASS" else
                "Cancelled/partial run is diagnostic only; complete the requested cascade plan before tuning."
                if gate_status == "CANCELLED" else
                "Do not add new controls yet; fix missing Event Core Body stages first."
            ) if isinstance(audit, dict) else "Audit unavailable.",
        }

    def _event_core_body_finalize(self, packet, execution_records, result_status, saved_video_path, failure_reason):
        packet = self._event_core_body_collect_from_records(packet, execution_records)
        body = packet.setdefault("metadata", {}).setdefault("event_core_body", {})
        body["result_status"] = str(result_status or "")
        body["saved_video_path"] = str(saved_video_path or "")
        body["failure_reason"] = str(failure_reason or "")
        body["finalized"] = True
        audit = self._event_core_body_consistency_audit(
            execution_records,
            result_status=result_status,
            saved_video_path=saved_video_path,
            failure_reason=failure_reason,
        )
        body["consistency_audit"] = audit
        execution_records.append(audit)
        order_audit = self._event_core_body_stage_order_audit(execution_records)
        body["stage_order_audit"] = order_audit
        execution_records.append(order_audit)
        gate = self._event_core_body_completion_gate(audit, order_audit)
        body["completion_gate"] = gate
        execution_records.append(gate)
        packet, integrated_conflicts = self._event_core_conflict_integration(packet, audit, order_audit, gate)
        body["event_conflicts"] = integrated_conflicts
        execution_records.append({
            "stage": "EventConflictIntegration",
            "status": "recorded",
            "conflict_count": len(integrated_conflicts),
            "severity_counts": {
                severity: sum(1 for c in integrated_conflicts if c.get("severity") == severity)
                for severity in sorted({c.get("severity") for c in integrated_conflicts if c.get("severity")})
            },
            "formula": "Audit findings become EventConflict objects with stage and formula-role context.",
        })
        local_sstates = self._event_core_local_sstate_breakdown(packet, execution_records, result_status, saved_video_path)
        body["local_sstates"] = local_sstates
        execution_records.append({
            "stage": "EventCoreBodyLocalSStateBreakdown",
            "status": "recorded",
            "sstate_count": len(local_sstates),
            "sstates": local_sstates,
            "formula": "Stage-level states are labeled by formula role rather than only technical stage name.",
        })
        strategy_matrix, vector_collisions = self._event_strategy_matrix_from_records(
            execution_records,
            result_status=result_status,
            saved_video_path=saved_video_path,
        )
        body["strategy_matrix"] = strategy_matrix
        body["vector_collision_records"] = vector_collisions
        local_micro_formula_records = []
        for collision in vector_collisions:
            local_formula = collision.get("local_formula") if isinstance(collision, dict) else None
            if not isinstance(local_formula, dict):
                continue
            collision_id = str(collision.get("collision_id") or "unknown")
            local_micro_formula_records.append({
                "stage": f"EventLocalMicroFormula_{collision_id}",
                "status": str(collision.get("status") or "recorded"),
                "collision_id": collision_id,
                "local_strategy_id": local_formula.get("local_strategy_id"),
                "scope": local_formula.get("scope"),
                "canonical_formula": local_formula.get("canonical_formula"),
                "left_side": local_formula.get("left_side"),
                "strategy_point": local_formula.get("strategy_point"),
                "right_side": local_formula.get("right_side"),
                "collision_math": local_formula.get("collision_math"),
                "intervention": local_formula.get("intervention"),
                "expansion_state": local_formula.get("expansion_state"),
                "carriers": collision.get("carriers"),
                "conflict_score": collision.get("conflict_score"),
                "drift_score": collision.get("drift_score"),
                "active_control_allowed": False,
                "control_mode": "REPORT_ONLY",
                "formula": "A local Strategy point unfolds the canonical equality at this carrier collision; report-only evidence, not generation control.",
            })
        body["local_micro_formula_records"] = local_micro_formula_records
        strategy_guidance = self._event_strategy_guidance_proposal(strategy_matrix, vector_collisions)
        strategy_matrix["guidance_proposal_count"] = int(strategy_guidance.get("proposal_count", 0) or 0)
        body["strategy_guidance_proposal"] = strategy_guidance
        execution_records.extend(vector_collisions)
        execution_records.extend(local_micro_formula_records)
        execution_records.append(strategy_guidance)
        execution_records.append(strategy_matrix)
        if body.get("runtime_monitor_summary"):
            execution_records.append({
                "stage": "EventRuntimeMonitorSummary",
                "status": "recorded",
                **body.get("runtime_monitor_summary", {}),
                "formula": "Runtime timing and memory are observer-only ObservedBehavior extensions.",
            })
        summary = self._event_core_body_summary_record(audit, order_audit, gate, body=body)
        body["summary"] = summary
        execution_records.append(summary)
        execution_records.append(self._event_core_body_report_card(audit, gate))
        execution_records.append({
            "stage": "EventCoreBodyFinalize",
            "status": "recorded",
            "stage_math_count": body.get("stage_math_count", 0),
            "boundary_count": body.get("boundary_count", 0),
            "s_wire_count": body.get("s_wire_count", 0),
            "result_status": str(result_status or ""),
            "saved_video_path": str(saved_video_path or ""),
            "failure_reason": str(failure_reason or ""),
            "one_node_integrity": body.get("one_node_integrity", {}),
        })
        packet = record_stage(
            packet,
            stage_name="EventCoreBody",
            action="FINALIZE_INTERNAL_BODY",
            observed_behavior="Internal one-node Event Core Body collected universal stage math, boundary math, S-Wire route, and final output state.",
            metadata={
                "stage_math_count": body.get("stage_math_count", 0),
                "boundary_count": body.get("boundary_count", 0),
                "s_wire_count": body.get("s_wire_count", 0),
                "result_status": str(result_status or ""),
                "saved_video_path": str(saved_video_path or ""),
                "one_node_integrity": body.get("one_node_integrity", {}),
            },
            formula_note="single external node -> internal Event Core Body -> internal Wan route -> final video/report output",
        )
        return packet

    def _resolve_output_dir(self, output_folder_mode, output_folder, custom_output_folder, subdir="event_equality_reports", output_target="COMFY_OUTPUT", media_type="video"):
        mode_target = str(output_target or "USER_D_AI_NSFW")
        media_type = str(media_type or "video").lower()

        def local_try_make_dir(path):
            try:
                Path(path).mkdir(parents=True, exist_ok=True)
                return Path(path)
            except Exception:
                return None

        # Legacy D:\AI NSFW branch kept for old workflows but no longer preferred.
        # Main Singularity node now forces COMFY_OUTPUT.
        if mode_target in ("USER_D_AI_NSFW", "USER_D_AI_AND_SFW", "DEFAULT_USER_PATH"):
            for candidate in _event_core_preferred_media_dirs(media_type):
                made = local_try_make_dir(candidate)
                if made is not None:
                    return made

        try:
            import folder_paths
            base_output = Path(folder_paths.get_output_directory())
        except Exception:
            base_output = Path.cwd() / "output"

        mode = str(output_folder_mode or "DEFAULT")
        if mode_target == "CUSTOM" and str(custom_output_folder or "").strip():
            out_dir = Path(str(custom_output_folder)).expanduser()
        elif mode_target == "PICKER" and str(output_folder or "").strip() and str(output_folder) != "default":
            candidate = Path(str(output_folder))
            out_dir = candidate if candidate.is_absolute() else base_output / candidate
        elif mode_target == "COMFY_OUTPUT":
            out_dir = base_output / (subdir or "")
        elif mode == "CUSTOM" and str(custom_output_folder or "").strip():
            out_dir = Path(str(custom_output_folder)).expanduser()
        elif mode == "PICKER" and str(output_folder or "").strip() and str(output_folder) != "default":
            candidate = Path(str(output_folder))
            out_dir = candidate if candidate.is_absolute() else base_output / candidate
        else:
            out_dir = base_output / (subdir or "")

        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir


    def _save_report_file(self, report, save_prefix, output_target, output_folder_mode, output_folder, custom_output_folder):
        safe_prefix = re.sub(r"[^a-zA-Z0-9_\\-]+", "_", str(save_prefix or "Singularity")).strip("_") or "Singularity"
        out_dir = self._resolve_output_dir(output_folder_mode, output_folder, custom_output_folder, subdir="", output_target=output_target, media_type="report")
        out_dir.mkdir(parents=True, exist_ok=True)

        text = "" if report is None else str(report)
        if not text.strip():
            text = (
                "# Singularity Report Fallback\n\n"
                f"- runtime_version: {EVENT_HORIZON_RUNTIME_VERSION}\n"
                f"- runtime_name: {EVENT_HORIZON_RUNTIME_NAME}\n"
                "- warning: build_markdown_report returned an empty report; fallback report was written to avoid a 0 KB file.\n"
            )

        path = out_dir / f"{safe_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.md"
        tmp_path = path.with_suffix(path.suffix + ".tmp")

        with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass

        size = tmp_path.stat().st_size if tmp_path.exists() else 0
        if size <= 0:
            raise RuntimeError(f"Report temp file is empty: {tmp_path}")

        tmp_path.replace(path)

        final_size = path.stat().st_size if path.exists() else 0
        if final_size <= 0:
            raise RuntimeError(f"Report final file is empty after replace: {path}")

        return str(path)

    def _rewrite_report_file(self, saved_report_path, report):
        path = Path(str(saved_report_path or ""))
        if not str(path):
            return 0
        text = "" if report is None else str(report)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        tmp_path.replace(path)
        return path.stat().st_size if path.exists() else 0

    def _runtime_monitor_rows(self, packet):
        meta = packet.get("metadata", {}) if isinstance(packet, dict) else {}
        body = meta.get("event_core_body", {}) if isinstance(meta.get("event_core_body", {}), dict) else {}
        records = body.get("runtime_monitor_records", []) if isinstance(body.get("runtime_monitor_records", []), list) else []

        rows = []
        first_perf = None
        previous_perf = None
        for index, rec in enumerate(records):
            memory = rec.get("memory", {}) if isinstance(rec.get("memory", {}), dict) else {}
            perf_raw = rec.get("perf_counter")
            try:
                perf = float(perf_raw)
            except Exception:
                perf = None
            if first_perf is None and perf is not None:
                first_perf = perf
            elapsed = (perf - first_perf) if (perf is not None and first_perf is not None) else None
            delta = (perf - previous_perf) if (perf is not None and previous_perf is not None) else None
            if perf is not None:
                previous_perf = perf

            rows.append({
                "index": index,
                "stage": rec.get("stage", ""),
                "record_type": rec.get("record_type", ""),
                "status": rec.get("status", ""),
                "elapsed_from_start_s": round(elapsed, 6) if elapsed is not None else "",
                "delta_since_previous_s": round(delta, 6) if delta is not None else "",
                "process_rss_mb": memory.get("process_rss_mb", ""),
                "torch_cuda_available": memory.get("torch_cuda_available", ""),
                "cuda_allocated_mb": memory.get("cuda_allocated_mb", ""),
                "cuda_reserved_mb": memory.get("cuda_reserved_mb", ""),
                "cuda_max_allocated_mb": memory.get("cuda_max_allocated_mb", ""),
                "cuda_max_reserved_mb": memory.get("cuda_max_reserved_mb", ""),
            })

        return rows

    def _runtime_motion_summary(self, packet):
        """
        Extract one canonical motion-math summary from execution records for sidecar diffing.
        Preference order:
          1) EventMath_concatenated_frame_motion (cascade-level full output)
          2) EventMath_decoded_frame_motion (single segment/final decode)
          3) latest *frame_motion record as fallback
        """
        meta = packet.get("metadata", {}) if isinstance(packet, dict) else {}
        execution_records = meta.get("execution_records", []) if isinstance(meta.get("execution_records", []), list) else []
        if not execution_records:
            return {"motion_stage": "", "available": False}

        preferred = None
        fallback = None
        for rec in execution_records:
            if not isinstance(rec, dict):
                continue
            stage = str(rec.get("stage", "") or "")
            if stage == "EventMath_concatenated_frame_motion":
                preferred = rec
                break
            if stage == "EventMath_decoded_frame_motion" and preferred is None:
                preferred = rec
            if stage.endswith("_frame_motion") or stage == "EventMathCascadeBoundary":
                fallback = rec

        selected = preferred if isinstance(preferred, dict) else (fallback if isinstance(fallback, dict) else None)
        if not isinstance(selected, dict):
            return {"motion_stage": "", "available": False}

        keys = [
            "frame_delta_count",
            "frame_delta_norm_mean",
            "frame_delta_norm_std",
            "frame_delta_norm_min",
            "frame_delta_norm_max",
            "frame_delta_norm_p25",
            "frame_delta_norm_p50",
            "frame_delta_norm_p75",
            "frame_delta_norm_p90",
            "frame_delta_norm_p95",
            "frame_delta_norm_iqr",
            "frame_delta_norm_cv_ratio",
            "frame_delta_p95_to_p50_ratio",
            "frame_delta_spike_ratio",
            "frame_delta_abs_mean",
            "frame_delta_abs_max",
            "frame_delta_cosine_mean",
            "frame_delta_reversal_ratio",
            "frame_delta_jerk_abs_mean",
            "frame_delta_jerk_ratio",
            "frame_motion_stability_score",
            "frame_motion_profile",
        ]
        out = {
            "motion_stage": str(selected.get("stage", "")),
            "status": str(selected.get("status", "")),
            "available": str(selected.get("status", "")) == "ok",
        }
        for key in keys:
            if key in selected:
                out[key] = selected.get(key)
        return out

    def _runtime_effective_cascade_count(self, packet):
        meta = packet.get("metadata", {}) if isinstance(packet, dict) else {}
        execution_records = meta.get("execution_records", []) if isinstance(meta.get("execution_records", []), list) else []
        for rec in execution_records:
            if not isinstance(rec, dict):
                continue
            stage = str(rec.get("stage", "") or "")
            if stage in ("SingularityCascadeExecutionGate", "SingularityCascadeBegin"):
                try:
                    return max(1, int(rec.get("cascade_count", 1)))
                except Exception:
                    return 1
        result = meta.get("result_status", {}) if isinstance(meta.get("result_status", {}), dict) else {}
        try:
            return max(1, int(result.get("cascade_count", 1)))
        except Exception:
            return 1

    def _runtime_monitor_settings_signature(self, packet):
        meta = packet.get("metadata", {}) if isinstance(packet, dict) else {}
        wan = meta.get("wan_workflow_interface", {}) if isinstance(meta.get("wan_workflow_interface", {}), dict) else {}
        result = meta.get("result_status", {}) if isinstance(meta.get("result_status", {}), dict) else {}
        effective_cascade_count = self._runtime_effective_cascade_count(packet)

        signature_source = {
            "width": wan.get("width"),
            "height": wan.get("height"),
            "frames": wan.get("frames"),
            "fps": wan.get("fps"),
            "seed": wan.get("seed"),
            "cascade_count": effective_cascade_count,
            "branch_mode_active": wan.get("branch_mode_active"),
            "ksampler_windows": wan.get("ksampler_windows"),

            "math_controls": getattr(self, "_event_requested_math_controls", {}),
            "sampler_trace": getattr(self, "_event_sampler_trace", {}),
            "runtime_controls": getattr(self, "_event_requested_runtime_controls", {}),
            "input_normalization_signature": (getattr(self, "_event_input_normalization", {}) or {}).get("normalized_signature", ""),
            "runtime_aliases": self._event_runtime_aliases(),
            "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
        }
        encoded = json.dumps(_event_json_safe(signature_source), sort_keys=True, ensure_ascii=True)
        signature = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]
        return signature, signature_source

    def _find_previous_runtime_monitor_json(self, out_dir, current_json_path, settings_signature):
        try:
            current = Path(current_json_path).resolve()
            candidates = sorted(
                Path(out_dir).glob("*_runtime_monitor.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for path in candidates:
                try:
                    if path.resolve() == current:
                        continue
                    with open(path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    if payload.get("settings_signature") == settings_signature:
                        return path, payload
                except Exception:
                    continue
        except Exception:
            pass
        return None, None

    def _write_runtime_monitor_csv(self, csv_path, rows):
        fieldnames = [
            "index",
            "stage",
            "record_type",
            "status",
            "elapsed_from_start_s",
            "delta_since_previous_s",
            "process_rss_mb",
            "torch_cuda_available",
            "cuda_allocated_mb",
            "cuda_reserved_mb",
            "cuda_max_allocated_mb",
            "cuda_max_reserved_mb",
        ]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})

    def _write_runtime_monitor_diff_csv(self, diff_path, current_payload, previous_payload, previous_path):
        def summary_value(payload, key, default=""):
            summary = payload.get("runtime_monitor_summary", {}) if isinstance(payload, dict) else {}
            return summary.get(key, default) if isinstance(summary, dict) else default

        def motion_value(payload, key, default=""):
            motion = payload.get("motion_summary", {}) if isinstance(payload, dict) else {}
            return motion.get(key, default) if isinstance(motion, dict) else default

        rows = []
        if not previous_payload:
            rows.append({
                "metric": "baseline",
                "current_value": current_payload.get("json_path", ""),
                "previous_value": "",
                "delta": "",
                "note": "no previous runtime monitor sidecar with matching settings_signature",
            })
        else:
            rows.append({
                "metric": "baseline",
                "current_value": current_payload.get("json_path", ""),
                "previous_value": str(previous_path or ""),
                "delta": "",
                "note": "matched by settings_signature",
            })
            comparisons = [
                ("record_count", summary_value(current_payload, "record_count"), summary_value(previous_payload, "record_count")),
                ("observed_stage_span_seconds", summary_value(current_payload, "observed_stage_span_seconds"), summary_value(previous_payload, "observed_stage_span_seconds")),
                ("result_status", current_payload.get("result_status", ""), previous_payload.get("result_status", "")),
                ("completion_gate", current_payload.get("completion_gate", ""), previous_payload.get("completion_gate", "")),
                ("video_path", current_payload.get("saved_video_path", ""), previous_payload.get("saved_video_path", "")),
            ]
            for metric, current_value, previous_value in comparisons:
                delta = ""
                try:
                    delta = round(float(current_value) - float(previous_value), 6)
                except Exception:
                    delta = ""
                rows.append({
                    "metric": metric,
                    "current_value": current_value,
                    "previous_value": previous_value,
                    "delta": delta,
                    "note": "",
                })

            motion_comparisons = [
                ("motion_stage", motion_value(current_payload, "motion_stage", ""), motion_value(previous_payload, "motion_stage", "")),
                ("motion_profile", motion_value(current_payload, "frame_motion_profile", ""), motion_value(previous_payload, "frame_motion_profile", "")),
                ("motion_stability_score", motion_value(current_payload, "frame_motion_stability_score", ""), motion_value(previous_payload, "frame_motion_stability_score", "")),
                ("motion_norm_mean", motion_value(current_payload, "frame_delta_norm_mean", ""), motion_value(previous_payload, "frame_delta_norm_mean", "")),
                ("motion_spike_ratio", motion_value(current_payload, "frame_delta_spike_ratio", ""), motion_value(previous_payload, "frame_delta_spike_ratio", "")),
                ("motion_reversal_ratio", motion_value(current_payload, "frame_delta_reversal_ratio", ""), motion_value(previous_payload, "frame_delta_reversal_ratio", "")),
                ("motion_cosine_mean", motion_value(current_payload, "frame_delta_cosine_mean", ""), motion_value(previous_payload, "frame_delta_cosine_mean", "")),
                ("motion_jerk_ratio", motion_value(current_payload, "frame_delta_jerk_ratio", ""), motion_value(previous_payload, "frame_delta_jerk_ratio", "")),
            ]
            for metric, current_value, previous_value in motion_comparisons:
                delta = ""
                try:
                    delta = round(float(current_value) - float(previous_value), 6)
                except Exception:
                    delta = ""
                rows.append({
                    "metric": metric,
                    "current_value": current_value,
                    "previous_value": previous_value,
                    "delta": delta,
                    "note": "",
                })

        with open(diff_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "current_value", "previous_value", "delta", "note"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _save_runtime_monitor_sidecars(self, packet, saved_report_path, saved_video_path, save_prefix):
        try:
            report_path = Path(saved_report_path) if saved_report_path else None
            if report_path and report_path.parent.exists():
                out_dir = report_path.parent
                stem = report_path.stem
            else:
                try:
                    import folder_paths
                    out_dir = Path(folder_paths.get_output_directory())
                except Exception:
                    out_dir = Path.cwd() / "output"
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                stem = f"{str(save_prefix or 'Singularity')}_{ts}"
            out_dir.mkdir(parents=True, exist_ok=True)

            json_path = out_dir / f"{stem}_runtime_monitor.json"
            csv_path = out_dir / f"{stem}_runtime_monitor.csv"
            diff_path = out_dir / f"{stem}_runtime_monitor_diff.csv"

            meta = packet.get("metadata", {}) if isinstance(packet, dict) else {}
            body = meta.get("event_core_body", {}) if isinstance(meta.get("event_core_body", {}), dict) else {}
            execution_records = meta.get("execution_records", []) if isinstance(meta.get("execution_records", []), list) else []

            settings_signature, settings_source = self._runtime_monitor_settings_signature(packet)
            rows = self._runtime_monitor_rows(packet)
            if not rows:
                rows = []
                for index, rec in enumerate(execution_records):
                    if not isinstance(rec, dict):
                        continue
                    rows.append({
                        "index": index,
                        "stage": rec.get("stage", ""),
                        "record_type": "execution_record",
                        "status": rec.get("status", ""),
                        "elapsed_from_start_s": "",
                        "delta_since_previous_s": "",
                        "process_rss_mb": "",
                        "torch_cuda_available": "",
                        "cuda_allocated_mb": "",
                        "cuda_reserved_mb": "",
                        "cuda_max_allocated_mb": "",
                        "cuda_max_reserved_mb": "",
                    })

            motion_summary = self._runtime_motion_summary(packet)
            summary = {
                "record_count": len(rows),
                "execution_record_count": len([r for r in execution_records if isinstance(r, dict)]),
                "stage_math_count": body.get("stage_math_count", 0),
                "boundary_count": body.get("boundary_count", 0),
                "math_tensor_record_count": body.get("math_tensor_record_count", 0),
                "observed_stage_span_seconds": "",
                "observer_only": True,
            }

            payload = {
                "schema_version": "singularity-runtime-monitor-v1",
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
                "runtime_name": EVENT_HORIZON_RUNTIME_NAME,
                "settings_signature": settings_signature,
                "settings_signature_source": settings_source,
                "report_path": str(saved_report_path or ""),
                "saved_video_path": str(saved_video_path or ""),
                "result_status": body.get("result_status", meta.get("result_status", "")),
                "completion_gate": (body.get("completion_gate", {}) if isinstance(body.get("completion_gate", {}), dict) else {}).get("status", ""),
                "runtime_monitor_summary": summary,
                "motion_summary": motion_summary,
                "execution_records": execution_records,
                "json_path": str(json_path),
                "csv_path": str(csv_path),
                "diff_path": str(diff_path),
            }

            previous_path, previous_payload = self._find_previous_runtime_monitor_json(out_dir, json_path, settings_signature)
            payload["previous_runtime_monitor_json"] = str(previous_path) if previous_path else ""

            with open(json_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(_event_json_safe(payload), f, ensure_ascii=False, indent=2)
                f.write("\n")
            self._write_runtime_monitor_csv(csv_path, rows)
            self._write_runtime_monitor_diff_csv(diff_path, payload, previous_payload, previous_path)

            return {
                "status": "ok",
                "schema_version": payload["schema_version"],
                "json_path": str(json_path),
                "csv_path": str(csv_path),
                "diff_path": str(diff_path),
                "previous_json_path": str(previous_path) if previous_path else "",
                "settings_signature": settings_signature,
                "record_count": len(rows),
                "motion_stage": motion_summary.get("motion_stage", ""),
                "motion_profile": motion_summary.get("frame_motion_profile", ""),
                "motion_stability_score": motion_summary.get("frame_motion_stability_score", ""),
                "observer_only": True,
            }
        except Exception as e:
            return {"status": "failed", "error": str(e), "observer_only": True}





