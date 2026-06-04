import os
import sys

# Ensure the node root is in sys.path for reliable imports when loaded by ComfyUI
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

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
from .core.enums import (
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
from .core.packet import (
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
from .core.signal import make_event_signal
from .core.relation import make_event_relation
from .core.conflict import make_conflict
from .core.event_sampler import EventSamplerCore, EventSamplerWindow, EventSamplerResult
from .reports.markdown_report import build_markdown_report
from .resolvers.role_resolver import resolve_role
from .resolvers.operator_registry import OperatorRegistry
from .resolvers.s_resolver import build_sstate_from_packet
from .utils.tensor_stats import compute_tensor_delta, extract_latent_samples, safe_shape
from .utils.frozen_helpers import build_input_signatures, build_passthrough_status, score_observability, collect_shared_targets, now_run_id
from .adapters.wan.wan_adapter import apply_wan_adapter

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

class EventDebugPing:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"message": ("STRING", {"default": "ping"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("message",)
    FUNCTION = "run"
    CATEGORY = "Singularity/Core"

    def run(self, message):
        return (f"Event Debug Ping: {message}",)


class EventInitPacket:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"debug_mode": (DEBUG_MODES, {"default": DEBUG_BASIC})}}

    RETURN_TYPES = ("EVENT_PACKET", "STRING")
    RETURN_NAMES = ("event_packet", "summary")
    FUNCTION = "run"
    CATEGORY = "Singularity/Core"

    def run(self, debug_mode):
        packet = make_event_packet(metadata={"debug_mode": debug_mode})
        packet = record_stage(
            packet,
            stage_name="EventInitPacket",
            action="INIT_PACKET",
            observed_behavior="EventPacket created with empty RouteMemory",
            metadata={"debug_mode": debug_mode},
        )
        return (packet, build_markdown_report(packet))


class EventReadText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "event_packet": ("EVENT_PACKET",),
                "text": ("STRING", {"default": "", "multiline": True}),
                "source_stage": ("STRING", {"default": "manual_text"}),
                "manual_role": (FORMULA_ROLES_CODE_PASS_6, {"default": "AUTO"}),
            }
        }

    RETURN_TYPES = ("EVENT_PACKET", "STRING")
    RETURN_NAMES = ("event_packet", "summary")
    FUNCTION = "run"
    CATEGORY = "Singularity/Readers"

    def run(self, event_packet, text, source_stage, manual_role):
        packet, signal, projection, conflict_ids = _read_signal(
            event_packet, TECH_TEXT, SPACE_TEXT, text, source_stage,
            manual_role, "text_strategy_route", "TextStrategyReader",
            metadata={"text_length": len(str(text or ""))},
        )
        sem = projection.get("semantic_summary", {})
        return (packet, f"Created {signal['id']} as TEXT/{signal['formula_role']}; projection={projection['id']}; words={sem.get('word_count', 0)}; conflicts={conflict_ids}")


class EventReadLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "event_packet": ("EVENT_PACKET",),
                "latent": ("LATENT",),
                "source_stage": ("STRING", {"default": "manual_latent"}),
                "manual_role": (FORMULA_ROLES_CODE_PASS_6, {"default": "AUTO"}),
                "route_position": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("EVENT_PACKET", "STRING")
    RETURN_NAMES = ("event_packet", "summary")
    FUNCTION = "run"
    CATEGORY = "Singularity/Readers"

    def run(self, event_packet, latent, source_stage, manual_role, route_position):
        packet, signal, projection, conflict_ids = _read_signal(
            event_packet, TECH_LATENT, SPACE_LATENT, latent, source_stage,
            manual_role, "latent_route", "LatentEventReader",
            route_position=route_position,
        )
        num = projection.get("numeric_summary", {})
        return (packet, f"Created {signal['id']} as LATENT/{signal['formula_role']}; projection={projection['id']}; shape={num.get('shape')}; conflicts={conflict_ids}")


class EventReadImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "event_packet": ("EVENT_PACKET",),
                "image": ("IMAGE",),
                "source_stage": ("STRING", {"default": "manual_image"}),
                "manual_role": (FORMULA_ROLES_CODE_PASS_6, {"default": "AUTO"}),
                "route_position": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("EVENT_PACKET", "STRING")
    RETURN_NAMES = ("event_packet", "summary")
    FUNCTION = "run"
    CATEGORY = "Singularity/Readers"

    def run(self, event_packet, image, source_stage, manual_role, route_position):
        packet, signal, projection, conflict_ids = _read_signal(
            event_packet, TECH_IMAGE, SPACE_IMAGE, image, source_stage,
            manual_role, "image_route", "ImageOutcomeReader",
            route_position=route_position,
        )
        num = projection.get("numeric_summary", {})
        return (packet, f"Created {signal['id']} as IMAGE/{signal['formula_role']}; projection={projection['id']}; shape={num.get('shape')}; conflicts={conflict_ids}")


class EventReadNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "event_packet": ("EVENT_PACKET",),
                "noise": ("LATENT",),
                "source_stage": ("STRING", {"default": "manual_noise"}),
                "manual_role": (FORMULA_ROLES_CODE_PASS_6, {"default": "AUTO"}),
            }
        }

    RETURN_TYPES = ("EVENT_PACKET", "STRING")
    RETURN_NAMES = ("event_packet", "summary")
    FUNCTION = "run"
    CATEGORY = "Singularity/Readers"

    def run(self, event_packet, noise, source_stage, manual_role):
        raw_noise = noise.get("samples") if isinstance(noise, dict) and "samples" in noise else noise
        packet, signal, projection, conflict_ids = _read_signal(
            event_packet, TECH_NOISE, SPACE_NOISE, raw_noise, source_stage,
            manual_role, "noise_route", "NoisePossibilityReader",
            metadata={"input_socket": "LATENT_as_noise_carrier"},
        )
        num = projection.get("numeric_summary", {})
        return (packet, f"Created {signal['id']} as NOISE/{signal['formula_role']}; projection={projection['id']}; shape={num.get('shape')}; conflicts={conflict_ids}")


class EventReadConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "event_packet": ("EVENT_PACKET",),
                "conditioning": ("CONDITIONING",),
                "source_stage": ("STRING", {"default": "manual_conditioning"}),
                "manual_role": (FORMULA_ROLES_CODE_PASS_6, {"default": "AUTO"}),
            }
        }

    RETURN_TYPES = ("EVENT_PACKET", "STRING")
    RETURN_NAMES = ("event_packet", "summary")
    FUNCTION = "run"
    CATEGORY = "Singularity/Readers"

    def run(self, event_packet, conditioning, source_stage, manual_role):
        packet, signal, projection, conflict_ids = _read_signal(
            event_packet, TECH_CONDITIONING, SPACE_CONDITIONING, conditioning, source_stage,
            manual_role, "conditioning_route", "ConditioningStrategyReader",
        )
        num = projection.get("numeric_summary", {})
        return (packet, f"Created {signal['id']} as CONDITIONING/{signal['formula_role']}; projection={projection['id']}; length={num.get('length')}; conflicts={conflict_ids}")


class EventStructuredStrategy:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "event_packet": ("EVENT_PACKET",),
                "main_strategy": ("STRING", {"default": "", "multiline": True}),
                "stable_anchors": ("STRING", {"default": "", "multiline": True}),
                "active_changes": ("STRING", {"default": "", "multiline": True}),
                "motion_axis": ("STRING", {"default": "", "multiline": True}),
                "contact_rule": ("STRING", {"default": "", "multiline": True}),
                "reciprocal_reaction": ("STRING", {"default": "", "multiline": True}),
                "deformation_rule": ("STRING", {"default": "", "multiline": True}),
                "material_rule": ("STRING", {"default": "", "multiline": True}),
                "support_rule": ("STRING", {"default": "", "multiline": True}),
                "endpoint": ("STRING", {"default": "", "multiline": True}),
                "loop_rule": ("STRING", {"default": "", "multiline": True}),
                "forbidden_drift": ("STRING", {"default": "", "multiline": True}),
                "negative_additions": ("STRING", {"default": "", "multiline": True}),
                "priority_notes": ("STRING", {"default": "", "multiline": True}),
                "strategy_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "source_stage": ("STRING", {"default": "StructuredStrategy"}),
            }
        }

    RETURN_TYPES = ("EVENT_PACKET", "STRING")
    RETURN_NAMES = ("event_packet", "strategy_summary")
    FUNCTION = "run"
    CATEGORY = "Singularity/Strategy"

    def run(
        self,
        event_packet,
        main_strategy,
        stable_anchors,
        active_changes,
        motion_axis,
        contact_rule,
        reciprocal_reaction,
        deformation_rule,
        material_rule,
        support_rule,
        endpoint,
        loop_rule,
        forbidden_drift,
        negative_additions,
        priority_notes,
        strategy_strength,
        source_stage,
    ):
        packet = ensure_packet(event_packet)
        structured = {
            "main_strategy": main_strategy,
            "stable_anchors": stable_anchors,
            "active_changes": active_changes,
            "motion_axis": motion_axis,
            "contact_rule": contact_rule,
            "reciprocal_reaction": reciprocal_reaction,
            "deformation_rule": deformation_rule,
            "material_rule": material_rule,
            "support_rule": support_rule,
            "endpoint": endpoint,
            "loop_rule": loop_rule,
            "forbidden_drift": forbidden_drift,
            "negative_additions": negative_additions,
            "priority_notes": priority_notes,
            "strategy_strength": strategy_strength,
        }

        combined_text = "\\n".join([
            str(main_strategy or ""),
            str(stable_anchors or ""),
            str(active_changes or ""),
            str(motion_axis or ""),
            str(contact_rule or ""),
            str(reciprocal_reaction or ""),
            str(deformation_rule or ""),
            str(material_rule or ""),
            str(support_rule or ""),
            str(endpoint or ""),
            str(loop_rule or ""),
            str(forbidden_drift or ""),
            str(negative_additions or ""),
            str(priority_notes or ""),
        ])

        packet, signal, projection, conflict_ids = _read_signal(
            packet,
            TECH_TEXT,
            SPACE_TEXT,
            combined_text,
            source_stage,
            "StrategyCurrent",
            "structured_strategy_route",
            "TextStrategyReader",
            metadata={
                "structured_strategy": structured,
                "text_length": len(combined_text),
            },
        )

        # Strategy completeness warnings.
        checks = [
            (stable_anchors, CONFLICT_NO_STABLE_ANCHORS, "Stable anchors are empty.", "Add what must remain fixed/preserved."),
            (active_changes, CONFLICT_NO_ACTIVE_CHANGES, "Active changes are empty.", "Add what should move/change."),
            (contact_rule, CONFLICT_NO_CONTACT_RULE, "Contact rule is empty.", "Add contact/alignment/interaction rule if relevant."),
            (endpoint, CONFLICT_NO_ENDPOINT, "Endpoint is empty.", "Add the intended end state/equilibrium."),
            (forbidden_drift, CONFLICT_NO_FORBIDDEN_DRIFT, "Forbidden drift is empty.", "Add what must not drift or change."),
        ]

        for value, conflict_type, symptom, suggestion in checks:
            if not str(value or "").strip():
                conflict = make_conflict(
                    conflict_type,
                    severity=SEV_LOW,
                    involved_signal_ids=[signal["id"]],
                    involved_projection_ids=[projection["id"]],
                    stage_position=source_stage,
                    suspected_cause="StructuredStrategy field is empty.",
                    observed_symptom=symptom,
                    suggested_response=suggestion,
                )
                packet = add_conflict(packet, conflict)
                conflict_ids.append(conflict["id"])

        packet = record_stage(
            packet,
            stage_name="EventStructuredStrategy",
            action="READ_STRUCTURED_STRATEGY",
            observed_behavior="Structured strategy read as explicit StrategyCurrent",
            output_signal_ids=[signal["id"]],
            projection_ids=[projection["id"]],
            conflict_ids=conflict_ids,
            formula_note="Structured Strategy creates expectations, priority map and route hints",
            metadata={
                "strategy_strength": strategy_strength,
                "has_priority_map": bool(projection.get("metadata", {}).get("priority_map")),
                "has_route_hints": bool(projection.get("metadata", {}).get("route_hints")),
            },
        )

        role_vector = projection.get("role_vector", {})
        summary = (
            f"StructuredStrategy signal={signal['id']} projection={projection['id']} "
            f"main={role_vector.get('has_main_strategy')} anchors={role_vector.get('has_anchors')} "
            f"active={role_vector.get('has_active_changes')} contact={role_vector.get('has_contact_rule')} "
            f"endpoint={role_vector.get('has_endpoint')} forbidden_drift={role_vector.get('has_forbidden_drift')} "
            f"conflicts={conflict_ids}"
        )
        return (packet, summary)


class EventRelationBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "event_packet": ("EVENT_PACKET",),
                "relation_type": (RELATION_TYPES_CODE_PASS_6, {"default": "EQUALS_AS_EVENT"}),
                "stage_name": ("STRING", {"default": "RelationBuilder"}),
                "source_signal_ids": ("STRING", {"default": ""}),
                "target_signal_ids": ("STRING", {"default": ""}),
                "formula_meaning": ("STRING", {"default": ""}),
                "local_strategy_id": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("EVENT_PACKET", "STRING")
    RETURN_NAMES = ("event_packet", "summary")
    FUNCTION = "run"
    CATEGORY = "Singularity/Research"

    def run(self, event_packet, relation_type, stage_name, source_signal_ids, target_signal_ids, formula_meaning, local_strategy_id):
        packet = ensure_packet(event_packet)
        source_ids = _parse_id_list(source_signal_ids)
        target_ids = _parse_id_list(target_signal_ids)
        conflict_ids = []

        if not source_ids and not target_ids:
            latest = get_latest_signals(packet, 2)
            if len(latest) >= 2:
                source_ids = [latest[-2]["id"]]
                target_ids = [latest[-1]["id"]]

        invalid_ids = []
        for sid in source_ids + target_ids:
            if not signal_exists(packet, sid):
                invalid_ids.append(sid)

        if invalid_ids:
            conflict = make_conflict(
                CONFLICT_INVALID_SIGNAL_ID,
                severity=SEV_MEDIUM,
                involved_signal_ids=invalid_ids,
                stage_position=stage_name,
                suspected_cause="RelationBuilder received signal ids not present in EventPacket.",
                observed_symptom=f"Invalid signal ids: {invalid_ids}",
                suggested_response="Use ids from report or leave fields empty to use latest two signals.",
            )
            packet = add_conflict(packet, conflict)
            conflict_ids.append(conflict["id"])

        if not source_ids or not target_ids:
            conflict = make_conflict(
                CONFLICT_EMPTY_RELATION,
                severity=SEV_MEDIUM,
                involved_signal_ids=source_ids + target_ids,
                stage_position=stage_name,
                suspected_cause="RelationBuilder did not receive enough source/target signals.",
                observed_symptom="Relation has empty source or target side.",
                suggested_response="Create/read at least two signals or provide explicit ids.",
            )
            packet = add_conflict(packet, conflict)
            conflict_ids.append(conflict["id"])

        source_projection_ids = []
        target_projection_ids = []
        for sid in source_ids:
            source_projection_ids.extend(get_projection_ids_for_signal(packet, sid))
        for tid in target_ids:
            target_projection_ids.extend(get_projection_ids_for_signal(packet, tid))

        if not formula_meaning:
            formula_meaning = f"{relation_type}: source signals relate to target signals"

        if not local_strategy_id:
            local_strategy_id = f"S0.{str(relation_type).lower()}"

        relation = make_event_relation(
            relation_type=relation_type,
            source_signal_ids=source_ids,
            target_signal_ids=target_ids,
            source_projection_ids=source_projection_ids,
            target_projection_ids=target_projection_ids,
            operator_name="EventRelationBuilder",
            formula_meaning=formula_meaning,
            local_strategy_id=local_strategy_id,
            equality_status=EQ_UNKNOWN,
            metadata={"stage_name": stage_name},
        )
        relation["conflict_ids"] = conflict_ids[:]

        packet = add_relation(packet, relation)
        packet = record_stage(
            packet,
            stage_name="EventRelationBuilder",
            action="CREATE_RELATION",
            observed_behavior=f"Relation created: {relation_type}",
            input_signal_ids=source_ids + target_ids,
            relation_ids=[relation["id"]],
            conflict_ids=conflict_ids,
            formula_note=formula_meaning,
        )

        return (packet, f"Created relation {relation['id']} type={relation_type} sources={source_ids} targets={target_ids}; conflicts={conflict_ids}")


class EventSStateBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "event_packet": ("EVENT_PACKET",),
                "position": ("STRING", {"default": "S0"}),
                "relation_ids": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("EVENT_PACKET", "STRING")
    RETURN_NAMES = ("event_packet", "summary")
    FUNCTION = "run"
    CATEGORY = "Singularity/Research"

    def run(self, event_packet, position, relation_ids):
        packet = ensure_packet(event_packet)
        rel_ids = _parse_id_list(relation_ids)
        if not rel_ids:
            rel_ids = None

        packet, sstate = build_sstate_from_packet(packet, position=position, active_relation_ids=rel_ids)
        packet = record_stage(
            packet,
            stage_name="EventSStateBuilder",
            action="BUILD_SSTATE",
            observed_behavior="SState resolved from active relations",
            input_signal_ids=sstate.get("active_signal_ids", []),
            relation_ids=sstate.get("active_relation_ids", []),
            sstate_ids=[sstate["id"]],
            conflict_ids=sstate.get("conflict_ids", []),
            formula_note="SState is relation-based, not signal-based",
        )

        return (packet, f"Created SState {sstate['id']} position={position}; active_relations={len(sstate.get('active_relation_ids', []))}; local_strategies={len(sstate.get('local_strategies', {}))}")


class EventNoiseStrategy:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "event_packet": ("EVENT_PACKET",),
                "mode": (["READ_EXISTING", "GENERATE_FROM_LATENT"], {"default": "READ_EXISTING"}),
                "latent_or_noise": ("LATENT",),
                "source_stage": ("STRING", {"default": "EventNoiseStrategy"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "noise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "normalize_noise": ("BOOLEAN", {"default": False}),
                "target_mean": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "target_std": ("FLOAT", {"default": 1.0, "min": 0.000001, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT", "EVENT_PACKET", "STRING")
    RETURN_NAMES = ("noise_or_latent", "event_packet", "noise_report")
    FUNCTION = "run"
    CATEGORY = "Singularity/Noise"

    def run(self, event_packet, mode, latent_or_noise, source_stage, seed, noise_strength, normalize_noise, target_mean, target_std):
        packet = ensure_packet(event_packet)
        conflict_ids = []

        raw_input = latent_or_noise
        input_samples = extract_latent_samples(raw_input)
        generated = False
        noise_tensor = None
        output_latent = latent_or_noise

        try:
            import torch
            if mode == "GENERATE_FROM_LATENT":
                if input_samples is None or not hasattr(input_samples, "shape"):
                    conflict = make_conflict(
                        CONFLICT_NOISE_MISSING,
                        severity=SEV_MEDIUM,
                        stage_position=source_stage,
                        suspected_cause="Cannot generate noise without tensor-like latent samples.",
                        observed_symptom="latent_or_noise has no tensor-like samples.",
                        suggested_response="Connect a valid LATENT object with samples.",
                    )
                    packet = add_conflict(packet, conflict)
                    conflict_ids.append(conflict["id"])
                    noise_tensor = input_samples
                else:
                    generator = None
                    try:
                        generator = torch.Generator(device=input_samples.device)
                        generator.manual_seed(int(seed))
                        noise_tensor = torch.randn(
                            input_samples.shape,
                            dtype=input_samples.dtype,
                            device=input_samples.device,
                            generator=generator,
                        )
                    except Exception:
                        torch.manual_seed(int(seed))
                        noise_tensor = torch.randn_like(input_samples)

                    if normalize_noise and noise_tensor is not None:
                        nf = noise_tensor.float()
                        current_mean = nf.mean()
                        current_std = nf.std()
                        eps = 1e-12
                        noise_tensor = ((nf - current_mean) / max(float(current_std.item()), eps)) * float(target_std) + float(target_mean)
                        noise_tensor = noise_tensor.to(dtype=input_samples.dtype)

                    noise_tensor = noise_tensor * float(noise_strength)
                    output_latent = dict(latent_or_noise) if isinstance(latent_or_noise, dict) else {"samples": noise_tensor}
                    output_latent["samples"] = noise_tensor
                    generated = True
            else:
                # READ_EXISTING: treat latent samples as the existing noise carrier.
                noise_tensor = input_samples

            if noise_tensor is None or not hasattr(noise_tensor, "shape"):
                conflict = make_conflict(
                    CONFLICT_NOISE_SHAPE_UNKNOWN,
                    severity=SEV_LOW,
                    stage_position=source_stage,
                    suspected_cause="Noise object is not tensor-like.",
                    observed_symptom="Noise shape could not be read.",
                    suggested_response="Use GENERATE_FROM_LATENT or connect a LATENT-like object.",
                )
                packet = add_conflict(packet, conflict)
                conflict_ids.append(conflict["id"])

        except Exception as e:
            conflict = make_conflict(
                CONFLICT_NOISE_READER_FAILED,
                severity=SEV_MEDIUM,
                stage_position=source_stage,
                suspected_cause="EventNoiseStrategy failed during noise read/generation.",
                observed_symptom=str(e),
                suggested_response="Check torch availability and latent input format.",
            )
            packet = add_conflict(packet, conflict)
            conflict_ids.append(conflict["id"])
            noise_tensor = input_samples

        packet, signal, projection, read_conflicts = _read_signal(
            packet,
            TECH_NOISE,
            SPACE_NOISE,
            noise_tensor,
            source_stage,
            "StrategyCandidate",
            "noise_strategy_route",
            "NoisePossibilityReader",
            metadata={
                "seed": seed,
                "noise_strength": noise_strength,
                "noise_mode": mode,
                "normalize_noise": normalize_noise,
                "target_mean": target_mean,
                "target_std": target_std,
                "generated_from_latent_shape": safe_shape(input_samples),
                "generated": generated,
            },
        )
        conflict_ids.extend(read_conflicts)

        packet = record_stage(
            packet,
            stage_name="EventNoiseStrategy",
            action="READ_NOISE_STRATEGY",
            observed_behavior=f"Noise route registered as StrategyCandidate mode={mode}",
            output_signal_ids=[signal["id"]],
            projection_ids=[projection["id"]],
            conflict_ids=conflict_ids,
            formula_note="NOISE is a StrategyCandidate / possibility field for future sampler behavior",
            metadata={
                "mode": mode,
                "seed": seed,
                "noise_strength": noise_strength,
                "generated": generated,
                "normalize_noise": normalize_noise,
            },
        )

        num = projection.get("numeric_summary", {})
        summary = (
            f"NoiseStrategy signal={signal['id']} projection={projection['id']} "
            f"mode={mode} generated={generated} shape={num.get('shape')} "
            f"mean={num.get('mean')} std={num.get('std')} norm={num.get('norm')} "
            f"seed={seed} strength={noise_strength} conflicts={conflict_ids}"
        )
        return (output_latent, packet, summary)


class EventLatentBoundaryReader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "event_packet": ("EVENT_PACKET",),
                "latent_before": ("LATENT",),
                "latent_after": ("LATENT",),
                "stage_name": ("STRING", {"default": "LatentBoundary"}),
                "position": ("STRING", {"default": "S_boundary"}),
                "next_requirement_hint": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("EVENT_PACKET", "STRING")
    RETURN_NAMES = ("event_packet", "boundary_summary")
    FUNCTION = "run"
    CATEGORY = "Singularity/Boundary"

    def run(self, event_packet, latent_before, latent_after, stage_name, position, next_requirement_hint):
        packet = ensure_packet(event_packet)
        conflict_ids = []

        # Create/read before latent as OutcomePrevious.
        packet, before_signal, before_projection, before_conflicts = _read_signal(
            packet,
            TECH_LATENT,
            SPACE_LATENT,
            latent_before,
            f"{stage_name}_before",
            ROLE_OUTCOME_PREVIOUS,
            "latent_before_boundary_route",
            "LatentEventReader",
            route_position="before",
        )
        conflict_ids.extend(before_conflicts)

        # Create/read after latent as OutcomeNext.
        packet, after_signal, after_projection, after_conflicts = _read_signal(
            packet,
            TECH_LATENT,
            SPACE_LATENT,
            latent_after,
            f"{stage_name}_after",
            ROLE_OUTCOME_NEXT,
            "latent_after_boundary_route",
            "LatentEventReader",
            route_position="after",
        )
        conflict_ids.extend(after_conflicts)

        before_samples = extract_latent_samples(latent_before)
        after_samples = extract_latent_samples(latent_after)

        delta, error = compute_tensor_delta(latent_before, latent_after)
        delta_signal = None
        delta_projection = None

        if delta is None:
            conflict_type = CONFLICT_BOUNDARY_SHAPE_MISMATCH if "shape mismatch" in str(error).lower() else CONFLICT_DELTA_UNAVAILABLE
            conflict = make_conflict(
                conflict_type,
                severity=SEV_MEDIUM,
                involved_signal_ids=[before_signal["id"], after_signal["id"]],
                stage_position=stage_name,
                suspected_cause="Could not compute delta between latent_before and latent_after.",
                observed_symptom=str(error),
                suggested_response="Check that both latents have compatible samples tensors.",
                metadata={
                    "before_shape": safe_shape(before_samples),
                    "after_shape": safe_shape(after_samples),
                },
            )
            packet = add_conflict(packet, conflict)
            conflict_ids.append(conflict["id"])
            source_ids = [before_signal["id"]]
            source_proj_ids = [before_projection["id"]]
            delta_id = None
            delta_norm = None
            relative_delta = None
        else:
            packet, delta_signal, delta_projection, delta_conflicts = _read_signal(
                packet,
                TECH_DELTA,
                SPACE_DELTA,
                delta,
                f"{stage_name}_delta",
                ROLE_OBSERVED_BEHAVIOR,
                "delta_boundary_route",
                "DeltaReader",
                metadata={
                    "before_signal_id": before_signal["id"],
                    "after_signal_id": after_signal["id"],
                    "before_ref": before_samples,
                    "next_requirement_hint": next_requirement_hint,
                },
            )
            conflict_ids.extend(delta_conflicts)
            source_ids = [before_signal["id"], delta_signal["id"]]
            source_proj_ids = [before_projection["id"], delta_projection["id"]]
            delta_id = delta_signal["id"]
            num = delta_projection.get("numeric_summary", {})
            delta_norm = num.get("delta_norm")
            relative_delta = num.get("relative_delta")

        relation = make_event_relation(
            relation_type=REL_TRANSFORMS_INTO,
            source_signal_ids=source_ids,
            target_signal_ids=[after_signal["id"]],
            source_projection_ids=source_proj_ids,
            target_projection_ids=[after_projection["id"]],
            operator_name="EventLatentBoundaryReader",
            formula_meaning="latent before plus observed delta becomes latent after",
            local_strategy_id=f"{position}.latent_boundary",
            equality_status=EQ_UNKNOWN,
            metadata={
                "stage_name": stage_name,
                "boundary_type": "latent",
                "next_requirement_hint": next_requirement_hint,
                "delta_signal_id": delta_id,
                "delta_norm": delta_norm,
                "relative_delta": relative_delta,
            },
        )
        relation["conflict_ids"] = conflict_ids[:]

        packet = add_relation(packet, relation)
        packet, sstate = build_sstate_from_packet(packet, position=position, active_relation_ids=[relation["id"]])
        packet = record_stage(
            packet,
            stage_name="EventLatentBoundaryReader",
            action="READ_BOUNDARY",
            observed_behavior="Latent boundary read as before + delta = after",
            input_signal_ids=[before_signal["id"], after_signal["id"]],
            output_signal_ids=[x for x in [delta_id] if x],
            projection_ids=[before_projection["id"], after_projection["id"]] + ([delta_projection["id"]] if delta_projection else []),
            relation_ids=[relation["id"]],
            sstate_ids=[sstate["id"]],
            conflict_ids=conflict_ids,
            formula_note="Outcome_before + ObservedBehavior_delta = S_boundary = Outcome_after",
            metadata={
                "delta_norm": delta_norm,
                "relative_delta": relative_delta,
                "next_requirement_hint": next_requirement_hint,
            },
        )

        summary = (
            f"Boundary {stage_name}: before={before_signal['id']} after={after_signal['id']} "
            f"delta={delta_id} relation={relation['id']} sstate={sstate['id']} "
            f"delta_norm={delta_norm} relative_delta={relative_delta} conflicts={conflict_ids}"
        )
        return (packet, summary)


class EventSamplerBoundaryReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "event_packet": ("EVENT_PACKET",),
                "latent_before_sampler": ("LATENT",),
                "latent_after_sampler": ("LATENT",),
                "position": ("STRING", {"default": "S_sampler_summary"}),
                "sampler_name": ("STRING", {"default": ""}),
                "scheduler": ("STRING", {"default": ""}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "cfg": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("EVENT_PACKET", "STRING")
    RETURN_NAMES = ("event_packet", "sampler_boundary_report")
    FUNCTION = "run"
    CATEGORY = "Singularity/Boundary"

    def run(self, event_packet, latent_before_sampler, latent_after_sampler, position, sampler_name, scheduler, steps, cfg, denoise, seed):
        packet = ensure_packet(event_packet)
        stage_name = "SamplerBoundaryReport"
        conflict_ids = []

        # Read before sampler latent as previous outcome.
        packet, before_signal, before_projection, before_conflicts = _read_signal(
            packet,
            TECH_LATENT,
            SPACE_LATENT,
            latent_before_sampler,
            "SamplerBoundary_before",
            ROLE_OUTCOME_PREVIOUS,
            "sampler_before_latent_route",
            "LatentEventReader",
            route_position="before_sampler",
        )
        conflict_ids.extend(before_conflicts)

        # Read after sampler latent as next outcome.
        packet, after_signal, after_projection, after_conflicts = _read_signal(
            packet,
            TECH_LATENT,
            SPACE_LATENT,
            latent_after_sampler,
            "SamplerBoundary_after",
            ROLE_OUTCOME_NEXT,
            "sampler_after_latent_route",
            "LatentEventReader",
            route_position="after_sampler",
        )
        conflict_ids.extend(after_conflicts)

        before_samples = extract_latent_samples(latent_before_sampler)
        after_samples = extract_latent_samples(latent_after_sampler)

        delta, error = compute_tensor_delta(latent_before_sampler, latent_after_sampler)
        delta_signal = None
        delta_projection = None
        delta_id = None
        delta_norm = None
        relative_delta = None

        if delta is None:
            conflict_type = CONFLICT_SAMPLER_BOUNDARY_SHAPE_MISMATCH if "shape mismatch" in str(error).lower() else CONFLICT_SAMPLER_DELTA_UNAVAILABLE
            conflict = make_conflict(
                conflict_type,
                severity=SEV_MEDIUM,
                involved_signal_ids=[before_signal["id"], after_signal["id"]],
                stage_position=stage_name,
                suspected_cause="Could not compute sampler-level delta.",
                observed_symptom=str(error),
                suggested_response="Check that sampler input/output latents have compatible samples tensors.",
                metadata={
                    "before_shape": safe_shape(before_samples),
                    "after_shape": safe_shape(after_samples),
                    "sampler_name": sampler_name,
                    "scheduler": scheduler,
                    "steps": steps,
                    "cfg": cfg,
                    "denoise": denoise,
                    "seed": seed,
                },
            )
            packet = add_conflict(packet, conflict)
            conflict_ids.append(conflict["id"])
            source_ids = [before_signal["id"]]
            source_proj_ids = [before_projection["id"]]
        else:
            packet, delta_signal, delta_projection, delta_conflicts = _read_signal(
                packet,
                TECH_DELTA,
                SPACE_DELTA,
                delta,
                "SamplerBoundary_delta",
                ROLE_OBSERVED_BEHAVIOR,
                "sampler_delta_route",
                "DeltaReader",
                metadata={
                    "before_signal_id": before_signal["id"],
                    "after_signal_id": after_signal["id"],
                    "before_ref": before_samples,
                    "sampler_name": sampler_name,
                    "scheduler": scheduler,
                    "steps": steps,
                    "cfg": cfg,
                    "denoise": denoise,
                    "seed": seed,
                    "trajectory_requirement": {
                        "sampler_name": sampler_name,
                        "scheduler": scheduler,
                        "steps": steps,
                        "cfg": cfg,
                        "denoise": denoise,
                        "seed": seed,
                    },
                },
            )
            conflict_ids.extend(delta_conflicts)
            delta_id = delta_signal["id"]
            source_ids = [before_signal["id"], delta_signal["id"]]
            source_proj_ids = [before_projection["id"], delta_projection["id"]]
            num = delta_projection.get("numeric_summary", {})
            delta_norm = num.get("delta_norm")
            relative_delta = num.get("relative_delta")

        missing_metadata = []
        if not sampler_name:
            missing_metadata.append("sampler_name")
        if not scheduler:
            missing_metadata.append("scheduler")
        if steps == 0:
            missing_metadata.append("steps")
        if missing_metadata:
            conflict = make_conflict(
                CONFLICT_SAMPLER_METADATA_MISSING,
                severity=SEV_LOW,
                involved_signal_ids=[before_signal["id"], after_signal["id"]],
                stage_position=stage_name,
                suspected_cause="Sampler metadata is incomplete.",
                observed_symptom=f"Missing or default metadata fields: {missing_metadata}",
                suggested_response="Fill sampler_name, scheduler and steps if available.",
                metadata={"missing_metadata": missing_metadata},
            )
            packet = add_conflict(packet, conflict)
            conflict_ids.append(conflict["id"])

        relation = make_event_relation(
            relation_type=REL_TRANSFORMS_INTO,
            source_signal_ids=source_ids,
            target_signal_ids=[after_signal["id"]],
            source_projection_ids=source_proj_ids,
            target_projection_ids=[after_projection["id"]],
            operator_name="EventSamplerBoundaryReport",
            formula_meaning="sampler transformed input latent into output latent through observed delta",
            local_strategy_id=f"{position}.latent_transition",
            equality_status=EQ_UNKNOWN,
            metadata={
                "stage_name": stage_name,
                "boundary_type": "sampler_summary",
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "steps": steps,
                "cfg": cfg,
                "denoise": denoise,
                "seed": seed,
                "delta_signal_id": delta_id,
                "delta_norm": delta_norm,
                "relative_delta": relative_delta,
                "trajectory_requirement": {
                    "sampler_name": sampler_name,
                    "scheduler": scheduler,
                    "steps": steps,
                    "cfg": cfg,
                    "denoise": denoise,
                    "seed": seed,
                },
            },
        )
        relation["conflict_ids"] = conflict_ids[:]

        packet = add_relation(packet, relation)
        packet, sstate = build_sstate_from_packet(packet, position=position, active_relation_ids=[relation["id"]])
        packet = record_stage(
            packet,
            stage_name="EventSamplerBoundaryReport",
            action="READ_SAMPLER_BOUNDARY",
            observed_behavior="Sampler summary boundary read as latent_before + sampler_delta = latent_after",
            input_signal_ids=[before_signal["id"], after_signal["id"]],
            output_signal_ids=[x for x in [delta_id] if x],
            projection_ids=[before_projection["id"], after_projection["id"]] + ([delta_projection["id"]] if delta_projection else []),
            relation_ids=[relation["id"]],
            sstate_ids=[sstate["id"]],
            conflict_ids=conflict_ids,
            formula_note="latent_before_sampler + sampler_delta = S_sampler_summary = latent_after_sampler",
            metadata={
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "steps": steps,
                "cfg": cfg,
                "denoise": denoise,
                "seed": seed,
                "delta_norm": delta_norm,
                "relative_delta": relative_delta,
            },
        )

        summary = (
            f"Sampler boundary: before={before_signal['id']} after={after_signal['id']} "
            f"delta={delta_id} relation={relation['id']} sstate={sstate['id']} "
            f"sampler={sampler_name} scheduler={scheduler} steps={steps} cfg={cfg} denoise={denoise} "
            f"delta_norm={delta_norm} relative_delta={relative_delta} conflicts={conflict_ids}"
        )
        return (packet, summary)


class EventBranchPacket:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "event_packet": ("EVENT_PACKET",),
                "branch_label": ("STRING", {"default": "branch"}),
            }
        }

    RETURN_TYPES = ("EVENT_PACKET", "STRING")
    RETURN_NAMES = ("event_packet", "summary")
    FUNCTION = "run"
    CATEGORY = "Singularity/Research"

    def run(self, event_packet, branch_label):
        packet = ensure_packet(event_packet)
        packet_copy = copy.deepcopy(packet)
        branch_id = now_run_id(prefix=str(branch_label or "branch"))
        packet_copy.setdefault("metadata", {})["branch_id"] = branch_id

        conflict = make_conflict(
            CONFLICT_PACKET_BRANCH_CREATED,
            severity=SEV_INFO,
            stage_position="EventBranchPacket",
            suspected_cause="Research EventPacket was explicitly branched.",
            observed_symptom="A deepcopy branch was created to avoid mutable dict graph corruption.",
            suggested_response="Use this branch independently from the source packet.",
            metadata={"branch_id": branch_id, "branch_label": branch_label},
        )
        packet_copy = add_conflict(packet_copy, conflict)
        packet_copy = record_stage(
            packet_copy,
            stage_name="EventBranchPacket",
            action="PACKET_BRANCH_CREATED",
            observed_behavior="EventPacket deep-copied for safe research branching",
            conflict_ids=[conflict["id"]],
            metadata={"branch_id": branch_id, "branch_label": branch_label},
        )
        return (packet_copy, f"Created EventPacket branch {branch_id}")


class EventSaveReportToFile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "report": ("STRING", {"default": "", "multiline": True}),
                "filename_prefix": ("STRING", {"default": "event_report"}),
                "output_directory": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("file_path", "report")
    FUNCTION = "run"
    CATEGORY = "Singularity/Core"

    def run(self, report, filename_prefix, output_directory):
        safe_prefix = re.sub(r"[^a-zA-Z0-9_\\-]+", "_", str(filename_prefix or "event_report")).strip("_")
        if not safe_prefix:
            safe_prefix = "event_report"

        if output_directory and str(output_directory).strip():
            out_dir = Path(str(output_directory)).expanduser()
        else:
            # Updated: reports now go directly to ComfyUI output (no more event_equality_reports)
            try:
                import folder_paths
                out_dir = Path(folder_paths.get_output_directory())
            except Exception:
                out_dir = Path.cwd() / "output"

        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = out_dir / f"{safe_prefix}_{timestamp}.md"
        path.write_text(str(report or ""), encoding="utf-8")
        return (str(path), report)


class EventCoreFrozen:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["BASIC", "RESEARCH", "WAN"], {"default": "BASIC"}),
                "report_detail": (["COMPACT", "STANDARD", "FULL"], {"default": "STANDARD"}),
                "text": ("STRING", {"default": "", "multiline": True}),
                "structured_strategy_text": ("STRING", {"default": "", "multiline": True}),
                "sampler_name": ("STRING", {"default": ""}),
                "scheduler": ("STRING", {"default": ""}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "cfg": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "workflow_hint": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "latent_before": ("LATENT",),
                "latent_after": ("LATENT",),
                "image": ("IMAGE",),
                "noise": ("LATENT",),
                "conditioning": ("CONDITIONING",),
                "decoded_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("EVENT_PACKET", "STRING", "STRING", "LATENT", "LATENT", "IMAGE", "LATENT", "CONDITIONING", "IMAGE")
    RETURN_NAMES = (
        "event_packet",
        "report",
        "short_summary",
        "latent_before_out",
        "latent_after_out",
        "image_out",
        "noise_out",
        "conditioning_out",
        "decoded_image_out",
    )
    FUNCTION = "run"
    CATEGORY = "Singularity/Core"

    def _missing_route(self, packet, route_name, mode, severity=None):
        sev = severity or (SEV_LOW if mode in ("RESEARCH", "WAN") else SEV_INFO)
        conflict = make_conflict(
            CONFLICT_FROZEN_ROUTE_MISSING,
            severity=sev,
            stage_position="EventCoreFrozen",
            suspected_cause=f"Optional route {route_name} was not connected.",
            observed_symptom=f"{route_name} route missing.",
            suggested_response="Connect this input only if the diagnostic requires it.",
            metadata={"route": route_name, "mode": mode},
        )
        return add_conflict(packet, conflict), conflict["id"]

    def _read_structured_text(self, packet, structured_strategy_text):
        # v0.1.1 accepts structured strategy as plain text, not parsing schema yet.
        structured = {
            "main_strategy": structured_strategy_text,
            "stable_anchors": "",
            "active_changes": "",
            "motion_axis": "",
            "contact_rule": "",
            "reciprocal_reaction": "",
            "deformation_rule": "",
            "material_rule": "",
            "support_rule": "",
            "endpoint": "",
            "loop_rule": "",
            "forbidden_drift": "",
            "negative_additions": "",
            "priority_notes": "",
            "temporal_scope": "",
            "identity_anchors": "",
            "camera_anchors": "",
            "motion_region": "",
            "stability_region": "",
            "strategy_strength": 1.0,
        }
        return _read_signal(
            packet,
            TECH_TEXT,
            SPACE_TEXT,
            structured_strategy_text,
            "Frozen_structured_strategy_text",
            "StrategyCurrent",
            "frozen_structured_strategy_route",
            "TextStrategyReader",
            metadata={"structured_strategy": structured, "text_length": len(str(structured_strategy_text or ""))},
        )

    def run(
        self,
        mode,
        report_detail,
        text,
        structured_strategy_text,
        sampler_name,
        scheduler,
        steps,
        cfg,
        denoise,
        seed,
        workflow_hint,
        latent_before=None,
        latent_after=None,
        image=None,
        noise=None,
        conditioning=None,
        decoded_image=None,
    ):
        run_id = now_run_id(prefix="frozen")
        packet = make_event_packet(metadata={
            "created_by": "EventCoreFrozen",
            "version": "0.1.1",
            "run_id": run_id,
        })

        packet = record_stage(
            packet,
            stage_name="EventCoreFrozen",
            action="INIT_FROZEN_NODE",
            observed_behavior="Frozen node created local EventPacket",
            metadata={"mode": mode, "report_detail": report_detail, "run_id": run_id},
        )

        relation_ids = []
        conflict_ids = []
        strategy_signal_ids = []
        source_signal_ids = []
        has_sampler_boundary = False
        has_decode = False
        has_structured = bool(str(structured_strategy_text or "").strip())
        has_noise = noise is not None

        input_signatures = build_input_signatures(
            text=text,
            structured_strategy_text=structured_strategy_text,
            latent_before=latent_before,
            latent_after=latent_after,
            image=image,
            noise=noise,
            conditioning=conditioning,
            decoded_image=decoded_image,
        )
        passthrough_status = build_passthrough_status(
            latent_before=latent_before,
            latent_after=latent_after,
            image=image,
            noise=noise,
            conditioning=conditioning,
            decoded_image=decoded_image,
        )

        # Missing-route diagnostics. These are not hard failures.
        for route_name, present in [
            ("text", bool(str(text or "").strip()) or has_structured),
            ("latent_before", latent_before is not None),
            ("latent_after", latent_after is not None),
            ("image", image is not None),
            ("noise", noise is not None),
            ("conditioning", conditioning is not None),
            ("decoded_image", decoded_image is not None),
        ]:
            if not present:
                packet, cid = self._missing_route(packet, route_name, mode)
                conflict_ids.append(cid)

        # Text strategy.
        if str(text or "").strip():
            packet, sig, proj, conf = _read_signal(
                packet,
                TECH_TEXT,
                SPACE_TEXT,
                text,
                "Frozen_text",
                "StrategyCurrent",
                "frozen_text_strategy_route",
                "TextStrategyReader",
                metadata={"text_length": len(str(text or ""))},
            )
            strategy_signal_ids.append(sig["id"])
            conflict_ids.extend(conf)

        # Structured strategy text.
        if has_structured:
            packet, sig, proj, conf = self._read_structured_text(packet, structured_strategy_text)
            strategy_signal_ids.append(sig["id"])
            conflict_ids.extend(conf)

        # Source / image route.
        if image is not None:
            packet, sig, proj, conf = _read_signal(
                packet,
                TECH_IMAGE,
                SPACE_IMAGE,
                image,
                "Frozen_image",
                "OutcomePrevious",
                "frozen_image_route",
                "ImageOutcomeReader",
                route_position="source",
            )
            source_signal_ids.append(sig["id"])
            conflict_ids.extend(conf)

        # Conditioning route.
        conditioning_signal_id = None
        if conditioning is not None:
            packet, sig, proj, conf = _read_signal(
                packet,
                TECH_CONDITIONING,
                SPACE_CONDITIONING,
                conditioning,
                "Frozen_conditioning",
                "StrategyCarrier",
                "frozen_conditioning_route",
                "ConditioningStrategyReader",
            )
            conditioning_signal_id = sig["id"]
            conflict_ids.extend(conf)

        # Noise route.
        noise_signal_id = None
        if noise is not None:
            raw_noise = noise.get("samples") if isinstance(noise, dict) and "samples" in noise else noise
            packet, sig, proj, conf = _read_signal(
                packet,
                TECH_NOISE,
                SPACE_NOISE,
                raw_noise,
                "Frozen_noise",
                "StrategyCandidate",
                "frozen_noise_route",
                "NoisePossibilityReader",
                metadata={"seed": seed, "noise_mode": "FROZEN_READ_EXISTING"},
            )
            noise_signal_id = sig["id"]
            conflict_ids.extend(conf)

        # Latent before / after + boundary.
        before_signal_id = None
        after_signal_id = None
        before_projection_id = None
        after_projection_id = None

        if latent_before is not None:
            packet, before_sig, before_proj, conf = _read_signal(
                packet,
                TECH_LATENT,
                SPACE_LATENT,
                latent_before,
                "Frozen_latent_before",
                "OutcomePrevious",
                "frozen_latent_before_route",
                "LatentEventReader",
                route_position="before_sampler",
            )
            before_signal_id = before_sig["id"]
            before_projection_id = before_proj["id"]
            source_signal_ids.append(before_signal_id)
            conflict_ids.extend(conf)

        if latent_after is not None:
            packet, after_sig, after_proj, conf = _read_signal(
                packet,
                TECH_LATENT,
                SPACE_LATENT,
                latent_after,
                "Frozen_latent_after",
                "OutcomeNext",
                "frozen_latent_after_route",
                "LatentEventReader",
                route_position="after_sampler",
            )
            after_signal_id = after_sig["id"]
            after_projection_id = after_proj["id"]
            conflict_ids.extend(conf)

        # Strategy relations.
        for sid in strategy_signal_ids:
            if conditioning_signal_id:
                rel = make_event_relation(
                    relation_type=REL_GUIDES,
                    source_signal_ids=[sid],
                    target_signal_ids=[conditioning_signal_id],
                    source_projection_ids=get_projection_ids_for_signal(packet, sid),
                    target_projection_ids=get_projection_ids_for_signal(packet, conditioning_signal_id),
                    operator_name="EventCoreFrozen",
                    formula_meaning="strategy guides conditioning route",
                    local_strategy_id="S0_frozen.strategy_conditioning",
                    equality_status=EQ_UNKNOWN,
                    metadata={"frozen_auto_relation": True},
                )
                packet = add_relation(packet, rel)
                relation_ids.append(rel["id"])

            target_candidates = []
            if after_signal_id:
                target_candidates.append(after_signal_id)
            target_candidates.extend(source_signal_ids)

            for target_id in target_candidates:
                if target_id == sid:
                    continue
                rel = make_event_relation(
                    relation_type=REL_GUIDES,
                    source_signal_ids=[sid],
                    target_signal_ids=[target_id],
                    source_projection_ids=get_projection_ids_for_signal(packet, sid),
                    target_projection_ids=get_projection_ids_for_signal(packet, target_id),
                    operator_name="EventCoreFrozen",
                    formula_meaning="strategy makes a guiding claim about target route",
                    local_strategy_id="S0_frozen.strategy_target",
                    equality_status=EQ_UNKNOWN,
                    metadata={"frozen_auto_relation": True},
                )
                packet = add_relation(packet, rel)
                relation_ids.append(rel["id"])

        # Boundary relation if before/after latents exist.
        if latent_before is not None and latent_after is not None and before_signal_id and after_signal_id:
            before_samples = extract_latent_samples(latent_before)
            delta, error = compute_tensor_delta(latent_before, latent_after)
            source_ids = [before_signal_id]
            source_proj_ids = [before_projection_id] if before_projection_id else []
            delta_id = None
            delta_norm = None
            relative_delta = None

            if delta is not None:
                packet, delta_sig, delta_proj, conf = _read_signal(
                    packet,
                    TECH_DELTA,
                    SPACE_DELTA,
                    delta,
                    "Frozen_sampler_delta",
                    "ObservedBehaviorCurrent",
                    "frozen_delta_route",
                    "DeltaReader",
                    metadata={
                        "before_signal_id": before_signal_id,
                        "after_signal_id": after_signal_id,
                        "before_ref": before_samples,
                        "sampler_name": sampler_name,
                        "scheduler": scheduler,
                        "steps": steps,
                        "cfg": cfg,
                        "denoise": denoise,
                        "seed": seed,
                    },
                )
                conflict_ids.extend(conf)
                delta_id = delta_sig["id"]
                source_ids.append(delta_id)
                source_proj_ids.append(delta_proj["id"])
                delta_norm = delta_proj.get("numeric_summary", {}).get("delta_norm")
                relative_delta = delta_proj.get("numeric_summary", {}).get("relative_delta")
            else:
                conflict = make_conflict(
                    CONFLICT_FROZEN_PARTIAL_OBSERVABILITY,
                    severity=SEV_LOW,
                    involved_signal_ids=[before_signal_id, after_signal_id],
                    stage_position="EventCoreFrozen",
                    suspected_cause="Frozen node could not compute latent delta.",
                    observed_symptom=str(error),
                    suggested_response="Check latent_before and latent_after shapes.",
                    metadata={"route": "sampler_boundary"},
                )
                packet = add_conflict(packet, conflict)
                conflict_ids.append(conflict["id"])

            rel = make_event_relation(
                relation_type=REL_TRANSFORMS_INTO,
                source_signal_ids=source_ids,
                target_signal_ids=[after_signal_id],
                source_projection_ids=source_proj_ids,
                target_projection_ids=[after_projection_id] if after_projection_id else [],
                operator_name="EventCoreFrozen",
                formula_meaning="latent_before plus observed delta transforms into latent_after",
                local_strategy_id="S0_frozen.sampler_boundary",
                equality_status=EQ_UNKNOWN,
                metadata={
                    "frozen_auto_relation": True,
                    "boundary_type": "sampler_summary",
                    "sampler_name": sampler_name,
                    "scheduler": scheduler,
                    "steps": steps,
                    "cfg": cfg,
                    "denoise": denoise,
                    "seed": seed,
                    "delta_signal_id": delta_id,
                    "delta_norm": delta_norm,
                    "relative_delta": relative_delta,
                },
            )
            packet = add_relation(packet, rel)
            relation_ids.append(rel["id"])
            has_sampler_boundary = True

            if noise_signal_id:
                rel_noise = make_event_relation(
                    relation_type=REL_CONSTRAINS,
                    source_signal_ids=[noise_signal_id],
                    target_signal_ids=[after_signal_id],
                    source_projection_ids=get_projection_ids_for_signal(packet, noise_signal_id),
                    target_projection_ids=get_projection_ids_for_signal(packet, after_signal_id),
                    operator_name="EventCoreFrozen",
                    formula_meaning="noise possibility field constrains latent_after event target",
                    local_strategy_id="S0_frozen.noise_target",
                    equality_status=EQ_UNKNOWN,
                    metadata={"frozen_auto_relation": True},
                )
                packet = add_relation(packet, rel_noise)
                relation_ids.append(rel_noise["id"])

        # Decoded image route.
        if decoded_image is not None:
            packet, img_sig, img_proj, conf = _read_signal(
                packet,
                TECH_IMAGE,
                SPACE_IMAGE,
                decoded_image,
                "Frozen_decoded_image",
                "OutcomeNext",
                "frozen_decoded_image_route",
                "ImageOutcomeReader",
                route_position="after_decode",
            )
            conflict_ids.extend(conf)
            has_decode = True

            if after_signal_id:
                rel = make_event_relation(
                    relation_type=REL_EXPANDS_TO,
                    source_signal_ids=[after_signal_id],
                    target_signal_ids=[img_sig["id"]],
                    source_projection_ids=get_projection_ids_for_signal(packet, after_signal_id),
                    target_projection_ids=[img_proj["id"]],
                    operator_name="EventCoreFrozen",
                    formula_meaning="latent_after expands to decoded visible output",
                    local_strategy_id="S0_frozen.decode",
                    equality_status=EQ_UNKNOWN,
                    metadata={"frozen_auto_relation": True, "boundary_type": "decode"},
                )
                packet = add_relation(packet, rel)
                relation_ids.append(rel["id"])

        # Build SState.
        if relation_ids:
            packet, sstate = build_sstate_from_packet(packet, position="S0_frozen", active_relation_ids=relation_ids)
        else:
            packet, sstate = build_sstate_from_packet(packet, position="S0_frozen", active_relation_ids=[])

        # Wan adapter only in WAN mode.
        has_wan = False
        if mode == "WAN":
            packet, wan_result = apply_wan_adapter(
                packet,
                mode="BASIC",
                hints={
                    "workflow_hint": workflow_hint or "EventCoreFrozen",
                    "has_high_sampler": False,
                    "has_low_sampler": False,
                    "has_temporal_module": False,
                    "has_lora_stack": False,
                    "has_accvid": False,
                    "has_lightx2v": False,
                },
            )
            relation_ids.extend(wan_result.get("created_relation_ids", []))
            conflict_ids.extend(wan_result.get("conflict_ids", []))
            has_wan = True

        observability = score_observability(
            input_signatures,
            has_structured=has_structured,
            has_noise=has_noise,
            has_sampler_boundary=has_sampler_boundary,
            has_decode=has_decode,
            has_wan=has_wan,
        )

        shared_targets = collect_shared_targets(packet)

        packet.setdefault("metadata", {})["frozen"] = {
            "enabled": True,
            "mode": mode,
            "report_detail": report_detail,
            "run_id": run_id,
            "semantic_mode": "simultaneous relational claims about one evolving event",
            "event_program_status": {
                "current_mode": "report_only",
                "program_state": "built_for_audit_not_control",
                "report_role": "visible audit/debug view of the internal Event Program",
                "final_target_output": "image_or_video_or_latent_or_decoded_media",
                "generation_control": "disabled_in_v0_1_1",
                "passthrough": "exact",
                "future_role": "generation_control_body_after_trace_and_correction_layers",
            },
            "input_signatures": input_signatures,
            "passthrough_status": passthrough_status,
            "observability": observability,
            "semantic_relation_mechanics": {
                "reading_mode": "simultaneous_event_claims",
                "sstate_meaning": "intersection_of_active_relations",
                "verdict_state": "EQ_UNKNOWN = relation recorded before judgment",
                "shared_targets": shared_targets,
            },
        }

        packet = record_stage(
            packet,
            stage_name="EventCoreFrozen",
            action="BUILD_FROZEN_REPORT",
            observed_behavior="Frozen node built internal Event Program audit report and returned exact passthrough objects",
            input_signal_ids=list(packet.get("signals", {}).keys()),
            relation_ids=relation_ids,
            sstate_ids=[sstate["id"]] if sstate else [],
            conflict_ids=conflict_ids,
            formula_note="partial inputs -> positional roles -> simultaneous relations -> SState intersection -> Event Program audit report -> exact passthrough",
            metadata={
                "mode": mode,
                "report_detail": report_detail,
                "run_id": run_id,
                "observability": observability,
            },
        )

        report = build_markdown_report(packet)
        short_summary = (
            f"EventCoreFrozen v0.1.1-r2: mode={mode}, current=report_only, target=image/video, "
            f"observability={observability.get('level')}:{observability.get('score')}, "
            f"signals={len(packet.get('signals', {}))}, "
            f"relations={len(packet.get('relations', {}))}, "
            f"sstates={len(packet.get('sstates', {}))}, "
            f"conflicts={len(packet.get('conflicts', {}))}, "
            f"passthrough=exact"
        )

        return (
            packet,
            report,
            short_summary,
            latent_before,
            latent_after,
            image,
            noise,
            conditioning,
            decoded_image,
        )


class EventWanAdapter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "event_packet": ("EVENT_PACKET",),
                "wan_mode": (["BASIC", "HIGH_LOW", "RESEARCH"], {"default": "BASIC"}),
                "workflow_hint": ("STRING", {"default": "", "multiline": True}),
                "has_high_sampler": ("BOOLEAN", {"default": False}),
                "has_low_sampler": ("BOOLEAN", {"default": False}),
                "has_temporal_module": ("BOOLEAN", {"default": False}),
                "has_lora_stack": ("BOOLEAN", {"default": False}),
                "has_accvid": ("BOOLEAN", {"default": False}),
                "has_lightx2v": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("EVENT_PACKET", "STRING")
    RETURN_NAMES = ("event_packet", "wan_report")
    FUNCTION = "run"
    CATEGORY = "Singularity/Adapters"

    def run(
        self,
        event_packet,
        wan_mode,
        workflow_hint,
        has_high_sampler,
        has_low_sampler,
        has_temporal_module,
        has_lora_stack,
        has_accvid,
        has_lightx2v,
    ):
        packet = ensure_packet(event_packet)
        hints = {
            "workflow_hint": workflow_hint,
            "has_high_sampler": has_high_sampler,
            "has_low_sampler": has_low_sampler,
            "has_temporal_module": has_temporal_module,
            "has_lora_stack": has_lora_stack,
            "has_accvid": has_accvid,
            "has_lightx2v": has_lightx2v,
        }
        packet, result = apply_wan_adapter(packet, mode=wan_mode, hints=hints)
        packet = record_stage(
            packet,
            stage_name="EventWanAdapter",
            action="APPLY_WAN_ADAPTER",
            observed_behavior="WanAdapter annotated generic EventPacket with Wan route labels and diagnostics",
            relation_ids=result.get("created_relation_ids", []),
            conflict_ids=result.get("conflict_ids", []),
            formula_note="WanAdapter labels first; it does not modify generation or generic relations.",
            metadata={
                "wan_mode": wan_mode,
                "route_status": result.get("route_status", {}),
            },
        )
        report = build_markdown_report(packet)
        return (packet, report)


class EventCoreNodeAlpha:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "debug_mode": (DEBUG_MODES, {"default": DEBUG_BASIC}),
                "adapter_mode": (["GENERIC", "WAN"], {"default": "GENERIC"}),
                "strategy_mode": (["NONE", "FREE_TEXT", "STRUCTURED", "BOTH"], {"default": "FREE_TEXT"}),

                "prompt_text": ("STRING", {"default": "", "multiline": True}),
                "stable_anchors": ("STRING", {"default": "", "multiline": True}),
                "active_changes": ("STRING", {"default": "", "multiline": True}),
                "motion_axis": ("STRING", {"default": "", "multiline": True}),
                "contact_rule": ("STRING", {"default": "", "multiline": True}),
                "reciprocal_reaction": ("STRING", {"default": "", "multiline": True}),
                "endpoint": ("STRING", {"default": "", "multiline": True}),
                "forbidden_drift": ("STRING", {"default": "", "multiline": True}),

                "read_sampler_boundary": ("BOOLEAN", {"default": False}),
                "read_decode_boundary": ("BOOLEAN", {"default": False}),
                "build_sstate": ("BOOLEAN", {"default": True}),

                "sampler_name": ("STRING", {"default": ""}),
                "scheduler": ("STRING", {"default": ""}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "cfg": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "latent_source": ("LATENT",),
                "conditioning": ("CONDITIONING",),
                "noise": ("LATENT",),
                "latent_before_sampler": ("LATENT",),
                "latent_after_sampler": ("LATENT",),
                "decoded_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("EVENT_PACKET", "STRING", "STRING")
    RETURN_NAMES = ("event_packet", "markdown_report", "short_summary")
    FUNCTION = "run"
    CATEGORY = "Singularity/Core"

    def run(
        self,
        debug_mode,
        adapter_mode,
        strategy_mode,
        prompt_text,
        stable_anchors,
        active_changes,
        motion_axis,
        contact_rule,
        reciprocal_reaction,
        endpoint,
        forbidden_drift,
        read_sampler_boundary,
        read_decode_boundary,
        build_sstate,
        sampler_name,
        scheduler,
        steps,
        cfg,
        denoise,
        seed,
        image=None,
        latent_source=None,
        conditioning=None,
        noise=None,
        latent_before_sampler=None,
        latent_after_sampler=None,
        decoded_image=None,
    ):
        packet = make_event_packet(metadata={
            "debug_mode": debug_mode,
            "adapter_mode": adapter_mode,
            "created_by": "EventCoreNodeAlpha",
            "alpha_report_only": True,
        })

        packet = record_stage(
            packet,
            stage_name="EventCoreNodeAlpha",
            action="INIT_ALPHA_BODY",
            observed_behavior="EventCoreNodeAlpha created internal report-only EventPacket",
            metadata={"debug_mode": debug_mode, "adapter_mode": adapter_mode, "strategy_mode": strategy_mode},
        )

        conflict_ids = []
        relation_ids = []
        strategy_signal_ids = []
        source_signal_ids = []
        sampler_relation_id = None
        decode_relation_id = None

        # Strategy route: free text.
        if strategy_mode in ("FREE_TEXT", "BOTH") and str(prompt_text or "").strip():
            packet, sig, proj, conf = _read_signal(
                packet,
                TECH_TEXT,
                SPACE_TEXT,
                prompt_text,
                "Alpha_prompt_text",
                "StrategyCurrent",
                "alpha_prompt_route",
                "TextStrategyReader",
                metadata={"text_length": len(str(prompt_text or ""))},
            )
            strategy_signal_ids.append(sig["id"])
            conflict_ids.extend(conf)

        # Strategy route: compact structured strategy inside Alpha.
        if strategy_mode in ("STRUCTURED", "BOTH"):
            structured = {
                "main_strategy": prompt_text,
                "stable_anchors": stable_anchors,
                "active_changes": active_changes,
                "motion_axis": motion_axis,
                "contact_rule": contact_rule,
                "reciprocal_reaction": reciprocal_reaction,
                "deformation_rule": "",
                "material_rule": "",
                "support_rule": "",
                "endpoint": endpoint,
                "loop_rule": "",
                "forbidden_drift": forbidden_drift,
                "negative_additions": "",
                "priority_notes": "",
                "strategy_strength": 1.0,
            }
            combined_text = "\\n".join([
                str(prompt_text or ""),
                str(stable_anchors or ""),
                str(active_changes or ""),
                str(motion_axis or ""),
                str(contact_rule or ""),
                str(reciprocal_reaction or ""),
                str(endpoint or ""),
                str(forbidden_drift or ""),
            ])
            packet, sig, proj, conf = _read_signal(
                packet,
                TECH_TEXT,
                SPACE_TEXT,
                combined_text,
                "Alpha_structured_strategy",
                "StrategyCurrent",
                "alpha_structured_strategy_route",
                "TextStrategyReader",
                metadata={"structured_strategy": structured, "text_length": len(combined_text)},
            )
            strategy_signal_ids.append(sig["id"])
            conflict_ids.extend(conf)

        if strategy_mode != "NONE" and not strategy_signal_ids:
            conflict = make_conflict(
                CONFLICT_ALPHA_ROUTE_MISSING,
                severity=SEV_LOW,
                stage_position="EventCoreNodeAlpha",
                suspected_cause="Strategy mode requested but no strategy content was provided.",
                observed_symptom="No strategy signal created.",
                suggested_response="Provide prompt_text or choose strategy_mode=NONE.",
                metadata={"route": "strategy"},
            )
            packet = add_conflict(packet, conflict)
            conflict_ids.append(conflict["id"])

        # Source image route.
        if image is not None:
            packet, sig, proj, conf = _read_signal(
                packet,
                TECH_IMAGE,
                SPACE_IMAGE,
                image,
                "Alpha_source_image",
                "OutcomePrevious",
                "alpha_source_image_route",
                "ImageOutcomeReader",
                route_position="source",
            )
            source_signal_ids.append(sig["id"])
            conflict_ids.extend(conf)

        # Source latent route.
        if latent_source is not None:
            packet, sig, proj, conf = _read_signal(
                packet,
                TECH_LATENT,
                SPACE_LATENT,
                latent_source,
                "Alpha_source_latent",
                "StrategyCarrier",
                "alpha_source_latent_route",
                "LatentEventReader",
                route_position="source",
            )
            source_signal_ids.append(sig["id"])
            conflict_ids.extend(conf)

        # Conditioning route.
        conditioning_signal_id = None
        if conditioning is not None:
            packet, sig, proj, conf = _read_signal(
                packet,
                TECH_CONDITIONING,
                SPACE_CONDITIONING,
                conditioning,
                "Alpha_conditioning",
                "StrategyCarrier",
                "alpha_conditioning_route",
                "ConditioningStrategyReader",
            )
            conditioning_signal_id = sig["id"]
            conflict_ids.extend(conf)

        # Noise route.
        noise_signal_id = None
        if noise is not None:
            raw_noise = noise.get("samples") if isinstance(noise, dict) and "samples" in noise else noise
            packet, sig, proj, conf = _read_signal(
                packet,
                TECH_NOISE,
                SPACE_NOISE,
                raw_noise,
                "Alpha_noise",
                "StrategyCandidate",
                "alpha_noise_route",
                "NoisePossibilityReader",
                metadata={"seed": seed, "noise_mode": "ALPHA_READ_EXISTING"},
            )
            noise_signal_id = sig["id"]
            conflict_ids.extend(conf)

        # Build simple strategy relations.
        for sid in strategy_signal_ids:
            if conditioning_signal_id:
                rel = make_event_relation(
                    relation_type=REL_GUIDES,
                    source_signal_ids=[sid],
                    target_signal_ids=[conditioning_signal_id],
                    source_projection_ids=get_projection_ids_for_signal(packet, sid),
                    target_projection_ids=get_projection_ids_for_signal(packet, conditioning_signal_id),
                    operator_name="EventCoreNodeAlpha",
                    formula_meaning="strategy guides conditioning route",
                    local_strategy_id="S0_core_alpha.strategy_conditioning",
                    equality_status=EQ_UNKNOWN,
                    metadata={"alpha_auto_relation": True},
                )
                packet = add_relation(packet, rel)
                relation_ids.append(rel["id"])

            for source_id in source_signal_ids:
                rel = make_event_relation(
                    relation_type=REL_GUIDES,
                    source_signal_ids=[sid],
                    target_signal_ids=[source_id],
                    source_projection_ids=get_projection_ids_for_signal(packet, sid),
                    target_projection_ids=get_projection_ids_for_signal(packet, source_id),
                    operator_name="EventCoreNodeAlpha",
                    formula_meaning="strategy guides source route interpretation",
                    local_strategy_id="S0_core_alpha.strategy_source",
                    equality_status=EQ_UNKNOWN,
                    metadata={"alpha_auto_relation": True},
                )
                packet = add_relation(packet, rel)
                relation_ids.append(rel["id"])

        # Sampler boundary route.
        sampler_after_signal_id = None
        if read_sampler_boundary:
            if latent_before_sampler is not None and latent_after_sampler is not None:
                packet, before_sig, before_proj, conf1 = _read_signal(
                    packet,
                    TECH_LATENT,
                    SPACE_LATENT,
                    latent_before_sampler,
                    "Alpha_sampler_before",
                    "OutcomePrevious",
                    "alpha_sampler_before_route",
                    "LatentEventReader",
                    route_position="before_sampler",
                )
                packet, after_sig, after_proj, conf2 = _read_signal(
                    packet,
                    TECH_LATENT,
                    SPACE_LATENT,
                    latent_after_sampler,
                    "Alpha_sampler_after",
                    "OutcomeNext",
                    "alpha_sampler_after_route",
                    "LatentEventReader",
                    route_position="after_sampler",
                )
                conflict_ids.extend(conf1 + conf2)
                sampler_after_signal_id = after_sig["id"]

                before_samples = extract_latent_samples(latent_before_sampler)
                delta, error = compute_tensor_delta(latent_before_sampler, latent_after_sampler)
                delta_id = None
                delta_proj_id = None
                source_ids = [before_sig["id"]]
                source_proj_ids = [before_proj["id"]]

                if delta is not None:
                    packet, delta_sig, delta_proj, conf3 = _read_signal(
                        packet,
                        TECH_DELTA,
                        SPACE_DELTA,
                        delta,
                        "Alpha_sampler_delta",
                        "ObservedBehaviorCurrent",
                        "alpha_sampler_delta_route",
                        "DeltaReader",
                        metadata={
                            "before_signal_id": before_sig["id"],
                            "after_signal_id": after_sig["id"],
                            "before_ref": before_samples,
                            "sampler_name": sampler_name,
                            "scheduler": scheduler,
                            "steps": steps,
                            "cfg": cfg,
                            "denoise": denoise,
                            "seed": seed,
                        },
                    )
                    conflict_ids.extend(conf3)
                    delta_id = delta_sig["id"]
                    delta_proj_id = delta_proj["id"]
                    source_ids.append(delta_id)
                    source_proj_ids.append(delta_proj_id)
                    delta_norm = delta_proj.get("numeric_summary", {}).get("delta_norm")
                    relative_delta = delta_proj.get("numeric_summary", {}).get("relative_delta")
                else:
                    conflict = make_conflict(
                        CONFLICT_ALPHA_PARTIAL_INPUT,
                        severity=SEV_LOW,
                        involved_signal_ids=[before_sig["id"], after_sig["id"]],
                        stage_position="EventCoreNodeAlpha",
                        suspected_cause="Alpha sampler boundary could not compute delta.",
                        observed_symptom=str(error),
                        suggested_response="Check sampler before/after latent shapes.",
                        metadata={"route": "sampler_boundary"},
                    )
                    packet = add_conflict(packet, conflict)
                    conflict_ids.append(conflict["id"])
                    delta_norm = None
                    relative_delta = None

                rel = make_event_relation(
                    relation_type=REL_TRANSFORMS_INTO,
                    source_signal_ids=source_ids,
                    target_signal_ids=[after_sig["id"]],
                    source_projection_ids=source_proj_ids,
                    target_projection_ids=[after_proj["id"]],
                    operator_name="EventCoreNodeAlpha",
                    formula_meaning="alpha sampler boundary: latent before plus delta becomes latent after",
                    local_strategy_id="S0_core_alpha.sampler_boundary",
                    equality_status=EQ_UNKNOWN,
                    metadata={
                        "alpha_auto_relation": True,
                        "boundary_type": "sampler_summary",
                        "sampler_name": sampler_name,
                        "scheduler": scheduler,
                        "steps": steps,
                        "cfg": cfg,
                        "denoise": denoise,
                        "seed": seed,
                        "delta_signal_id": delta_id,
                        "delta_norm": delta_norm,
                        "relative_delta": relative_delta,
                    },
                )
                packet = add_relation(packet, rel)
                relation_ids.append(rel["id"])
                sampler_relation_id = rel["id"]

                if noise_signal_id:
                    rel_noise = make_event_relation(
                        relation_type=REL_CONSTRAINS,
                        source_signal_ids=[noise_signal_id],
                        target_signal_ids=[after_sig["id"]],
                        source_projection_ids=get_projection_ids_for_signal(packet, noise_signal_id),
                        target_projection_ids=[after_proj["id"]],
                        operator_name="EventCoreNodeAlpha",
                        formula_meaning="noise possibility field constrains sampler output route",
                        local_strategy_id="S0_core_alpha.noise_sampler",
                        equality_status=EQ_UNKNOWN,
                        metadata={"alpha_auto_relation": True},
                    )
                    packet = add_relation(packet, rel_noise)
                    relation_ids.append(rel_noise["id"])

            else:
                conflict = make_conflict(
                    CONFLICT_ALPHA_ROUTE_MISSING,
                    severity=SEV_LOW,
                    stage_position="EventCoreNodeAlpha",
                    suspected_cause="read_sampler_boundary enabled but one or both sampler latents are missing.",
                    observed_symptom="Sampler boundary route missing.",
                    suggested_response="Connect latent_before_sampler and latent_after_sampler or disable read_sampler_boundary.",
                    metadata={"route": "sampler_boundary"},
                )
                packet = add_conflict(packet, conflict)
                conflict_ids.append(conflict["id"])

        # Decode route.
        if read_decode_boundary:
            if latent_after_sampler is not None and decoded_image is not None:
                if sampler_after_signal_id is None:
                    packet, lat_sig, lat_proj, conf = _read_signal(
                        packet,
                        TECH_LATENT,
                        SPACE_LATENT,
                        latent_after_sampler,
                        "Alpha_decode_latent",
                        "OutcomePrevious",
                        "alpha_decode_latent_route",
                        "LatentEventReader",
                        route_position="after_sampler",
                    )
                    conflict_ids.extend(conf)
                    sampler_after_signal_id = lat_sig["id"]
                else:
                    lat_sig = packet["signals"][sampler_after_signal_id]
                    lat_proj_ids = get_projection_ids_for_signal(packet, sampler_after_signal_id)

                packet, img_sig, img_proj, conf = _read_signal(
                    packet,
                    TECH_IMAGE,
                    SPACE_IMAGE,
                    decoded_image,
                    "Alpha_decoded_image",
                    "OutcomeNext",
                    "alpha_decoded_image_route",
                    "ImageOutcomeReader",
                    route_position="after_decode",
                )
                conflict_ids.extend(conf)

                rel = make_event_relation(
                    relation_type=REL_EXPANDS_TO,
                    source_signal_ids=[sampler_after_signal_id],
                    target_signal_ids=[img_sig["id"]],
                    source_projection_ids=get_projection_ids_for_signal(packet, sampler_after_signal_id),
                    target_projection_ids=[img_proj["id"]],
                    operator_name="EventCoreNodeAlpha",
                    formula_meaning="latent after sampler expands to decoded image outcome",
                    local_strategy_id="S0_core_alpha.decode",
                    equality_status=EQ_UNKNOWN,
                    metadata={"alpha_auto_relation": True, "boundary_type": "decode"},
                )
                packet = add_relation(packet, rel)
                relation_ids.append(rel["id"])
                decode_relation_id = rel["id"]
            else:
                conflict = make_conflict(
                    CONFLICT_ALPHA_ROUTE_MISSING,
                    severity=SEV_LOW,
                    stage_position="EventCoreNodeAlpha",
                    suspected_cause="read_decode_boundary enabled but latent_after_sampler or decoded_image is missing.",
                    observed_symptom="Decode boundary route missing.",
                    suggested_response="Connect latent_after_sampler and decoded_image or disable read_decode_boundary.",
                    metadata={"route": "decode_boundary"},
                )
                packet = add_conflict(packet, conflict)
                conflict_ids.append(conflict["id"])

        if build_sstate:
            active_rel_ids = relation_ids if relation_ids else None
            packet, sstate = build_sstate_from_packet(packet, position="S0_core_alpha", active_relation_ids=active_rel_ids)
            packet = record_stage(
                packet,
                stage_name="EventCoreNodeAlpha",
                action="BUILD_ALPHA_SSTATE",
                observed_behavior="Alpha body SState built from internal auto-relations",
                input_signal_ids=sstate.get("active_signal_ids", []),
                relation_ids=sstate.get("active_relation_ids", []),
                sstate_ids=[sstate["id"]],
                conflict_ids=conflict_ids,
                formula_note="EventCoreNodeAlpha is report-only body: routes -> relations -> S0_core_alpha",
                metadata={
                    "adapter_mode": adapter_mode,
                    "sampler_relation_id": sampler_relation_id,
                    "decode_relation_id": decode_relation_id,
                },
            )

        # Adapter layer: label and diagnose after generic routes are built.
        if adapter_mode == "WAN":
            packet, wan_result = apply_wan_adapter(
                packet,
                mode="BASIC",
                hints={
                    "workflow_hint": "EventCoreNodeAlpha",
                    "has_high_sampler": False,
                    "has_low_sampler": False,
                    "has_temporal_module": False,
                    "has_lora_stack": False,
                    "has_accvid": False,
                    "has_lightx2v": False,
                },
            )
            relation_ids.extend(wan_result.get("created_relation_ids", []))
            conflict_ids.extend(wan_result.get("conflict_ids", []))

        packet = record_stage(
            packet,
            stage_name="EventCoreNodeAlpha",
            action="BUILD_ALPHA_REPORT",
            observed_behavior="Alpha report-only body assembled available routes",
            conflict_ids=conflict_ids,
            metadata={
                "strategy_signals": strategy_signal_ids,
                "source_signals": source_signal_ids,
                "relation_ids": relation_ids,
                "adapter_mode": adapter_mode,
            },
        )

        report = build_markdown_report(packet)
        short_summary = (
            f"EventCoreNodeAlpha: signals={len(packet.get('signals', {}))} "
            f"relations={len(packet.get('relations', {}))} "
            f"sstates={len(packet.get('sstates', {}))} "
            f"conflicts={len(packet.get('conflicts', {}))} "
            f"adapter={adapter_mode}"
        )
        return (packet, report, short_summary)


class EventUNIReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"event_packet": ("EVENT_PACKET",)}}

    RETURN_TYPES = ("STRING", "EVENT_PACKET")
    RETURN_NAMES = ("markdown_report", "event_packet")
    FUNCTION = "run"
    CATEGORY = "Singularity/Core"

    def run(self, event_packet):
        packet = ensure_packet(event_packet)
        packet = record_stage(
            packet,
            stage_name="EventUNIReport",
            action="BUILD_REPORT",
            observed_behavior="Markdown report built from EventPacket",
        )
        return (build_markdown_report(packet), packet)


def _event_core_cleanup_memory(label="cleanup"):
    info = {
        "label": label,
        "gc": False,
        "torch_cuda_empty_cache": False,
        "torch_cuda_ipc_collect": False,
        "error": None,

        "memory_before": _event_core_memory_snapshot(f"{label}_before"),
    }
    try:
        import gc
        gc.collect()
        info["gc"] = True
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                info["torch_cuda_empty_cache"] = True
                try:
                    torch.cuda.ipc_collect()
                    info["torch_cuda_ipc_collect"] = True
                except Exception:
                    pass
        except Exception as e:
            info["error"] = f"torch cleanup skipped: {e}"
    except Exception as e:
        info["error"] = str(e)
    info["memory_after"] = _event_core_memory_snapshot(f"{label}_after")
    return info


def _event_core_memory_snapshot(label="snapshot"):
    info = {
        "label": str(label or "snapshot"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "perf_counter": float(time.perf_counter()),
    }
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        info["process_rss_mb"] = round(float(mem.rss) / (1024 * 1024), 3)
        info["process_vms_mb"] = round(float(mem.vms) / (1024 * 1024), 3)
    except Exception as e:
        info["process_memory_error"] = str(e)
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        info["torch_cuda_available"] = cuda_available
        if cuda_available:
            info["cuda_device"] = int(torch.cuda.current_device())
            info["cuda_allocated_mb"] = round(float(torch.cuda.memory_allocated()) / (1024 * 1024), 3)
            info["cuda_reserved_mb"] = round(float(torch.cuda.memory_reserved()) / (1024 * 1024), 3)
            info["cuda_max_allocated_mb"] = round(float(torch.cuda.max_memory_allocated()) / (1024 * 1024), 3)
            info["cuda_max_reserved_mb"] = round(float(torch.cuda.max_memory_reserved()) / (1024 * 1024), 3)
    except Exception as e:
        info["cuda_memory_error"] = str(e)
    return info


def _event_core_numeric_delta(before, after):
    try:
        if before is None or after is None:
            return None
        return round(float(after) - float(before), 6)
    except Exception:
        return None


def _event_core_memory_delta(before_snapshot, after_snapshot):
    before = before_snapshot if isinstance(before_snapshot, dict) else {}
    after = after_snapshot if isinstance(after_snapshot, dict) else {}
    fields = (
        "process_rss_mb",
        "process_vms_mb",
        "cuda_allocated_mb",
        "cuda_reserved_mb",
        "cuda_max_allocated_mb",
        "cuda_max_reserved_mb",
    )
    delta = {}
    for key in fields:
        value = _event_core_numeric_delta(before.get(key), after.get(key))
        if value is not None:
            delta[key] = value
    return delta


def _event_core_state_descriptor(value, name="state"):
    desc = {
        "name": str(name or "state"),
        "type": type(value).__name__,
    }
    tensor = value
    if isinstance(value, dict) and "samples" in value:
        tensor = value.get("samples")
        desc["carrier_type"] = "latent_dict"
    try:
        if hasattr(tensor, "shape"):
            desc["shape"] = list(tensor.shape)
        if hasattr(tensor, "dtype"):
            desc["dtype"] = str(tensor.dtype)
        if hasattr(tensor, "device"):
            desc["device"] = str(tensor.device)
    except Exception as e:
        desc["descriptor_error"] = str(e)
    return desc


from .core.orchestrator import WanEventWorkflowCore
class SingularityCascadeSimple(WanEventWorkflowCore):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "primary_model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "source_image_file": (_event_core_list_input_images(), {"image_upload": True}),

                "positive_prompt": ("STRING", {"default": "", "multiline": True, "height": 180, "dynamicPrompts": False}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True, "height": 180, "dynamicPrompts": False}),
                "temporal_texture_lock": ("BOOLEAN", {"default": True}),

                                "cascade_count": ("INT", {"default": 1, "min": 1, "max": 5}),
                "pause_after_cascade_1": ("BOOLEAN", {"default": False}),
                "pause_after_cascade_2": ("BOOLEAN", {"default": False}),
                "pause_after_cascade_3": ("BOOLEAN", {"default": False}),
                "pause_after_cascade_4": ("BOOLEAN", {"default": False}),
                "frames_per_cascade": ("INT", {"default": 49, "min": 1, "max": 4096}),
                "width": ("INT", {"default": 416, "min": 16, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 608, "min": 16, "max": 8192, "step": 8}),
                "fps": ("INT", {"default": 16, "min": 1, "max": 240}),
                "seed": ("INT", {"default": 359, "min": 0, "max": 0xffffffffffffffff}),

                "sampler_name": ("STRING", {"default": "euler"}),
                "scheduler": ("STRING", {"default": "simple"}),
                "global_steps": ("INT", {"default": 4, "min": 0, "max": 10000}),
                "primary_cfg": ("FLOAT", {"default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "secondary_cfg": ("FLOAT", {"default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "primary_start_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "primary_end_step": ("INT", {"default": 1, "min": 0, "max": 10000}),
                "secondary_start_step": ("INT", {"default": 1, "min": 0, "max": 10000}),
                "secondary_end_step": ("INT", {"default": 4, "min": 0, "max": 10000}),

                # Keep combo UX, but include lowercase legacy tokens so stale workflows do not hard-fail before normalization.
                "math_control_mode": (["OBSERVE_ONLY", "LATENT_DELTA_SCALE", "DEEP_STEP_DELTA_CONTROL", "observe_only", "latent_delta_scale", "deep_step_delta_control"], {"default": "LATENT_DELTA_SCALE"}),
                "high_delta_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.0001, "round": 0.0001}),
                "low_delta_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.0001, "round": 0.0001}),

                "decode_tile_size": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                # Allow wider transport range; runtime normalization will clamp to safe decode constraints.
                "decode_overlap": ("INT", {"default": 64, "min": 0, "max": 65535, "step": 8}),
                "decode_temporal_size": ("INT", {"default": 32, "min": 1, "max": 4096}),
                "decode_temporal_overlap": ("INT", {"default": 12, "min": 0, "max": 65535}),

                "image_upscale_method": (["nearest-exact", "nearest", "bilinear", "area", "bicubic", "lanczos"], {"default": "nearest-exact"}),
                "image_crop": (["disabled", "center"], {"default": "disabled"}),

                "save_video": ("BOOLEAN", {"default": True}),
                "video_format": (["video/h264-mp4", "video/h265-mp4", "image/webp", "image/gif"], {"default": "video/h264-mp4"}),
                "save_report": ("BOOLEAN", {"default": True}),
                "save_prefix": ("STRING", {"default": "Singularity"}),
                "sampler_trace_mode": (["OFF", "SHADOW_STEP_TRACE"], {"default": "OFF"}),
                "sampler_trace_max_steps": ("INT", {"default": 64, "min": 1, "max": 65535}),

                # Tail frame selection UI layer control (always manual green outline primary)
                "use_formula_recommendation": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "secondary_model": ("MODEL",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                # Synced from Tail 3 bar clicks (green outline). Explicit param (no **kwargs).
                # Placed in optional so adding it doesn't break existing workflows' widgets_values (old saves won't have the INT value).
                # Default 0. The bar (not this widget) is the primary UI for manual choice.
                "selected_tail_index": ("INT", {"default": 0, "min": 0, "max": 2, "step": 1}),
            },
            "hidden": {
                "id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("status", "saved_video_path", "saved_report_path", "report", "tail_frame_0", "tail_frame_1", "tail_frame_2")
    FUNCTION = "run"
    CATEGORY = "Singularity/Singularity"
    OUTPUT_NODE = True

    @staticmethod
    def _append_input_adjustment(adjustments, field, original, normalized, reason):
        if original == normalized:
            return
        adjustments.append({
            "field": str(field),
            "original": _event_json_safe(original),
            "normalized": _event_json_safe(normalized),
            "reason": str(reason),
        })

    def _get_ksampler_allowed_values(self):
        samplers = []
        schedulers = []
        try:
            cls = self._get_node_class("KSamplerAdvanced")
            input_types = cls.INPUT_TYPES() if hasattr(cls, "INPUT_TYPES") else {}
            required = input_types.get("required", {}) if isinstance(input_types, dict) else {}

            sampler_def = required.get("sampler_name")
            scheduler_def = required.get("scheduler")

            if isinstance(sampler_def, (list, tuple)) and sampler_def:
                options = sampler_def[0]
                if isinstance(options, (list, tuple)):
                    samplers = [str(v) for v in options if str(v).strip()]
            if isinstance(scheduler_def, (list, tuple)) and scheduler_def:
                options = scheduler_def[0]
                if isinstance(options, (list, tuple)):
                    schedulers = [str(v) for v in options if str(v).strip()]
        except Exception:
            pass

        if not samplers:
            samplers = ["euler"]
        if not schedulers:
            schedulers = ["simple"]
        return samplers, schedulers

    @staticmethod
    def _sanitize_save_prefix_text(value):
        text = str(value or "").strip()
        if not text:
            return ""
        text = re.sub(r'[\x00-\x1f]', "_", text)
        text = re.sub(r'[<>:"/\\|?*]+', "_", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = text.rstrip(" .")
        if len(text) > 120:
            text = text[:120].rstrip(" .")
        return text

    def _event_runtime_aliases(self):
        aliases = []
        try:
            mappings = globals().get("NODE_CLASS_MAPPINGS", {}) or {}
            for key, value in mappings.items():
                if value is self.__class__:
                    aliases.append(str(key))
        except Exception:
            return []
        return sorted(set(aliases))

    def _normalize_clean_inputs(
        self,
        *,
        secondary_model,
        cascade_count,
        pause_after_cascade_1=False,
        pause_after_cascade_2=False,
        pause_after_cascade_3=False,
        pause_after_cascade_4=False,
        frames_per_cascade,
        width,
        height,
        fps,
        seed,
        sampler_name,
        scheduler,
        global_steps,
        primary_cfg,
        secondary_cfg,
        primary_start_step,
        primary_end_step,
        secondary_start_step,
        secondary_end_step,
        math_control_mode,
        high_delta_strength,
        low_delta_strength,
        decode_tile_size,
        decode_overlap,
        decode_temporal_size,
        decode_temporal_overlap,
        image_upscale_method,
        image_crop,
        video_format,
        save_prefix,
        sampler_trace_mode,
        sampler_trace_max_steps,
    ):
        adjustments = []
        allowed_samplers, allowed_schedulers = self._get_ksampler_allowed_values()

        def clamp_int(field, value, default, min_value, max_value):
            original = value
            try:
                out = int(value)
            except Exception:
                out = int(default)
                self._append_input_adjustment(adjustments, field, original, out, "invalid_int_fallback")
            clamped = max(int(min_value), min(int(max_value), int(out)))
            if clamped != out:
                self._append_input_adjustment(adjustments, field, out, clamped, f"clamped_to_{min_value}_{max_value}")
            return clamped

        def clamp_float(field, value, default, min_value, max_value, digits=None):
            original = value
            try:
                out = float(value)
            except Exception:
                out = float(default)
                self._append_input_adjustment(adjustments, field, original, out, "invalid_float_fallback")
            clamped = max(float(min_value), min(float(max_value), float(out)))
            if digits is not None:
                rounded = round(clamped, int(digits))
                if rounded != clamped:
                    self._append_input_adjustment(adjustments, field, clamped, rounded, f"rounded_{digits}_digits")
                clamped = rounded
            if clamped != out:
                self._append_input_adjustment(adjustments, field, out, clamped, f"clamped_to_{min_value}_{max_value}")
            return clamped

        def clamp_enum(field, value, allowed_values, default_value, casefold=False, reason="invalid_enum_fallback"):
            original = value
            text = str(value if value is not None else default_value)
            if casefold:
                text_cmp = text.upper()
                allowed_cmp = {str(v).upper(): str(v) for v in allowed_values}
                if text_cmp in allowed_cmp:
                    out = allowed_cmp[text_cmp]
                else:
                    out = str(default_value)
                    self._append_input_adjustment(adjustments, field, original, out, reason)
                return out
            if text in allowed_values:
                return text
            out = str(default_value)
            self._append_input_adjustment(adjustments, field, original, out, reason)
            return out

        def coerce_bool(field, value):
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "y", "on")
            return bool(value)

        def align_to_step(field, value, step, min_value):
            if step <= 1:
                return int(value)
            aligned = int(value) - (int(value) % int(step))
            if aligned < int(min_value):
                aligned = int(min_value)
            if aligned != int(value):
                self._append_input_adjustment(adjustments, field, value, aligned, f"aligned_to_step_{step}")
            return int(aligned)

        dual_branch = True  # force DUAL_HIGH_LOW support for high/low acceleration (0-1 / 1-4 etc.) even without secondary_model connected; execution falls back to primary_model for low phase.
        # Per full study of entire _knowledge_base + VERY LAST r58_InputIntegrityHardening node (Gemini's final):
        # r49/r55/r58 _normalize_clean_inputs did "secondary_start_n = primary_end_n" + "forced_to_primary_end_for_dual_branch" adjustment + clamp min=primary_end_n,
        # and first-body low only under "if DUAL_HIGH_LOW and secondary_model is not None".
        # That caused exactly the "low never starts after high in img-to-vid start + pause_after_cascade_1 + cascade=1"
        # and "the step route is treated like the formula is the sampler".
        # We deliberately do NOT link steps here (user steps are sacred, formula only on values/raw_delta). See execution.py first-body direct bypass.

        cascade_count_n = clamp_int("cascade_count", cascade_count, 1, 1, 5)
        pause_after_flags_n = {
            1: coerce_bool("pause_after_cascade_1", pause_after_cascade_1),
            2: coerce_bool("pause_after_cascade_2", pause_after_cascade_2),
            3: coerce_bool("pause_after_cascade_3", pause_after_cascade_3),
            4: coerce_bool("pause_after_cascade_4", pause_after_cascade_4),
        }
        pause_after_segments_n = [
            int(segment_index)
            for segment_index in range(1, int(cascade_count_n))
            if pause_after_flags_n.get(segment_index, False)
        ]
        ignored_pause_after_segments_n = [
            int(segment_index)
            for segment_index, enabled in pause_after_flags_n.items()
            if enabled and segment_index >= int(cascade_count_n)
        ]
        frames_per_cascade_n = clamp_int("frames_per_cascade", frames_per_cascade, 49, 1, 4096)
        if cascade_count_n > 1 and frames_per_cascade_n < 2:
            self._append_input_adjustment(
                adjustments,
                "frames_per_cascade",
                frames_per_cascade_n,
                2,
                "cascade_requires_at_least_two_frames",
            )
            frames_per_cascade_n = 2

        width_n = align_to_step(
            "width",
            clamp_int("width", width, 416, 16, 8192),
            step=8,
            min_value=16,
        )
        height_n = align_to_step(
            "height",
            clamp_int("height", height, 608, 16, 8192),
            step=8,
            min_value=16,
        )
        fps_n = clamp_int("fps", fps, 16, 1, 240)
        seed_n = clamp_int("seed", seed, 359, 0, 0xFFFFFFFFFFFFFFFF)

        sampler_default = allowed_samplers[0] if allowed_samplers else "euler"
        scheduler_default = allowed_schedulers[0] if allowed_schedulers else "simple"
        sampler_name_n = clamp_enum(
            "sampler_name",
            str(sampler_name or "").strip() or sampler_default,
            allowed_samplers,
            sampler_default,
            reason="unsupported_sampler_fallback",
        )
        scheduler_n = clamp_enum(
            "scheduler",
            str(scheduler or "").strip() or scheduler_default,
            allowed_schedulers,
            scheduler_default,
            reason="unsupported_scheduler_fallback",
        )

        global_steps_n = clamp_int("global_steps", global_steps, 4, 1, 10000)
        if dual_branch and global_steps_n < 2:
            self._append_input_adjustment(adjustments, "global_steps", global_steps_n, 2, "dual_branch_min_steps")
            global_steps_n = 2

        # Raw philosophy: respect the user's explicit step ranges for high/low acceleration.
        # Do NOT let the "formula" concept (high->low handoff) mutate the step numbers the user typed.
        # The formula only affects the *content* (raw delta added to low cfg), not the *ranges*.
        # The old Gemini coupling (secondary_start = primary_end + adjustment) was exactly the source of the
        # broken img2vid-start + pause confusion remembered from earlier tests.
        primary_start_n = clamp_int("primary_start_step", primary_start_step, 0, 0, max(0, global_steps_n - 1))
        primary_end_n = clamp_int("primary_end_step", primary_end_step, max(1, primary_start_n + 1), primary_start_n + 1, global_steps_n)
        secondary_start_n = clamp_int("secondary_start_step", secondary_start_step, 0, 0, max(0, global_steps_n - 1))
        secondary_end_n = clamp_int("secondary_end_step", secondary_end_step, max(1, secondary_start_n + 1), secondary_start_n + 1, global_steps_n)

        primary_cfg_n = clamp_float("primary_cfg", primary_cfg, 1.0, -1000.0, 1000.0, digits=4)
        secondary_cfg_n = clamp_float("secondary_cfg", secondary_cfg, 1.0, -1000.0, 1000.0, digits=4)

        math_control_mode_n = clamp_enum(
            "math_control_mode",
            math_control_mode,
            {"OBSERVE_ONLY", "LATENT_DELTA_SCALE", "DEEP_STEP_DELTA_CONTROL"},
            "OBSERVE_ONLY",
            casefold=True,
        )
        high_delta_strength_n = clamp_float("high_delta_strength", high_delta_strength, 1.0, 0.0, 2.0, digits=4)
        low_delta_strength_n = clamp_float("low_delta_strength", low_delta_strength, 1.0, 0.0, 2.0, digits=4)

        decode_tile_size_n = align_to_step(
            "decode_tile_size",
            clamp_int("decode_tile_size", decode_tile_size, 512, 64, 8192),
            step=8,
            min_value=64,
        )
        decode_overlap_n = align_to_step(
            "decode_overlap",
            clamp_int("decode_overlap", decode_overlap, 64, 0, 8192),
            step=8,
            min_value=0,
        )
        max_overlap = max(0, decode_tile_size_n - 8)
        if decode_overlap_n > max_overlap:
            self._append_input_adjustment(adjustments, "decode_overlap", decode_overlap_n, max_overlap, "decode_overlap_cannot_exceed_tile_size_minus_step")
            decode_overlap_n = max_overlap

        decode_temporal_size_n = clamp_int("decode_temporal_size", decode_temporal_size, 32, 1, 4096)
        decode_temporal_overlap_n = clamp_int("decode_temporal_overlap", decode_temporal_overlap, 12, 0, 4096)
        max_temporal_overlap = max(0, decode_temporal_size_n - 1)
        if decode_temporal_overlap_n > max_temporal_overlap:
            self._append_input_adjustment(
                adjustments,
                "decode_temporal_overlap",
                decode_temporal_overlap_n,
                max_temporal_overlap,
                "decode_temporal_overlap_cannot_exceed_temporal_size_minus_one",
            )
            decode_temporal_overlap_n = max_temporal_overlap

        image_upscale_method_n = clamp_enum(
            "image_upscale_method",
            image_upscale_method,
            {"nearest-exact", "nearest", "bilinear", "area", "bicubic", "lanczos"},
            "nearest-exact",
        )
        image_crop_n = clamp_enum("image_crop", image_crop, {"disabled", "center"}, "disabled")
        video_format_n = clamp_enum(
            "video_format",
            video_format,
            {"video/h264-mp4", "video/h265-mp4", "image/webp", "image/gif"},
            "video/h264-mp4",
        )
        sampler_trace_mode_n = clamp_enum(
            "sampler_trace_mode",
            sampler_trace_mode,
            {"OFF", "SHADOW_STEP_TRACE"},
            "OFF",
            casefold=True,
        )
        sampler_trace_max_steps_n = clamp_int("sampler_trace_max_steps", sampler_trace_max_steps, 64, 1, 65535)

        save_prefix_n = self._sanitize_save_prefix_text(save_prefix)
        if save_prefix_n != str(save_prefix or "").strip():
            self._append_input_adjustment(adjustments, "save_prefix", save_prefix, save_prefix_n, "sanitized_for_windows_filename")
        if not save_prefix_n:
            save_prefix_n = "Singularity"
            self._append_input_adjustment(adjustments, "save_prefix", save_prefix, save_prefix_n, "empty_string_fallback")

        normalized = {
            "cascade_count": cascade_count_n,
            "cascade_pause_policy": "LEGACY_FLAGS",
            "pause_after_cascade_1": pause_after_flags_n.get(1, False),
            "pause_after_cascade_2": pause_after_flags_n.get(2, False),
            "pause_after_cascade_3": pause_after_flags_n.get(3, False),
            "pause_after_cascade_4": pause_after_flags_n.get(4, False),
            "pause_after_segments": pause_after_segments_n,
            "ignored_pause_after_segments": ignored_pause_after_segments_n,
            "frames_per_cascade": frames_per_cascade_n,
            "width": width_n,
            "height": height_n,
            "fps": fps_n,
            "seed": seed_n,
            "sampler_name": sampler_name_n,
            "scheduler": scheduler_n,
            "global_steps": global_steps_n,
            "primary_cfg": primary_cfg_n,
            "secondary_cfg": secondary_cfg_n,
            "primary_start_step": primary_start_n,
            "primary_end_step": primary_end_n,
            "secondary_start_step": secondary_start_n,
            "secondary_end_step": secondary_end_n,
            "math_control_mode": math_control_mode_n,
            "high_delta_strength": high_delta_strength_n,
            "low_delta_strength": low_delta_strength_n,
            "decode_tile_size": decode_tile_size_n,
            "decode_overlap": decode_overlap_n,
            "decode_temporal_size": decode_temporal_size_n,
            "decode_temporal_overlap": decode_temporal_overlap_n,
            "image_upscale_method": image_upscale_method_n,
            "image_crop": image_crop_n,
            "video_format": video_format_n,
            "save_prefix": save_prefix_n,
            "sampler_trace_mode": sampler_trace_mode_n,
            "sampler_trace_max_steps": sampler_trace_max_steps_n,
            "branch_mode": "DUAL_HIGH_LOW" if dual_branch else "SINGLE",
            "cascade_mode": "SOLO_1" if cascade_count_n <= 1 else f"CASCADE_{cascade_count_n}",
        }
        signature_source = {
            "normalized": normalized,
            "adjustment_count": len(adjustments),
            "adjustment_reasons": sorted(
                str(adj.get("reason", "unknown") or "unknown")
                for adj in adjustments
                if isinstance(adj, dict)
            ),
        }
        encoded = json.dumps(_event_json_safe(signature_source), sort_keys=True, ensure_ascii=True)
        normalized_signature = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]
        return {
            "normalized": normalized,
            "adjustments": adjustments,
            "normalized_signature": normalized_signature,
            "normalized_signature_source": signature_source,
        }

    def run(
        self,
        primary_model,
        clip,
        vae,
        source_image_file,
        positive_prompt,
        negative_prompt,
        temporal_texture_lock,
        cascade_count,
        frames_per_cascade,
        width,
        height,
        fps,
        seed,
        sampler_name,
        scheduler,
        global_steps,
        primary_cfg,
        secondary_cfg,
        primary_start_step,
        primary_end_step,
        secondary_start_step,
        secondary_end_step,
        math_control_mode,
        high_delta_strength,
        low_delta_strength,
        decode_tile_size,
        decode_overlap,
        decode_temporal_size,
        decode_temporal_overlap,
        image_upscale_method,
        image_crop,
        save_video,
        video_format,
        save_report,
        save_prefix,
        sampler_trace_mode,
        sampler_trace_max_steps,
        pause_after_cascade_1=False,
        pause_after_cascade_2=False,
        pause_after_cascade_3=False,
        pause_after_cascade_4=False,
        secondary_model=None,
        image=None,
        mask=None,
        use_formula_recommendation=False,
        selected_tail_index=0,
        id=None,
    ):
        self._event_strategy_coupling = {"low_strength_multiplier": 1.0}
        self._singularity_node_id = id
        normalization = self._normalize_clean_inputs(
            secondary_model=secondary_model,
            cascade_count=cascade_count,
            pause_after_cascade_1=pause_after_cascade_1,
            pause_after_cascade_2=pause_after_cascade_2,
            pause_after_cascade_3=pause_after_cascade_3,
            pause_after_cascade_4=pause_after_cascade_4,
            frames_per_cascade=frames_per_cascade,
            width=width,
            height=height,
            fps=fps,
            seed=seed,
            sampler_name=sampler_name,
            scheduler=scheduler,
            global_steps=global_steps,
            primary_cfg=primary_cfg,
            secondary_cfg=secondary_cfg,
            primary_start_step=primary_start_step,
            primary_end_step=primary_end_step,
            secondary_start_step=secondary_start_step,
            secondary_end_step=secondary_end_step,
            math_control_mode=math_control_mode,
            high_delta_strength=high_delta_strength,
            low_delta_strength=low_delta_strength,
            decode_tile_size=decode_tile_size,
            decode_overlap=decode_overlap,
            decode_temporal_size=decode_temporal_size,
            decode_temporal_overlap=decode_temporal_overlap,
            image_upscale_method=image_upscale_method,
            image_crop=image_crop,
            video_format=video_format,
            save_prefix=save_prefix,
            sampler_trace_mode=sampler_trace_mode,
            sampler_trace_max_steps=sampler_trace_max_steps,
        )
        normalized = normalization.get("normalized", {})
        self._event_input_normalization = normalization

        cascade_count = int(normalized.get("cascade_count", 1))
        frames_per_cascade = int(normalized.get("frames_per_cascade", 49))
        width = int(normalized.get("width", 416))
        height = int(normalized.get("height", 608))
        fps = int(normalized.get("fps", 16))
        seed = int(normalized.get("seed", 359))
        sampler_name = str(normalized.get("sampler_name", "euler"))
        scheduler = str(normalized.get("scheduler", "simple"))
        global_steps = int(normalized.get("global_steps", 4))
        primary_cfg = float(normalized.get("primary_cfg", 1.0))
        secondary_cfg = float(normalized.get("secondary_cfg", 1.0))
        primary_start_step = int(normalized.get("primary_start_step", 0))
        primary_end_step = int(normalized.get("primary_end_step", 1))
        secondary_start_step = int(normalized.get("secondary_start_step", 1))
        secondary_end_step = int(normalized.get("secondary_end_step", 4))
        math_control_mode = str(normalized.get("math_control_mode", "OBSERVE_ONLY"))
        high_delta_strength = float(normalized.get("high_delta_strength", 1.0))
        low_delta_strength = float(normalized.get("low_delta_strength", 1.0))
        decode_tile_size = int(normalized.get("decode_tile_size", 512))
        decode_overlap = int(normalized.get("decode_overlap", 64))
        decode_temporal_size = int(normalized.get("decode_temporal_size", 32))
        decode_temporal_overlap = int(normalized.get("decode_temporal_overlap", 12))
        image_upscale_method = str(normalized.get("image_upscale_method", "nearest-exact"))
        image_crop = str(normalized.get("image_crop", "disabled"))
        video_format = str(normalized.get("video_format", "video/h264-mp4"))
        save_prefix = str(normalized.get("save_prefix", "Singularity"))
        sampler_trace_mode = str(normalized.get("sampler_trace_mode", "OFF"))
        sampler_trace_max_steps = int(normalized.get("sampler_trace_max_steps", 64))
        branch_mode = str(normalized.get("branch_mode", "SINGLE"))
        cascade_mode = str(normalized.get("cascade_mode", "SOLO_1"))

        self._event_requested_runtime_controls = {
            "cascade_count": int(cascade_count),
            "frames_per_cascade": int(frames_per_cascade or 49),
            "width": int(width),
            "height": int(height),
            "fps": int(fps),
            "seed": int(seed),
            "branch_mode": str(branch_mode),
        }

        # Shift is intentionally disabled in the clean node.
        # If a model needs SD3 shift, apply it outside this node before connecting MODEL.
        if bool(temporal_texture_lock):
            positive_prompt = (str(positive_prompt or "").rstrip() + "\n"
                "Small liquid droplets, sweat beads, reflections, and tiny highlights keep a stable temporal identity across frames; "
                "any visible liquid motion follows one continuous gravity-consistent direction with no sudden reversal or reset.")
            negative_prompt = (str(negative_prompt or "").rstrip() + "\n"
                "upward moving droplets, reversing sweat, liquid moving against gravity, flickering specular noise, "
                "jumping highlights, temporal reset, frame snap, duplicated motion boundary, jittering droplets, crawling texture")
        self._event_math_control_mode = str(math_control_mode or "OBSERVE_ONLY")
        self._event_delta_strengths = {
            "high": float(high_delta_strength),
            "low": float(low_delta_strength),
        }
        self._event_requested_math_controls = {
            "math_control_mode": str(math_control_mode or "OBSERVE_ONLY"),
            "high_delta_strength": float(high_delta_strength),
            "low_delta_strength": float(low_delta_strength),
            "active_math_sampler_path": "semantic_overlay_native_sampler_when_latent_delta_scale",
            "high_low_strategy_carrier_coupling": True,
            "precision_step": 0.0001,
            "precision_round": 0.0001,
            "sampler_trace_mode": str(sampler_trace_mode or "OFF").upper(),
            "sampler_trace_max_steps": int(sampler_trace_max_steps or 64),
        }
        self._event_sampler_trace = {
            "mode": str(sampler_trace_mode or "OFF").upper(),
            "max_steps": int(sampler_trace_max_steps or 64),
        }

        return super().run(
            primary_model=primary_model,
            clip=clip,
            vae=vae,
            source_image_file=source_image_file,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            event_strategy="r44: one external Singularity node with internal Event Core Body, RouteMemory, S-Wire, and universal stage math",
            generation_target="VIDEO",
            terminal_mode=True,
            enable_continuation_outputs=False,
            execution_mode="RUN",
            branch_mode=branch_mode,
            cascade_count=cascade_count,
            cascade_mode=cascade_mode,
            frames_per_cascade=int(frames_per_cascade or 49),
            width=int(width),
            height=int(height),
            frames=int(frames_per_cascade or 49),
            batch_size=1,
            fps=int(fps),
            seed=int(seed),
            sampler_name=sampler_name,
            scheduler=scheduler,
            global_steps=int(global_steps),
            primary_cfg=float(primary_cfg),
            secondary_cfg=float(secondary_cfg),
            primary_start_step=int(primary_start_step),
            primary_end_step=int(primary_end_step),
            secondary_start_step=int(secondary_start_step),
            secondary_end_step=int(secondary_end_step),
            primary_sd3_shift=0.0,
            secondary_sd3_shift=0.0,
            decode_tile_size=int(decode_tile_size),
            decode_overlap=int(decode_overlap),
            decode_temporal_size=int(decode_temporal_size),
            decode_temporal_overlap=int(decode_temporal_overlap),
            image_upscale_method=image_upscale_method,
            image_crop=image_crop,
            stage_delay_seconds=0.0,
            save_video=bool(save_video),
            video_format=video_format,
            force_vhs_video_combine=True,
            save_frames=False,
            save_report=bool(save_report),
            output_target="COMFY_OUTPUT",
            save_output_image=False,
            save_prefix=save_prefix,
            pause_after_cascade_1=pause_after_cascade_1,
            pause_after_cascade_2=pause_after_cascade_2,
            pause_after_cascade_3=pause_after_cascade_3,
            pause_after_cascade_4=pause_after_cascade_4,
            use_formula_recommendation=bool(use_formula_recommendation),
            selected_tail_index=int(selected_tail_index or 0),
            output_folder_mode="DEFAULT",
            output_folder="default",
            custom_output_folder="",
            report_detail="STANDARD",
            secondary_model=secondary_model,
            image=image,
            mask=mask,
        )


NODE_CLASS_MAPPINGS = {
    # Public clean release: one visible ComfyUI node.
    # Development/debug classes remain internal implementation details.
    "Singularity": SingularityCascadeSimple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Singularity": "Singularity",
}



