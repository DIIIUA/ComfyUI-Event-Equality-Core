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
EVENT_HORIZON_RUNTIME_NAME = "Event Horizon R59 Strategy Math Native Loop"
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
        entries = ["default", r"D:\AI NSFW\VID", r"D:\AI NSFW\PIC"]
        if os.path.isdir(output_dir):
            for root, dirs, files in os.walk(output_dir):
                rel = os.path.relpath(root, output_dir)
                if rel != ".":
                    entries.append(rel.replace("\\", "/"))
                if len(entries) >= 100:
                    break
        return sorted(set(entries))
    except Exception:
        return ["default", r"D:\AI NSFW\VID", r"D:\AI NSFW\PIC"]



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

    roots = [Path(r"D:\AI NSFW"), Path(r"D:/AI NSFW")]
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
    CATEGORY = "Event Equality/Core"

    def run(self, message):
        return (f"Event Debug Ping: {message}",)


class EventInitPacket:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"debug_mode": (DEBUG_MODES, {"default": DEBUG_BASIC})}}

    RETURN_TYPES = ("EVENT_PACKET", "STRING")
    RETURN_NAMES = ("event_packet", "summary")
    FUNCTION = "run"
    CATEGORY = "Event Equality/Core"

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
    CATEGORY = "Event Equality/Readers"

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
    CATEGORY = "Event Equality/Readers"

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
    CATEGORY = "Event Equality/Readers"

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
    CATEGORY = "Event Equality/Readers"

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
    CATEGORY = "Event Equality/Readers"

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
    CATEGORY = "Event Equality/Strategy"

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
    CATEGORY = "Event Equality/Research"

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
    CATEGORY = "Event Equality/Research"

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
    CATEGORY = "Event Equality/Noise"

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
    CATEGORY = "Event Equality/Boundary"

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
    CATEGORY = "Event Equality/Boundary"

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
    CATEGORY = "Event Equality/Research"

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
    CATEGORY = "Event Equality/Core"

    def run(self, report, filename_prefix, output_directory):
        safe_prefix = re.sub(r"[^a-zA-Z0-9_\\-]+", "_", str(filename_prefix or "event_report")).strip("_")
        if not safe_prefix:
            safe_prefix = "event_report"

        if output_directory and str(output_directory).strip():
            out_dir = Path(str(output_directory)).expanduser()
        else:
            out_dir = Path.cwd() / "output" / "event_equality_reports"

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
    CATEGORY = "Event Equality/Core"

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
    CATEGORY = "Event Equality/Adapters"

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
    CATEGORY = "Event Equality/Core"

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
    CATEGORY = "Event Equality/Core"

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
        "barrier_policy": "preserve pass-through tensor/state; cleanup must not destroy StrategyCarrier",
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


class WanEventWorkflowCore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "primary_model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "source_image_file": (_event_core_list_input_images(),),

                "positive_prompt": ("STRING", {"default": "", "multiline": True, "height": 180, "dynamicPrompts": False}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True, "height": 180, "dynamicPrompts": False}),
                "event_strategy": ("STRING", {"default": "", "multiline": True, "height": 90}),

                "generation_target": (["VIDEO", "IMAGE", "AUTO"], {"default": "VIDEO"}),
                "terminal_mode": ("BOOLEAN", {"default": True}),
                "enable_continuation_outputs": ("BOOLEAN", {"default": False}),
                "execution_mode": (["RUN"], {"default": "RUN"}),
                "branch_mode": (["AUTO", "SINGLE", "DUAL_HIGH_LOW"], {"default": "AUTO"}),

                                "cascade_count": ("INT", {"default": 1, "min": 1, "max": 5}),
                "pause_after_cascade_1": ("BOOLEAN", {"default": False}),
                "pause_after_cascade_2": ("BOOLEAN", {"default": False}),
                "pause_after_cascade_3": ("BOOLEAN", {"default": False}),
                "pause_after_cascade_4": ("BOOLEAN", {"default": False}),
                "resume_frame_index": ("INT", {"default": -1, "min": -1, "max": 4096}),
                "cascade_mode": (["SOLO_1", "CASCADE_2", "CASCADE_3", "CASCADE_4", "CASCADE_5"], {"default": "SOLO_1"}),
                "frames_per_cascade": ("INT", {"default": 49, "min": 1, "max": 4096}),

                "width": ("INT", {"default": 608, "min": 16, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 416, "min": 16, "max": 8192, "step": 8}),
                "frames": ("INT", {"default": 49, "min": 1, "max": 4096}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "fps": ("INT", {"default": 16, "min": 1, "max": 240}),
                "seed": ("INT", {"default": 359, "min": 0, "max": 0xffffffffffffffff}),

                "sampler_name": ("STRING", {"default": "euler"}),
                "scheduler": ("STRING", {"default": "simple"}),
                "global_steps": ("INT", {"default": 4, "min": 0, "max": 10000}),
                "primary_cfg": ("FLOAT", {"default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "secondary_cfg": ("FLOAT", {"default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "primary_start_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "primary_end_step": ("INT", {"default": 3, "min": 0, "max": 10000}),
                "secondary_start_step": ("INT", {"default": 3, "min": 0, "max": 10000}),
                "secondary_end_step": ("INT", {"default": 4, "min": 0, "max": 10000}),
                "primary_sd3_shift": ("FLOAT", {"default": 8.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "secondary_sd3_shift": ("FLOAT", {"default": 8.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),

                "decode_tile_size": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "decode_overlap": ("INT", {"default": 64, "min": 0, "max": 8192, "step": 8}),
                "decode_temporal_size": ("INT", {"default": 32, "min": 1, "max": 4096}),
                "decode_temporal_overlap": ("INT", {"default": 12, "min": 0, "max": 4096}),

                "image_upscale_method": (["nearest-exact", "nearest", "bilinear", "area", "bicubic", "lanczos"], {"default": "nearest-exact"}),
                "image_crop": (["disabled", "center"], {"default": "disabled"}),

                "cleanup_timing": ([
                    "NONE", "BEFORE_GENERATION", "BETWEEN_SAMPLERS", "AFTER_GENERATION", "BEFORE_AND_AFTER", "ALL",
                    "none", "before_generation", "between_samplers", "after_generation", "before_and_after", "all",
                ], {"default": "ALL"}),
                "stage_delay_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "use_custom_cleanup_nodes": ("BOOLEAN", {"default": True}),

                "save_video": ("BOOLEAN", {"default": True}),
                "video_format": (["video/h264-mp4", "video/h265-mp4", "image/webp", "image/gif"], {"default": "video/h264-mp4"}),
                "force_vhs_video_combine": ("BOOLEAN", {"default": True}),
                "save_frames": ("BOOLEAN", {"default": False}),
                "save_report": ("BOOLEAN", {"default": True}),
                "output_target": (["COMFY_OUTPUT"], {"default": "COMFY_OUTPUT"}),
                "save_output_image": ("BOOLEAN", {"default": False}),
                "save_prefix": ("STRING", {"default": "wansolo"}),
                "output_folder_mode": (["DEFAULT", "PICKER", "CUSTOM"], {"default": "DEFAULT"}),
                "output_folder": (["default"],),
                "custom_output_folder": ("STRING", {"default": ""}),
                "report_detail": (["COMPACT", "STANDARD", "FULL"], {"default": "STANDARD"}),
            },
            "optional": {
                "secondary_model": ("MODEL",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    # Terminal output node.
    # IMPORTANT r15:
    # No IMAGE/LATENT outputs here. IMAGE outputs caused ComfyUI/PreviewImage-style
    # behavior and made the node look like an image-producing endpoint.
    # Video/image files are saved internally; strings are the final visible outputs.
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "status",
        "saved_video_path",
        "saved_report_path",
        "report",
    )
    FUNCTION = "run"
    CATEGORY = "Event Equality/Event Horizon"
    OUTPUT_NODE = True

    def _placeholder_image(self, width=64, height=64):
        try:
            import torch
            return torch.zeros((1, int(height), int(width), 3), dtype=torch.float32)
        except Exception:
            return None

    def _ensure_preview_image(self, image, fallback_width=64, fallback_height=64):
        if image is None:
            return self._placeholder_image(fallback_width, fallback_height)
        try:
            import torch
            if hasattr(image, "dim"):
                if image.dim() == 3:
                    return image.unsqueeze(0)
                return image
        except Exception:
            pass
        return image

    def _representative_preview_frame(self, image, fallback_width=64, fallback_height=64, mode="last"):
        """
        PreviewImage saves every image in a batch. A decoded video is usually an IMAGE batch
        shaped [frames, H, W, C]. Sending it directly to PreviewImage creates 65+ PNG files.
        This function reduces any IMAGE batch to one representative frame for UI preview.
        """
        img = self._ensure_preview_image(image, fallback_width, fallback_height)
        if img is None:
            return self._placeholder_image(fallback_width, fallback_height)
        try:
            if hasattr(img, "dim") and img.dim() == 4 and img.shape[0] > 1:
                idx = 0 if str(mode).lower() == "first" else int(img.shape[0]) - 1
                return img[idx:idx + 1]
        except Exception:
            pass
        return img

    def _load_image_from_upload(self, source_image_file, records):
        if not source_image_file or str(source_image_file).lower() in ("none", "null", ""):
            records.append({"stage": "source_image_upload", "status": "skipped", "reason": "no source_image_file"})
            return None
        try:
            import folder_paths
            import numpy as np
            import torch
            from PIL import Image, ImageOps, ImageSequence

            image_path = folder_paths.get_annotated_filepath(source_image_file)
            img = Image.open(image_path)
            frames = []
            for frame in ImageSequence.Iterator(img):
                frame = ImageOps.exif_transpose(frame)
                if frame.mode == "I":
                    frame = frame.point(lambda i: i * (1 / 255))
                frame = frame.convert("RGB")
                arr = np.array(frame).astype(np.float32) / 255.0
                frames.append(torch.from_numpy(arr)[None,])
                break

            if not frames:
                raise RuntimeError("No frame loaded from source_image_file")

            out = torch.cat(frames, dim=0)
            records.append({"stage": "source_image_upload", "status": "ok", "file": str(source_image_file), "shape": list(out.shape)})
            return out
        except Exception as e:
            records.append({"stage": "source_image_upload", "status": "failed", "file": str(source_image_file), "error": str(e)})
            return None

    def _make_ui_previews(self, source_preview, result_preview, save_prefix,
        records, include_result_preview=False):
        """
        r15: no internal PreviewImage calls.

        Reason:
        PreviewImage writes PNG files. The main terminal node must not create image
        outputs or preview dumps during video generation. Source selection is handled
        by the image picker widget; final result is saved by VHS/video export.
        """
        records.append({
            "stage": "preview_ui",
            "status": "disabled",
            "preview_policy": "no_internal_previewimage_calls_in_terminal_node",
            "reason": "avoid PNG dumps and avoid making video workflow behave like image output",
        })
        return []

    def _save_pause_frames_to_temp(self, frames):
        import folder_paths
        import os
        import random
        from PIL import Image
        import numpy as np

        ui_images = []
        try:
            temp_dir = folder_paths.get_temp_directory()
            
            total_f = frames.shape[0]
            # Wan2.1 valid frames must be 4*k + 1. 
            # We want up to 7 valid frames from the tail.
            valid_indices = []
            for i in range(total_f):
                # i is 0-indexed. The frame count is i+1.
                if (i + 1 - 1) % 4 == 0:
                    valid_indices.append(i)
                    
            # Take the last 7 valid indices
            selected_indices = valid_indices[-7:] if len(valid_indices) >= 7 else valid_indices
            
            prefix = "cascade_preview_" + str(random.randint(100000, 999999))
            
            for idx, frame_idx in enumerate(selected_indices):
                frame = frames[frame_idx]
                img_array = (255. * frame.cpu().numpy()).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(img_array)
                
                filename = f"{prefix}_{idx}.png"
                filepath = os.path.join(temp_dir, filename)
                img.save(filepath)
                
                ui_images.append({
                    "filename": filename,
                    "subfolder": "",
                    "type": "temp",
                    "resume_index": frame_idx + 1  # 1-indexed for the UI resume_frame_index
                })
        except Exception as e:
            print(f"Failed to save pause frames: {e}")
        return ui_images


    def _import_comfy_nodes(self):
        return importlib.import_module("nodes")

    def _get_node_class(self, class_name):
        comfy_nodes = self._import_comfy_nodes()
        cls = getattr(comfy_nodes, class_name, None)
        if cls is None and hasattr(comfy_nodes, "NODE_CLASS_MAPPINGS"):
            cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(class_name)
        if cls is None:
            raise RuntimeError(f"ComfyUI node class {class_name} not found")
        return cls

    def _call_node_method(self, class_name, method_names, *args, **kwargs):
        cls = self._get_node_class(class_name)
        inst = cls()
        if isinstance(method_names, str):
            method_names = [method_names]

        fn_name = getattr(cls, "FUNCTION", None) or getattr(inst, "FUNCTION", None)
        candidate_names = []
        for name in method_names:
            if name not in candidate_names:
                candidate_names.append(name)
        if fn_name and fn_name not in candidate_names:
            candidate_names.append(fn_name)
        for fallback in ("run", "execute", "process", "generate", "encode", "decode", "sample", "patch", "upscale", "save_images", "combine_video", "combine"):
            if fallback not in candidate_names:
                candidate_names.append(fallback)

        last_error = None
        tried = []
        for name in candidate_names:
            if hasattr(inst, name):
                tried.append(name)
                method = getattr(inst, name)
                try:
                    return method(*args, **kwargs)
                except TypeError as e:
                    last_error = e
                    try:
                        sig = inspect.signature(method)
                        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
                        return method(*args, **filtered)
                    except Exception as e2:
                        last_error = e2
                except Exception as e:
                    last_error = e

        raise RuntimeError(
            f"Could not call {class_name}. tried={tried}, requested={method_names}, "
            f"class_FUNCTION={fn_name}, last_error={last_error}"
        )


    def _describe_raw_output(self, obj, max_items=80):
        try:
            attrs = []
            for name in dir(obj):
                if name.startswith("__"):
                    continue
                attrs.append(name)
                if len(attrs) >= max_items:
                    break
            return {
                "type": f"{type(obj).__module__}.{type(obj).__name__}",
                "repr": str(obj)[:500],
                "dir_sample": attrs,
                "has_dict": hasattr(obj, "__dict__"),
                "dict_keys": list(getattr(obj, "__dict__", {}).keys())[:max_items] if hasattr(obj, "__dict__") else [],
            }
        except Exception as e:
            return {"type": str(type(obj)), "describe_error": str(e)}

    def _node_output_to_sequence(self, result, preferred_names=None, stage="node_output", records=None):
        preferred_names = preferred_names or []
        if isinstance(result, tuple) or isinstance(result, list):
            return list(result)
        if isinstance(result, dict):
            if "result" in result:
                return self._node_output_to_sequence(result["result"], preferred_names, stage, records)
            values = []
            for name in preferred_names:
                if name in result:
                    values.append(result[name])
            return values if values else list(result.values())

        values = []
        for name in preferred_names:
            if hasattr(result, name):
                try:
                    values.append(getattr(result, name))
                except Exception:
                    pass
        if values:
            if records is not None:
                records.append({"stage": f"{stage}_node_output_unwrapped", "status": "ok", "mode": "preferred_attrs", "names": preferred_names})
            return values

        for attr in ("result", "results", "value", "values", "output", "outputs", "data"):
            if hasattr(result, attr):
                try:
                    obj = getattr(result, attr)
                    seq = self._node_output_to_sequence(obj, preferred_names, stage, records)
                    if seq:
                        if records is not None:
                            records.append({"stage": f"{stage}_node_output_unwrapped", "status": "ok", "mode": attr})
                        return seq
                except Exception:
                    pass

        try:
            d = getattr(result, "__dict__", None)
            if isinstance(d, dict) and d:
                values = []
                for name in preferred_names:
                    if name in d:
                        values.append(d[name])
                if values:
                    return values
                # common private storages in API output wrappers
                for storage_name in ("_values", "_outputs", "_data", "_result", "_results"):
                    if storage_name in d:
                        try:
                            return self._node_output_to_sequence(d[storage_name], preferred_names, stage, records)
                        except Exception:
                            pass
                plain = [v for k, v in d.items() if not k.startswith("_") and not callable(v)]
                if plain:
                    return plain
                private_plain = [v for k, v in d.items() if not callable(v)]
                if private_plain:
                    return private_plain
        except Exception:
            pass

        indexed = []
        try:
            for i in range(8):
                try:
                    indexed.append(result[i])
                except Exception:
                    break
            if indexed:
                if records is not None:
                    records.append({"stage": f"{stage}_node_output_unwrapped", "status": "ok", "mode": "getitem"})
                return indexed
        except Exception:
            pass

        try:
            seq = list(result)
            if seq:
                if records is not None:
                    records.append({"stage": f"{stage}_node_output_unwrapped", "status": "ok", "mode": "iter"})
                return seq
        except Exception:
            pass

        desc = self._describe_raw_output(result)
        if records is not None:
            records.append({"stage": f"{stage}_node_output_unwrap_failed", "status": "failed", "raw_output": desc})
        raise RuntimeError(f"{stage} returned unrecognized NodeOutput/container: {desc}")

    def _first_output(self, result, preferred_names=None, stage="node_output", records=None):
        seq = self._node_output_to_sequence(result, preferred_names=preferred_names, stage=stage, records=records)
        if not seq:
            raise RuntimeError(f"{stage} returned no outputs")
        return seq[0]

    def _extract_tensors_from_obj(self, obj, prefix="root", max_items=16):
        tensors = []
        try:
            if hasattr(obj, "detach") and hasattr(obj, "shape"):
                tensors.append((prefix, obj))
                return tensors
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if len(tensors) >= max_items:
                        break
                    tensors.extend(self._extract_tensors_from_obj(v, f"{prefix}.{k}", max_items=max_items-len(tensors)))
                return tensors
            if isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    if len(tensors) >= max_items:
                        break
                    tensors.extend(self._extract_tensors_from_obj(v, f"{prefix}[{i}]", max_items=max_items-len(tensors)))
                return tensors
        except Exception:
            pass
        return tensors

    def _finite_guard(self, obj, records, stage, strict=True):
        """
        Detect NaN/Inf tensors before black-video save.
        Returns True if all discovered tensors are finite.
        """
        try:
            import torch
            tensors = self._extract_tensors_from_obj(obj)
            if not tensors:
                records.append({"stage": stage, "status": "no_tensor_found"})
                return True
            all_ok = True
            reports = []
            for name, t in tensors:
                try:
                    td = t.detach()
                    finite = torch.isfinite(td)
                    total = int(td.numel())
                    finite_count = int(finite.sum().item()) if total else 0
                    nonfinite_count = total - finite_count
                    finite_ratio = float(finite_count / total) if total else 1.0
                    reports.append({
                        "tensor": name,
                        "shape": list(td.shape),
                        "dtype": str(td.dtype),
                        "device": str(td.device),
                        "total": total,
                        "nonfinite_count": nonfinite_count,
                        "finite_ratio": finite_ratio,
                    })
                    if nonfinite_count > 0:
                        all_ok = False
                except Exception as e:
                    reports.append({"tensor": name, "status": "inspect_failed", "error": str(e)})
            records.append({"stage": stage, "status": "ok" if all_ok else "nonfinite_detected", "reports": reports})
            if strict and not all_ok:
                raise RuntimeError(f"{stage}: non-finite tensor values detected; refusing to save black/corrupt video")
            return all_ok
        except Exception as e:
            records.append({"stage": stage, "status": "failed", "error": str(e)})
            if strict:
                raise
            return False

    def _encode_text(self, clip, text, records=None, label="TextEncode"):
        result = self._call_node_method("CLIPTextEncode", ["encode"], clip=clip, text=str(text or ""))
        conditioning = self._first_output(result, preferred_names=["conditioning", "CONDITIONING"], stage=f"Event{label}")
        if records is not None:
            self._event_universal_stage_math(
                records,
                f"Event{label}",
                input_state=text,
                output_state=conditioning,
                observed_behavior="text prompt encoded into conditioning strategy",
                formula_role="TEXT/CLIP -> CONDITIONING as NumericStrategy",
                route_id=f"route_{str(label).lower()}",
                next_requirement="WanImageToVideo and sampler require conditioning compatible with model route",
                control_mode="REPORT_ONLY",
                metadata={"text_length": len(str(text or ""))},
            )
        return conditioning

    def _scale_image(self, image, width, height, method, crop, records):
        if image is None:
            records.append({"stage": "EventImageScaleStart", "status": "skipped", "reason": "no image"})
            return None
        try:
            result = self._call_node_method(
                "ImageScale",
                ["upscale"],
                image=image,
                upscale_method=str(method or "nearest-exact"),
                width=int(width),
                height=int(height),
                crop=str(crop or "disabled"),
            )
            scaled = self._first_output(result, preferred_names=["image", "images", "IMAGE"], stage="EventImageScaleStart", records=records)
            records.append({"stage": "EventImageScaleStart", "status": "ok", "width": width, "height": height})
            self._event_universal_stage_math(
                records,
                "EventImageScaleStart",
                input_state=image,
                output_state=scaled,
                observed_behavior="source image resized/cropped into model-compatible source outcome",
                formula_role="IMAGE SourceAnchor -> IMAGE scaled OutcomePrevious",
                route_id="route_source_image",
                next_requirement="WanImageToVideo requires scaled image, width, height, frame count and conditioning",
                control_mode="REPORT_ONLY",
                metadata={"width": int(width), "height": int(height), "method": str(method or "nearest-exact"), "crop": str(crop or "disabled")},
            )
            return scaled
        except Exception as e:
            records.append({"stage": "EventImageScaleStart", "status": "failed_no_fake_success", "error": str(e)})
            raise

    def _apply_sd3_shift(self, model, shift, label, records):
        try:
            shift_value = float(shift or 0.0)
        except Exception:
            shift_value = 0.0
        if abs(shift_value) < 1e-9:
            records.append({
                "stage": f"EventModelShift_{label}",
                "status": "skipped_zero_shift",
                "shift": shift_value,
                "formula": "shift=0 -> original model passthrough; do not apply ModelSamplingSD3 patch",
            })
            self._event_universal_stage_math(
                records,
                f"EventModelShift_{label}",
                input_state=model,
                output_state=model,
                observed_behavior="shift value is zero; model route passes through unchanged",
                formula_role="MODEL Operator -> MODEL Operator passthrough",
                route_id=f"route_model_shift_{label}",
                next_requirement="sampler requires model route compatible with current branch",
                control_mode="REPORT_ONLY",
                metadata={"shift": shift_value, "applied": False},
            )
            return model
        try:
            result = self._call_node_method("ModelSamplingSD3", ["patch"], model=model, shift=shift_value)
            patched = self._first_output(result, preferred_names=["model", "MODEL"], stage=f"EventModelShift_{label}", records=records)
            records.append({"stage": f"EventModelShift_{label}", "status": "ok", "shift": shift_value, "formula": "model + shift behavior = EventSingularity_model_shift = shifted model"})
            self._event_universal_stage_math(
                records,
                f"EventModelShift_{label}",
                input_state=model,
                output_state=patched,
                observed_behavior="ModelSamplingSD3 shift patches model route for sampler branch",
                formula_role="MODEL Operator + shift behavior -> MODEL shifted Operator",
                route_id=f"route_model_shift_{label}",
                next_requirement="sampler requires shifted model route compatible with branch window",
                control_mode="REPORT_ONLY",
                metadata={"shift": shift_value, "applied": True},
            )
            return patched
        except Exception as e:
            records.append({"stage": f"event_model_shift_{label}", "status": "failed", "shift": shift_value, "error": str(e)})
            raise

    def _wan_image_to_video(self, positive, negative, vae, start_image, width, height, frames, batch_size, records):
        try:
            result = self._call_node_method(
                "WanImageToVideo",
                ["encode", "generate", "process", "run"],
                positive=positive,
                negative=negative,
                vae=vae,
                clip_vision_output=None,
                start_image=start_image,
                width=int(width),
                height=int(height),
                length=int(frames),
                batch_size=int(batch_size),
            )
        except Exception as first_error:
            records.append({"stage": "EventWanImageToVideoSeed_first_signature_failed", "status": "retrying", "error": str(first_error)})
            result = self._call_node_method(
                "WanImageToVideo",
                ["generate", "process", "run", "encode"],
                positive=positive,
                negative=negative,
                vae=vae,
                clip_vision_output=None,
                image=start_image,
                start_image=start_image,
                width=int(width),
                height=int(height),
                frames=int(frames),
                length=int(frames),
                batch_size=int(batch_size),
            )

        seq = self._node_output_to_sequence(
            result,
            preferred_names=["positive", "negative", "latent", "conditioning", "conditioning_positive", "conditioning_negative", "samples"],
            stage="EventWanImageToVideoSeed",
            records=records,
        )
        if len(seq) < 3:
            raise RuntimeError(
                f"WanImageToVideo returned fewer than 3 usable outputs after unwrapping: "
                f"count={len(seq)}, raw={self._describe_raw_output(result)}"
            )

        records.append({
            "stage": "EventWanImageToVideoSeed",
            "status": "ok",
            "width": width,
            "height": height,
            "frames": frames,
            "formula": "source image + conditioning + VAE behavior = EventSingularity_i2v = video latent seed",
            "event_node": "EventWanImageToVideo",
            "unwrapped_output_count": len(seq),
        })
        self._event_universal_stage_math(
            records,
            "EventWanImageToVideoSeed",
            input_state=start_image,
            output_state=seq[2],
            observed_behavior="scaled source image, positive conditioning, VAE and dimensions created initial video latent seed",
            formula_role="IMAGE + CONDITIONING + VAE -> LATENT seed OutcomePrevious",
            route_id="route_wan_i2v_seed",
            next_requirement="High sampler requires latent seed as previous outcome and noise-enabled first denoise window",
            control_mode="REPORT_ONLY",
            metadata={"width": int(width), "height": int(height), "frames": int(frames), "batch_size": int(batch_size)},
        )
        return seq[0], seq[1], seq[2]


    def _low_level_sampler_operation(self, *, model, positive, negative, latent, seed, steps, cfg, sampler_name, scheduler, start_at_step, end_at_step, add_noise, return_leftover_noise):
        result = self._call_node_method(
            "KSamplerAdvanced",
            ["sample"],
            model=model,
            add_noise=add_noise,
            noise_seed=int(seed),
            steps=int(steps),
            cfg=float(cfg),
            sampler_name=str(sampler_name or "euler"),
            scheduler=str(scheduler or "simple"),
            positive=positive,
            negative=negative,
            latent_image=latent,
            start_at_step=int(start_at_step),
            end_at_step=int(end_at_step),
            return_with_leftover_noise=return_leftover_noise,
        )
        return self._first_output(result, preferred_names=["latent", "samples", "LATENT"], stage="EventSamplerLowLevel")

    def _tensor_from_latent_like(self, obj):
        try:
            if isinstance(obj, dict):
                if "samples" in obj:
                    return obj["samples"]
                for v in obj.values():
                    t = self._tensor_from_latent_like(v)
                    if t is not None:
                        return t
            if hasattr(obj, "detach") and hasattr(obj, "shape"):
                return obj
            if isinstance(obj, (list, tuple)) and obj:
                return self._tensor_from_latent_like(obj[0])
        except Exception:
            return None
        return None

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

    def _dual_branch_delta_coupling_math(self, *, delta_high, delta_low, records, active_branch_mode, cascade_count):
        """
        Observer-only branch coupling math.
        Compares high/low sampler deltas to expose alignment and energy split diagnostics.
        """
        rec = {
            "stage": "EventMathDualBranchDeltaCoupling",
            "active_branch_mode": str(active_branch_mode or ""),
            "cascade_count": int(cascade_count or 1),
        }
        try:
            import math
            import torch

            if str(active_branch_mode or "") != "DUAL_HIGH_LOW":
                rec.update({"status": "skipped", "reason": "single_branch_mode"})
                records.append(rec)
                return rec

            high_t = self._tensor_from_latent_like(delta_high)
            low_t = self._tensor_from_latent_like(delta_low)
            if high_t is None or low_t is None:
                rec.update({
                    "status": "unavailable",
                    "reason": "missing_delta_tensor",
                    "high_present": bool(high_t is not None),
                    "low_present": bool(low_t is not None),
                })
                records.append(rec)
                return rec

            high_f = torch.nan_to_num(high_t.detach().float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
            low_f = torch.nan_to_num(low_t.detach().float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
            if high_f.numel() < 1 or low_f.numel() < 1:
                rec.update({"status": "unavailable", "reason": "empty_delta"})
                records.append(rec)
                return rec

            n = int(min(high_f.numel(), low_f.numel()))
            high_f = high_f[:n]
            low_f = low_f[:n]

            max_elements = 200000
            stride = max(1, n // max_elements) if n > max_elements else 1
            if stride > 1:
                high_f = high_f[::stride]
                low_f = low_f[::stride]

            sample_count = int(high_f.numel())
            high_norm = float(torch.linalg.vector_norm(high_f).item()) if sample_count else 0.0
            low_norm = float(torch.linalg.vector_norm(low_f).item()) if sample_count else 0.0
            dot = float((high_f * low_f).sum().item()) if sample_count else 0.0
            denom = (high_norm * low_norm) + 1e-12
            cosine = float(dot / denom) if sample_count else 0.0
            cosine = max(-1.0, min(1.0, cosine))
            angle_deg = float(math.degrees(math.acos(cosine)))

            if cosine > 0.20:
                direction_relation = "aligned"
            elif cosine < -0.20:
                direction_relation = "opposed"
            else:
                direction_relation = "orthogonal_mixed"

            energy_total = high_norm + low_norm
            high_energy_fraction = float(high_norm / (energy_total + 1e-12))
            low_energy_fraction = float(low_norm / (energy_total + 1e-12))
            low_to_high_norm_ratio = float(low_norm / (high_norm + 1e-12))
            alignment_score = float(0.5 * (1.0 + cosine))
            refinement_coupling_score = float(alignment_score * low_energy_fraction)

            rec.update({
                "status": "ok",
                "sample_count": sample_count,
                "sample_stride": int(stride),
                "high_delta_norm": high_norm,
                "low_delta_norm": low_norm,
                "low_to_high_norm_ratio": low_to_high_norm_ratio,
                "delta_cosine_alignment": cosine,
                "delta_alignment_angle_deg": angle_deg,
                "direction_relation": direction_relation,
                "high_energy_fraction": high_energy_fraction,
                "low_energy_fraction": low_energy_fraction,
                "alignment_score_01": alignment_score,
                "refinement_coupling_score_01": refinement_coupling_score,
                "formula": "ObservedBehavior(high) and ObservedBehavior(low) are compared as coupled branch deltas over a shared latent basis.",
            })
            records.append(rec)
            return rec
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
        This is not a separate visual ComfyUI node. It is the internal body of the single Event Horizon node.
        """
        packet = ensure_packet(packet)
        body = {
            "body_version": EVENT_HORIZON_BODY_VERSION,
            "body_name": "One Node Event Core Body + Runtime Monitor Body",
            "external_node": "Event Horizon",
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
            "message": "Event Core Body is internal to this one Event Horizon node; no manual Event graph required.",
        })
        packet = record_stage(
            packet,
            stage_name="EventCoreBody",
            action="INIT_INTERNAL_BODY",
            observed_behavior="Single Event Horizon node created internal EventPacket, S-Wire, RouteMemory, and Formula route body.",
            metadata=body,
            formula_note=body["formula"],
        )
        return packet

    def _event_core_live_record(self, stage_name, record_type="stage", status="", formula_role="", route_id="", observed_behavior="", metadata=None):
        """
        r49 live RouteMemory/S-Wire hook.
        This is observer-only: it records route timing and memory snapshots without changing tensors or routing.
        """
        live = getattr(self, "_event_live_body", None)
        if not isinstance(live, dict):
            return None
        timeline = live.setdefault("route_timeline", [])
        entry = {
            "index": len(timeline),
            "stage": str(stage_name or "unknown"),
            "record_type": str(record_type or "stage"),
            "status": str(status or ""),
            "formula_role": str(formula_role or ""),
            "route_id": str(route_id or ""),
            "observed_behavior": str(observed_behavior or ""),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "perf_counter": float(time.perf_counter()),
            "memory": _event_core_memory_snapshot(str(stage_name or "unknown")),
            "metadata": metadata or {},
        }
        timeline.append(entry)
        live.setdefault("runtime_monitor", []).append({
            "stage": entry["stage"],
            "record_type": entry["record_type"],
            "status": entry["status"],
            "perf_counter": entry["perf_counter"],
            "memory": entry["memory"],
        })
        if record_type in ("stage_math", "boundary_math", "manual"):
            live.setdefault("s_wire_live", []).append({
                "stage": entry["stage"],
                "formula_role": entry["formula_role"],
                "route_id": entry["route_id"],
                "status": entry["status"],
            })
        return entry

    def _event_core_body_collect_from_records(self, packet, execution_records):
        """
        Builds a compact internal S-Wire / RouteMemory summary from execution_records.
        This keeps r44 one-node: records are collected inside the same Event Horizon node.
        """
        packet = ensure_packet(packet)
        body = packet.setdefault("metadata", {}).setdefault("event_core_body", {})
        s_wire = []
        stage_math_records = []
        boundary_records = []
        for rec in execution_records or []:
            stage = str(rec.get("stage", ""))
            if stage.startswith("EventUniversalMath_"):
                item = {
                    "stage": stage,
                    "stage_name": rec.get("stage_name", stage.replace("EventUniversalMath_", "")),
                    "formula_role": rec.get("formula_role", ""),
                    "route_id": rec.get("route_id", ""),
                    "observed_behavior": rec.get("observed_behavior", ""),
                    "control_mode": rec.get("control_mode", ""),
                    "status": rec.get("status", ""),
                }
                stage_math_records.append(item)
                s_wire.append(item)
            elif stage.startswith("EventUniversalBoundary_"):
                item = {
                    "stage": stage,
                    "boundary_name": rec.get("boundary_name", stage.replace("EventUniversalBoundary_", "")),
                    "route_id": rec.get("route_id", ""),
                    "observed_behavior": rec.get("observed_behavior", ""),
                    "control_mode": rec.get("control_mode", ""),
                    "status": rec.get("status", ""),
                }
                boundary_records.append(item)
                s_wire.append(item)

        body["s_wire"] = s_wire
        body["stage_math_records"] = stage_math_records
        body["boundary_records"] = boundary_records
        body["stage_math_count"] = len(stage_math_records)
        body["boundary_count"] = len(boundary_records)
        body["s_wire_count"] = len(s_wire)
        live = getattr(self, "_event_live_body", {})
        if isinstance(live, dict):
            body["live_route_timeline"] = list(live.get("route_timeline", []))
            body["runtime_monitor_records"] = list(live.get("runtime_monitor", []))
            body["s_wire_live"] = list(live.get("s_wire_live", []))
            body["live_route_count"] = len(body["live_route_timeline"])
            body["runtime_monitor_count"] = len(body["runtime_monitor_records"])
            if body["runtime_monitor_records"]:
                first = body["runtime_monitor_records"][0].get("perf_counter")
                last = body["runtime_monitor_records"][-1].get("perf_counter")
                try:
                    observed_span = float(last) - float(first)
                except Exception:
                    observed_span = None
                body["runtime_monitor_summary"] = {
                    "record_count": len(body["runtime_monitor_records"]),
                    "observed_stage_span_seconds": round(observed_span, 6) if observed_span is not None else None,
                    "observer_only": True,
                    "has_process_memory": any("process_rss_mb" in (x.get("memory") or {}) for x in body["runtime_monitor_records"]),
                    "has_cuda_memory": any((x.get("memory") or {}).get("torch_cuda_available") for x in body["runtime_monitor_records"]),
                }
        body["one_node_integrity"] = {
            "external_node_count": 1,
            "internal_event_body": True,
            "manual_event_graph_required": False,
            "route_memory_kept_inside_node": True,
        }
        return packet

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

        duplicate_counts = {}
        for s in stages:
            duplicate_counts[s] = duplicate_counts.get(s, 0) + 1
        duplicates = {k: v for k, v in duplicate_counts.items() if v > 1 and k not in ("EventHorizonStageDelay",)}

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

        pass_gate = (
            one_node_ok and
            missing_total == 0 and
            stage_math_count >= 7 and
            order_ok
        )

        return {
            "stage": "EventCoreBodyCompletionGate",
            "status": "PASS" if pass_gate else "BLOCKED",
            "formula": "Tuning may begin only after the one-node Event Core Body is coherent, ordered, and complete enough.",
            "one_node_ok": one_node_ok,
            "stage_order_ok": order_ok,
            "missing_total": missing_total,
            "stage_math_count": stage_math_count,
            "boundary_math_count": int(checks.get("boundary_math_count", 0) or 0),
            "math_tensor_record_count": int(checks.get("math_tensor_record_count", 0) or 0),
            "result_status": audit.get("result_status", "") if isinstance(audit, dict) else "",
            "saved_video_path": audit.get("saved_video_path", "") if isinstance(audit, dict) else "",
            "next_action": (
                "Core body gate passed. Tuning/control work may begin on the next iteration."
                if pass_gate else
                "Core body gate blocked. Fix missing stages or order violations before tuning."
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
            packet = add_core_conflict(
                "EventCoreBodyCompletionGateBlocked",
                "BLOCKED",
                "EventCoreBodyCompletionGate",
                f"CompletionGate={gate.get('status')}, missing_total={gate.get('missing_total')}, stage_order_ok={gate.get('stage_order_ok')}",
                "Do not enable active tuning/cache/compile paths until the gate is PASS.",
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
            "live_route_count": body.get("live_route_count", 0),
            "runtime_monitor_count": body.get("runtime_monitor_count", 0),
            "local_sstate_count": len(body.get("local_sstates", []) or []),
            "event_conflict_count": len(body.get("event_conflicts", []) or []),
            "required_exact_missing": checks.get("required_exact_missing", []),
            "required_prefix_missing": checks.get("required_prefix_missing", []),
            "video_stage_missing": checks.get("video_stage_missing", []),
            "order_violations": order_audit.get("violations", []) if isinstance(order_audit, dict) else [],
            "next_action": gate.get("next_action", "") if isinstance(gate, dict) else "",
        }

    def _event_core_body_report_card(self, audit):
        checks = audit.get("checks", {}) if isinstance(audit, dict) else {}
        return {
            "stage": "EventCoreBodyReportCard",
            "status": audit.get("status", "unknown") if isinstance(audit, dict) else "unknown",
            "severity": audit.get("severity", "UNKNOWN") if isinstance(audit, dict) else "UNKNOWN",
            "one_node_ok": bool(checks.get("one_external_node_policy_present")) and bool(checks.get("event_core_body_initialized")),
            "stage_math_count": checks.get("stage_math_count", 0),
            "boundary_math_count": checks.get("boundary_math_count", 0),
            "missing_total": (
                len(checks.get("required_exact_missing", []) or []) +
                len(checks.get("required_prefix_missing", []) or []) +
                len(checks.get("video_stage_missing", []) or [])
            ),
            "next_action": (
                "Continue only if EventCoreBodyCompletionGate is PASS on normal and cascade runs."
                if audit.get("status") == "pass" else
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
        execution_records.append(self._event_core_body_report_card(audit))
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
        return controlled_latent_after, result

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

    def _run_custom_cleanup_chain(self, value, label, records):
        current = value
        try:
            result = self._call_node_method(
                "RAMCleanup",
                ["cleanup", "clean"],
                anything=current,
                clean_file_cache=True,
                clean_processes=True,
                clean_dlls=True,
                retry_times=3,
            )
            current = self._first_output(result, preferred_names=["anything", "output"], stage=f"EventRAMCleanup_{label}", records=records)
            records.append({"stage": f"EventRAMCleanup_{label}", "status": "ok", "formula": "memory before + cleanup behavior = EventSingularity_cleanup_ram = memory after"})
            self._event_universal_stage_math(
                records,
                f"EventRAMCleanup_{label}",
                input_state=value,
                output_state=current,
                observed_behavior="RAM cleanup executed while preserving pass-through value",
                formula_role="MEMORY pressure + cleanup behavior -> pass-through state",
                route_id=f"route_cleanup_ram_{label}",
                next_requirement="next stage should receive same technical value with reduced memory pressure",
                control_mode="REPORT_ONLY",
                metadata={"cleanup_type": "RAMCleanup"},
            )
        except Exception as e:
            records.append({"stage": f"event_ram_cleanup_{label}", "status": "fallback_python_cleanup", "error": str(e)})
            _event_core_cleanup_memory(f"ram_cleanup_fallback_{label}")

        try:
            result = self._call_node_method(
                "VRAMCleanup",
                ["cleanup", "clean"],
                anything=current,
                offload_model=False,
                offload_cache=True,
            )
            current = self._first_output(result, preferred_names=["anything", "output"], stage=f"EventVRAMCleanup_{label}", records=records)
            records.append({"stage": f"EventVRAMCleanup_{label}", "status": "ok", "formula": "vram before + cleanup behavior = EventSingularity_cleanup_vram = vram after"})
            self._event_universal_stage_math(
                records,
                f"EventVRAMCleanup_{label}",
                input_state=value,
                output_state=current,
                observed_behavior="VRAM cleanup executed while preserving pass-through value",
                formula_role="VRAM pressure + cleanup behavior -> pass-through state",
                route_id=f"route_cleanup_vram_{label}",
                next_requirement="next stage should receive same technical value with reduced VRAM pressure",
                control_mode="REPORT_ONLY",
                metadata={"cleanup_type": "VRAMCleanup"},
            )
        except Exception as e:
            records.append({"stage": f"event_vram_cleanup_{label}", "status": "fallback_python_cleanup", "error": str(e)})
            _event_core_cleanup_memory(f"vram_cleanup_fallback_{label}")

        return current

    def _cleanup_mode_includes(self, cleanup_timing, phase_name):
        mode = str(cleanup_timing or "NONE").upper()
        phase = str(phase_name or "").upper()
        matrix = {
            "BEFORE_GENERATION": {"BEFORE_GENERATION", "BEFORE_AND_AFTER", "ALL"},
            "BETWEEN_SAMPLERS": {"BETWEEN_SAMPLERS", "ALL"},
            "AFTER_GENERATION": {"AFTER_GENERATION", "BEFORE_AND_AFTER", "ALL"},
        }
        return mode in matrix.get(phase, set())

    def _run_branch_barrier(
        self,
        *,
        phase_name,
        cleanup_timing,
        strategy_state,
        strategy_label,
        records,
        cleanup_records,
        barrier_records,
        route_id,
        stage_name,
        formula_role,
        observed_behavior,
        next_requirement,
        use_custom_cleanup_nodes,
        use_custom_chain,
        cleanup_label,
    ):
        phase = str(phase_name or "").upper()
        mode = str(cleanup_timing or "NONE").upper()
        if not self._cleanup_mode_includes(mode, phase):
            return strategy_state, None

        preserved_before = _event_core_state_descriptor(strategy_state, strategy_label)
        current_state = strategy_state
        custom_cleanup_used = bool(use_custom_chain and use_custom_cleanup_nodes and strategy_state is not None)
        if custom_cleanup_used:
            current_state = self._run_custom_cleanup_chain(current_state, cleanup_label, records)

        cleanup_info = _event_core_cleanup_memory(f"{cleanup_label}_python")
        if isinstance(cleanup_records, list):
            cleanup_records.append(cleanup_info)

        preserved_after = _event_core_state_descriptor(current_state, strategy_label)
        shape_same = preserved_before.get("shape") == preserved_after.get("shape")
        dtype_same = preserved_before.get("dtype") == preserved_after.get("dtype")
        type_same = preserved_before.get("type") == preserved_after.get("type")
        strategy_state_preserved = bool(type_same and (shape_same or "shape" not in preserved_before) and (dtype_same or "dtype" not in preserved_before))

        released_actions = []
        if custom_cleanup_used:
            released_actions.extend(["RAMCleanup pass-through", "VRAMCleanup pass-through"])
        if cleanup_info.get("gc"):
            released_actions.append("python gc.collect")
        if cleanup_info.get("torch_cuda_empty_cache"):
            released_actions.append("torch.cuda.empty_cache")
        if cleanup_info.get("torch_cuda_ipc_collect"):
            released_actions.append("torch.cuda.ipc_collect")

        memory_before = cleanup_info.get("memory_before", {}) if isinstance(cleanup_info.get("memory_before", {}), dict) else {}
        memory_after = cleanup_info.get("memory_after", {}) if isinstance(cleanup_info.get("memory_after", {}), dict) else {}
        barrier_record = {
            "barrier_phase": phase.lower(),
            "cleanup_timing_requested": mode,
            "observer_only": True,
            "strategy_state_required": strategy_state is not None,
            "strategy_state_preserved": strategy_state_preserved,
            "preserved": {
                "before": preserved_before,
                "after": preserved_after,
            },
            "released": {
                "actions": released_actions,
                "python_cleanup": {
                    "gc": bool(cleanup_info.get("gc")),
                    "torch_cuda_empty_cache": bool(cleanup_info.get("torch_cuda_empty_cache")),
                    "torch_cuda_ipc_collect": bool(cleanup_info.get("torch_cuda_ipc_collect")),
                },
            },
            "memory_before": memory_before,
            "memory_after": memory_after,
            "memory_delta": _event_core_memory_delta(memory_before, memory_after),
        }
        if cleanup_info.get("error"):
            barrier_record["cleanup_error"] = str(cleanup_info.get("error"))

        records.append({
            "stage": f"EventBranchBarrier_{phase}",
            "status": "recorded",
            "cleanup_timing": mode,
            "strategy_state_preserved": strategy_state_preserved,
            "released_actions": released_actions,
        })
        if isinstance(barrier_records, list):
            barrier_records.append(barrier_record)

        self._event_universal_stage_math(
            records,
            stage_name,
            input_state=strategy_state,
            output_state=current_state,
            observed_behavior=observed_behavior,
            formula_role=formula_role,
            route_id=route_id,
            next_requirement=next_requirement,
            control_mode="REPORT_ONLY",
            metadata=barrier_record,
        )
        return current_state, barrier_record

    def _resolve_output_dir(self, output_folder_mode, output_folder, custom_output_folder, subdir="event_equality_reports", output_target="USER_D_AI_NSFW", media_type="video"):
        mode_target = str(output_target or "USER_D_AI_NSFW")
        media_type = str(media_type or "video").lower()

        def local_try_make_dir(path):
            try:
                Path(path).mkdir(parents=True, exist_ok=True)
                return Path(path)
            except Exception:
                return None

        # Preferred explicit user folder:
        # D:\AI NSFW\VID for video, D:\AI NSFW\PIC for images.
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
        safe_prefix = re.sub(r"[^a-zA-Z0-9_\\-]+", "_", str(save_prefix or "Event Horizon")).strip("_") or "Event_Horizon"
        out_dir = self._resolve_output_dir(output_folder_mode, output_folder, custom_output_folder, subdir="event_equality_reports", output_target=output_target, media_type="report")
        out_dir.mkdir(parents=True, exist_ok=True)

        text = "" if report is None else str(report)
        if not text.strip():
            text = (
                "# Event Horizon Report Fallback\n\n"
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
            if stage in ("EventHorizonCascadeExecutionGate", "EventHorizonCascadeBegin"):
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
            "cleanup_timing": wan.get("cleanup_timing"),
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
        if not saved_report_path:
            return {"status": "skipped_no_report_path"}

        report_path = Path(saved_report_path)
        out_dir = report_path.parent
        base = report_path.with_suffix("")
        json_path = Path(str(base) + "_runtime_monitor.json")
        csv_path = Path(str(base) + "_runtime_monitor.csv")
        diff_path = Path(str(base) + "_runtime_monitor_diff.csv")

        meta = packet.get("metadata", {}) if isinstance(packet, dict) else {}
        body = meta.get("event_core_body", {}) if isinstance(meta.get("event_core_body", {}), dict) else {}
        result = meta.get("result_status", {}) if isinstance(meta.get("result_status", {}), dict) else {}
        gate = body.get("completion_gate", {}) if isinstance(body.get("completion_gate", {}), dict) else {}
        settings_signature, signature_source = self._runtime_monitor_settings_signature(packet)
        rows = self._runtime_monitor_rows(packet)
        motion_summary = self._runtime_motion_summary(packet)

        payload = {
            "schema_version": "event_horizon.runtime_monitor.v2",
            "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
            "runtime_name": EVENT_HORIZON_RUNTIME_NAME,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "save_prefix": str(save_prefix or ""),
            "run_id": body.get("run_id", ""),
            "settings_signature": settings_signature,
            "settings_signature_source": signature_source,
            "result_status": result.get("result_status", body.get("result_status", "")),
            "completion_gate": gate.get("status", ""),
            "saved_video_path": str(saved_video_path or result.get("saved_video_path", "")),
            "report_path": str(saved_report_path),
            "json_path": str(json_path),
            "csv_path": str(csv_path),
            "diff_path": str(diff_path),
            "event_core_summary": body.get("summary", {}),
            "runtime_monitor_summary": body.get("runtime_monitor_summary", {}),
            "runtime_monitor_records": body.get("runtime_monitor_records", []),
            "runtime_monitor_rows": rows,
            "motion_summary": motion_summary,
            "observer_only": True,
            "formula": "Runtime timing and memory are ObservedBehavior extensions recorded without mutating generation.",
        }

        out_dir.mkdir(parents=True, exist_ok=True)
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


    def _save_image_attempt(self, image, save_prefix, records):
        result = self._call_node_method("SaveImage", ["save_images"], images=image, filename_prefix=str(save_prefix or "wansolo"))
        records.append({"stage": "EventSaveFrames", "status": "ok", "result": str(result)[:300]})
        self._event_universal_stage_math(
            records,
            "EventSaveFrames",
            input_state=image,
            output_state=str(result)[:300],
            observed_behavior="decoded frame/image tensor saved as image assets",
            formula_role="IMAGE/FRAMES VisibleOutcome -> FILE output reference",
            route_id="route_save_frames",
            next_requirement="saved image assets should match decoded visible outcome",
            control_mode="REPORT_ONLY",
            metadata={"save_prefix": str(save_prefix or "wansolo")},
        )
        return str(result)

    def _normalize_video_path_candidate(self, candidate):
        if candidate is None:
            return ""
        c = str(candidate)
        if not c:
            return ""
        exts = (".mp4", ".webp", ".gif", ".mov", ".mkv")
        if not c.lower().endswith(exts):
            return ""

        # If absolute, return it even if the file existence check fails during Comfy async timing.
        if os.path.isabs(c):
            return c

        try:
            import folder_paths
            output_dir = Path(folder_paths.get_output_directory())
        except Exception:
            output_dir = Path.cwd() / "output"

        direct = output_dir / c
        if direct.exists():
            return str(direct)

        # VHS sometimes gives only filename and subfolder separately; search by basename.
        if output_dir.exists():
            matches = list(output_dir.rglob(Path(c).name))
            if matches:
                return str(max(matches, key=lambda x: x.stat().st_mtime))

        # As a last resort return the candidate, but report will show it was unresolved.
        return c

    def _extract_path_from_video_result(self, result, save_prefix, records, *args, **kwargs):
        """
        r19: extract path only from standard VHS return / standard ComfyUI output.
        """
        raw_description = str(result)[:1000]

        try:
            import folder_paths
            output_dir = Path(folder_paths.get_output_directory())
        except Exception:
            output_dir = Path.cwd() / "output"

        try:
            if isinstance(result, dict):
                gifs = result.get("ui", {}).get("gifs", []) or []
                for item in gifs:
                    if isinstance(item, dict):
                        fullpath = item.get("fullpath")
                        path = self._normalize_video_path_candidate(fullpath)
                        if path:
                            records.append({"stage": "EventVideoCombine_extract_path", "status": "ok", "mode": "ui.gifs.fullpath", "path": path})
                            return path
                        filename = item.get("filename")
                        subfolder = item.get("subfolder", "")
                        if filename:
                            candidate = output_dir / str(subfolder) / str(filename)
                            path = self._normalize_video_path_candidate(str(candidate))
                            if path:
                                records.append({"stage": "EventVideoCombine_extract_path", "status": "ok", "mode": "ui.gifs.filename", "path": path})
                                return path
        except Exception as e:
            records.append({"stage": "EventVideoCombine_extract_path", "status": "ui_parse_failed", "error": str(e)})

        candidates = []
        def visit(obj, depth=0):
            if obj is None or depth > 8:
                return
            if isinstance(obj, str):
                candidates.append(obj)
                return
            if isinstance(obj, (list, tuple, set)):
                for x in obj:
                    visit(x, depth + 1)
                return
            if isinstance(obj, dict):
                for key in ("fullpath", "path", "filename", "file", "files", "result", "results", "ui", "gifs", "images"):
                    if key in obj:
                        visit(obj[key], depth + 1)
                return
            for attr in ("fullpath", "path", "filename", "file", "files", "result", "results", "value", "values", "output", "outputs", "data"):
                if hasattr(obj, attr):
                    try:
                        visit(getattr(obj, attr), depth + 1)
                    except Exception:
                        pass
            d = getattr(obj, "__dict__", None)
            if isinstance(d, dict):
                visit(d, depth + 1)

        visit(result)
        for ext in (".mp4", ".mov", ".mkv", ".webp", ".gif"):
            for c in candidates:
                if isinstance(c, str) and c.lower().endswith(ext):
                    path = self._normalize_video_path_candidate(c)
                    if path:
                        records.append({"stage": "EventVideoCombine_extract_path", "status": "ok", "mode": "recursive_standard_vhs", "path": path})
                        return path

        newest = self._find_latest_standard_output_video(save_prefix, records)
        if newest:
            return newest

        records.append({
            "stage": "EventVideoCombine_extract_path",
            "status": "not_found",
            "raw_result_type": f"{type(result).__module__}.{type(result).__name__}",
            "raw_result": raw_description,
            "candidates": [str(x)[:200] for x in candidates[:20]],
        })
        return ""


    def _get_first_available_node_class(self, class_names, records=None, stage="node_class_lookup"):
        errors = {}
        for class_name in class_names:
            try:
                cls = self._get_node_class(class_name)
                if records is not None:
                    records.append({
                        "stage": stage,
                        "status": "found",
                        "class_name": class_name,
                        "class": str(cls),
                    })
                return class_name, cls
            except Exception as e:
                errors[class_name] = str(e)
        if records is not None:
            records.append({
                "stage": stage,
                "status": "not_found",
                "class_names": list(class_names),
                "errors": errors,
            })
        raise RuntimeError(f"No candidate node class found for {class_names}; errors={errors}")

    def _call_candidate_node_method(self, class_names, method_names, records=None, stage="candidate_node_call", *args, **kwargs):
        last_error = None
        tried = []
        for class_name in class_names:
            try:
                # Check class exists first so report shows whether VHS is installed / mapped.
                self._get_first_available_node_class([class_name], records=records, stage=f"{stage}_class_check")
                tried.append(class_name)
                return self._call_node_method(class_name, method_names, *args, **kwargs)
            except Exception as e:
                last_error = e
                if records is not None:
                    records.append({
                        "stage": stage,
                        "status": "candidate_failed",
                        "class_name": class_name,
                        "error": str(e),
                    })
        raise RuntimeError(f"All candidate node calls failed. tried={tried}; last_error={last_error}")

    def _frames_to_uint8_numpy(self, image, records=None, stage="frames_to_uint8"):
        try:
            import numpy as np
            frames = image
            if not hasattr(frames, "detach"):
                raise RuntimeError(f"{stage} expected torch-like IMAGE batch, got {type(frames)}")
            arr = frames.detach().cpu().numpy()
            if arr.ndim != 4:
                raise RuntimeError(f"{stage} expected IMAGE batch [N,H,W,C], got shape={getattr(arr, 'shape', None)}")
            finite = np.isfinite(arr)
            nonfinite_count = int(arr.size - finite.sum())
            if nonfinite_count > 0:
                raise RuntimeError(f"{stage} refusing to encode non-finite frames: nonfinite_count={nonfinite_count}, shape={arr.shape}")
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype("uint8")
            if records is not None:
                records.append({"stage": stage, "status": "ok", "shape": list(arr.shape), "nonfinite_count": 0})
            return arr
        except Exception as e:
            if records is not None:
                records.append({"stage": stage, "status": "failed", "error": str(e)})
            raise

    def _fallback_animated_export(self, image, fps, save_prefix, video_format, records, output_target="USER_D_AI_NSFW", output_folder_mode="DEFAULT", output_folder="default", custom_output_folder=""):
        """
        Emergency media writer if VHS is absent or incompatible.
        Tries mp4 first, then animated WebP. It must physically write a file or return "".
        """
        try:
            arr = self._frames_to_uint8_numpy(image, records=records, stage="EventVideoFallback_frames_to_numpy")
            output_dir = self._resolve_output_dir(output_folder_mode, output_folder, custom_output_folder, subdir="", output_target=output_target, media_type="video")
            output_dir.mkdir(parents=True, exist_ok=True)
            safe_prefix = re.sub(r"[^a-zA-Z0-9_\\-]+", "_", str(save_prefix or "wansolo")).strip("_") or "wansolo"
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            # 1) imageio v3 / ffmpeg if available.
            if "mp4" in str(video_format).lower():
                try:
                    import imageio.v3 as iio
                    path = output_dir / f"{safe_prefix}_{ts}_fallback_imageio.mp4"
                    iio.imwrite(str(path), arr, fps=float(fps), codec="libx264")
                    if path.exists() and path.stat().st_size > 0:
                        records.append({"stage": "EventVideoFallbackImageIO", "status": "ok", "path": str(path), "size": path.stat().st_size})
                        return str(path)
                    records.append({"stage": "EventVideoFallbackImageIO", "status": "wrote_no_file_or_empty", "path": str(path)})
                except Exception as e:
                    records.append({"stage": "EventVideoFallbackImageIO", "status": "failed", "error": str(e)})

                # 2) OpenCV mp4 if available.
                try:
                    import cv2
                    path = output_dir / f"{safe_prefix}_{ts}_fallback_cv2.mp4"
                    h, w = int(arr.shape[1]), int(arr.shape[2])
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
                    if not writer.isOpened():
                        raise RuntimeError("cv2.VideoWriter did not open")
                    for frame in arr:
                        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    writer.release()
                    if path.exists() and path.stat().st_size > 0:
                        records.append({"stage": "EventVideoFallbackCV2", "status": "ok", "path": str(path), "size": path.stat().st_size})
                        return str(path)
                    records.append({"stage": "EventVideoFallbackCV2", "status": "wrote_no_file_or_empty", "path": str(path)})
                except Exception as e:
                    records.append({"stage": "EventVideoFallbackCV2", "status": "failed", "error": str(e)})

            # 3) Animated WebP fallback.
            try:
                from PIL import Image
                path = output_dir / f"{safe_prefix}_{ts}_fallback.webp"
                pil_frames = [Image.fromarray(frame) for frame in arr]
                duration = int(1000 / max(1, float(fps)))
                pil_frames[0].save(path, save_all=True, append_images=pil_frames[1:], duration=duration, loop=0, lossless=True)
                if path.exists() and path.stat().st_size > 0:
                    records.append({"stage": "EventVideoFallbackWebP", "status": "ok", "path": str(path), "size": path.stat().st_size})
                    return str(path)
                records.append({"stage": "EventVideoFallbackWebP", "status": "wrote_no_file_or_empty", "path": str(path)})
            except Exception as e:
                records.append({"stage": "EventVideoFallbackWebP", "status": "failed", "error": str(e)})

            return ""
        except Exception as e:
            records.append({"stage": "EventVideoFallback", "status": "failed", "error": str(e)})
            return ""


    def _extract_vhs_ui_payload_from_records(self, records):
        """
        r21: recover the VHS UI payload so ComfyUI can display the generated video
        in the node UI without adding IMAGE outputs.
        """
        try:
            import ast
            for rec in reversed(records or []):
                if rec.get("stage") == "EventVideoCombine" and rec.get("raw_vhs_result"):
                    raw = rec.get("raw_vhs_result")
                    try:
                        payload = ast.literal_eval(raw)
                    except Exception:
                        payload = None
                    if isinstance(payload, tuple) and len(payload) > 0:
                        payload = payload[0]
                    if isinstance(payload, dict) and isinstance(payload.get("ui"), dict):
                        ui = payload.get("ui")
                        if "gifs" in ui:
                            return {"gifs": ui.get("gifs")}
                        if "images" in ui:
                            return {"images": ui.get("images")}
        except Exception:
            pass
        return None

    def _video_combine_attempt(self, image, fps, save_prefix, records, video_format="video/h264-mp4", force_vhs=True, output_target="COMFY_OUTPUT", output_folder_mode="DEFAULT", output_folder="default", custom_output_folder=""):
        """
        r19: Standard VHS-only save.

        This intentionally matches the working JSON:
        VHS_VideoCombine receives filename_prefix, format, frame_rate, save_output=True.
        It does NOT receive output_dir or custom path arguments.
        VHS saves into the standard ComfyUI output folder.
        """
        records.append({
            "stage": "EventVideoSaveBegin",
            "status": "start_standard_vhs",
            "video_format": str(video_format),
            "fps": fps,
            "filename_prefix": str(save_prefix or "wansolo"),
            "path_policy": "standard_comfyui_vhs_output_no_custom_output_dir",
            "matches_working_json": True,
            "loop_count": 0,
            "pingpong": False,
        })
        self._event_universal_stage_math(
            records,
            "EventVideoSaveBegin",
            input_state=image,
            output_state=image,
            observed_behavior="decoded frames prepared for VHS video combine without loop/pingpong modification",
            formula_role="FRAMES VisibleOutcome -> FRAMES video input state",
            route_id="route_video_save_begin",
            next_requirement="VHS_VideoCombine receives ordered frames with frame_rate and save_output",
            control_mode="REPORT_ONLY",
            metadata={"video_format": str(video_format), "fps": float(fps), "loop_count": 0, "pingpong": False},
        )

        try:
            self._frames_to_uint8_numpy(image, records=records, stage="EventVideoSave_validate_frames")
        except Exception as e:
            raise RuntimeError(f"Cannot save video because decoded frames are invalid: {e}")

        try:
            result = self._call_candidate_node_method(
                ["VHS_VideoCombine"],
                ["combine_video", "combine", "run", "execute"],
                records=records,
                stage="EventVideoCombineVHS",
                images=image,
                audio=None,
                meta_batch=None,
                vae=None,
                frame_rate=float(fps),
                loop_count=0,
                filename_prefix=str(save_prefix or "wansolo"),
                format=str(video_format or "video/h264-mp4"),
                pingpong=False,
                save_output=True,
                pix_fmt="yuv420p",
                crf=19,
                save_metadata=True,
                trim_to_audio=False,
            )
            path = self._extract_path_from_video_result(result, save_prefix, records)
            if not path:
                path = self._find_latest_standard_output_video(save_prefix, records)

            records.append({
                "stage": "EventVideoCombine",
                "status": "ok" if path else "ok_no_path_returned",
                "saved_video_path": path,
                "raw_vhs_result": str(result)[:5000],
                "formula": "decoded frames + VHS_VideoCombine(save_output=True) = saved video in standard ComfyUI output",
            })
            self._event_universal_stage_math(
                records,
                "EventVideoCombine",
                input_state=image,
                output_state=path or str(result)[:500],
                observed_behavior="VHS_VideoCombine packaged decoded frames into video file/reference",
                formula_role="FRAMES VisibleOutcome -> VIDEO file OutcomeFinal",
                route_id="route_video_combine",
                next_requirement="final report and UI preview should point to saved video path",
                control_mode="REPORT_ONLY",
                metadata={"video_format": str(video_format), "fps": float(fps), "saved_video_path": path or ""},
            )

            if path:
                return path
            raise RuntimeError(f"VHS_VideoCombine ran but no output path was found. raw={str(result)[:500]}")
        except Exception as e:
            records.append({"stage": "EventVideoCombine", "status": "failed_standard_vhs", "error": str(e)})
            raise

    def _find_latest_standard_output_video(self, save_prefix, records):
        try:
            import folder_paths
            output_dir = Path(folder_paths.get_output_directory())
        except Exception:
            output_dir = Path.cwd() / "output"

        safe_prefix = str(save_prefix or "wansolo")
        found = []
        if output_dir.exists():
            for ext in (".mp4", ".webp", ".gif", ".mov", ".mkv"):
                found.extend(output_dir.rglob(f"{safe_prefix}*{ext}"))

        if found:
            newest = max(found, key=lambda x: x.stat().st_mtime)
            records.append({
                "stage": "EventVideoCombine_standard_output_search",
                "status": "ok",
                "path": str(newest),
                "output_dir": str(output_dir),
            })
            return str(newest)

        records.append({
            "stage": "EventVideoCombine_standard_output_search",
            "status": "not_found",
            "output_dir": str(output_dir),
            "prefix": safe_prefix,
        })
        return ""


    def _record_mirror_cut(self, packet, resume_frame_index, target_t, start_segment, execution_records):
        """Зеркальная логика MIRROR_CUT boundary event в EventPacket."""
        packet, prev_sig, prev_proj, _ = _read_signal(
            packet,
            TECH_LATENT,
            SPACE_LATENT,
            {"resume_frame_index": resume_frame_index, "is_reference": True},
            f"MirrorCut_{start_segment}_OutcomePrevious",
            "OutcomePrevious",
            f"route_mirror_cut_{start_segment}",
            "MirrorCutReader",
            metadata={"resume_frame_index": resume_frame_index, "target_t": target_t, "is_reference": True}
        )

        cut_payload = {
            "resume_frame_index": resume_frame_index,
            "target_t": target_t,
            "reason": "user_filmstrip_selection"
        }
        packet, cut_sig, cut_proj, _ = _read_signal(
            packet,
            TECH_DELTA,
            SPACE_EVENT,
            cut_payload,
            f"MirrorCut_{start_segment}",
            "ObservedBehaviorCurrent",
            f"route_mirror_cut_{start_segment}",
            "MirrorCutReader",
            metadata=cut_payload
        )

        rel = make_event_relation(
            relation_type=REL_TRANSFORMS_INTO,
            source_signal_ids=[prev_sig["id"], cut_sig["id"]],
            target_signal_ids=[],  # будет заполнено новым StrategyCarrier
            operator_name="Unchaining_MirrorCut",
            formula_meaning="Outcome at pause + MirrorCut = new Strategy for continuation",
            local_strategy_id=f"S_mirror_cut_{start_segment}",
            metadata={
                "boundary_type": "pause_resume",
                "resume_frame_index": resume_frame_index,
                "target_t": target_t
            }
        )
        packet = add_relation(packet, rel)

        packet, sstate = build_sstate_from_packet(
            packet,
            position=f"S_mirror_cut_{start_segment}",
            active_relation_ids=[rel["id"]]
        )

        packet = record_stage(
            packet,
            stage_name="EventHorizonMirrorCut",
            action="UNCHAINING",
            status="ok",
            metadata={"resume_frame_index": resume_frame_index, "target_t": target_t}
        )
        return packet, rel, sstate

    def _complete_mirror_cut_relation(self, packet, new_strategy_carrier_sig_id):
        """Дополнение зеркальной связи новым StrategyCarrier."""
        rel_id = _EVENT_HORIZON_CASCADE_CACHE.get("pending_mirror_cut_rel_id")
        if not rel_id:
            return packet

        import time
        for rel in packet.get("relations", []):
            if rel.get("id") == rel_id:
                rel.setdefault("target_signal_ids", []).append(new_strategy_carrier_sig_id)
                rel["metadata"]["completed"] = True
                break

        packet, continuation_sstate = build_sstate_from_packet(
            packet,
            position=f"S_after_mirror_cut_{int(time.time())}",
            active_relation_ids=[rel_id]
        )

        packet = record_stage(
            packet,
            stage_name="EventHorizonUnchainedContinuation",
            action="RESTART_FROM_OBSERVED",
            status="ok",
            metadata={"rel_id": rel_id}
        )
        return packet

    def _make_text_signal(self, packet, text, role, route, source_stage):
        return _read_signal(
            packet,
            TECH_TEXT,
            SPACE_TEXT,
            text,
            source_stage,
            role,
            route,
            "TextStrategyReader",
            metadata={"text_length": len(str(text or ""))},
        )

    def _fail_conflict(self, packet, stage, message, metadata=None):
        conflict = make_conflict(
            CONFLICT_FROZEN_PARTIAL_OBSERVABILITY,
            severity=SEV_MEDIUM,
            stage_position=stage,
            suspected_cause="Terminal Wan workflow generation did not produce the requested target.",
            observed_symptom=str(message),
            suggested_response="Send Execution Records and ComfyUI traceback; next patch will fix exact incompatible stage.",
            metadata=metadata or {},
        )
        packet = add_conflict(packet, conflict)
        return packet, conflict["id"]

    def _stage_delay(self, seconds, records, label):
        try:
            seconds = float(seconds or 0.0)
        except Exception:
            seconds = 0.0
        if seconds <= 0:
            return
        try:
            import time
            records.append({"stage": "EventHorizonStageDelay", "status": "sleep", "seconds": seconds, "label": str(label)})
            time.sleep(seconds)
        except Exception as e:
            records.append({"stage": "EventHorizonStageDelay", "status": "failed", "seconds": seconds, "label": str(label), "error": str(e)})

    def _last_frame_image(self, frames, width=64, height=64):
        return self._representative_preview_frame(frames, width, height, mode="last")

    def _drop_first_frame_batch(self, frames, records, segment_index):
        """
        r22 temporal continuity guard:
        segment N starts from the last frame of segment N-1.
        Keeping that first generated frame duplicates the previous terminal frame and can look like
        a motion reset or vector reversal at the cascade boundary.
        """
        try:
            if hasattr(frames, "dim") and frames.dim() == 4 and frames.shape[0] > 1:
                out = frames[1:]
                records.append({
                    "stage": "EventHorizonCascadeDropFirstFrame",
                    "status": "ok",
                    "segment_index": int(segment_index),
                    "before_shape": list(frames.shape),
                    "after_shape": list(out.shape),
                    "reason": "remove duplicated source/continuation frame at cascade boundary",
                })
                return out
        except Exception as e:
            records.append({
                "stage": "EventHorizonCascadeDropFirstFrame",
                "status": "failed",
                "segment_index": int(segment_index),
                "error": str(e),
            })
        return frames

    def _concat_frame_batches(self, batches, records):
        valid = [b for b in batches if b is not None]
        if not valid:
            return None
        if len(valid) == 1:
            return valid[0]
        try:
            import torch
            out = torch.cat(valid, dim=0)
            records.append({
                "stage": "EventHorizonCascadeFrameConcat",
                "status": "ok",
                "segments": len(valid),
                "shape": list(out.shape) if hasattr(out, "shape") else str(type(out)),
            })
            return out
        except Exception as e:
            records.append({
                "stage": "EventHorizonCascadeFrameConcat",
                "status": "failed_using_last_segment_only",
                "segments": len(valid),
                "error": str(e),
            })
            return valid[-1]

    def _run_event_horizon_segment_core(
        self,
        *,
        segment_index,
        source_image,
        primary_model,
        secondary_model,
        clip,
        vae,
        positive_prompt,
        negative_prompt,
        active_branch_mode,
        width,
        height,
        frames,
        batch_size,
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
        primary_sd3_shift,
        secondary_sd3_shift,
        decode_tile_size,
        decode_overlap,
        decode_temporal_size,
        decode_temporal_overlap,
        image_upscale_method,
        image_crop,
        cleanup_timing,
        use_custom_cleanup_nodes,
        stage_delay_seconds,
        records,
        cleanup_records,
        barrier_records,
    ):
        segment_label = f"cascade_{segment_index}"
        records.append({
            "stage": "EventHorizonCascadeSegmentBegin",
            "status": "begin",
            "segment_index": segment_index,
            "frames": int(frames),
            "formula": "source frame + segment transition = EventSingularity_segment = decoded frame batch",
        })

        positive = self._encode_text(clip, positive_prompt, records, label=f'{segment_label}_TextEncodePositive')
        negative = self._encode_text(clip, negative_prompt, records, label=f'{segment_label}_TextEncodeNegative')
        self._stage_delay(stage_delay_seconds, records, f"cascade_{segment_index}_after_text_encode")
        scaled_image = self._scale_image(source_image, width, height, image_upscale_method, image_crop, records)

        wan_positive, wan_negative, wan_latent = self._wan_image_to_video(
            positive, negative, vae, scaled_image, width, height, frames, batch_size, records
        )
        self._stage_delay(stage_delay_seconds, records, f"cascade_{segment_index}_after_wan_image_to_video")

        high_model = self._apply_sd3_shift(primary_model, primary_sd3_shift, f"{segment_label}_high", records)
        high_window = EventSamplerWindow(
            branch_name=f"{segment_label}_high",
            branch_role="cascade_segment_high_motion_structure",
            seed=int(seed) + int(segment_index) - 1,
            steps=int(global_steps),
            cfg=float(primary_cfg),
            sampler_name=str(sampler_name),
            scheduler=str(scheduler),
            start_at_step=int(primary_start_step),
            end_at_step=int(primary_end_step),
            add_noise="enable",
            return_with_leftover_noise="enable",
            sd3_shift=float(primary_sd3_shift),
        )
        latent_after_high, high_result = self._event_sample_window(
            high_model, wan_positive, wan_negative, wan_latent, high_window, records
        )
        self._stage_delay(stage_delay_seconds, records, f"cascade_{segment_index}_after_high_sampler")
        delta_high, delta_high_err = compute_tensor_delta(wan_latent, latent_after_high)
        if delta_high is None:
            records.append({
                "stage": f"EventMathDualBranchDeltaCoupling_{segment_label}",
                "status": "high_delta_unavailable",
                "segment_index": int(segment_index),
                "error": str(delta_high_err),
            })
        delta_low = None

        final_latent = latent_after_high
        if active_branch_mode == "DUAL_HIGH_LOW" and secondary_model is not None:
            latent_before_low = latent_after_high
            latent_before_low, _ = self._run_branch_barrier(
                phase_name="BETWEEN_SAMPLERS",
                cleanup_timing=cleanup_timing,
                strategy_state=latent_before_low,
                strategy_label=f"{segment_label}_strategy_carrier_low",
                records=records,
                cleanup_records=cleanup_records,
                barrier_records=barrier_records,
                route_id=f"route_cleanup_{segment_label}",
                stage_name=f"EventCleanup_{segment_label}_between_high_low",
                formula_role="LATENT cascade high output + cleanup behavior -> LATENT cascade low input",
                observed_behavior="Smart Branch Barrier between cascade high and low preserves StrategyCarrier while releasing disposable memory.",
                next_requirement="cascade low sampler receives preserved high output latent",
                use_custom_cleanup_nodes=use_custom_cleanup_nodes,
                use_custom_chain=True,
                cleanup_label=f"{segment_label}_between_high_low",
            )

            low_model = self._apply_sd3_shift(secondary_model, secondary_sd3_shift, f"{segment_label}_low", records)
            low_window = EventSamplerWindow(
                branch_name=f"{segment_label}_low",
                branch_role="cascade_segment_low_detail_refinement",
                seed=int(seed) + int(segment_index) - 1,
                steps=int(global_steps),
                cfg=float(secondary_cfg),
                sampler_name=str(sampler_name),
                scheduler=str(scheduler),
                start_at_step=int(secondary_start_step),
                end_at_step=int(secondary_end_step),
                add_noise="disable",
                return_with_leftover_noise="disable",
                sd3_shift=float(secondary_sd3_shift),
            )
            latent_after_low, low_result = self._event_sample_window(
                low_model, wan_positive, wan_negative, latent_before_low, low_window, records
            )
            self._stage_delay(stage_delay_seconds, records, f"cascade_{segment_index}_after_low_sampler")
            delta_low, delta_low_err = compute_tensor_delta(latent_before_low, latent_after_low)
            if delta_low is None:
                records.append({
                    "stage": f"EventMathDualBranchDeltaCoupling_{segment_label}",
                    "status": "low_delta_unavailable",
                    "segment_index": int(segment_index),
                    "error": str(delta_low_err),
                })
            final_latent = latent_after_low

        self._dual_branch_delta_coupling_math(
            delta_high=delta_high,
            delta_low=delta_low,
            records=records,
            active_branch_mode=active_branch_mode,
            cascade_count=int(segment_index),
        )

        frames_out = self._decode_tiled(
            vae, final_latent,
            decode_tile_size, decode_overlap, decode_temporal_size, decode_temporal_overlap,
            records
        )
        self._math_tensor_summary(frames_out, records, f"EventMath_cascade_{segment_index}_decoded_frames", strict=False)
        self._frame_motion_math(frames_out, records, f"EventMath_cascade_{segment_index}_frame_motion")
        self._stage_delay(stage_delay_seconds, records, f"cascade_{segment_index}_after_decode")
        last_frame = self._last_frame_image(frames_out, width, height)
        records.append({
            "stage": "EventHorizonCascadeSegmentEnd",
            "status": "ok",
            "segment_index": segment_index,
            "frames": int(frames),
            "last_frame_for_next_segment": True,
        })
        return frames_out, last_frame, final_latent

    def run(
        self,
        primary_model,
        clip,
        vae,
        source_image_file,
        positive_prompt,
        negative_prompt,
        event_strategy,
        generation_target,
        terminal_mode,
        enable_continuation_outputs,
        execution_mode,
        branch_mode,
        cascade_count,
        cascade_mode,
        frames_per_cascade,
        width,
        height,
        frames,
        batch_size,
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
        primary_sd3_shift,
        secondary_sd3_shift,
        decode_tile_size,
        decode_overlap,
        decode_temporal_size,
        decode_temporal_overlap,
        image_upscale_method,
        image_crop,
        cleanup_timing,
        stage_delay_seconds,
        use_custom_cleanup_nodes,
        save_video,
        video_format,
        force_vhs_video_combine,
        save_frames,
        save_report,
        output_target,
        save_output_image,
        save_prefix,
        output_folder_mode,
        output_folder,
        custom_output_folder,
        report_detail,
        pause_after_cascade_1=False,
        pause_after_cascade_2=False,
        pause_after_cascade_3=False,
        pause_after_cascade_4=False,
        resume_frame_index=-1,
        secondary_model=None,
        image=None,
        mask=None,
    ):
        run_id = now_run_id(prefix="terminal_wansolo")
        self._event_strategy_coupling = {"low_strength_multiplier": 1.0}
        cleanup_records = []
        branch_barrier_records = []
        execution_records = []
        runtime_aliases = self._event_runtime_aliases()
        execution_records.append({
            "stage": "EventHorizonRuntimeVersion",
            "status": "loaded",
            "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
            "runtime_name": EVENT_HORIZON_RUNTIME_NAME,
            "math_report_enabled": True,
            "expected_math_stages": [
                "EventMath_high_latent_before",
                "EventMath_high_latent_after",
                "EventMath_low_latent_before",
                "EventMath_low_latent_after",
                "EventMathStrategyCarrierCoupling",
                "EventMathDualBranchDeltaCoupling",
                "EventMath_decoded_frame_motion",
                "EventUniversalMath_EventTextEncodePositive",
                "EventUniversalMath_EventTextEncodeNegative",
                "EventUniversalMath_EventImageScaleStart",
                "EventUniversalMath_EventWanImageToVideoSeed",
                "EventUniversalMath_EventSamplerHigh",
                "EventUniversalMath_EventCleanupBetweenSamplers",
                "EventUniversalMath_EventSamplerLow",
                "EventUniversalMath_EventVAEDecodeTiled",
                "EventUniversalMath_EventVideoCombine",
                "EventUniversalMath_EventModelShift_high",
                "EventUniversalMath_EventModelShift_low",
                "EventUniversalMath_EventCleanupBetweenSamplers",
                "EventUniversalMath_EventVideoSaveBegin",
                "EventUniversalBoundary_EventCascadeBoundary",
                "EventCoreBodyInit",
                "EventOneNodePolicy",
                "EventCoreBodyConsistencyAudit",
                "EventCoreBodyStageOrderAudit",
                "EventCoreBodyCompletionGate",
                "EventCoreBodySummary",
                "EventCoreBodyReportCard",
                "EventCoreBodyFinalize",
            ],
        })
        execution_records.append({
            "stage": "EventWorkflowBinding",
            "status": "recorded",
            "runtime_class": self.__class__.__name__,
            "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
            "runtime_name": EVENT_HORIZON_RUNTIME_NAME,
            "alias_count": len(runtime_aliases),
            "aliases": runtime_aliases,
            "compatibility": {
                "EventHorizon": bool("EventHorizon" in runtime_aliases),
            },
            "formula": "The public workflow node alias resolves to one internal Event Core Body runtime implementation.",
        })
        input_normalization = getattr(self, "_event_input_normalization", None)
        if isinstance(input_normalization, dict):
            adjustments = input_normalization.get("adjustments", [])
            reason_histogram = {}
            for adj in adjustments:
                if not isinstance(adj, dict):
                    continue
                reason = str(adj.get("reason", "unknown") or "unknown")
                reason_histogram[reason] = int(reason_histogram.get(reason, 0)) + 1
            execution_records.append({
                "stage": "EventInputNormalization",
                "status": "recorded",
                "normalized_values": input_normalization.get("normalized", {}),
                "adjustment_count": len(adjustments),
                "adjustment_reason_histogram": reason_histogram,
                "normalized_signature": input_normalization.get("normalized_signature", ""),
                "formula": "input configuration is normalized before stage execution to preserve deterministic route semantics",
            })
            if adjustments:
                execution_records.append({
                    "stage": "EventInputNormalizationAdjustments",
                    "status": "recorded",
                    "adjustments": adjustments,
                })
            self._event_core_live_record(
                "EventInputNormalization",
                record_type="manual",
                status="recorded",
                formula_role="ConfigGuard",
                route_id="route_input_normalization",
                observed_behavior="Input values were normalized before generation.",
                metadata={
                    "adjustment_count": len(adjustments),
                    "normalized_signature": input_normalization.get("normalized_signature", ""),
                },
            )
        self._event_control_warning(
            execution_records,
            getattr(self, "_event_math_control_mode", "OBSERVE_ONLY"),
            getattr(self, "_event_delta_strengths", {}).get("high", 1.0),
            getattr(self, "_event_delta_strengths", {}).get("low", 1.0),
        )
        execution_records.append({
            "stage": "EventMathControlSummary",
            "status": "recorded",
            "math_control_mode": getattr(self, "_event_math_control_mode", "OBSERVE_ONLY"),
            "high_delta_strength_requested": getattr(self, "_event_delta_strengths", {}).get("high", 1.0),
            "low_delta_strength_requested": getattr(self, "_event_delta_strengths", {}).get("low", 1.0),
            "sampler_trace_mode": getattr(self, "_event_sampler_trace", {}).get("mode", "OFF"),
            "sampler_trace_max_steps": getattr(self, "_event_sampler_trace", {}).get("max_steps", 64),
            "active_generation_math_path": (
                "semantic_overlay_native_sampler"
                if str(getattr(self, "_event_math_control_mode", "OBSERVE_ONLY")).upper() == "LATENT_DELTA_SCALE"
                else "native_step_loop" if str(getattr(self, "_event_math_control_mode", "OBSERVE_ONLY")).upper() == "DEEP_STEP_DELTA_CONTROL"
                else "boundary_sampler_wrapper"
            ),
            "strategy_carrier_coupling": "enabled_for_low_branch_without_step_replacement",
            "precision_step": 0.0001,
            "precision_round": 0.0001,
        })
        execution_records.append({
            "stage": "EventUniversalPipelineMap",
            "status": "recorded",
            "formula": "Every internal stage is read as NodeInputState + NodeObservedBehavior = NodeSState = NodeOutputState",
            "route": [
                "EventTextEncodePositive",
                "EventTextEncodeNegative",
                "EventImageScaleStart",
                "EventWanImageToVideoSeed",
                "EventModelShiftHigh",
                "EventSamplerHigh",
                "EventCleanupBetweenSamplers",
                "EventModelShiftLow",
                "EventSamplerLow",
                "EventVAEDecodeTiled",
                "EventHorizonCascadeBoundary",
                "EventVideoSaveBegin",
                "EventVideoCombine",
                "EventSaveReport",
                "EventCleanupAfterGeneration",
            ],
            "status_note": "r44 keeps the Event Core Body inside one Event Horizon node; EVENT_PACKET/S-Wire are internal runtime body, not a manual visual graph",
        })
        execution_records.append({
            "stage": "EventOneNodePolicy",
            "status": "active",
            "external_visual_node": "Event Horizon",
            "manual_event_graph_required": False,
            "internal_body": [
                "EventPacket",
                "FormulaReader",
                "RoleResolver",
                "RouteMemory",
                "SState",
                "UniversalStageMath",
                "UniversalBoundaryMath",
                "ReportBuilder",
            ],
            "reason": "Avoids manual ComfyUI workflow errors, wire-order drift, lost route memory, and mismatched Event logic between hidden stages.",
        })
        self._event_core_live_record(
            "EventOneNodePolicy",
            record_type="manual",
            status="active",
            formula_role="ArchitectureBoundary",
            route_id="route_one_node_policy",
            observed_behavior="One external node owns the internal Runtime Layer and Event Core Body.",
        )

        prompt_key_source = (
            str(positive_prompt or "") + "\n---NEGATIVE---\n" + str(negative_prompt or "") +
            f"\nclip={type(clip).__name__}:{id(clip)}"
        )
        encoder_cache_key = hashlib.sha256(prompt_key_source.encode("utf-8", errors="ignore")).hexdigest()[:24]
        execution_records.append({
            "stage": "EventRuntimeLayerProbes",
            "status": "observer_only",
            "runtime_monitor": "active",
            "compile_guard": "observed_not_compiling",
            "encoder_cache": "key_computed_cache_disabled_until_equivalence_proof",
            "branch_barrier": "smart branch barrier records preserved StrategyCarrier and released memory actions per phase",
            "lora_matrix_switch": "external_model_clip_route_observed_no_internal_switch",
            "prompt_operator_panel": "prompt fields observed as StrategyCandidate carriers; no prompt rewrite",
            "test_runner": "not active inside generation node; report data is structured for external runner",
            "universal_input_normalization": "readers and RoleResolver remain internal; fixed Wan interface still current",
            "encoder_cache_key_preview": encoder_cache_key,
            "formula": "Forgotten runtime-layer ideas are present as internal observer records before active intervention.",
        })
        self._event_core_live_record(
            "EventRuntimeLayerProbes",
            record_type="manual",
            status="observer_only",
            formula_role="Infrastructure / ObservedBehavior extension",
            route_id="route_runtime_layer_probes",
            observed_behavior="Runtime layer probes were recorded without changing generation tensors.",
            metadata={"encoder_cache_key_preview": encoder_cache_key},
        )

        saved_video_path = ""
        saved_report_path = ""
        video_ui_payload = None
        generated_frames = None
        generated_latent = None
        source_preview = None
        result_preview = None
        result_status = "NONE"
        failure_reason = ""

        # r14d compatibility guard: old saved workflows or stale node widgets
        # must not crash if output_target is missing during execution.
        if "output_target" not in locals() or output_target is None:
            output_target = "USER_D_AI_NSFW"

        # Internal image picker: if external IMAGE socket is not connected, use source_image_file.
        uploaded_image = None
        if image is None:
            uploaded_image = self._load_image_from_upload(source_image_file, execution_records)
            if uploaded_image is not None:
                image = uploaded_image
        source_preview = self._representative_preview_frame(image, width, height, mode="first")

        requested_video = generation_target in ("VIDEO", "AUTO")
        requested_image = generation_target in ("IMAGE", "AUTO")

        dual_available = secondary_model is not None
        if branch_mode == "SINGLE":
            active_branch_mode = "SINGLE"
        elif branch_mode == "DUAL_HIGH_LOW":
            active_branch_mode = "DUAL_HIGH_LOW" if dual_available else "SINGLE_FALLBACK_SECONDARY_MISSING"
        else:
            active_branch_mode = "DUAL_HIGH_LOW" if dual_available else "SINGLE"

        mode_to_count = {
            "SOLO_1": 1,
            "CASCADE_2": 2,
            "CASCADE_3": 3,
            "CASCADE_4": 4,
            "CASCADE_5": 5,
        }
        mode_count = int(mode_to_count.get(str(cascade_mode), int(cascade_count or 1)))
        numeric_count = int(cascade_count or mode_count or 1)
        requested_cascade_count = max(mode_count, numeric_count)
        requested_cascade_count = max(1, min(5, requested_cascade_count))
        frames = int(frames_per_cascade or frames)

        packet = make_event_packet(metadata={
            "created_by": "EventHorizonCascade",
            "version": EVENT_HORIZON_RUNTIME_VERSION,
            "run_id": run_id,
            "node_role": "terminal_event_horizon_cascade",
        })

        packet = self._event_core_body_init(packet, execution_records, run_id, route_name="wan_terminal_one_node")

        conflict_ids = []
        relation_ids = []
        signal_ids = []

        packet = record_stage(
            packet,
            stage_name="WanEventWorkflowCore",
            action="INIT_EVENT_HORIZON_CASCADE_WORKFLOW",
            observed_behavior="Terminal-first Event Horizon workflow initialized",
            metadata={
                "run_id": run_id,
                "generation_target": generation_target,
                "cascade_mode": cascade_mode,
                "cascade_count": requested_cascade_count,
                "frames_per_cascade": frames,
                "terminal_mode": terminal_mode,
                "enable_continuation_outputs": enable_continuation_outputs,
                "execution_mode": execution_mode,
                "active_branch_mode": active_branch_mode,
            },
        )

        packet["metadata"]["event_program_status"] = {
            "current_mode": execution_mode,
            "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
            "runtime_name": EVENT_HORIZON_RUNTIME_NAME,
            "math_report_enabled": True,
            "terminal_mode": bool(terminal_mode),
            "generation_target": generation_target,
            "result_status": result_status,
            "program_state": "terminal_save_first_generation_body",
            "final_target_output": "saved_video_or_saved_frames",
            "normal_outputs": "status_paths_report_only_no_image_sockets",
            "continuation_outputs": "removed_from_main_terminal_node_in_r15_use_future_extractor_node",
            "sampler_replacement": "EventSamplerCore boundary replacement; native step loop pending",
            "no_fake_success_rule": "source image is never presented as successful VIDEO result",
        }

        packet["metadata"]["terminal_ui_model"] = {
            "start_preview": "input image",
            "result_preview": "generated frames/video representative frame if available",
            "save_first": True,
            "default_target": "VIDEO",
        }

        packet["metadata"]["wan_workflow_interface"] = {
            "interface_variant": "terminal_B_no_lora_strings",
            "source_workflow": "Event Horizon / Wan2.2 I2V Quant 14B",
            "lora_policy": "LoRA is applied outside; node receives already patched MODEL/CLIP.",
            "required_external": ["primary_model", "clip", "vae", "source_image_file or image for VIDEO"],
            "optional_external": ["secondary_model", "mask", "external image socket"],
            "width": width,
            "height": height,
            "frames": frames,
            "batch_size": batch_size,
            "fps": fps,
            "seed": seed,
            "generation_target": generation_target,
            "video_format": video_format,
            "force_vhs_video_combine": force_vhs_video_combine,
            "output_target": output_target,
            "output_folder_mode": output_folder_mode,
            "output_folder": output_folder,
            "custom_output_folder": custom_output_folder,
            "preferred_video_dir": r"D:\AI NSFW\VID",
            "preferred_image_dir": r"D:\AI NSFW\PIC",
            "preview_policy": "source preview only; result last-frame only when continuation is enabled",
            "event_strategy_note": "optional future control field; currently stored in Event Program report, not required for animation",
            "execution_mode": execution_mode,
            "branch_mode_requested": branch_mode,
            "branch_mode_active": active_branch_mode,
            "cleanup_timing": cleanup_timing,
            "ksampler_replacement": "EventSamplerCore",
            "ksampler_windows": {
                "primary": {"start": primary_start_step, "end": primary_end_step, "add_noise": "enable", "return_leftover_noise": "enable"},
                "secondary": {"start": secondary_start_step, "end": secondary_end_step, "add_noise": "disable", "return_leftover_noise": "disable"},
                "global_steps": global_steps,
            },
        }

        packet["metadata"]["wan_event_internal_topology"] = {
            "universal_event_node_rule": "every internal stage must pass through an Event Horizon; relation center is Event Singularity",
            "event_horizon_model": {
                "EventHorizon": "boundary layer where a technical node input/output transition becomes Event-Horizon-aware",
                "EventSingularity": "EventSingularity center where input state, observed behavior, and output state collapse into one relation point",
                "formula": "NodeInputState + NodeObservedBehavior = EventSingularity = NodeOutputState"
            },
            "terminal_route": [
                "EventTextEncodePositive",
                "EventTextEncodeNegative",
                "EventImageScaleStart",
                "EventWanImageToVideoSeed",
                "EventModelShiftHigh",
                "EventSamplerHigh",
                "EventCleanupBetweenSamplers",
                "EventModelShiftLow",
                "EventSamplerLow",
                "EventVAEDecodeTiled",
                "EventVideoCombine",
                "EventSaveReport",
            ],
            "internal_hidden_states": [
                "conditioning_positive",
                "conditioning_negative",
                "wan_latent_seed",
                "latent_before_high",
                "latent_after_high",
                "delta_high",
                "latent_before_low",
                "latent_after_low",
                "delta_low",
                "decoded_frames",
                "video_file",
                "S0_wan_terminal",
            ],
        }

        if str(positive_prompt or "").strip():
            packet, sig, proj, conf = self._make_text_signal(packet, positive_prompt, "StrategyCurrent", "wan_positive_prompt_route", "EventTextEncodePositive")
            signal_ids.append(sig["id"])
            conflict_ids.extend(conf)
        if str(negative_prompt or "").strip():
            packet, sig, proj, conf = self._make_text_signal(packet, negative_prompt, "NegativeConstraint", "wan_negative_prompt_route", "EventTextEncodeNegative")
            signal_ids.append(sig["id"])
            conflict_ids.extend(conf)
        if str(event_strategy or "").strip():
            packet, sig, proj, conf = self._make_text_signal(packet, event_strategy, "EventProgramStrategy", "wan_event_strategy_route", "Wan_event_strategy")
            signal_ids.append(sig["id"])
            conflict_ids.extend(conf)

        if image is not None:
            packet, sig, proj, conf = _read_signal(
                packet, TECH_IMAGE, SPACE_IMAGE, image, "EventSourcePreview", "SourceAnchor",
                "wan_source_image_route", "ImageOutcomeReader", route_position="start_preview"
            )
            signal_ids.append(sig["id"])
            conflict_ids.extend(conf)

        if generation_target == "VIDEO" and image is None:
            failure_reason = "VIDEO target requires source image for WanImageToVideo; no image was connected."
            result_status = "FAILED"
            packet, cid = self._fail_conflict(packet, "WanEventWorkflowCore/InputValidation", failure_reason, {"generation_target": generation_target})
            conflict_ids.append(cid)

        if secondary_model is None and branch_mode == "DUAL_HIGH_LOW":
            conflict = make_conflict(
                CONFLICT_FROZEN_ROUTE_MISSING,
                severity=SEV_LOW,
                stage_position="WanEventWorkflowCore",
                suspected_cause="DUAL_HIGH_LOW requested but secondary_model was not connected.",
                observed_symptom="secondary branch disabled; falling back to single branch.",
                suggested_response="Connect secondary_model for Low branch, or set branch_mode=SINGLE/AUTO.",
                metadata={"route": "secondary_model", "fallback": "single_branch"},
            )
            packet = add_conflict(packet, conflict)
            conflict_ids.append(conflict["id"])

        _, _ = self._run_branch_barrier(
            phase_name="BEFORE_GENERATION",
            cleanup_timing=cleanup_timing,
            strategy_state=None,
            strategy_label="before_generation_context",
            records=execution_records,
            cleanup_records=cleanup_records,
            barrier_records=branch_barrier_records,
            route_id="route_cleanup_before_generation",
            stage_name="EventCleanupBeforeGeneration",
            formula_role="MEMORY pressure -> MEMORY prepared state",
            observed_behavior="Smart Branch Barrier before generation releases disposable memory without mutating generation tensors.",
            next_requirement="generation starts with reduced stale memory pressure",
            use_custom_cleanup_nodes=use_custom_cleanup_nodes,
            use_custom_chain=False,
            cleanup_label="before_generation",
        )

        if execution_mode in ("RUN", "TRY_EVENT_HORIZON_CASCADE", "TRY_WAN_SOLO_EXECUTION") and result_status != "FAILED":
            execution_records.append({
                "stage": "EventHorizonCascadeExecutionGate",
                "status": "running",
                "execution_mode": execution_mode,
                "cascade_count": requested_cascade_count,
                "frames_per_cascade": frames,
            })
            try:
                if image is None:
                    raise RuntimeError("Event Horizon VIDEO route requires source image input.")

                positive = self._encode_text(clip, positive_prompt, execution_records, label='TextEncodePositive')
                negative = self._encode_text(clip, negative_prompt, execution_records, label='TextEncodeNegative')
                execution_records.append({"stage": "EventTextEncode", "status": "ok", "formula": "prompt + CLIP behavior = S_text = conditioning"})
                self._stage_delay(stage_delay_seconds, execution_records, "after_text_encode")

                scaled_image = self._scale_image(image, width, height, image_upscale_method, image_crop, execution_records)

                wan_positive, wan_negative, wan_latent = self._wan_image_to_video(
                    positive, negative, vae, scaled_image, width, height, frames, batch_size, execution_records
                )
                self._stage_delay(stage_delay_seconds, execution_records, "after_wan_image_to_video")

                packet, wan_latent_sig, wan_latent_proj, conf = _read_signal(
                    packet, TECH_LATENT, SPACE_LATENT, wan_latent,
                    "EventWanImageToVideoSeed", "OutcomePrevious",
                    "wan_i2v_latent_seed_route", "LatentEventReader", route_position="wan_i2v_latent_seed"
                )
                conflict_ids.extend(conf)

                high_model = self._apply_sd3_shift(primary_model, primary_sd3_shift, "high", execution_records)

                high_window = EventSamplerWindow(
                    branch_name="high",
                    branch_role="coarse_motion_global_trajectory",
                    seed=int(seed),
                    steps=int(global_steps),
                    cfg=float(primary_cfg),
                    sampler_name=str(sampler_name),
                    scheduler=str(scheduler),
                    start_at_step=int(primary_start_step),
                    end_at_step=int(primary_end_step),
                    add_noise="enable",
                    return_with_leftover_noise="enable",
                    sd3_shift=float(primary_sd3_shift),
                )

                latent_after_high, high_result = self._event_sample_window(
                    high_model, wan_positive, wan_negative, wan_latent, high_window, execution_records
                )
                self._stage_delay(stage_delay_seconds, execution_records, "after_high_sampler")
                packet["metadata"].setdefault("event_sampler_results", {})["high"] = high_result.to_metadata()

                packet, high_sig, high_proj, conf = _read_signal(
                    packet, TECH_LATENT, SPACE_LATENT, latent_after_high,
                    "EventSamplerHigh_latent_after", "OutcomeNext",
                    "wan_high_after_route", "LatentEventReader", route_position="after_event_sampler_high"
                )
                conflict_ids.extend(conf)

                delta_high, delta_err = compute_tensor_delta(wan_latent, latent_after_high)
                if delta_high is not None:
                    packet, delta_sig, delta_proj, conf = _read_signal(
                        packet, TECH_DELTA, SPACE_DELTA, delta_high,
                        "EventSamplerHigh_delta", "ObservedBehaviorCurrent",
                        "wan_high_delta_route", "DeltaReader",
                        metadata={"before_signal_id": wan_latent_sig["id"], "after_signal_id": high_sig["id"], "before_ref": extract_latent_samples(wan_latent)}
                    )
                    conflict_ids.extend(conf)
                    rel = make_event_relation(
                        relation_type=REL_TRANSFORMS_INTO,
                        source_signal_ids=[wan_latent_sig["id"], delta_sig["id"]],
                        target_signal_ids=[high_sig["id"]],
                        source_projection_ids=[wan_latent_proj["id"], delta_proj["id"]],
                        target_projection_ids=[high_proj["id"]],
                        operator_name="EventSamplerCore",
                        formula_meaning="EventSamplerHigh transforms Wan latent seed into high/noise motion latent",
                        local_strategy_id="S0_wan_terminal.high_sampler",
                        equality_status=EQ_UNKNOWN,
                        metadata={"branch": "high", "sampler_replacement": "EventSamplerCore"},
                    )
                    packet = add_relation(packet, rel)
                    relation_ids.append(rel["id"])
                else:
                    execution_records.append({"stage": "EventSamplerHigh_delta", "status": "unavailable", "error": str(delta_err)})

                final_latent = latent_after_high
                delta_low = None

                if active_branch_mode == "DUAL_HIGH_LOW" and secondary_model is not None:
                    latent_before_low = latent_after_high

                    latent_before_low, _ = self._run_branch_barrier(
                        phase_name="BETWEEN_SAMPLERS",
                        cleanup_timing=cleanup_timing,
                        strategy_state=latent_before_low,
                        strategy_label="strategy_carrier_low",
                        records=execution_records,
                        cleanup_records=cleanup_records,
                        barrier_records=branch_barrier_records,
                        route_id="route_cleanup_between_samplers",
                        stage_name="EventCleanupBetweenSamplers",
                        formula_role="LATENT high output + cleanup behavior -> LATENT low input",
                        observed_behavior="Smart Branch Barrier between samplers preserves high latent StrategyCarrier while releasing disposable memory.",
                        next_requirement="low sampler receives preserved high output latent",
                        use_custom_cleanup_nodes=use_custom_cleanup_nodes,
                        use_custom_chain=True,
                        cleanup_label="between_high_low",
                    )

                    packet, low_before_sig, low_before_proj, conf = _read_signal(
                        packet, TECH_LATENT, SPACE_LATENT, latent_before_low,
                        "EventSamplerLow_latent_before", "OutcomePrevious",
                        "wan_low_before_route", "LatentEventReader", route_position="before_event_sampler_low"
                    )
                    conflict_ids.extend(conf)

                    low_model = self._apply_sd3_shift(secondary_model, secondary_sd3_shift, "low", execution_records)

                    low_window = EventSamplerWindow(
                        branch_name="low",
                        branch_role="refinement_detail_stabilization",
                        seed=int(seed),
                        steps=int(global_steps),
                        cfg=float(secondary_cfg),
                        sampler_name=str(sampler_name),
                        scheduler=str(scheduler),
                        start_at_step=int(secondary_start_step),
                        end_at_step=int(secondary_end_step),
                        add_noise="disable",
                        return_with_leftover_noise="disable",
                        sd3_shift=float(secondary_sd3_shift),
                    )

                    latent_after_low, low_result = self._event_sample_window(
                        low_model, wan_positive, wan_negative, latent_before_low, low_window, execution_records
                    )
                    self._stage_delay(stage_delay_seconds, execution_records, "after_low_sampler")
                    packet["metadata"].setdefault("event_sampler_results", {})["low"] = low_result.to_metadata()

                    packet, low_after_sig, low_after_proj, conf = _read_signal(
                        packet, TECH_LATENT, SPACE_LATENT, latent_after_low,
                        "EventSamplerLow_latent_after", "OutcomeNext",
                        "wan_low_after_route", "LatentEventReader", route_position="after_event_sampler_low"
                    )
                    conflict_ids.extend(conf)

                    delta_low, delta_err = compute_tensor_delta(latent_before_low, latent_after_low)
                    if delta_low is not None:
                        packet, delta2_sig, delta2_proj, conf = _read_signal(
                            packet, TECH_DELTA, SPACE_DELTA, delta_low,
                            "EventSamplerLow_delta", "ObservedBehaviorCurrent",
                            "wan_low_delta_route", "DeltaReader",
                            metadata={"before_signal_id": low_before_sig["id"], "after_signal_id": low_after_sig["id"], "before_ref": extract_latent_samples(latent_before_low)}
                        )
                        conflict_ids.extend(conf)
                        rel = make_event_relation(
                            relation_type=REL_TRANSFORMS_INTO,
                            source_signal_ids=[low_before_sig["id"], delta2_sig["id"]],
                            target_signal_ids=[low_after_sig["id"]],
                            source_projection_ids=[low_before_proj["id"], delta2_proj["id"]],
                            target_projection_ids=[low_after_proj["id"]],
                            operator_name="EventSamplerCore",
                            formula_meaning="EventSamplerLow refines high latent into low/detail final latent",
                            local_strategy_id="S0_wan_terminal.low_sampler",
                            equality_status=EQ_UNKNOWN,
                            metadata={"branch": "low", "sampler_replacement": "EventSamplerCore"},
                        )
                        packet = add_relation(packet, rel)
                        relation_ids.append(rel["id"])
                    else:
                        execution_records.append({"stage": "EventSamplerLow_delta", "status": "unavailable", "error": str(delta_err)})

                    final_latent = latent_after_low

                self._dual_branch_delta_coupling_math(
                    delta_high=delta_high,
                    delta_low=delta_low,
                    records=execution_records,
                    active_branch_mode=active_branch_mode,
                    cascade_count=requested_cascade_count,
                )

                generated_latent = final_latent

                generated_frames = self._decode_tiled(
                    vae, generated_latent,
                    decode_tile_size, decode_overlap, decode_temporal_size, decode_temporal_overlap,
                    execution_records
                )
                self._math_tensor_summary(generated_frames, execution_records, "EventMath_decoded_frames", strict=False)
                self._frame_motion_math(generated_frames, execution_records, "EventMath_decoded_frame_motion")
                self._stage_delay(stage_delay_seconds, execution_records, "after_decode")

                packet, img_sig, img_proj, conf = _read_signal(
                    packet, TECH_IMAGE, SPACE_IMAGE, generated_frames,
                    "EventVAEDecodeTiled_frames", "VisibleOutcome",
                "wan_decoded_frames_route", "ImageOutcomeReader", route_position="decoded_frames"
                )
                conflict_ids.extend(conf)

                # Event Horizon extension.
                self._pause_flag_triggered = False
                if resume_frame_index != -1 and _EVENT_HORIZON_CASCADE_CACHE["latent"] is not None:
                    # Resuming from a pause
                    generated_latent = _EVENT_HORIZON_CASCADE_CACHE["latent"]
                    generated_frames = _EVENT_HORIZON_CASCADE_CACHE["frames"]
                    
                    # Trim the latent tensor in the temporal dimension T.
                    # For Wan2.1, T = (frames - 1) // 4 + 1
                    target_t = max(1, (resume_frame_index - 1) // 4 + 1)
                    if generated_latent['samples'].shape[2] > target_t:
                        generated_latent['samples'] = generated_latent['samples'][:, :, :target_t, :, :]
                    
                    if generated_frames is not None and generated_frames.shape[0] > resume_frame_index:
                        generated_frames = generated_frames[:resume_frame_index, :, :, :]
                        
                    _EVENT_HORIZON_CASCADE_CACHE["latent"] = generated_latent
                    _EVENT_HORIZON_CASCADE_CACHE["frames"] = generated_frames
                        
                    segment_batches = [generated_frames]
                    current_cascade_image = self._last_frame_image(generated_frames, width, height)
                    
                    start_segment = _EVENT_HORIZON_CASCADE_CACHE["segment_index"] + 1
                    execution_records.append({"stage": "EventHorizonCascadeResume", "status": "resumed", "resume_frame_index": resume_frame_index, "start_segment": start_segment})
                    
                    packet, mirror_rel, mirror_sstate = self._record_mirror_cut(
                        packet, resume_frame_index, target_t, start_segment, execution_records
                    )
                    _EVENT_HORIZON_CASCADE_CACHE["pending_mirror_cut_rel_id"] = mirror_rel["id"]
                    
                    packet, cut_latent_sig, cut_latent_proj, _ = _read_signal(
                        packet, TECH_LATENT, SPACE_LATENT, generated_latent,
                        f"MirrorCut_{start_segment}_StrategyCarrier", "StrategyCarrier",
                        f"route_mirror_cut_{start_segment}", "MirrorCutReader"
                    )
                    packet = self._complete_mirror_cut_relation(packet, cut_latent_sig["id"])
                else:
                    segment_batches = [generated_frames]
                    current_cascade_image = self._last_frame_image(generated_frames, width, height)
                    start_segment = 2

                # Handle pause at cascade 1
                if pause_after_cascade_1 and start_segment == 2 and resume_frame_index == -1:
                    _EVENT_HORIZON_CASCADE_CACHE["latent"] = generated_latent
                    _EVENT_HORIZON_CASCADE_CACHE["frames"] = generated_frames
                    _EVENT_HORIZON_CASCADE_CACHE["segment_index"] = 1
                    execution_records.append({"stage": "EventHorizonCascadePause_1", "status": "paused"})
                    self._pause_flag_triggered = True
                    # Skip the cascade loop
                    requested_cascade_count = 1

                if requested_cascade_count >= start_segment:
                    execution_records.append({
                        "stage": "EventHorizonCascadeBegin",
                        "status": "begin",
                        "cascade_count": requested_cascade_count,
                        "frames_per_cascade": frames,
                    })
                    for segment_index in range(start_segment, requested_cascade_count + 1):
                        next_frames, current_cascade_image, generated_latent = self._run_event_horizon_segment_core(
                            segment_index=segment_index,
                            source_image=current_cascade_image,
                            primary_model=primary_model,
                            secondary_model=secondary_model,
                            clip=clip,
                            vae=vae,
                            positive_prompt=positive_prompt,
                            negative_prompt=negative_prompt,
                            active_branch_mode=active_branch_mode,
                            width=width,
                            height=height,
                            frames=frames,
                            batch_size=batch_size,
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
                            primary_sd3_shift=primary_sd3_shift,
                            secondary_sd3_shift=secondary_sd3_shift,
                            decode_tile_size=decode_tile_size,
                            decode_overlap=decode_overlap,
                            decode_temporal_size=decode_temporal_size,
                            decode_temporal_overlap=decode_temporal_overlap,
                            image_upscale_method=image_upscale_method,
                            image_crop=image_crop,
                            cleanup_timing=cleanup_timing,
                            use_custom_cleanup_nodes=use_custom_cleanup_nodes,
                            stage_delay_seconds=stage_delay_seconds,
                            records=execution_records,
                            cleanup_records=cleanup_records,
                            barrier_records=branch_barrier_records,
                        )
                        if segment_index > 1:
                            next_frames = self._drop_first_frame_batch(next_frames, execution_records, segment_index)
                        self._cascade_boundary_math(segment_batches[-1], next_frames, execution_records, segment_index)
                        segment_batches.append(next_frames)
                        
                        pause_flags = {
                            2: pause_after_cascade_2,
                            3: pause_after_cascade_3,
                            4: pause_after_cascade_4
                        }
                        if pause_flags.get(segment_index, False):
                            _EVENT_HORIZON_CASCADE_CACHE["latent"] = generated_latent
                            _EVENT_HORIZON_CASCADE_CACHE["frames"] = self._concat_frame_batches(segment_batches, execution_records)
                            _EVENT_HORIZON_CASCADE_CACHE["segment_index"] = segment_index
                            execution_records.append({"stage": f"EventHorizonCascadePause_{segment_index}", "status": "paused"})
                            self._pause_flag_triggered = True
                            break

                    generated_frames = self._concat_frame_batches(segment_batches, execution_records)
                    self._frame_motion_math(generated_frames, execution_records, "EventMath_concatenated_frame_motion")
                    execution_records.append({
                        "stage": "EventHorizonCascadeEnd",
                        "status": "ok",
                        "segments": requested_cascade_count,
                        "total_requested_frames": requested_cascade_count * int(frames),
                    })
                    self._event_universal_stage_math(
                        execution_records,
                        "EventHorizonCascadeEnd",
                        input_state=segment_batches[0] if segment_batches else None,
                        output_state=generated_frames,
                        observed_behavior="multiple cascade frame batches concatenated into one continuous output sequence",
                        formula_role="FRAME_BATCHES segment outcomes -> FRAMES full cascade outcome",
                        route_id="route_cascade_concat",
                        next_requirement="video combine requires one ordered frame batch",
                        control_mode="REPORT_ONLY",
                        metadata={"segments": requested_cascade_count, "frames_per_cascade": int(frames)},
                    )

                if save_frames or save_output_image or generation_target == "IMAGE":
                    try:
                        self._save_image_attempt(generated_frames, save_prefix, execution_records)
                    except Exception as e:
                        execution_records.append({"stage": "EventSaveFrames", "status": "failed", "error": str(e)})

                if requested_video and save_video:
                    try:
                        saved_video_path = self._video_combine_attempt(generated_frames, fps, save_prefix, execution_records, video_format=video_format, force_vhs=force_vhs_video_combine, output_target=output_target, output_folder_mode=output_folder_mode, output_folder=output_folder, custom_output_folder=custom_output_folder)
                        result_status = "VIDEO"
                    except Exception as e:
                        execution_records.append({"stage": "EventVideoCombine", "status": "failed", "error": str(e)})
                        if generation_target == "VIDEO":
                            result_status = "FAILED"
                            failure_reason = f"VIDEO target requested but video combine failed: {e}"
                            packet, cid = self._fail_conflict(packet, "EventVideoCombine", failure_reason, {"generation_target": generation_target})
                            conflict_ids.append(cid)
                        else:
                            result_status = "FRAMES"
                else:
                    result_status = "FRAMES" if generated_frames is not None else "NONE"

                ui_images = []
                if getattr(self, "_pause_flag_triggered", False):
                    result_status = "PAUSED"
                    if generated_frames is not None:
                        ui_images = self._save_pause_frames_to_temp(generated_frames)

                video_ui_payload = self._extract_vhs_ui_payload_from_records(execution_records)

                generated_frames, _ = self._run_branch_barrier(
                    phase_name="AFTER_GENERATION",
                    cleanup_timing=cleanup_timing,
                    strategy_state=generated_frames,
                    strategy_label="final_frames_outcome",
                    records=execution_records,
                    cleanup_records=cleanup_records,
                    barrier_records=branch_barrier_records,
                    route_id="route_cleanup_after_generation",
                    stage_name="EventCleanupAfterGeneration",
                    formula_role="OUTPUT frames + cleanup behavior -> saved/previewable output state",
                    observed_behavior="Smart Branch Barrier after generation preserves final frames while releasing disposable memory.",
                    next_requirement="report/output finalization continues after cleanup",
                    use_custom_cleanup_nodes=use_custom_cleanup_nodes,
                    use_custom_chain=False,
                    cleanup_label="after_generation",
                )

            except Exception as e:
                failure_reason = str(e)
                result_status = "FAILED"
                execution_records.append({"stage": "terminal_wan_execution", "status": "failed", "error": failure_reason})
                packet, cid = self._fail_conflict(packet, "WanEventWorkflowCore/Execution", failure_reason, {"execution_mode": execution_mode, "generation_target": generation_target})
                conflict_ids.append(cid)

        elif execution_mode == "REPORT_ONLY":
            result_status = "INCOMPLETE"
            execution_records.append({"stage": "execution", "status": "skipped_report_only"})
        else:
            result_status = "INCOMPLETE"
            execution_records.append({
                "stage": "execution",
                "status": "skipped_unsupported_execution_mode",
                "execution_mode": execution_mode,
                "expected_modes": ["RUN", "TRY_EVENT_HORIZON_CASCADE", "TRY_WAN_SOLO_EXECUTION", "REPORT_ONLY"],
            })

        # r12 safety: saved_video_path must be a clean string path, never raw VHS payload.
        if not isinstance(saved_video_path, str):
            raw_saved_video_payload = str(saved_video_path)[:1000]
            try:
                saved_video_path = self._extract_path_from_video_result(saved_video_path, save_prefix, execution_records)
            except Exception as e:
                execution_records.append({"stage": "EventVideoPathFinalCoerce", "status": "failed", "error": str(e), "raw_payload": raw_saved_video_payload})
                saved_video_path = ""
            else:
                execution_records.append({"stage": "EventVideoPathFinalCoerce", "status": "ok", "path": saved_video_path, "raw_payload": raw_saved_video_payload})

        barrier_phase_counts = {}
        barrier_with_strategy = 0
        barrier_preserved = 0
        for item in branch_barrier_records:
            phase = str(item.get("barrier_phase", "unknown"))
            barrier_phase_counts[phase] = barrier_phase_counts.get(phase, 0) + 1
            if item.get("strategy_state_required"):
                barrier_with_strategy += 1
                if item.get("strategy_state_preserved"):
                    barrier_preserved += 1

        packet["metadata"]["cleanup_records"] = cleanup_records
        packet["metadata"]["branch_barrier_records"] = branch_barrier_records
        packet["metadata"]["branch_barrier_summary"] = {
            "record_count": len(branch_barrier_records),
            "phase_counts": barrier_phase_counts,
            "strategy_state_checks": barrier_with_strategy,
            "strategy_state_preserved_count": barrier_preserved,
            "observer_only": True,
        }
        packet["metadata"]["execution_records"] = execution_records
        packet["metadata"]["result_status"] = {
            "generation_target": generation_target,
            "result_status": result_status,
            "failure_reason": failure_reason,
            "saved_video_path": saved_video_path,
            "fake_success_prevented": True,
            "source_image_was_not_returned_as_successful_video": True,
            "source_image_file": str(source_image_file) if source_image_file else "",
        }
        packet["metadata"]["program_output_policy"] = {
            "terminal_mode": bool(terminal_mode),
            "normal_result": "saved files + status, no IMAGE outputs",
            "continuation_outputs": "removed from main terminal node in r15",
            "image_passthrough_policy": "never claim source image as generated VIDEO result",
            "video_required_for_success": generation_target == "VIDEO",
        }
        # update program status with final result
        packet["metadata"]["event_program_status"]["result_status"] = result_status
        packet["metadata"]["event_program_status"]["saved_video_path"] = saved_video_path
        packet["metadata"]["event_program_status"]["failure_reason"] = failure_reason

        packet = self._event_core_body_finalize(packet, execution_records, result_status, saved_video_path, failure_reason)

        packet, sstate = build_sstate_from_packet(packet, position="S0_wan_terminal", active_relation_ids=relation_ids)

        packet = record_stage(
            packet,
            stage_name="WanEventWorkflowCore",
            action="BUILD_TERMINAL_WAN_REPORT",
            observed_behavior="Built terminal Wan Event Program report with save-first output policy",
            input_signal_ids=signal_ids,
            relation_ids=relation_ids,
            sstate_ids=[sstate["id"]] if sstate else [],
            conflict_ids=conflict_ids,
            formula_note="terminal external inputs -> internal Event-Horizon-aware subnodes -> EventSamplerCore high/low -> decoded frames/video -> saved result",
            metadata={
                "result_status": result_status,
                "failure_reason": failure_reason,
                "saved_video_path": saved_video_path,
                "execution_records": execution_records,
            },
        )

        if save_report:
            report = build_markdown_report(packet)
            if not str(report or "").strip():
                execution_records.append({
                    "stage": "EventSaveReportBuild",
                    "status": "empty_report_from_builder",
                    "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
                })
            else:
                execution_records.append({
                    "stage": "EventSaveReportBuild",
                    "status": "ok",
                    "chars": len(str(report)),
                    "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
                })
            try:
                saved_report_path = self._save_report_file(report, save_prefix, "COMFY_OUTPUT", "DEFAULT", "default", "")
                try:
                    report_size = Path(saved_report_path).stat().st_size
                except Exception:
                    report_size = -1
                execution_records.append({
                    "stage": "EventSaveReport",
                    "status": "standard_comfy_output_ok",
                    "path": saved_report_path,
                    "bytes": report_size,
                    "nonempty": bool(report_size and report_size > 0),
                    "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
                })
                if report_size <= 0:
                    raise RuntimeError(f"Report was saved but is empty: {saved_report_path}")
                sidecar_info = self._save_runtime_monitor_sidecars(packet, saved_report_path, saved_video_path, save_prefix)
                packet.setdefault("metadata", {})["runtime_monitor_sidecars"] = sidecar_info
                execution_records.append({
                    "stage": "EventRuntimeMonitorSidecars",
                    "status": sidecar_info.get("status", "unknown"),
                    "json_path": sidecar_info.get("json_path", ""),
                    "csv_path": sidecar_info.get("csv_path", ""),
                    "diff_path": sidecar_info.get("diff_path", ""),
                    "previous_json_path": sidecar_info.get("previous_json_path", ""),
                    "record_count": sidecar_info.get("record_count", 0),
                    "motion_stage": sidecar_info.get("motion_stage", ""),
                    "motion_profile": sidecar_info.get("motion_profile", ""),
                    "motion_stability_score": sidecar_info.get("motion_stability_score", ""),
                    "observer_only": True,
                    "formula": "Runtime Monitor writes machine-readable ObservedBehavior sidecars without changing generation.",
                })
                report = build_markdown_report(packet)
                report_size = self._rewrite_report_file(saved_report_path, report)
            except Exception as e:
                execution_records.append({"stage": "EventSaveReport", "status": "failed", "error": str(e), "runtime_version": EVENT_HORIZON_RUNTIME_VERSION})
                saved_report_path = ""
        else:
            report = ""
            saved_report_path = ""
            execution_records.append({"stage": "EventSaveReport", "status": "disabled_by_user", "runtime_version": EVENT_HORIZON_RUNTIME_VERSION})

        # Preview model:
        # - source_preview always shows the input/source.
        # - result_preview shows generated frames only if real generation produced frames.
        # - if generation failed, result_preview is a blank placeholder, not the source image.
        if enable_continuation_outputs and generated_frames is not None and result_status in ("FRAMES", "VIDEO"):
            result_preview = self._representative_preview_frame(generated_frames, width, height, mode="last")
        else:
            result_preview = self._placeholder_image(width, height)

        if not ui_images:
            ui_images = self._make_ui_previews(source_preview, result_preview, save_prefix, execution_records, include_result_preview=enable_continuation_outputs)
        packet["metadata"]["ui_preview"] = {
            "source_preview": "source image or upload",
            "result_preview": "disabled; no PreviewImage calls in terminal node",
            "continuation_seed_frame": "not emitted by main terminal node in r15; use future extractor/chain node",
            "ui_images_count": len(ui_images),
            "video_ui_payload_returned": bool(video_ui_payload),
        }

        status = (
            f"EventHorizon v{EVENT_HORIZON_RUNTIME_VERSION} | target={generation_target} | result={result_status} | "
            f"terminal={terminal_mode} | continuation={enable_continuation_outputs} | "
            f"video_path={saved_video_path or 'none'} | report_path={saved_report_path or 'none'}"
        )
        if failure_reason:
            status += f" | failure={failure_reason}"

        continuation_image = self._representative_preview_frame(generated_frames, width, height, mode="last") if (enable_continuation_outputs and generated_frames is not None) else self._placeholder_image(width, height)
        continuation_latent = generated_latent if enable_continuation_outputs else None
        continuation_packet = packet if enable_continuation_outputs else None

        result_tuple = (
            status,
            saved_video_path,
            saved_report_path,
            report,
        )

        if not video_ui_payload:
            video_ui_payload = {}
        if ui_images:
            if getattr(self, "_pause_flag_triggered", False):
                video_ui_payload = {} # Wipe VHS video payload to prevent overlap
            video_ui_payload["pause_frames"] = ui_images
            
        if video_ui_payload:
            return {"ui": video_ui_payload, "result": result_tuple}

        return result_tuple



class EventHorizonCascadeSimple(WanEventWorkflowCore):
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
                "resume_frame_index": ("INT", {"default": -1, "min": -1, "max": 4096}),
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

                "cleanup_timing": ([
                    "NONE", "BEFORE_GENERATION", "BETWEEN_SAMPLERS", "AFTER_GENERATION", "BEFORE_AND_AFTER", "ALL",
                    "none", "before_generation", "between_samplers", "after_generation", "before_and_after", "all",
                ], {"default": "ALL"}),
                "save_video": ("BOOLEAN", {"default": True}),
                "video_format": (["video/h264-mp4", "video/h265-mp4", "image/webp", "image/gif"], {"default": "video/h264-mp4"}),
                "save_report": ("BOOLEAN", {"default": True}),
                "save_prefix": ("STRING", {"default": "Event Horizon"}),
                "sampler_trace_mode": (["OFF", "SHADOW_STEP_TRACE"], {"default": "OFF"}),
                "sampler_trace_max_steps": ("INT", {"default": 64, "min": 1, "max": 65535}),
            },
            "optional": {
                "secondary_model": ("MODEL",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("status", "saved_video_path", "saved_report_path", "report")
    FUNCTION = "run"
    CATEGORY = "Event Equality/Event Horizon"
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
        cleanup_timing,
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

        def align_to_step(field, value, step, min_value):
            if step <= 1:
                return int(value)
            aligned = int(value) - (int(value) % int(step))
            if aligned < int(min_value):
                aligned = int(min_value)
            if aligned != int(value):
                self._append_input_adjustment(adjustments, field, value, aligned, f"aligned_to_step_{step}")
            return int(aligned)

        dual_branch = secondary_model is not None

        cascade_count_n = clamp_int("cascade_count", cascade_count, 1, 1, 5)
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

        if dual_branch:
            primary_start_max = max(0, global_steps_n - 2)
            primary_start_n = clamp_int("primary_start_step", primary_start_step, 0, 0, primary_start_max)
            primary_end_min = primary_start_n + 1
            primary_end_max = max(primary_end_min, global_steps_n - 1)
            primary_end_n = clamp_int("primary_end_step", primary_end_step, primary_end_min, primary_end_min, primary_end_max)
            secondary_start_n = primary_end_n
            try:
                secondary_start_original = int(secondary_start_step)
            except Exception:
                secondary_start_original = secondary_start_step
            if secondary_start_original != secondary_start_n:
                self._append_input_adjustment(
                    adjustments,
                    "secondary_start_step",
                    secondary_start_original,
                    secondary_start_n,
                    "forced_to_primary_end_for_dual_branch",
                )
            secondary_end_n = clamp_int(
                "secondary_end_step",
                secondary_end_step,
                min(global_steps_n, secondary_start_n + 1),
                secondary_start_n + 1,
                global_steps_n,
            )
        else:
            primary_start_n = clamp_int("primary_start_step", primary_start_step, 0, 0, max(0, global_steps_n - 1))
            primary_end_n = clamp_int("primary_end_step", primary_end_step, max(1, primary_start_n + 1), primary_start_n + 1, global_steps_n)
            secondary_start_n = clamp_int("secondary_start_step", secondary_start_step, primary_end_n, 0, max(0, global_steps_n - 1))
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
        cleanup_timing_n = clamp_enum(
            "cleanup_timing",
            cleanup_timing,
            {"NONE", "BEFORE_GENERATION", "BETWEEN_SAMPLERS", "AFTER_GENERATION", "BEFORE_AND_AFTER", "ALL"},
            "ALL",
            casefold=True,
        )
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
            save_prefix_n = "Event Horizon"
            self._append_input_adjustment(adjustments, "save_prefix", save_prefix, save_prefix_n, "empty_string_fallback")

        normalized = {
            "cascade_count": cascade_count_n,
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
            "cleanup_timing": cleanup_timing_n,
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
        cleanup_timing,
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
        resume_frame_index=-1,
        secondary_model=None,
        image=None,
        mask=None,
    ):
        self._event_strategy_coupling = {"low_strength_multiplier": 1.0}
        normalization = self._normalize_clean_inputs(
            secondary_model=secondary_model,
            cascade_count=cascade_count,
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
            cleanup_timing=cleanup_timing,
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
        cleanup_timing = str(normalized.get("cleanup_timing", "ALL"))
        video_format = str(normalized.get("video_format", "video/h264-mp4"))
        save_prefix = str(normalized.get("save_prefix", "Event Horizon"))
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
            event_strategy="r44: one external Event Horizon node with internal Event Core Body, RouteMemory, S-Wire, and universal stage math",
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
            cleanup_timing=cleanup_timing,
            stage_delay_seconds=0.0,
            use_custom_cleanup_nodes=True,
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
            resume_frame_index=resume_frame_index,
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
    "EventHorizon": EventHorizonCascadeSimple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EventHorizon": "Event Horizon",
}
