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
        strategy_control_plans = [
            r for r in records
            if str(r.get("stage", "") or "") == "EventStrategyControlSurfacePlan"
        ]
        strategy_control_apply_records = [
            r for r in records
            if str(r.get("stage", "") or "").startswith("EventStrategyControlSurfaceApply_")
        ]
        public_surface_contracts = [
            r for r in records
            if str(r.get("stage", "") or "") == "EventPublicSurfaceContract"
        ]
        public_package_static_scans = [
            r for r in records
            if str(r.get("stage", "") or "") == "EventPublicPackageStaticScan"
        ]

        body["collection_disabled"] = False
        body["collection_mode"] = "runtime_record_derived_minimal"
        body["stage_math_count"] = len(stage_math)
        body["boundary_count"] = len(boundary_math)
        body["math_tensor_record_count"] = len(math_tensor)
        body["s_wire_count"] = int(body.get("s_wire_count", 0) or 0)
        body["live_route_count"] = int(body.get("live_route_count", 0) or 0)
        body["runtime_monitor_count"] = int(body.get("runtime_monitor_count", 0) or 0)
        if strategy_control_plans:
            body["strategy_control_surface_plan"] = strategy_control_plans[-1]
        if public_surface_contracts:
            body["public_surface_contract"] = public_surface_contracts[-1]
        if public_package_static_scans:
            body["public_package_static_scan"] = public_package_static_scans[-1]
        body["strategy_control_surface_apply_records"] = strategy_control_apply_records
        return packet

    def _event_public_package_static_scan(self):
        package_root = Path(__file__).resolve().parent.parent
        required_files = [
            "VERSION",
            "__init__.py",
            "nodes.py",
            "README.md",
            "CHANGELOG.md",
            "HOMEPAGE_DESCRIPTION.md",
            "FORMULA_INTEGRITY.md",
            "LICENSE",
            "pyproject.toml",
            "requirements.txt",
            "core/cascade.py",
            "core/execution.py",
            "core/orchestrator.py",
            "core/telemetry.py",
            "reports/markdown_report.py",
            "web/singularity_cascade_ui_v2.js",
        ]
        public_docs = [
            "README.md",
            "CHANGELOG.md",
            "HOMEPAGE_DESCRIPTION.md",
            "FORMULA_INTEGRITY.md",
        ]
        required_dirs = [
            "core",
            "reports",
            "web",
            "adapters",
            "readers",
            "resolvers",
            "utils",
        ]

        def issue(reason, evidence=None):
            out = {"reason": str(reason or "unknown")}
            if evidence is not None:
                out["evidence"] = evidence
            return out

        def rel(path):
            try:
                return str(path.relative_to(package_root)).replace("\\", "/")
            except Exception:
                return str(path)

        blockers = []
        warnings = []

        missing_required_files = []
        required_file_stats = []
        for item in required_files:
            path = package_root / item
            exists = path.is_file()
            size = path.stat().st_size if exists else 0
            required_file_stats.append({"path": item, "exists": exists, "bytes": size})
            if not exists:
                missing_required_files.append(item)
            elif size <= 0:
                blockers.append(issue("required_file_empty", {"path": item}))

        missing_required_dirs = [
            item for item in required_dirs
            if not (package_root / item).is_dir()
        ]
        if missing_required_files:
            blockers.append(issue("required_public_package_files_missing", {"paths": missing_required_files}))
        if missing_required_dirs:
            blockers.append(issue("required_public_package_dirs_missing", {"paths": missing_required_dirs}))

        public_doc_stats = []
        for item in public_docs:
            path = package_root / item
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                blockers.append(issue("public_doc_read_failed", {"path": item, "error": str(e)}))
                continue
            cyrillic_count = len(re.findall(r"[\u0400-\u04FF]", text))
            non_ascii_count = len(re.findall(r"[^\x00-\x7F]", text))
            public_doc_stats.append({
                "path": item,
                "bytes": path.stat().st_size,
                "chars": len(text),
                "cyrillic_count": cyrillic_count,
                "non_ascii_count": non_ascii_count,
            })
            if cyrillic_count > 0:
                blockers.append(issue("public_doc_contains_cyrillic", {"path": item, "cyrillic_count": cyrillic_count}))
            elif non_ascii_count > 0:
                warnings.append(issue("public_doc_contains_non_ascii", {"path": item, "non_ascii_count": non_ascii_count}))

        runtime_cache_dirs = []
        runtime_cache_files = []
        forbidden_dirs = []
        forbidden_files = []
        source_checkout_markers = []
        transient_dirs = {".pytest_cache", ".mypy_cache", ".ruff_cache", "build", "dist"}
        for root, dirs, files in os.walk(package_root):
            root_path = Path(root)
            for name in list(dirs):
                path = root_path / name
                relative = rel(path)
                if name == "__pycache__":
                    runtime_cache_dirs.append(relative)
                elif name in transient_dirs or name.endswith(".egg-info"):
                    forbidden_dirs.append(relative)
                elif name == ".git":
                    source_checkout_markers.append(relative)
            for name in files:
                path = root_path / name
                relative = rel(path)
                lower = name.lower()
                if lower.endswith((".pyc", ".pyo")):
                    runtime_cache_files.append(relative)
                elif lower in (".coverage", "coverage.xml"):
                    forbidden_files.append(relative)
        if forbidden_dirs:
            blockers.append(issue("forbidden_cache_or_build_dirs_present", {"paths": forbidden_dirs[:24], "count": len(forbidden_dirs)}))
        if forbidden_files:
            blockers.append(issue("forbidden_compiled_or_coverage_files_present", {"paths": forbidden_files[:24], "count": len(forbidden_files)}))
        if runtime_cache_dirs:
            warnings.append(issue("runtime_generated_pycache_dirs_present", {"paths": runtime_cache_dirs[:24], "count": len(runtime_cache_dirs)}))
        if runtime_cache_files:
            warnings.append(issue("runtime_generated_bytecode_files_present", {"paths": runtime_cache_files[:24], "count": len(runtime_cache_files)}))
        if source_checkout_markers:
            warnings.append(issue("source_checkout_metadata_present_not_for_zip", {"paths": source_checkout_markers[:8], "count": len(source_checkout_markers)}))

        version_text = ""
        version_path = package_root / "VERSION"
        if version_path.is_file():
            try:
                version_text = version_path.read_text(encoding="utf-8", errors="replace").strip()
            except Exception:
                version_text = ""
        if version_text and version_text != EVENT_HORIZON_RUNTIME_VERSION:
            blockers.append(issue(
                "version_file_mismatch",
                {"VERSION": version_text, "runtime_version": EVENT_HORIZON_RUNTIME_VERSION},
            ))

        if blockers:
            status = "public_package_static_blocked"
            severity = "BLOCKED"
            next_action = "Fix package blockers before creating a public zip or release candidate."
        elif warnings:
            status = "public_package_static_warning"
            severity = "WARNING"
            next_action = "Static package surface is usable for development, but review warnings before public zip."
        else:
            status = "public_package_static_clean"
            severity = "PASS"
            next_action = "Static package surface is clean; runtime gate and human video review still apply."

        return {
            "stage": "EventPublicPackageStaticScan",
            "status": status,
            "severity": severity,
            "scan_version": "public_package_static_scan_v1",
            "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
            "runtime_name": EVENT_HORIZON_RUNTIME_NAME,
            "formula": "Public package files are release Outcome carriers; README, changelog, version, web assets, and cache cleanliness must return to the same public Strategy before packaging.",
            "control_mode": "REPORT_ONLY",
            "does_not_change_generation": True,
            "package_root": str(package_root),
            "version_file": version_text,
            "required_file_count": len(required_files),
            "missing_required_files": missing_required_files,
            "missing_required_dirs": missing_required_dirs,
            "public_doc_stats": public_doc_stats,
            "required_file_stats": required_file_stats,
            "forbidden_dir_count": len(forbidden_dirs),
            "forbidden_file_count": len(forbidden_files),
            "runtime_cache_dir_count": len(runtime_cache_dirs),
            "runtime_cache_file_count": len(runtime_cache_files),
            "source_checkout_marker_count": len(source_checkout_markers),
            "forbidden_dirs": forbidden_dirs[:48],
            "forbidden_files": forbidden_files[:48],
            "runtime_cache_dirs": runtime_cache_dirs[:48],
            "runtime_cache_files": runtime_cache_files[:48],
            "source_checkout_markers": source_checkout_markers[:16],
            "blockers": blockers,
            "warnings": warnings,
            "next_action": next_action,
        }

    def _event_math_topology_ledger_from_records(self, execution_records):
        records = [r for r in (execution_records or []) if isinstance(r, dict)]

        def stage_name(record):
            return str(record.get("stage", "") or "")

        def matches(record, exact=None, prefix=None):
            stage = stage_name(record)
            if exact and stage in exact:
                return True
            if prefix and any(stage.startswith(p) for p in prefix):
                return True
            return False

        def collect(exact=None, prefix=None):
            exact = set(exact or [])
            prefix = list(prefix or [])
            return [r for r in records if matches(r, exact=exact, prefix=prefix)]

        def has_any(exact=None, prefix=None):
            return bool(collect(exact=exact, prefix=prefix))

        def as_bool(value, default=False):
            if isinstance(value, bool):
                return value
            if value is None:
                return bool(default)
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)

        def add_surface(
            surfaces,
            *,
            surface_id,
            label,
            exact=None,
            prefix=None,
            formula_role="",
            risk_layer="observer",
            control_mode="REPORT_ONLY",
            generation_side_effect="none",
            public_release_role="diagnostic",
            active_statuses=None,
            active_when_control_allowed=False,
            active_when_non_report_control=False,
            research_when_present=False,
            next_route="",
        ):
            found = collect(exact=exact, prefix=prefix)
            stages = [stage_name(r) for r in found]
            statuses = sorted({str(r.get("status", "") or "") for r in found if str(r.get("status", "") or "")})
            active_statuses = set(active_statuses or ["active", "applied", "active_control"])
            active_by_status = any(str(r.get("status", "") or "").lower() in active_statuses for r in found)
            active_by_allowed = any(as_bool(r.get("active_control_allowed", False)) for r in found) if active_when_control_allowed else False
            active_by_mode = any(str(r.get("control_mode", "REPORT_ONLY") or "REPORT_ONLY").upper() != "REPORT_ONLY" for r in found) if active_when_non_report_control else False
            active = active_by_status or active_by_allowed or active_by_mode
            report_only = all(str(r.get("control_mode", "REPORT_ONLY") or "REPORT_ONLY").upper() == "REPORT_ONLY" for r in found) if found else False
            research = bool(research_when_present and found) or risk_layer in ("research", "active_research", "research_control")
            surfaces.append({
                "surface_id": str(surface_id),
                "label": str(label),
                "present": bool(found),
                "stage_count": len(found),
                "stages": stages[:24],
                "statuses": statuses,
                "formula_role": str(formula_role),
                "risk_layer": str(risk_layer),
                "control_mode": str(control_mode),
                "report_only": bool(report_only),
                "active_generation_control": bool(active),
                "research_surface": bool(research),
                "generation_side_effect": str(generation_side_effect),
                "public_release_role": str(public_release_role),
                "next_route": str(next_route),
            })

        surfaces = []
        add_surface(
            surfaces,
            surface_id="finite_input_guard",
            label="Finite Input Guard",
            exact=["EventInputNormalization", "EventInputNormalizationAdjustments"],
            formula_role="corrupted UI/runtime numeric carrier returns to declared default before Strategy math",
            risk_layer="public_guard",
            public_release_role="required_boundary",
            next_route="Read adjustments before interpreting any math value.",
        )
        add_surface(
            surfaces,
            surface_id="prompt_purity_lock",
            label="Prompt Purity Lock",
            exact=["EventPromptPurityLock", "EventPromptStrategyTranscodeApply"],
            formula_role="prompt stays clean StrategyCandidate; semantic math must not become prompt prose",
            risk_layer="public_guard",
            public_release_role="required_boundary",
            next_route="If prompt transform is enabled, treat the run as diagnostic/research evidence.",
        )
        add_surface(
            surfaces,
            surface_id="math_control_summary",
            label="Math Control Summary",
            exact=["EventMathControlSummary"],
            formula_role="UI math mode selects observer, delta overlay, or research Strategy pressure route",
            risk_layer="public_or_research_switch",
            public_release_role="required_boundary",
        )
        add_surface(
            surfaces,
            surface_id="public_surface_contract",
            label="Public Surface Contract",
            exact=["EventPublicSurfaceContract"],
            formula_role="visible UI/runtime mode carriers return to public/research boundary before final readiness",
            risk_layer="public_guard",
            public_release_role="required_boundary",
        )
        add_surface(
            surfaces,
            surface_id="public_package_static_scan",
            label="Public Package Static Scan",
            exact=["EventPublicPackageStaticScan"],
            formula_role="public README/changelog/version/web package files return to the same release Strategy before packaging",
            risk_layer="public_guard",
            public_release_role="required_boundary",
            next_route="Block public zip if required files, English-only docs, version, or cache cleanliness fail.",
        )
        add_surface(
            surfaces,
            surface_id="semantic_relation_pressure_router",
            label="Semantic Relation Pressure Router",
            exact=["EventSemanticRelationPressureRouter"],
            formula_role="prompt/topology relation pressure is read as local Strategy evidence without text injection",
            risk_layer="research",
            public_release_role="diagnostic_or_research",
            active_when_control_allowed=True,
            research_when_present=True,
            next_route="Keep this as pressure evidence unless a bounded model-native route proves visual benefit.",
        )
        add_surface(
            surfaces,
            surface_id="strategy_control_surface",
            label="Strategy Control Surface",
            exact=["EventStrategyControlSurfacePlan"],
            prefix=["EventStrategyControlSurfaceApply_"],
            formula_role="local Strategy pressure folds back into S_global_event_route and model attractor",
            risk_layer="research_control",
            public_release_role="research_evidence",
            active_when_control_allowed=True,
            research_when_present=True,
            next_route="Active use requires safe-mode comparison and visible video review.",
        )
        add_surface(
            surfaces,
            surface_id="visible_motion_strategy_return_gate",
            label="Visible Motion Strategy Return Gate",
            exact=["EventVisibleMotionStrategyReturnGate"],
            formula_role="visible Outcome(t+1) and ObservedBehavior(t+1) return as next-run Strategy evidence, not same-run damping",
            risk_layer="observer",
            public_release_role="visual_diagnostic",
            next_route="Use visible motion evidence to choose the next bounded topology route; never damp globally from frame pressure alone.",
        )
        add_surface(
            surfaces,
            surface_id="true_region_topology_evidence",
            label="True Region Topology Evidence",
            exact=["EventTrueRegionTopologyEvidence"],
            formula_role="visible/background/spatial/object/tail pressures are separated into region-role evidence before any active topology route is allowed",
            risk_layer="observer_to_route",
            public_release_role="visual_diagnostic",
            next_route="Require region separation proof before spatial/background pressure can become active control.",
        )
        add_surface(
            surfaces,
            surface_id="fractal_strategy_intersection_map",
            label="Fractal Strategy Intersection Map",
            exact=["EventFractalStrategyIntersectionMap"],
            formula_role="all dynamic intersections unfold the same Strategy equality recursively and return to S_global_event_route across depth 7",
            risk_layer="observer_to_route",
            public_release_role="topology_diagnostic",
            next_route="Use the dominant intersection axis to choose the next bounded report-only route; do not promote derived strategies into independent controllers.",
        )
        add_surface(
            surfaces,
            surface_id="region_weighted_fractal_strategy_return",
            label="Region-Weighted Fractal Strategy Return",
            exact=["EventRegionWeightedFractalStrategyReturn"],
            formula_role="dominant fractal Strategy axes are weighted by true region and visible-video evidence before they can nominate a next route",
            risk_layer="observer_to_route",
            public_release_role="topology_diagnostic",
            next_route="Require dominant_axis_evidence_match and region_axis_confidence before any region-derived active control.",
        )
        add_surface(
            surfaces,
            surface_id="pixel_region_motion_map",
            label="Pixel Region Motion Map",
            exact=["EventPixelRegionMotionMap"],
            formula_role="decoded visible Outcome is split into center, object/contact, seam, top/bottom, and edge/background pixel motion before pressure evidence can nominate a next route",
            risk_layer="observer_to_route",
            public_release_role="visual_diagnostic",
            next_route="Compare pressure-derived background leakage with actual pixel-region motion before choosing action/background control.",
        )
        add_surface(
            surfaces,
            surface_id="cascade_seam_impulse_review",
            label="Cascade Seam Impulse Review",
            exact=["EventCascadeSeamImpulseReview"],
            formula_role="tail Outcome(previous segment), boundary ObservedBehavior, and post-continue Outcome are compared as one cascade Strategy transition",
            risk_layer="observer_to_route",
            public_release_role="continuity_diagnostic",
            next_route="If high, route to tail-next-source Strategy continuity; do not apply damping from one run.",
        )
        add_surface(
            surfaces,
            surface_id="tail_next_source_continuity_proposal",
            label="Tail Next Source Continuity Proposal",
            exact=["EventTailNextSourceStrategyContinuityProposal"],
            formula_role="selected tail Outcome(previous segment), tail motion ObservedBehavior, next source frame, and first post-continue motion are compared as the same StrategyCarrier",
            risk_layer="observer_to_route",
            public_release_role="continuity_diagnostic",
            next_route="If repeated high, design a tail-next-source continuity bridge; keep this proposal report-only.",
        )
        add_surface(
            surfaces,
            surface_id="cascade_seam_phase_classifier",
            label="Cascade Seam Phase Classifier",
            exact=["EventCascadeSeamPhaseClassifier"],
            formula_role="seam pressure is classified as semantic phase re-entry, prompt text change, latent mismatch, background anchor conflict, center-action overdrive, or sampler handoff reset before active control",
            risk_layer="observer_to_route",
            public_release_role="continuity_diagnostic",
            next_route="Use repeated fixed-seed classifier evidence to pick one next active surface; do not strengthen the selected-tail echo blindly.",
        )
        add_surface(
            surfaces,
            surface_id="semantic_phase_schedule_proposal",
            label="Semantic Phase Schedule Proposal",
            exact=["EventCascadeSemanticPhaseScheduleProposal"],
            formula_role="clean global StrategyCandidate text is separated from local cascade phase identity so the next segment can be diagnosed without prompt text injection",
            risk_layer="observer_to_route",
            public_release_role="continuity_diagnostic",
            next_route="If repeated high, design a local Strategy phase window that keeps model-facing prompt text unchanged.",
        )
        add_surface(
            surfaces,
            surface_id="semantic_phase_window_carrier",
            label="Semantic Phase Report Fence",
            exact=["EventCascadeSemanticPhaseScheduleProposal"],
            formula_role="selected tail progress is converted into bounded local Strategy phase evidence without prompt text injection or tensor mutation",
            risk_layer="observer_to_route",
            public_release_role="continuity_diagnostic",
            active_statuses=["phase_window_tensor_applied"],
            next_route="Keep report-only until a non-concat sampler-entry route proves lower seam impulse without background/detail loss.",
        )
        add_surface(
            surfaces,
            surface_id="selected_tail_source_reconstruction_safe",
            label="Selected Tail Source Reconstruction SAFE Package",
            exact=["EventSelectedTailSourceReconstructionPackage"],
            formula_role="selected tail OutcomePrevious and tail motion ObservedBehaviorPrevious are reconstructed as next-source StrategyCarrier evidence without tensor mutation",
            risk_layer="observer_to_route",
            public_release_role="continuity_diagnostic",
            next_route="Use this as the safe package before any latent-memory or sampler-entry mutation.",
        )
        add_surface(
            surfaces,
            surface_id="max_risk_strategy_ring_package",
            label="MAX RISK Strategy Ring Package",
            exact=["EventMaxRiskStrategyRingPackage"],
            formula_role="previous latent tail may become a tiny forced Strategy ring at next segment entry only when explicitly selected",
            risk_layer="active_research",
            control_mode="ACTIVE_RESEARCH",
            public_release_role="research_evidence",
            active_statuses=["max_risk_package_active"],
            active_when_control_allowed=True,
            research_when_present=True,
            generation_side_effect="can override the latent bridge hard guard and alter next segment entry in MAX_RISK_STRATEGY_RING mode",
            next_route="Run only as fixed-seed A/B against SAFE and OBSERVE_ONLY; inspect video for color/noise/identity artifacts.",
        )
        add_surface(
            surfaces,
            surface_id="boundary_background_anchor_control",
            label="Boundary Background Anchor Control",
            exact=["EventCascadeBoundaryBackgroundAnchorControlBind", "EventCascadeBoundaryBackgroundAnchorCard"],
            formula_role="selected tail visible frame becomes background/source evidence before the next sampler",
            risk_layer="observer_to_route",
            public_release_role="continuity_diagnostic",
            active_statuses=["active"],
            next_route="Use as cascade-boundary evidence; do not confuse with tail-only pressure.",
        )
        add_surface(
            surfaces,
            surface_id="r126_low_mid_window_route",
            label="R126 Low Mid-Window Route Guard",
            prefix=["EventR126LowMidWindowSpatialControlRoute_"],
            formula_role="route-key guarded low mid-window Strategy intersection for spatial carrier preservation",
            risk_layer="active_research",
            public_release_role="research_evidence",
            active_statuses=["active"],
            research_when_present=True,
            generation_side_effect="possible additional low-branch sampler calls when active",
            next_route="Verify route_key_matches and additional_sampler_calls before visual conclusions.",
        )
        add_surface(
            surfaces,
            surface_id="denoise_phase_map",
            label="Denoise Phase Map",
            prefix=["EventDenoisePhaseMap_"],
            formula_role="classifies high/low/post-window/endpoint phase before any local math can act",
            risk_layer="public_guard",
            public_release_role="required_for_step_control",
            next_route="Only low mid-window should become active-control eligible.",
        )
        add_surface(
            surfaces,
            surface_id="spatial_carrier_preservation_map",
            label="Spatial Carrier Preservation Map",
            prefix=["EventSpatialCarrierPreservationMap_"],
            formula_role="background/source/tail carriers become bounded spatial preservation pressure",
            risk_layer="active_research",
            public_release_role="research_evidence",
            active_statuses=["active"],
            research_when_present=True,
            generation_side_effect="bounded ROI gain only when active and phase-safe",
            next_route="If guarded_report_only, do not expect visible improvement from this surface.",
        )
        add_surface(
            surfaces,
            surface_id="action_background_separation_gate",
            label="Action / Background Separation Evidence Gate",
            exact=["EventActionBackgroundSeparationEvidence"],
            prefix=["EventActionBackgroundSeparationGate_"],
            formula_role="center action, object/contact, seam, and background carriers return as separable sub-strategies before model-attractor pressure",
            risk_layer="active_research",
            public_release_role="research_evidence",
            active_statuses=["active"],
            research_when_present=True,
            generation_side_effect="bounded StrategyField/window compression when background pressure competes with action center",
            next_route="Compare background/center motion ratio before and after; this gate should reduce whole-scene motion leakage, not freeze action.",
        )
        add_surface(
            surfaces,
            surface_id="pixel_pressure_disagreement_review",
            label="Pixel Pressure Disagreement Review",
            exact=["EventPixelPressureDisagreementReview"],
            formula_role="selected visible pixel Outcome reweights pressure-derived background/action interpretation before any route nomination",
            risk_layer="observer_to_route",
            public_release_role="visual_diagnostic",
            next_route="Use corrected pixel/pressure axis before active control; never let scalar pressure overrule visible Outcome alone.",
        )
        add_surface(
            surfaces,
            surface_id="pressure_pixel_reweighting_proposal",
            label="Pressure / Pixel Reweighting Proposal",
            exact=["EventPressurePixelReweightingProposal"],
            formula_role="pixel-corrected pressure becomes bounded future control proposal while preserving the model-attractor route",
            risk_layer="observer_to_route",
            public_release_role="visual_diagnostic",
            next_route="Use only for fixed-seed A/B before any active low-mid-window pressure/pixel reweighting.",
        )
        add_surface(
            surfaces,
            surface_id="pressure_pixel_reweighting_active_candidate",
            label="Pressure / Pixel Reweighting Active Candidate",
            prefix=["EventPressurePixelReweightingActiveCandidate_"],
            formula_role="R147/R149 pixel-corrected pressure evidence returns as a quality-guarded local/spatial low-branch active A/B candidate",
            risk_layer="active_research",
            public_release_role="research_evidence",
            active_statuses=["active_candidate"],
            research_when_present=True,
            generation_side_effect="quality-guarded tiny low-branch delta overlay with local/spatial background gain only when math_control_mode is PRESSURE_PIXEL_REWEIGHTING",
            next_route="Accept only if fixed-seed R150 preserves R149 quality while reducing edge/background pressure artifacts.",
        )
        add_surface(
            surfaces,
            surface_id="noise_field_strategy_bridge",
            label="Noise Field Strategy Bridge",
            prefix=["EventNoiseFieldStrategyBridge_"],
            formula_role="source/noise evidence names the next safe Strategy surface; R171 can birth tiny spatial gain, anchor-only low-frequency source-image carrier, and a bounded two-slice selected-tail post-drop seam-entry micro echo with regional/background guard before high sampler while keeping unsafe post-window delta report-only",
            risk_layer="observer_to_research",
            public_release_role="diagnostic",
            active_statuses=["pre_high_seed_active_candidate"],
            research_when_present=True,
            next_route="If SOURCE_NOISE_FIELD_SHAPING is active, inspect EventSourceNoiseBirthShaping before judging video.",
        )
        add_surface(
            surfaces,
            surface_id="source_noise_birth_shaping",
            label="R171 Regional Tail Guard",
            prefix=["EventSourceNoiseBirthShaping_"],
            formula_role="Wan latent seed and Wan positive concat conditioning are lightly shaped before high sampler so source/noise StrategyCarrier returns to model-attractor without prompt text or forced microtexture",
            risk_layer="active_research",
            public_release_role="research_evidence",
            active_statuses=["applied"],
            research_when_present=True,
            generation_side_effect="tiny feathered low-frequency spatial gain/source carrier on Wan latent seed and Wan positive concat conditioning before high sampler when math_control_mode is SOURCE_NOISE_FIELD_SHAPING",
            next_route="Compare fixed-seed A/B against R159 SAFE and R162; reject if it creates background/face pixel noise, finger/detail smearing, or stronger seam reversal.",
        )
        add_surface(
            surfaces,
            surface_id="segment_entry_latent_memory_bridge",
            label="Segment Entry Latent Memory Bridge",
            exact=["EventSegmentEntryLatentMemoryBridge"],
            formula_role="previous latent OutcomePrevious may return to next segment entry as explicit bounded memory",
            risk_layer="research_control",
            public_release_role="research_evidence",
            active_statuses=["active"],
            research_when_present=True,
            generation_side_effect="can alter next segment latent entry only when explicit bridge mode is active",
            next_route="Use only with same-seed comparison against OBSERVE_ONLY.",
        )
        add_surface(
            surfaces,
            surface_id="strategy_matrix",
            label="Strategy Matrix And Local Micro-Formulae",
            exact=["EventStrategyMatrix"],
            prefix=["EventLocalMicroFormula_"],
            formula_role="carrier collisions unfold local Strategy equalities as report-only evidence",
            risk_layer="observer",
            public_release_role="diagnostic",
            next_route="Use this to choose the next active surface, not as proof of visual quality.",
        )

        required_ids = {
            "finite_input_guard",
            "prompt_purity_lock",
            "math_control_summary",
            "public_surface_contract",
            "public_package_static_scan",
        }
        present_ids = {s["surface_id"] for s in surfaces if s["present"]}
        missing_required = sorted(required_ids - present_ids)
        active_surfaces = [s for s in surfaces if s["active_generation_control"]]
        research_surfaces = [s for s in surfaces if s["research_surface"] and s["present"]]
        report_only_surfaces = [s for s in surfaces if s["present"] and s["report_only"]]

        if missing_required:
            status = "incomplete_math_topology"
            severity = "WARNING"
            next_action = "Missing required math topology boundaries; treat the report as incomplete evidence."
        elif active_surfaces:
            status = "active_math_present"
            severity = "RESEARCH"
            next_action = "Use this run as internal math evidence and compare against a safe-mode baseline before release."
        elif research_surfaces:
            status = "research_or_diagnostic_math_present"
            severity = "WARNING"
            next_action = "Research/diagnostic math is present but no active generation surface is proven active."
        else:
            status = "public_observer_topology"
            severity = "PASS"
            next_action = "Math topology is observer/public-boundary only; final release still needs video inspection."

        return {
            "stage": "EventMathTopologyLedger",
            "status": status,
            "severity": severity,
            "ledger_version": "math_topology_ledger_v1",
            "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
            "runtime_name": EVENT_HORIZON_RUNTIME_NAME,
            "formula": "Every math organ is classified by formula role, risk layer, control mode, and generation side-effect so Strategy evidence can return to one release-readable topology.",
            "control_mode": "REPORT_ONLY",
            "does_not_change_generation": True,
            "surface_count": len(surfaces),
            "present_surface_count": len(present_ids),
            "required_surface_ids": sorted(required_ids),
            "missing_required_surface_ids": missing_required,
            "active_generation_surface_count": len(active_surfaces),
            "active_generation_surface_ids": [s["surface_id"] for s in active_surfaces],
            "research_surface_count": len(research_surfaces),
            "research_surface_ids": [s["surface_id"] for s in research_surfaces],
            "report_only_surface_count": len(report_only_surfaces),
            "surfaces": surfaces,
            "next_action": next_action,
        }

    def _event_math_topology_dependency_graph(self, math_topology_ledger):
        ledger = math_topology_ledger if isinstance(math_topology_ledger, dict) else {}
        surfaces = [
            s for s in (ledger.get("surfaces", []) or [])
            if isinstance(s, dict)
        ]
        surface_by_id = {str(s.get("surface_id", "") or ""): s for s in surfaces}

        def surface_present(surface_id):
            return bool(surface_by_id.get(surface_id, {}).get("present", False))

        def surface_active(surface_id):
            return bool(surface_by_id.get(surface_id, {}).get("active_generation_control", False))

        def surface_research(surface_id):
            return bool(surface_by_id.get(surface_id, {}).get("research_surface", False))

        def node(node_id, label, formula_role, source_surface_id="", node_type="surface"):
            surface = surface_by_id.get(source_surface_id, {}) if source_surface_id else {}
            if source_surface_id:
                present = surface_present(source_surface_id)
                active = surface_active(source_surface_id)
                research = surface_research(source_surface_id)
                risk_layer = str(surface.get("risk_layer", "") or "")
                public_release_role = str(surface.get("public_release_role", "") or "")
            else:
                present = True
                active = False
                research = False
                risk_layer = "concept"
                public_release_role = "context"
            return {
                "node_id": str(node_id),
                "label": str(label),
                "node_type": str(node_type),
                "source_surface_id": str(source_surface_id or ""),
                "present": bool(present),
                "active_generation_control": bool(active),
                "research_surface": bool(research),
                "risk_layer": risk_layer,
                "public_release_role": public_release_role,
                "formula_role": str(formula_role),
            }

        nodes = [
            node("ui_runtime_carrier", "UI / runtime numeric carriers", "Raw UI values enter as candidate Strategy carriers before normalization.", node_type="carrier"),
            node("finite_input_guard", "Finite Input Guard", "Invalid numeric carrier returns to declared default before Strategy math.", "finite_input_guard"),
            node("prompt_strategy_candidate", "Clean Prompt StrategyCandidate", "Prompt remains clean text carrier; math must not become prompt prose.", node_type="carrier"),
            node("prompt_purity_lock", "Prompt Purity Lock", "Locks semantic/math topology outside prompt text.", "prompt_purity_lock"),
            node("math_control_summary", "Math Control Summary", "UI math mode selects observer/safe delta/research route.", "math_control_summary"),
            node("semantic_relation_pressure_router", "Semantic Relation Pressure Router", "Reads prompt/topology relation pressure as local Strategy evidence.", "semantic_relation_pressure_router"),
            node("strategy_control_surface", "Strategy Control Surface", "Folds bounded local pressure back into S_global_event_route.", "strategy_control_surface"),
            node("visible_motion_strategy_return_gate", "Visible Motion Strategy Return Gate", "Visible video behavior returns as next Strategy evidence without same-run active damping.", "visible_motion_strategy_return_gate"),
            node("true_region_topology_evidence", "True Region Topology Evidence", "Region-role evidence separates center action, edge/background, top band, object/contact, and selected-tail carriers before active control can be nominated.", "true_region_topology_evidence"),
            node("fractal_strategy_intersection_map", "Fractal Strategy Intersection Map", "All dynamic intersections recursively unfold the Strategy equality to depth 7 and return to the parent route.", "fractal_strategy_intersection_map"),
            node("region_weighted_fractal_strategy_return", "Region-Weighted Fractal Strategy Return", "Dominant fractal axes must agree with true-region and visible-video evidence before nominating a route.", "region_weighted_fractal_strategy_return"),
            node("pixel_region_motion_map", "Pixel Region Motion Map", "Decoded pixel regions verify whether pressure-derived background leakage is visible evidence or scalar artifact.", "pixel_region_motion_map"),
            node("cascade_seam_impulse_review", "Cascade Seam Impulse Review", "Tail, boundary, and post-continue motion vectors expose whether the next segment continues or is reborn.", "cascade_seam_impulse_review"),
            node("tail_next_source_continuity_proposal", "Tail Next Source Continuity Proposal", "Selected tail source, tail motion, next entry frame, and visible seam evidence are gathered into a report-only bridge proposal.", "tail_next_source_continuity_proposal"),
            node("cascade_seam_phase_classifier", "Cascade Seam Phase Classifier", "Classifies the seam cause before active control: semantic phase, prompt text change, latent mismatch, background conflict, center overdrive, or sampler handoff reset.", "cascade_seam_phase_classifier"),
            node("semantic_phase_schedule_proposal", "Semantic Phase Schedule Proposal", "Separates clean prompt text identity from local cascade phase identity before any active phase-window route.", "semantic_phase_schedule_proposal"),
            node("selected_tail_source_reconstruction_safe", "Selected Tail Source Reconstruction SAFE Package", "Reconstructs source/tail inheritance as report-only StrategyCarrier evidence.", "selected_tail_source_reconstruction_safe"),
            node("max_risk_strategy_ring_package", "MAX RISK Strategy Ring Package", "Explicitly allows a tiny hard-guard override at the next segment latent entry.", "max_risk_strategy_ring_package"),
            node("pixel_pressure_disagreement_review", "Pixel Pressure Disagreement Review", "Selected visible pixels correct scalar pressure interpretation before future route nomination.", "pixel_pressure_disagreement_review"),
            node("pressure_pixel_reweighting_proposal", "Pressure / Pixel Reweighting Proposal", "Corrected pixel-pressure evidence returns as bounded future weights, never same-run tensor control.", "pressure_pixel_reweighting_proposal"),
            node("pressure_pixel_reweighting_active_candidate", "Pressure / Pixel Reweighting Quality Guard", "Fixed-seed R147 proof returns as a quality-guarded low-branch A/B delta after the R148 grain verdict.", "pressure_pixel_reweighting_active_candidate"),
            node("source_tail_carrier", "Source / selected tail visible carrier", "Visible tail/source OutcomePrevious becomes next sampler Strategy evidence.", node_type="carrier"),
            node("boundary_background_anchor_control", "Boundary Background Anchor Control", "Moves selected-tail background/source evidence before next sampler.", "boundary_background_anchor_control"),
            node("r126_low_mid_window_route", "R126 Low Mid-Window Route Guard", "Places spatial control only into route-key matched low mid-window.", "r126_low_mid_window_route"),
            node("denoise_phase_map", "Denoise Phase Map", "Classifies phase before local math can touch delta.", "denoise_phase_map"),
            node("spatial_carrier_preservation_map", "Spatial Carrier Preservation Map", "Converts spatial/source carriers into bounded phase-safe pressure.", "spatial_carrier_preservation_map"),
            node("action_background_separation_gate", "Action Background Separation Gate", "Keeps center action and edge background as separable sub-strategies before pressure returns to the model.", "action_background_separation_gate"),
            node("source_noise_field", "Source / noise field carrier", "Noise/source field is read before deciding whether shaping is safe.", node_type="carrier"),
            node("noise_field_strategy_bridge", "Noise Field Strategy Bridge", "Names safe next Strategy surface from source/noise evidence.", "noise_field_strategy_bridge"),
            node("source_noise_birth_shaping", "Source / Noise Birth Shaping", "Tiny guarded pre-high seed gain acts before high sampler while prompt and CFG stay native.", "source_noise_birth_shaping"),
            node("segment_entry_latent_memory_bridge", "Segment Entry Latent Memory Bridge", "Optional previous latent OutcomePrevious returns at next segment entry.", "segment_entry_latent_memory_bridge"),
            node("strategy_matrix", "Strategy Matrix / micro-formulae", "Carrier collisions unfold report-only local Strategy equalities.", "strategy_matrix"),
            node("public_surface_contract", "Public Surface Contract", "Classifies visible UI/runtime modes before final readiness.", "public_surface_contract"),
            node("public_package_static_scan", "Public Package Static Scan", "Public docs/version/web/cache package surface returns before release packaging.", "public_package_static_scan"),
            node("math_topology_ledger", "Math Topology Ledger", "Collects all math organs into one release-readable topology.", node_type="ledger"),
            node("public_release_readiness_gate", "Public Release Readiness Gate", "Returns public/research/not-ready verdict after finalized route.", node_type="gate"),
            node("public_release_candidate_manifest", "Public Release Candidate Manifest", "Post-save artifacts and release gates return to one package/no-package Strategy verdict.", node_type="manifest"),
            node("human_report_top_summary", "Human Report Top Summary", "Returns graph/readiness/manifest meaning to the human reader before raw records.", node_type="report"),
            node("public_package_verdict", "Public Package Verdict", "A package handoff is allowed only after manifest convergence plus human video review.", node_type="release_outcome"),
            node("model_attractor", "Model Attractor", "The model remains the topological center; math should help it understand, not replace it.", node_type="formula_center"),
            node("visible_video_outcome", "Visible Video Outcome", "Decoded video is the visible Outcome(t+1) that still needs human inspection.", node_type="outcome"),
        ]

        def edge(edge_id, source, target, formula_link, required=True, source_surface_id="", risk="observer"):
            source_node = next((n for n in nodes if n["node_id"] == source), {})
            target_node = next((n for n in nodes if n["node_id"] == target), {})
            linked_surfaces = [sid for sid in [source_surface_id, source_node.get("source_surface_id", ""), target_node.get("source_surface_id", "")] if sid]
            present = bool(source_node.get("present", False)) and bool(target_node.get("present", False))
            active = any(surface_active(sid) for sid in linked_surfaces)
            research = any(surface_research(sid) for sid in linked_surfaces) or str(risk) in ("research", "active_research")
            return {
                "edge_id": str(edge_id),
                "from": str(source),
                "to": str(target),
                "present": bool(present),
                "required": bool(required),
                "active_generation_edge": bool(active),
                "research_edge": bool(research),
                "risk": str(risk),
                "formula_link": str(formula_link),
            }

        edges = [
            edge("ui_to_finite_guard", "ui_runtime_carrier", "finite_input_guard", "corrupted carrier -> fallback Strategy boundary", source_surface_id="finite_input_guard", risk="public_guard"),
            edge("finite_guard_to_math_mode", "finite_input_guard", "math_control_summary", "finite numeric carrier -> selected math route", source_surface_id="math_control_summary", risk="public_guard"),
            edge("prompt_to_purity", "prompt_strategy_candidate", "prompt_purity_lock", "prompt remains StrategyCandidate, not formula prose", source_surface_id="prompt_purity_lock", risk="public_guard"),
            edge("purity_to_semantic_router", "prompt_purity_lock", "semantic_relation_pressure_router", "semantic pressure can be read only outside prompt text", source_surface_id="semantic_relation_pressure_router", required=False, risk="research"),
            edge("semantic_router_to_control_surface", "semantic_relation_pressure_router", "strategy_control_surface", "local relation pressure returns to S_global_event_route", source_surface_id="strategy_control_surface", required=False, risk="research"),
            edge("visible_outcome_to_motion_return", "visible_video_outcome", "visible_motion_strategy_return_gate", "visible Outcome(t+1) returns as next-run Strategy evidence", source_surface_id="visible_motion_strategy_return_gate", required=False, risk="observer"),
            edge("motion_return_to_control_surface", "visible_motion_strategy_return_gate", "strategy_control_surface", "visible motion may nominate the next bounded route but cannot force same-run damping", source_surface_id="visible_motion_strategy_return_gate", required=False, risk="research"),
            edge("motion_return_to_true_region_topology", "visible_motion_strategy_return_gate", "true_region_topology_evidence", "coupled visible motion must be resolved into region-role evidence before active control", source_surface_id="true_region_topology_evidence", required=False, risk="observer_to_route"),
            edge("strategy_matrix_to_fractal_intersections", "strategy_matrix", "fractal_strategy_intersection_map", "local collision formulae become primary intersections for recursive Strategy unfold", source_surface_id="fractal_strategy_intersection_map", required=False, risk="observer_to_route"),
            edge("true_region_to_fractal_intersections", "true_region_topology_evidence", "fractal_strategy_intersection_map", "region-role pressures become local intersections that must return to S_global_event_route", source_surface_id="fractal_strategy_intersection_map", required=False, risk="observer_to_route"),
            edge("fractal_to_region_weighted_return", "fractal_strategy_intersection_map", "region_weighted_fractal_strategy_return", "dominant fractal axis is reweighted by region/visible evidence before route nomination", source_surface_id="region_weighted_fractal_strategy_return", required=False, risk="observer_to_route"),
            edge("true_region_to_region_weighted_return", "true_region_topology_evidence", "region_weighted_fractal_strategy_return", "true region confidence checks whether a dominant region axis is real evidence or scalar overweight", source_surface_id="region_weighted_fractal_strategy_return", required=False, risk="observer_to_route"),
            edge("region_weighted_return_to_control_surface", "region_weighted_fractal_strategy_return", "strategy_control_surface", "only evidence-matched axes may nominate future bounded control surfaces", source_surface_id="region_weighted_fractal_strategy_return", required=False, risk="research"),
            edge("region_weighted_return_to_model_attractor", "region_weighted_fractal_strategy_return", "model_attractor", "region-weighted axes must return to the model attractor instead of becoming independent controllers", source_surface_id="region_weighted_fractal_strategy_return", required=False, risk="observer_to_route"),
            edge("region_weighted_return_to_ledger", "region_weighted_fractal_strategy_return", "math_topology_ledger", "axis confidence becomes release-readable topology evidence", source_surface_id="region_weighted_fractal_strategy_return", required=False, risk="observer_to_route"),
            edge("region_weighted_return_to_action_background", "region_weighted_fractal_strategy_return", "action_background_separation_gate", "guarded fractal axis is split into action/background/seam carriers before any route nomination", source_surface_id="action_background_separation_gate", required=False, risk="observer_to_route"),
            edge("visible_outcome_to_pixel_region_map", "visible_video_outcome", "pixel_region_motion_map", "visible Outcome(t+1) is segmented into pixel-region motion before scalar pressure is trusted", source_surface_id="pixel_region_motion_map", required=False, risk="observer_to_route"),
            edge("visible_outcome_to_seam_impulse", "visible_video_outcome", "cascade_seam_impulse_review", "visible segment tails and entries expose cascade Strategy continuity or rebirth", source_surface_id="cascade_seam_impulse_review", required=False, risk="observer_to_route"),
            edge("tail_carrier_to_seam_impulse", "source_tail_carrier", "cascade_seam_impulse_review", "selected/tail Outcome(previous segment) is compared against post-continue movement", source_surface_id="cascade_seam_impulse_review", required=False, risk="observer_to_route"),
            edge("seam_impulse_to_control_surface", "cascade_seam_impulse_review", "strategy_control_surface", "a repeated seam impulse may nominate tail-next-source Strategy continuity but remains report-only here", source_surface_id="cascade_seam_impulse_review", required=False, risk="research"),
            edge("seam_impulse_to_model_attractor", "cascade_seam_impulse_review", "model_attractor", "cascade continuity evidence must return to the model attractor instead of becoming blind damping", source_surface_id="cascade_seam_impulse_review", required=False, risk="observer_to_route"),
            edge("seam_impulse_to_ledger", "cascade_seam_impulse_review", "math_topology_ledger", "cascade seam impulse becomes release-readable topology evidence", source_surface_id="cascade_seam_impulse_review", required=False, risk="observer_to_route"),
            edge("seam_impulse_to_tail_next_source_proposal", "cascade_seam_impulse_review", "tail_next_source_continuity_proposal", "visible/vector seam evidence is translated into the carriers a future continuity bridge would need", source_surface_id="tail_next_source_continuity_proposal", required=False, risk="observer_to_route"),
            edge("tail_carrier_to_tail_next_source_proposal", "source_tail_carrier", "tail_next_source_continuity_proposal", "selected tail frame and tail motion become the left side of the next-source Strategy equality", source_surface_id="tail_next_source_continuity_proposal", required=False, risk="observer_to_route"),
            edge("tail_next_source_proposal_to_control_surface", "tail_next_source_continuity_proposal", "strategy_control_surface", "a repeated high proposal may nominate report-only tail-next-source inheritance before active control", source_surface_id="tail_next_source_continuity_proposal", required=False, risk="research"),
            edge("tail_next_source_proposal_to_model_attractor", "tail_next_source_continuity_proposal", "model_attractor", "continuity bridge design must return to the model attractor instead of becoming local damping", source_surface_id="tail_next_source_continuity_proposal", required=False, risk="observer_to_route"),
            edge("tail_next_source_proposal_to_ledger", "tail_next_source_continuity_proposal", "math_topology_ledger", "tail-next-source continuity proposal becomes release-readable topology evidence", source_surface_id="tail_next_source_continuity_proposal", required=False, risk="observer_to_route"),
            edge("seam_impulse_to_phase_classifier", "cascade_seam_impulse_review", "cascade_seam_phase_classifier", "seam impulse evidence becomes phase-cause classification before active route choice", source_surface_id="cascade_seam_phase_classifier", required=False, risk="observer_to_route"),
            edge("tail_next_source_to_phase_classifier", "tail_next_source_continuity_proposal", "cascade_seam_phase_classifier", "tail/source continuity pressure helps separate latent mismatch from prompt or sampler reset", source_surface_id="cascade_seam_phase_classifier", required=False, risk="observer_to_route"),
            edge("pixel_region_map_to_phase_classifier", "pixel_region_motion_map", "cascade_seam_phase_classifier", "visible region motion separates background conflict from center action overdrive", source_surface_id="cascade_seam_phase_classifier", required=False, risk="observer_to_route"),
            edge("latent_memory_to_phase_classifier", "segment_entry_latent_memory_bridge", "cascade_seam_phase_classifier", "latent bridge admissibility helps classify carrier mismatch before more alpha", source_surface_id="cascade_seam_phase_classifier", required=False, risk="observer_to_route"),
            edge("phase_classifier_to_semantic_schedule", "cascade_seam_phase_classifier", "semantic_phase_schedule_proposal", "semantic phase re-entry is separated from prompt text changes before selecting a next route", source_surface_id="semantic_phase_schedule_proposal", required=False, risk="observer_to_route"),
            edge("semantic_schedule_to_control_surface", "semantic_phase_schedule_proposal", "strategy_control_surface", "a repeated semantic phase schedule can nominate a future local Strategy window without prompt text injection", source_surface_id="semantic_phase_schedule_proposal", required=False, risk="research"),
            edge("semantic_schedule_to_model_attractor", "semantic_phase_schedule_proposal", "model_attractor", "phase windows must return to the model attractor instead of becoming independent text rewrites", source_surface_id="semantic_phase_schedule_proposal", required=False, risk="observer_to_route"),
            edge("semantic_schedule_to_ledger", "semantic_phase_schedule_proposal", "math_topology_ledger", "semantic phase schedule evidence becomes release-readable topology evidence", source_surface_id="semantic_phase_schedule_proposal", required=False, risk="observer_to_route"),
            edge("semantic_schedule_to_phase_window_carrier", "semantic_phase_schedule_proposal", "semantic_phase_window_carrier", "the report-only schedule becomes a bounded local numeric carrier on the next segment route", source_surface_id="semantic_phase_window_carrier", required=False, risk="active_research"),
            edge("remaining_strategy_to_phase_window_carrier", "source_tail_carrier", "semantic_phase_window_carrier", "selected tail progress and tail motion define the local phase window before the next high sampler", source_surface_id="semantic_phase_window_carrier", required=False, risk="active_research"),
            edge("phase_window_carrier_to_latent_bridge", "semantic_phase_window_carrier", "segment_entry_latent_memory_bridge", "R177 reports phase-window pressure at the existing tiny concat-only seam-entry bridge but restores the R173 region guard baseline and does not let phase pressure mutate tensors", source_surface_id="semantic_phase_window_carrier", required=False, risk="report_only"),
            edge("phase_window_carrier_to_model_attractor", "semantic_phase_window_carrier", "model_attractor", "local phase control must return to model-readable continuity rather than becoming an independent local loop", source_surface_id="semantic_phase_window_carrier", required=False, risk="active_research"),
            edge("phase_window_carrier_to_ledger", "semantic_phase_window_carrier", "math_topology_ledger", "R177 phase-window rejection evidence is visible in the topology ledger for A/B comparison without being counted as an active generation surface", source_surface_id="semantic_phase_window_carrier", required=False, risk="report_only"),
            edge("phase_classifier_to_control_surface", "cascade_seam_phase_classifier", "strategy_control_surface", "phase classification nominates exactly one future route but remains report-only here", source_surface_id="cascade_seam_phase_classifier", required=False, risk="research"),
            edge("phase_classifier_to_model_attractor", "cascade_seam_phase_classifier", "model_attractor", "classified seam phase returns to the model attractor before any active control", source_surface_id="cascade_seam_phase_classifier", required=False, risk="observer_to_route"),
            edge("phase_classifier_to_ledger", "cascade_seam_phase_classifier", "math_topology_ledger", "seam phase classification becomes release-readable topology evidence", source_surface_id="cascade_seam_phase_classifier", required=False, risk="observer_to_route"),
            edge("tail_next_source_proposal_to_safe_package", "tail_next_source_continuity_proposal", "selected_tail_source_reconstruction_safe", "tail/source evidence is reconstructed into a safe report package before mutation is allowed", source_surface_id="selected_tail_source_reconstruction_safe", required=False, risk="observer_to_route"),
            edge("safe_package_to_model_attractor", "selected_tail_source_reconstruction_safe", "model_attractor", "safe reconstruction must return to the model attractor instead of becoming prompt prose", source_surface_id="selected_tail_source_reconstruction_safe", required=False, risk="observer_to_route"),
            edge("safe_package_to_max_risk_package", "selected_tail_source_reconstruction_safe", "max_risk_strategy_ring_package", "only repeated high rebirth risk should justify explicit max-risk Strategy ring A/B", source_surface_id="max_risk_strategy_ring_package", required=False, risk="active_research"),
            edge("max_risk_package_to_latent_bridge", "max_risk_strategy_ring_package", "segment_entry_latent_memory_bridge", "max-risk package can override bridge hard guard with a tiny single-slice latent Strategy ring", source_surface_id="max_risk_strategy_ring_package", required=False, risk="active_research"),
            edge("max_risk_package_to_model_attractor", "max_risk_strategy_ring_package", "model_attractor", "max-risk mutation must still serve the model attractor and not become independent physics", source_surface_id="max_risk_strategy_ring_package", required=False, risk="active_research"),
            edge("max_risk_package_to_ledger", "max_risk_strategy_ring_package", "math_topology_ledger", "max-risk route is release-readable evidence and never silent behavior", source_surface_id="max_risk_strategy_ring_package", required=False, risk="active_research"),
            edge("pixel_region_map_to_action_background", "pixel_region_motion_map", "action_background_separation_gate", "pixel-region motion checks pressure-derived leakage before action/background route nomination", source_surface_id="pixel_region_motion_map", required=False, risk="observer_to_route"),
            edge("pixel_region_map_to_model_attractor", "pixel_region_motion_map", "model_attractor", "visible pixel evidence must return to the model attractor instead of becoming an independent controller", source_surface_id="pixel_region_motion_map", required=False, risk="observer_to_route"),
            edge("pixel_region_map_to_ledger", "pixel_region_motion_map", "math_topology_ledger", "pixel-region motion becomes release-readable topology evidence", source_surface_id="pixel_region_motion_map", required=False, risk="observer_to_route"),
            edge("pixel_region_map_to_pressure_disagreement", "pixel_region_motion_map", "pixel_pressure_disagreement_review", "selected visible Outcome reweights scalar pressure before interpretation", source_surface_id="pixel_pressure_disagreement_review", required=False, risk="observer_to_route"),
            edge("action_background_to_pressure_disagreement", "action_background_separation_gate", "pixel_pressure_disagreement_review", "pressure-derived action/background split is checked against selected visible pixels", source_surface_id="pixel_pressure_disagreement_review", required=False, risk="observer_to_route"),
            edge("pressure_disagreement_to_control_surface", "pixel_pressure_disagreement_review", "strategy_control_surface", "corrected pixel/pressure axis may nominate a future bounded route only after fixed-seed proof", source_surface_id="pixel_pressure_disagreement_review", required=False, risk="research"),
            edge("pressure_disagreement_to_model_attractor", "pixel_pressure_disagreement_review", "model_attractor", "pixel-corrected pressure must return to the model attractor instead of becoming independent control", source_surface_id="pixel_pressure_disagreement_review", required=False, risk="observer_to_route"),
            edge("pressure_disagreement_to_ledger", "pixel_pressure_disagreement_review", "math_topology_ledger", "pixel/pressure disagreement becomes release-readable topology evidence", source_surface_id="pixel_pressure_disagreement_review", required=False, risk="observer_to_route"),
            edge("pressure_disagreement_to_reweighting_proposal", "pixel_pressure_disagreement_review", "pressure_pixel_reweighting_proposal", "corrected disagreement becomes bounded next-run weights, not current-run mutation", source_surface_id="pressure_pixel_reweighting_proposal", required=False, risk="observer_to_route"),
            edge("reweighting_proposal_to_control_surface", "pressure_pixel_reweighting_proposal", "strategy_control_surface", "bounded weights may nominate a future low-mid-window A/B route only after fixed-seed proof", source_surface_id="pressure_pixel_reweighting_proposal", required=False, risk="research"),
            edge("reweighting_proposal_to_model_attractor", "pressure_pixel_reweighting_proposal", "model_attractor", "pressure/pixel weights must serve the model attractor instead of becoming independent physics", source_surface_id="pressure_pixel_reweighting_proposal", required=False, risk="observer_to_route"),
            edge("reweighting_proposal_to_ledger", "pressure_pixel_reweighting_proposal", "math_topology_ledger", "bounded pressure/pixel weights become release-readable topology evidence", source_surface_id="pressure_pixel_reweighting_proposal", required=False, risk="observer_to_route"),
            edge("reweighting_proposal_to_active_candidate", "pressure_pixel_reweighting_proposal", "pressure_pixel_reweighting_active_candidate", "fixed-seed proof may become a quality-guarded low-branch A/B candidate", source_surface_id="pressure_pixel_reweighting_active_candidate", required=False, risk="active_research"),
            edge("active_candidate_to_control_surface", "pressure_pixel_reweighting_active_candidate", "strategy_control_surface", "candidate delta returns through the normal Strategy control surface before latent delta control", source_surface_id="pressure_pixel_reweighting_active_candidate", required=False, risk="active_research"),
            edge("active_candidate_to_model_attractor", "pressure_pixel_reweighting_active_candidate", "model_attractor", "candidate remains subordinate to the model attractor instead of becoming independent physics", source_surface_id="pressure_pixel_reweighting_active_candidate", required=False, risk="active_research"),
            edge("active_candidate_to_ledger", "pressure_pixel_reweighting_active_candidate", "math_topology_ledger", "active candidate becomes release-readable topology evidence", source_surface_id="pressure_pixel_reweighting_active_candidate", required=False, risk="active_research"),
            edge("action_background_report_to_control_surface", "action_background_separation_gate", "strategy_control_surface", "separated action/background evidence may nominate a bounded future route, never same-run tensor control", source_surface_id="action_background_separation_gate", required=False, risk="research"),
            edge("action_background_to_model_attractor", "action_background_separation_gate", "model_attractor", "separated carrier roles must return to the model attractor instead of becoming independent controllers", source_surface_id="action_background_separation_gate", required=False, risk="observer_to_route"),
            edge("action_background_to_ledger", "action_background_separation_gate", "math_topology_ledger", "action/background separation becomes release-readable topology evidence", source_surface_id="action_background_separation_gate", required=False, risk="observer_to_route"),
            edge("fractal_intersections_to_control_surface", "fractal_strategy_intersection_map", "strategy_control_surface", "dominant fractal axis may nominate a bounded route, but remains report-only until fixed-seed proof", source_surface_id="fractal_strategy_intersection_map", required=False, risk="research"),
            edge("fractal_intersections_to_model_attractor", "fractal_strategy_intersection_map", "model_attractor", "recursive Strategy intersections align around the model attractor instead of replacing model physics", source_surface_id="fractal_strategy_intersection_map", required=False, risk="observer_to_route"),
            edge("fractal_intersections_to_ledger", "fractal_strategy_intersection_map", "math_topology_ledger", "seven-layer intersection map becomes release-readable topology evidence", source_surface_id="fractal_strategy_intersection_map", required=False, risk="observer_to_route"),
            edge("true_region_to_spatial_preservation", "true_region_topology_evidence", "spatial_carrier_preservation_map", "only proven region topology may become spatial preservation pressure", source_surface_id="true_region_topology_evidence", required=False, risk="research"),
            edge("true_region_to_action_background", "true_region_topology_evidence", "action_background_separation_gate", "region roles may separate center action from edge/background before returning to the model attractor", source_surface_id="true_region_topology_evidence", required=False, risk="research"),
            edge("tail_to_boundary_background", "source_tail_carrier", "boundary_background_anchor_control", "selected tail OutcomePrevious becomes boundary/source evidence", source_surface_id="boundary_background_anchor_control", required=False, risk="observer_to_route"),
            edge("boundary_to_r126", "boundary_background_anchor_control", "r126_low_mid_window_route", "boundary evidence may enter only a route-key matched low mid-window", source_surface_id="r126_low_mid_window_route", required=False, risk="active_research"),
            edge("r126_to_denoise_phase", "r126_low_mid_window_route", "denoise_phase_map", "active route must prove low mid-window denoise phase", source_surface_id="denoise_phase_map", required=False, risk="active_research"),
            edge("denoise_to_spatial_preservation", "denoise_phase_map", "spatial_carrier_preservation_map", "only phase-safe delta can receive spatial preservation pressure", source_surface_id="spatial_carrier_preservation_map", required=False, risk="active_research"),
            edge("spatial_preservation_to_control_surface", "spatial_carrier_preservation_map", "strategy_control_surface", "bounded spatial pressure folds back to global Strategy", source_surface_id="strategy_control_surface", required=False, risk="active_research"),
            edge("spatial_preservation_to_action_background", "spatial_carrier_preservation_map", "action_background_separation_gate", "spatial role pressure is separated into center action vs edge background before model-attractor pressure", source_surface_id="action_background_separation_gate", required=False, risk="active_research"),
            edge("action_background_to_control_surface", "action_background_separation_gate", "strategy_control_surface", "background motion leakage is compressed while center action returns to S_global_event_route", source_surface_id="action_background_separation_gate", required=False, risk="active_research"),
            edge("noise_to_bridge", "source_noise_field", "noise_field_strategy_bridge", "source/noise field names the safe future shaping surface", source_surface_id="noise_field_strategy_bridge", required=False, risk="observer"),
            edge("bridge_to_birth_shaping", "noise_field_strategy_bridge", "source_noise_birth_shaping", "only pre-high seed evidence may become tiny source/noise shaping", source_surface_id="source_noise_birth_shaping", required=False, risk="active_research"),
            edge("birth_shaping_to_control_surface", "source_noise_birth_shaping", "strategy_control_surface", "pre-high seed shaping is still subordinate to the selected Strategy control surface", source_surface_id="strategy_control_surface", required=False, risk="active_research"),
            edge("birth_shaping_to_model_attractor", "source_noise_birth_shaping", "model_attractor", "source/noise shaping must help the model understand the event, not replace sampler physics", source_surface_id="source_noise_birth_shaping", required=False, risk="active_research"),
            edge("latent_memory_to_control_surface", "segment_entry_latent_memory_bridge", "strategy_control_surface", "previous latent OutcomePrevious returns only through explicit bounded memory", source_surface_id="segment_entry_latent_memory_bridge", required=False, risk="research"),
            edge("strategy_matrix_to_ledger", "strategy_matrix", "math_topology_ledger", "local micro-formula observations become release-readable topology", source_surface_id="strategy_matrix", required=False, risk="observer"),
            edge("math_mode_to_public_surface", "math_control_summary", "public_surface_contract", "UI/runtime modes must be classified before release readiness", source_surface_id="public_surface_contract", risk="public_guard"),
            edge("public_surface_to_package_static_scan", "public_surface_contract", "public_package_static_scan", "UI/runtime public surface must agree with package files", source_surface_id="public_package_static_scan", risk="public_guard"),
            edge("public_surface_to_ledger", "public_surface_contract", "math_topology_ledger", "public/research boundary becomes part of math topology", source_surface_id="math_topology_ledger", risk="public_guard"),
            edge("package_static_scan_to_ledger", "public_package_static_scan", "math_topology_ledger", "public package files become release-readable topology", source_surface_id="public_package_static_scan", risk="public_guard"),
            edge("ledger_to_readiness", "math_topology_ledger", "public_release_readiness_gate", "all math surfaces return to one public/research verdict", source_surface_id="math_topology_ledger", risk="public_guard"),
            edge("package_static_scan_to_release_manifest", "public_package_static_scan", "public_release_candidate_manifest", "package file cleanliness becomes package/no-package evidence", source_surface_id="public_package_static_scan", risk="public_guard"),
            edge("readiness_to_release_manifest", "public_release_readiness_gate", "public_release_candidate_manifest", "public/research readiness becomes package/no-package evidence", risk="public_guard"),
            edge("ledger_to_release_manifest", "math_topology_ledger", "public_release_candidate_manifest", "math topology proof becomes package/no-package evidence", source_surface_id="math_topology_ledger", risk="public_guard"),
            edge("release_manifest_to_human_report", "public_release_candidate_manifest", "human_report_top_summary", "package verdict must be visible before raw records", risk="report"),
            edge("release_manifest_to_package_verdict", "public_release_candidate_manifest", "public_package_verdict", "release package is an Outcome only after manifest convergence", risk="release_gate"),
            edge("readiness_to_human_report", "public_release_readiness_gate", "human_report_top_summary", "readiness verdict must stay visible before raw records", risk="report"),
            edge("control_surface_to_model_attractor", "strategy_control_surface", "model_attractor", "math acts as bounded attractor guidance, not model replacement", source_surface_id="strategy_control_surface", required=False, risk="research"),
            edge("model_attractor_to_video_outcome", "model_attractor", "visible_video_outcome", "the final proof is visible Outcome(t+1), not report fields alone", required=False, risk="visual_review"),
        ]

        required_missing = [e["edge_id"] for e in edges if e["required"] and not e["present"]]
        active_edges = [e for e in edges if e["active_generation_edge"]]
        research_edges = [e for e in edges if e["research_edge"] and e["present"]]

        if not ledger:
            status = "missing_math_topology_ledger"
            severity = "WARNING"
            next_action = "Record EventMathTopologyLedger before interpreting math dependencies."
        elif required_missing:
            status = "incomplete_math_topology_graph"
            severity = "WARNING"
            next_action = "Required dependency edges are missing; treat public/release interpretation as incomplete."
        elif active_edges:
            status = "active_math_dependency_graph"
            severity = "RESEARCH"
            next_action = "Active math dependency edges exist; compare against safe baseline and inspect video before release."
        elif research_edges:
            status = "research_dependency_graph"
            severity = "WARNING"
            next_action = "Research/diagnostic dependency edges are visible, but no active generation edge is proven active."
        else:
            status = "observer_dependency_graph"
            severity = "PASS"
            next_action = "Graph is observer/public-boundary only; release still needs VIDEO + visual inspection."

        return {
            "stage": "EventMathTopologyDependencyGraph",
            "status": status,
            "severity": severity,
            "graph_version": "math_topology_dependency_graph_v1",
            "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
            "runtime_name": EVENT_HORIZON_RUNTIME_NAME,
            "formula": "The math body is returned as a dependency graph: each local Strategy surface must declare where it receives evidence, where it sends pressure, and whether it can affect generation.",
            "control_mode": "REPORT_ONLY",
            "does_not_change_generation": True,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "present_node_count": sum(1 for n in nodes if n.get("present")),
            "present_edge_count": sum(1 for e in edges if e.get("present")),
            "required_missing_edge_ids": required_missing,
            "active_generation_edge_count": len(active_edges),
            "active_generation_edge_ids": [e["edge_id"] for e in active_edges],
            "research_edge_count": len(research_edges),
            "research_edge_ids": [e["edge_id"] for e in research_edges],
            "nodes": nodes,
            "edges": edges,
            "next_action": next_action,
        }

    def _event_strategy_matrix_from_records(self, execution_records, result_status="", saved_video_path=""):
        """
        Report-only map of places where carriers collide into Strategy.
        This is deliberately observer-only: it creates evidence records, never sampler control.
        """
        records = [r for r in (execution_records or []) if isinstance(r, dict)]

        def _flatten_record_text(value, depth=0):
            if depth > 3:
                return ""
            if value is None:
                return ""
            if isinstance(value, (str, int, float, bool)):
                return str(value)
            if isinstance(value, dict):
                parts = []
                for key, item in value.items():
                    parts.append(str(key))
                    parts.append(_flatten_record_text(item, depth + 1))
                return " ".join(parts)
            if isinstance(value, (list, tuple)):
                return " ".join(_flatten_record_text(item, depth + 1) for item in value[:24])
            return str(type(value).__name__)

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
                str(record.get("compiler_version", "") or ""),
                str(record.get("transcoder_version", "") or ""),
                str(record.get("object_topology_status", "") or ""),
                str(record.get("object_relation_ontology_status", "") or ""),
                str(record.get("object_relation_strategy_point", "") or ""),
                _flatten_record_text(record.get("strategy_graph")),
                _flatten_record_text(record.get("object_topology_map")),
                _flatten_record_text(record.get("object_relation_ontology")),
                _flatten_record_text(record.get("model_language_transcode")),
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
            "text": hits("eventtextencode", "text encode", "conditioning", "eventpromptstrategycompiler", "eventpromptstrategytranscode", "prompt strategy", "prompt transcode"),
            "image": hits("eventimagescalestart", "image scale", "source image", "eventimagescale"),
            "latent_seed": hits("eventwanimagetovideoseed", "wan_i2v_latent_seed", "wan image to video"),
            "high_sampler": hits("eventsamplerhigh", "samplerhigh", "branch_name high", "branch high"),
            "low_sampler": hits("eventsamplerlow", "samplerlow", "branch_name low", "branch low"),
            "delta_control": hits("eventmathdeltacontrol", "eventstrategypressurewindow", "eventstrategycontrolsurface", "latent_delta_scale", "strategy_pressure_window", "delta control"),
            "cfg_policy": hits("eventstrategycfg", "eventmathsamplerpathpolicy", "cfg_policy"),
            "motion": hits("eventmath_decoded_frame_motion", "eventmath_concatenated_frame_motion", "frame_motion"),
            "cascade": hits("singularitycascadebegin", "singularitycascadesegmentend", "singularitycascadeend"),
            "pause": hits("singularitycascadepause", "singularitycascaderesume", "mirrorcut"),
            "boundary": hits("eventuniversalboundary", "cascadeboundary", "cascade boundary"),
            "tail_formula": hits("formula_tail_mirror_break", "tailframesselect", "admissible_continuation"),
            "object_topology": hits("eventobjecttopologycarrier", "object_topology", "objecttopologycarrier", "rigid_object", "rigidity_lock"),
            "object_relation": hits("eventobjectrelationontology", "object_relation_ontology", "object_contact_strategy", "carrier_roles", "contact_boundary_carrier", "rigid_physical_carrier"),
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
            "object_relation_sentence_count",
            "rigid_object_count",
            "topology_pressure_score",
            "contact_pressure_score",
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

            elif collision_id == "object_relation_ontology":
                sentence_count = latest_snapshot_value(snapshots, "object_relation_sentence_count", None)
                rigid_count = latest_snapshot_value(snapshots, "rigid_object_count", None)
                if sentence_count is not None:
                    basis["metrics_used"].append("object_relation_sentence_count")
                    basis["object_relation_sentence_count"] = sentence_count
                if rigid_count is not None:
                    basis["metrics_used"].append("rigid_object_count")
                    basis["rigid_object_count"] = rigid_count
                if status == "observed":
                    conflict_score = max(conflict_score, 0.0)
                    drift_score = None
                    basis["reason"] = "ontology carrier is present; visible carrier/contact scoring is not implemented yet"

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
                "intervention_surface": ["STRATEGY_PRESSURE_WINDOW", "LATENT_DELTA_SCALE", "bounded coupling multiplier", "deep-step research only after evidence"],
                "public_safe_control": "bounded_latent_delta_research",
            },
            "object_relation_ontology": {
                "left_outcome": ["source image object layout", "ObjectTopologyCarrier", "prompt relation map"],
                "left_observed_behavior": ["positive Strategy transform", "object/contact/motion relation pressure"],
                "strategy_point": "Object identity, contact boundary, and relative motion must describe the same event.",
                "right_observed_behavior": ["sampler response to object relation", "high-low refinement pressure", "visible contact motion"],
                "right_outcome": ["visible carrier identity", "readable contact boundary", "continued object relation"],
                "collision_math": ["carrier_persistence_score", "contact_boundary_continuity_score", "topology_seam_score", "object_relation_drift_score"],
                "intervention_surface": ["ObjectRelationReview", "future ROI/contact scorer", "conditioning relation weighting", "high-low route attribution"],
                "public_safe_control": "report_only_visual_scoring",
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
                "object_relation_ontology",
                ["rigid_physical_carrier", "soft_contact_carrier", "contact_boundary_carrier", "relative_motion_path"],
                ["object_topology", "object_relation"],
                ["text_any", "image", "high_sampler", "low_sampler", "motion", "boundary"],
                "Object carrier identity collides with contact/motion behavior and must survive into visible frames.",
                "Object relation Strategy point / carrier identity + contact boundary",
                "This is the missing bridge between prompt ontology and visible/sampler evidence; add ObjectRelationReview before stronger control.",
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
            "EventStrategyPressureWindow",
            "EventStrategyControlSurface",
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
            "OBJECT_TOPOLOGY": bool(categories["object_topology"]),
            "OBJECT_RELATION_ONTOLOGY": bool(categories["object_relation"]),
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
            "matrix_version": "strategy_matrix_v4_object_relation_review_report_only",
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

    def _event_object_relation_review_from_records(self, execution_records, strategy_matrix=None, vector_collisions=None):
        """
        Report-only review of the object/contact Strategy point.
        This reads already-recorded prompt, sampler, boundary, and frame-motion evidence.
        It never modifies conditioning, tensors, samplers, or routing.
        """
        records = [r for r in (execution_records or []) if isinstance(r, dict)]
        vector_collisions = [c for c in (vector_collisions or []) if isinstance(c, dict)]

        def safe_float(value, default=None):
            try:
                out = float(value)
            except Exception:
                return default
            return out if math.isfinite(out) else default

        def clamp01(value):
            value = safe_float(value, 0.0)
            return max(0.0, min(1.0, value))

        def first_record(stage_name):
            for rec in records:
                if str(rec.get("stage", "") or "") == stage_name:
                    return rec
            return {}

        def latest_record(stage_name):
            for rec in reversed(records):
                if str(rec.get("stage", "") or "") == stage_name:
                    return rec
            return {}

        prompt_apply = latest_record("EventPromptStrategyTranscodeApply")
        relation_collision = {}
        motion_collision = {}
        high_low_collision = {}
        tail_collision = {}
        for item in vector_collisions:
            cid = str(item.get("collision_id") or "")
            if cid == "object_relation_ontology":
                relation_collision = item
            elif cid == "previous_next_frame_motion":
                motion_collision = item
            elif cid == "high_low_sampler_strategy":
                high_low_collision = item
            elif cid == "tail_next_source":
                tail_collision = item

        relation_active = bool(
            prompt_apply.get("object_relation_ontology_applied")
            or prompt_apply.get("object_relation_ontology_status") == "active"
            or relation_collision.get("status") == "observed"
        )

        motion_candidates = []
        for idx, rec in enumerate(records):
            stage = str(rec.get("stage", "") or "")
            if stage in ("EventMath_concatenated_frame_motion", "EventMath_decoded_frame_motion") or stage.endswith("_frame_motion"):
                if str(rec.get("status", "") or "") == "ok":
                    motion_candidates.append((idx, rec))

        def motion_rank(item):
            idx, rec = item
            stage = str(rec.get("stage", "") or "")
            if stage == "EventMath_concatenated_frame_motion":
                rank = 4
            elif stage == "EventMath_decoded_frame_motion":
                rank = 3
            elif stage.startswith("EventMath_cascade_"):
                rank = 2
            else:
                rank = 1
            return (rank, idx)

        motion_record = sorted(motion_candidates, key=motion_rank)[-1][1] if motion_candidates else {}
        motion_available = bool(motion_record)

        stability = safe_float(motion_record.get("frame_motion_stability_score"), None)
        spike = safe_float(motion_record.get("frame_delta_spike_ratio"), 1.0)
        reversal = safe_float(motion_record.get("frame_delta_reversal_ratio"), 0.0)
        jerk = safe_float(motion_record.get("frame_delta_jerk_ratio"), 0.0)
        cv_ratio = safe_float(motion_record.get("frame_delta_norm_cv_ratio"), 0.0)
        motion_abs_mean = safe_float(motion_record.get("frame_delta_abs_mean"), None)
        cascade_seam_review = latest_record("EventCascadeSeamMotionReview")
        cascade_seam_status = str(cascade_seam_review.get("status", "") if isinstance(cascade_seam_review, dict) else "")
        cascade_post_seam_score = safe_float(
            cascade_seam_review.get("post_seam_acceleration_score") if isinstance(cascade_seam_review, dict) else None,
            None,
        )
        cascade_post_seam_attribution = str(
            cascade_seam_review.get("attribution", "") if isinstance(cascade_seam_review, dict) else ""
        )
        cascade_max_post_segment_ratio = safe_float(
            cascade_seam_review.get("max_post_segment_to_previous_segment_mean_ratio") if isinstance(cascade_seam_review, dict) else None,
            None,
        )
        cascade_max_boundary_ratio = safe_float(
            cascade_seam_review.get("max_boundary_to_previous_segment_mean_ratio") if isinstance(cascade_seam_review, dict) else None,
            None,
        )

        boundary_records = [
            rec for rec in records
            if str(rec.get("stage", "") or "") == "EventMathCascadeBoundary"
            and str(rec.get("status", "") or "") == "ok"
        ]
        boundary_abs_values = [
            safe_float(rec.get("boundary_delta_abs_mean"), None)
            for rec in boundary_records
        ]
        boundary_abs_values = [v for v in boundary_abs_values if v is not None]
        max_boundary_abs_mean = max(boundary_abs_values) if boundary_abs_values else None
        seam_to_motion_abs_ratio = None
        if motion_abs_mean is not None and motion_abs_mean > 0 and max_boundary_abs_mean is not None:
            seam_to_motion_abs_ratio = float(max_boundary_abs_mean / (motion_abs_mean + 1e-12))

        high_low_pressure = max(
            clamp01(high_low_collision.get("conflict_score", 0.0)),
            clamp01(high_low_collision.get("drift_score", 0.0)),
        ) if high_low_collision else 0.0
        tail_pressure = max(
            clamp01(tail_collision.get("conflict_score", 0.0)),
            clamp01(tail_collision.get("drift_score", 0.0)),
        ) if tail_collision else 0.0

        scores_available = relation_active and motion_available
        carrier_persistence_score = None
        contact_boundary_continuity_score = None
        topology_seam_score = None
        object_relation_drift_score = None
        attribution = "not_scored"
        pressure_terms = {}

        if scores_available:
            motion_stability = clamp01(stability if stability is not None else 0.5)
            spike_pressure = clamp01(((spike if spike is not None else 1.0) - 1.0) / 1.5)
            reversal_pressure = clamp01(reversal if reversal is not None else 0.0)
            jerk_pressure = clamp01(jerk if jerk is not None else 0.0)
            cv_pressure = clamp01(cv_ratio if cv_ratio is not None else 0.0)
            seam_pressure = (
                clamp01((float(seam_to_motion_abs_ratio) - 1.0) / 1.5)
                if seam_to_motion_abs_ratio is not None
                else 0.0
            )
            post_seam_acceleration_pressure = clamp01(cascade_post_seam_score if cascade_post_seam_score is not None else 0.0)

            carrier_persistence_score = clamp01(
                0.50 * motion_stability
                + 0.18 * (1.0 - spike_pressure)
                + 0.14 * (1.0 - jerk_pressure)
                + 0.10 * (1.0 - reversal_pressure)
                + 0.08 * (1.0 - cv_pressure)
            )
            contact_boundary_continuity_score = clamp01(
                0.40 * motion_stability
                + 0.20 * (1.0 - seam_pressure)
                + 0.18 * (1.0 - spike_pressure)
                + 0.12 * (1.0 - jerk_pressure)
                + 0.10 * (1.0 - reversal_pressure)
            )
            if cascade_seam_status == "reviewed":
                contact_boundary_continuity_score = clamp01(
                    contact_boundary_continuity_score * (1.0 - 0.18 * post_seam_acceleration_pressure)
                )
            topology_seam_score = clamp01(1.0 - seam_pressure) if seam_to_motion_abs_ratio is not None else None

            score_terms = [carrier_persistence_score, contact_boundary_continuity_score]
            if topology_seam_score is not None:
                score_terms.append(topology_seam_score)
            relation_coherence = sum(score_terms) / max(1, len(score_terms))
            sampler_tail_pressure = max(high_low_pressure, tail_pressure)
            object_relation_drift_score = clamp01(
                0.70 * (1.0 - relation_coherence)
                + 0.20 * sampler_tail_pressure
                + 0.10 * max(spike_pressure, jerk_pressure, reversal_pressure, post_seam_acceleration_pressure)
            )

            pressure_terms = {
                "motion_stability": motion_stability,
                "spike_pressure": spike_pressure,
                "reversal_pressure": reversal_pressure,
                "jerk_pressure": jerk_pressure,
                "cv_pressure": cv_pressure,
                "seam_pressure": seam_pressure,
                "post_seam_acceleration_pressure": post_seam_acceleration_pressure,
                "high_low_pressure": high_low_pressure,
                "tail_pressure": tail_pressure,
            }

            pressure_map = {
                "cascade_boundary_pressure": seam_pressure,
                "post_seam_acceleration_pressure": post_seam_acceleration_pressure,
                "visible_motion_pressure": max(spike_pressure, jerk_pressure, reversal_pressure),
                "high_low_sampler_pressure": high_low_pressure,
                "tail_resume_pressure": tail_pressure,
            }
            attribution = max(pressure_map, key=pressure_map.get)
            if pressure_map.get(attribution, 0.0) < 0.25:
                attribution = "no_single_dominant_pressure"

        if not relation_active:
            status = "inactive_no_object_relation_ontology"
            next_action = "Verify the report-only semantic density/object-relation map before scoring object/contact continuity; do not inject formula text into the prompt."
        elif not motion_available:
            status = "awaiting_visible_motion_evidence"
            next_action = "Run a completed VIDEO pass so ObjectRelationReview can read frame-motion evidence."
        else:
            status = "reviewed"
            next_action = "Compare fixed-seed A/B videos and inspect whether carrier identity and contact boundary match the scores."

        review = {
            "stage": "EventObjectRelationReview",
            "status": status,
            "review_version": "object_relation_review_v1_report_only",
            "formula": "Object identity + contact boundary = Strategy(relation) = visible carrier outcome + motion behavior.",
            "control_mode": "REPORT_ONLY",
            "active_control_allowed": False,
            "object_relation_ontology_active": bool(relation_active),
            "object_relation_sentence_count": prompt_apply.get("object_relation_sentence_count", 0),
            "rigid_object_count": prompt_apply.get("rigid_object_count", 0),
            "prompt_transform_applied": bool(prompt_apply.get("positive_prompt_transformed")),
            "negative_prompt_transformed": bool(prompt_apply.get("negative_prompt_transformed")),
            "motion_stage": str(motion_record.get("stage", "") if isinstance(motion_record, dict) else ""),
            "motion_profile": str(motion_record.get("frame_motion_profile", "") if isinstance(motion_record, dict) else ""),
            "frame_motion_stability_score": stability,
            "frame_delta_spike_ratio": spike,
            "frame_delta_reversal_ratio": reversal,
            "frame_delta_jerk_ratio": jerk,
            "frame_delta_norm_cv_ratio": cv_ratio,
            "boundary_count": len(boundary_records),
            "max_boundary_delta_abs_mean": max_boundary_abs_mean,
            "seam_to_motion_abs_ratio": seam_to_motion_abs_ratio,
            "cascade_seam_review_status": cascade_seam_status,
            "cascade_post_seam_acceleration_score": cascade_post_seam_score,
            "cascade_post_seam_attribution": cascade_post_seam_attribution,
            "cascade_max_post_segment_to_previous_segment_mean_ratio": cascade_max_post_segment_ratio,
            "cascade_max_boundary_to_previous_segment_mean_ratio": cascade_max_boundary_ratio,
            "carrier_persistence_score": carrier_persistence_score,
            "contact_boundary_continuity_score": contact_boundary_continuity_score,
            "topology_seam_score": topology_seam_score,
            "object_relation_drift_score": object_relation_drift_score,
            "attribution": attribution,
            "pressure_terms": pressure_terms,
            "score_policy": "Proxy scores from global motion and cascade boundary records; no ROI/contact detector yet.",
            "next_metric_route": [
                "carrier_persistence_score -> future object mask / feature-region tracker",
                "contact_boundary_continuity_score -> future contact ROI continuity scorer",
                "topology_seam_score -> cascade boundary relation continuity",
                "object_relation_drift_score -> gate for any future bounded object-relation guidance",
            ],
            "next_action": next_action,
        }
        return review

    def _event_apply_object_relation_review(self, object_relation_review, strategy_matrix, vector_collisions):
        if not isinstance(object_relation_review, dict):
            return strategy_matrix, vector_collisions
        strategy_matrix = strategy_matrix if isinstance(strategy_matrix, dict) else {}
        vector_collisions = [c for c in (vector_collisions or []) if isinstance(c, dict)]

        def safe_float(value, default=None):
            try:
                out = float(value)
            except Exception:
                return default
            return out if math.isfinite(out) else default

        drift = safe_float(object_relation_review.get("object_relation_drift_score"), None)
        carrier = safe_float(object_relation_review.get("carrier_persistence_score"), None)
        contact = safe_float(object_relation_review.get("contact_boundary_continuity_score"), None)
        seam = safe_float(object_relation_review.get("topology_seam_score"), None)

        strategy_matrix["object_relation_review_status"] = object_relation_review.get("status", "")
        strategy_matrix["object_relation_review_version"] = object_relation_review.get("review_version", "")
        strategy_matrix["object_relation_drift_score"] = drift
        strategy_matrix["carrier_persistence_score"] = carrier
        strategy_matrix["contact_boundary_continuity_score"] = contact
        strategy_matrix["topology_seam_score"] = seam
        if drift is not None:
            current_max = safe_float(strategy_matrix.get("max_drift_score"), None)
            if current_max is None or drift >= current_max:
                strategy_matrix["max_drift_score"] = drift
                strategy_matrix["top_drift_collision"] = "object_relation_ontology"

        for item in vector_collisions:
            if str(item.get("collision_id") or "") != "object_relation_ontology":
                continue
            basis = item.get("score_basis", {}) if isinstance(item.get("score_basis"), dict) else {}
            basis["object_relation_review"] = {
                "review_version": object_relation_review.get("review_version", ""),
                "status": object_relation_review.get("status", ""),
                "carrier_persistence_score": carrier,
                "contact_boundary_continuity_score": contact,
                "topology_seam_score": seam,
                "object_relation_drift_score": drift,
                "cascade_seam_review_status": object_relation_review.get("cascade_seam_review_status", ""),
                "cascade_post_seam_acceleration_score": object_relation_review.get("cascade_post_seam_acceleration_score", None),
                "cascade_post_seam_attribution": object_relation_review.get("cascade_post_seam_attribution", ""),
                "cascade_max_post_segment_to_previous_segment_mean_ratio": object_relation_review.get("cascade_max_post_segment_to_previous_segment_mean_ratio", None),
                "cascade_max_boundary_to_previous_segment_mean_ratio": object_relation_review.get("cascade_max_boundary_to_previous_segment_mean_ratio", None),
                "attribution": object_relation_review.get("attribution", ""),
                "pressure_terms": object_relation_review.get("pressure_terms", {}),
            }
            metrics = basis.setdefault("metrics_used", [])
            if isinstance(metrics, list):
                for key in (
                    "carrier_persistence_score",
                    "contact_boundary_continuity_score",
                    "topology_seam_score",
                    "object_relation_drift_score",
                ):
                    if key not in metrics:
                        metrics.append(key)
            item["score_basis"] = basis
            if drift is not None:
                item["drift_score"] = drift
                item["conflict_score"] = max(
                    safe_float(item.get("conflict_score"), 0.0) or 0.0,
                    drift,
                )
                item["recommendation"] = (
                    "ObjectRelationReview is now measured. Keep this report-only and compare fixed-seed videos before any active guidance."
                )
            local_formula = item.get("local_formula", {}) if isinstance(item.get("local_formula"), dict) else {}
            collision_math = local_formula.get("collision_math", {}) if isinstance(local_formula.get("collision_math"), dict) else {}
            collision_math["measured_now"] = True
            collision_math["object_relation_review"] = basis.get("object_relation_review", {})
            collision_math["score_basis"] = basis
            local_formula["collision_math"] = collision_math
            item["local_formula"] = local_formula
            break

        return strategy_matrix, vector_collisions

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

            if cid == "object_relation_ontology":
                review = basis.get("object_relation_review", {}) if isinstance(basis.get("object_relation_review", {}), dict) else {}
                review_status = str(review.get("status") or "")
                review_drift = safe_float(review.get("object_relation_drift_score"), None)
                seam_acceleration = safe_float(review.get("cascade_post_seam_acceleration_score"), None)
                seam_attribution = str(review.get("cascade_post_seam_attribution") or "")
                priority = "high"
                message = "Object relation ontology is active, but visible carrier/contact continuity is not scored yet."
                if review_status == "reviewed":
                    if review_drift is not None and review_drift < 0.25:
                        priority = "low"
                    elif review_drift is not None and review_drift < 0.45:
                        priority = "medium"
                    message = "ObjectRelationReview measured carrier/contact continuity as report-only evidence."
                    if seam_acceleration is not None and seam_acceleration >= 0.35:
                        priority = "high"
                        message = "Cascade seam review detected post-seam motion pressure; compare prompt-per-segment Strategy before damping."
                add(
                    cid,
                    "object_relation_review",
                    priority,
                    message,
                    "Compare fixed-seed runs by carrier identity, contact boundary continuity, seam-local relation drift, and high-low attribution.",
                    control_surface="report_only_visual_scoring",
                    evidence={
                        "status": status,
                        "score_basis": basis,
                        "object_relation_review": review,
                        "cascade_post_seam_acceleration_score": seam_acceleration,
                        "cascade_post_seam_attribution": seam_attribution,
                    },
                )

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
            "proposal_version": "strategy_guidance_v2_object_relation_review",
            "formula": "Strategy Matrix scores become test proposals; they do not modify generation.",
            "active_control_allowed": False,
            "control_mode": "REPORT_ONLY",
            "top_conflict_collision": top_conflict,
            "top_drift_collision": top_drift,
            "proposal_count": len(proposals),
            "proposals": proposals,
            "public_default_policy": "Keep public defaults high_delta_strength=1.0 and low_delta_strength=1.0 until fixed-seed visual evidence proves a bounded preset.",
        }

    def _event_relation_pressure_cards_from_records(
        self,
        execution_records,
        strategy_matrix=None,
        object_relation_review=None,
        vector_collisions=None,
    ):
        """
        Compact report-only cards for the current Strategy pressure map.
        These records turn scattered evidence into readable diagnostics and never
        modify prompts, tensors, sampler state, or cascade routing.
        """
        records = [r for r in (execution_records or []) if isinstance(r, dict)]
        strategy_matrix = strategy_matrix if isinstance(strategy_matrix, dict) else {}
        object_relation_review = object_relation_review if isinstance(object_relation_review, dict) else {}
        vector_collisions = [c for c in (vector_collisions or []) if isinstance(c, dict)]

        def safe_float(value, default=None):
            try:
                out = float(value)
            except Exception:
                return default
            return out if math.isfinite(out) else default

        def clamp01(value):
            value = safe_float(value, 0.0)
            return max(0.0, min(1.0, value))

        def latest_record(stage_name):
            for rec in reversed(records):
                if str(rec.get("stage", "") or "") == stage_name:
                    return rec
            return {}

        def first_record(stage_name):
            for rec in records:
                if str(rec.get("stage", "") or "") == stage_name:
                    return rec
            return {}

        def records_by_stage(stage_name):
            return [rec for rec in records if str(rec.get("stage", "") or "") == stage_name]

        def records_by_prefix(prefix):
            return [rec for rec in records if str(rec.get("stage", "") or "").startswith(prefix)]

        def collision_by_id(collision_id):
            for item in vector_collisions:
                if str(item.get("collision_id") or "") == collision_id:
                    return item
            return {}

        def compact_record_refs(stage_names):
            out = []
            for stage in stage_names:
                if stage and stage not in out:
                    out.append(stage)
            return out[:12]

        def status_from_pressure(value, prefix):
            value = clamp01(value)
            if value >= 0.55:
                return f"{prefix}_high"
            if value >= 0.30:
                return f"{prefix}_watch"
            return f"{prefix}_nominal"

        prompt_updates = records_by_stage("EventCascadePromptRuntimeUpdate")
        reused_updates = [
            rec for rec in prompt_updates
            if bool(rec.get("prompt_continuity_reused"))
            or str(rec.get("status") or "") == "reused_active_strategy"
        ]
        changed_updates = [
            rec for rec in prompt_updates
            if str(rec.get("status") or "") == "applied"
            and not bool(rec.get("prompt_continuity_reused"))
        ]
        protected_negative_updates = [
            rec for rec in prompt_updates
            if bool(rec.get("negative_prompt_payload_reused_previous_active"))
        ]
        negative_payload_drift = [
            rec for rec in prompt_updates
            if bool(rec.get("negative_payload_missing"))
            or bool(rec.get("negative_payload_truncated"))
            or bool(rec.get("negative_prompt_changed_from_previous_active"))
        ]
        if not prompt_updates:
            prompt_status = "no_runtime_prompt_update"
        elif changed_updates:
            prompt_status = "changed_runtime_strategy"
        elif protected_negative_updates:
            prompt_status = "protected_negative_payload_drift"
        else:
            prompt_status = "clean_same_strategy"

        prompt_card = {
            "stage": "EventPromptCarrierContinuityCard",
            "status": prompt_status,
            "card_version": "relation_pressure_cards_v1",
            "formula": "Prompt carriers across pause/continue should preserve Strategy unless the user changes them.",
            "control_mode": "REPORT_ONLY",
            "active_control_allowed": False,
            "update_count": len(prompt_updates),
            "reused_update_count": len(reused_updates),
            "changed_update_count": len(changed_updates),
            "protected_negative_payload_reuse_count": len(protected_negative_updates),
            "negative_payload_drift_count": len(negative_payload_drift),
            "last_same_prompt_match_basis": str(prompt_updates[-1].get("same_prompt_match_basis", "") if prompt_updates else ""),
            "last_prompt_continuity_policy": str(prompt_updates[-1].get("prompt_continuity_policy", "") if prompt_updates else ""),
            "last_positive_identity_matches": bool(prompt_updates[-1].get("positive_identity_matches", False)) if prompt_updates else False,
            "last_positive_payload_transforms_to_current_active": bool(prompt_updates[-1].get("positive_payload_transforms_to_current_active", False)) if prompt_updates else False,
            "last_positive_strategy_identity_matches": bool(prompt_updates[-1].get("positive_strategy_identity_matches", False)) if prompt_updates else False,
            "last_negative_identity_matches": bool(prompt_updates[-1].get("negative_identity_matches", False)) if prompt_updates else False,
            "evidence_stages": compact_record_refs([str(rec.get("stage", "") or "") for rec in prompt_updates]),
            "next_action": (
                "If the user did not change prompt text, any changed_runtime_strategy result should be treated as prompt payload drift before tuning math."
                if prompt_status == "changed_runtime_strategy"
                else "Prompt continuity is usable for the next fixed-seed comparison."
            ),
        }

        low_trace_values = []
        low_trace_stages = []
        for rec in records_by_prefix("EventSamplerStepTraceSummary_"):
            stage = str(rec.get("stage", "") or "")
            branch = str(rec.get("branch", rec.get("branch_name", "")) or "").lower()
            if "low" not in stage.lower() and "low" not in branch:
                continue
            for key in ("trace_vs_window_relative_delta", "relative_delta"):
                value = safe_float(rec.get(key), None)
                if value is not None:
                    low_trace_values.append(value)
                    low_trace_stages.append(stage)
        for rec in records_by_prefix("EventMath_"):
            stage = str(rec.get("stage", "") or "")
            if "step_trace_vs_window_output" not in stage.lower() or "low" not in stage.lower():
                continue
            value = safe_float(rec.get("relative_delta"), None)
            if value is not None:
                low_trace_values.append(value)
                low_trace_stages.append(stage)

        high_low_collision = collision_by_id("high_low_sampler_strategy")
        high_low_pressure = max(
            clamp01(high_low_collision.get("conflict_score", 0.0)),
            clamp01(high_low_collision.get("drift_score", 0.0)),
            clamp01((object_relation_review.get("pressure_terms", {}) or {}).get("high_low_pressure", 0.0))
            if isinstance(object_relation_review.get("pressure_terms", {}), dict)
            else 0.0,
        )
        max_low_trace_relative_delta = max(low_trace_values) if low_trace_values else None
        low_trace_pressure = clamp01(((max_low_trace_relative_delta or 1.0) - 1.0) / 1.0)
        low_pressure = max(high_low_pressure, low_trace_pressure)
        low_card = {
            "stage": "EventLowBranchRelationPressureCard",
            "status": status_from_pressure(low_pressure, "low_pressure"),
            "card_version": "relation_pressure_cards_v1",
            "formula": "OutcomeNext(high) becomes StrategyCarrier(low); low ObservedBehavior must refine it without breaking relation identity.",
            "control_mode": "REPORT_ONLY",
            "active_control_allowed": False,
            "high_low_pressure": high_low_pressure,
            "low_trace_pressure": low_trace_pressure,
            "max_low_trace_relative_delta": max_low_trace_relative_delta,
            "low_trace_sample_count": len(low_trace_values),
            "top_drift_collision": strategy_matrix.get("top_drift_collision", ""),
            "evidence_stages": compact_record_refs(low_trace_stages + [str(high_low_collision.get("stage", "") or "")]),
            "next_action": "Inspect low-branch pressure before changing low_delta_strength; this card does not recommend active control by itself.",
        }

        carrier = safe_float(object_relation_review.get("carrier_persistence_score"), None)
        contact = safe_float(object_relation_review.get("contact_boundary_continuity_score"), None)
        seam = safe_float(object_relation_review.get("topology_seam_score"), None)
        drift = safe_float(object_relation_review.get("object_relation_drift_score"), None)
        review_status = str(object_relation_review.get("status") or "not_recorded")
        if review_status != "reviewed":
            object_status = review_status
        elif (drift is not None and drift >= 0.45) or (carrier is not None and carrier < 0.45) or (contact is not None and contact < 0.45):
            object_status = "object_identity_watch"
        elif (drift is not None and drift < 0.25) and (carrier is not None and carrier >= 0.65) and (contact is not None and contact >= 0.65):
            object_status = "object_identity_stable"
        else:
            object_status = "object_identity_measured"
        object_card = {
            "stage": "EventObjectCarrierIdentityCard",
            "status": object_status,
            "card_version": "relation_pressure_cards_v1",
            "formula": "Object carrier identity + contact boundary should remain one readable Strategy relation across visible frames.",
            "control_mode": "REPORT_ONLY",
            "active_control_allowed": False,
            "object_relation_review_status": review_status,
            "carrier_persistence_score": carrier,
            "contact_boundary_continuity_score": contact,
            "topology_seam_score": seam,
            "object_relation_drift_score": drift,
            "object_relation_attribution": object_relation_review.get("attribution", ""),
            "score_policy": object_relation_review.get("score_policy", ""),
            "next_action": "Use this as a visual checklist: carrier identity, boundary continuity, and seam topology must agree before stronger math.",
        }

        seam_review = latest_record("EventCascadeSeamMotionReview")
        tail_collision = collision_by_id("tail_next_source")
        tail_pressure = max(
            clamp01(tail_collision.get("conflict_score", 0.0)),
            clamp01(tail_collision.get("drift_score", 0.0)),
            clamp01((object_relation_review.get("pressure_terms", {}) or {}).get("tail_pressure", 0.0))
            if isinstance(object_relation_review.get("pressure_terms", {}), dict)
            else 0.0,
        )
        post_seam_score = safe_float(seam_review.get("post_seam_acceleration_score"), None) if isinstance(seam_review, dict) else None
        seam_pressure_terms = seam_review.get("pressure_terms", {}) if isinstance(seam_review.get("pressure_terms", {}), dict) else {}
        tail_status = (
            "tail_not_applicable_single_segment"
            if str(seam_review.get("status", "") or "") == "not_applicable"
            else status_from_pressure(max(tail_pressure, clamp01(post_seam_score or 0.0)), "tail_strategy")
        )
        tail_card = {
            "stage": "EventTailStrategyContinuityCard",
            "status": tail_status,
            "card_version": "relation_pressure_cards_v1",
            "formula": "Selected visible tail becomes OutcomePrevious for the next segment; seam behavior must stay attached to the same Strategy.",
            "control_mode": "REPORT_ONLY",
            "active_control_allowed": False,
            "tail_pressure": tail_pressure,
            "cascade_seam_review_status": seam_review.get("status", "") if isinstance(seam_review, dict) else "",
            "observed_segments": seam_review.get("observed_segments", None) if isinstance(seam_review, dict) else None,
            "segment_frame_counts": seam_review.get("segment_frame_counts", []) if isinstance(seam_review, dict) else [],
            "post_seam_acceleration_score": post_seam_score,
            "post_seam_attribution": seam_review.get("attribution", "") if isinstance(seam_review, dict) else "",
            "seam_pressure_terms": seam_pressure_terms,
            "next_action": "If post-seam acceleration repeats, compare prompt-per-segment identity before adding any motion damping.",
        }

        motion_candidates = []
        for rec in records:
            stage = str(rec.get("stage", "") or "")
            if stage in ("EventMath_concatenated_frame_motion", "EventMath_decoded_frame_motion") or stage.endswith("_frame_motion"):
                motion_candidates.append(rec)
        motion_record = motion_candidates[-1] if motion_candidates else {}
        stability = safe_float(motion_record.get("frame_motion_stability_score"), None)
        spike = safe_float(motion_record.get("frame_delta_spike_ratio"), None)
        reversal = safe_float(motion_record.get("frame_delta_reversal_ratio"), None)
        jerk = safe_float(motion_record.get("frame_delta_jerk_ratio"), None)
        spike_pressure = clamp01(((spike if spike is not None else 1.0) - 1.0) / 1.5)
        jerk_pressure = clamp01(jerk or 0.0)
        reversal_pressure = clamp01(reversal or 0.0)
        instability_pressure = clamp01(1.0 - (stability if stability is not None else 1.0))
        frame_pressure = max(spike_pressure, jerk_pressure, reversal_pressure, instability_pressure, clamp01(post_seam_score or 0.0))
        frame_card = {
            "stage": "EventFrameSpikeAttributionCard",
            "status": status_from_pressure(frame_pressure, "frame_spike"),
            "card_version": "relation_pressure_cards_v1",
            "formula": "Visible frame-to-frame motion exposes whether Strategy continuity becomes smooth motion or unstable spike behavior.",
            "control_mode": "REPORT_ONLY",
            "active_control_allowed": False,
            "motion_stage": str(motion_record.get("stage", "") or ""),
            "motion_profile": str(motion_record.get("frame_motion_profile", "") or ""),
            "frame_motion_stability_score": stability,
            "frame_delta_spike_ratio": spike,
            "frame_delta_reversal_ratio": reversal,
            "frame_delta_jerk_ratio": jerk,
            "frame_pressure": frame_pressure,
            "object_relation_attribution": object_relation_review.get("attribution", ""),
            "cascade_post_seam_attribution": seam_review.get("attribution", "") if isinstance(seam_review, dict) else "",
            "next_action": "Analyze the mp4 alongside this card; a numeric spike is only useful when tied to visible behavior.",
        }

        input_normalization = latest_record("EventInputNormalization")
        image_scale = latest_record("EventUniversalMath_EventImageScaleStart")
        source_upload = latest_record("source_image_upload")
        if not source_upload:
            source_upload = first_record("EventSourceImageUpload")
        normalized_values = input_normalization.get("normalized_values", {}) if isinstance(input_normalization.get("normalized_values", {}), dict) else {}
        image_crop = normalized_values.get("image_crop", image_scale.get("image_crop", ""))
        image_upscale_method = normalized_values.get("image_upscale_method", image_scale.get("image_upscale_method", ""))
        width = normalized_values.get("width", image_scale.get("width", ""))
        height = normalized_values.get("height", image_scale.get("height", ""))
        source_present = bool(image_scale or source_upload or normalized_values)
        if not source_present:
            source_status = "source_anchor_missing_evidence"
        elif str(image_crop or "").lower() == "disabled":
            source_status = "source_anchor_crop_disabled"
        else:
            source_status = "source_anchor_recorded"
        source_card = {
            "stage": "EventSourceAnchorPreservationCard",
            "status": source_status,
            "card_version": "relation_pressure_cards_v1",
            "formula": "Source image is OutcomePrevious / SourceAnchor; scale and crop choices define what the sampler is asked to preserve.",
            "control_mode": "REPORT_ONLY",
            "active_control_allowed": False,
            "source_image_present": source_present,
            "image_crop": image_crop,
            "image_upscale_method": image_upscale_method,
            "width": width,
            "height": height,
            "input_adjustment_count": input_normalization.get("adjustment_count", 0) if isinstance(input_normalization, dict) else 0,
            "formula_role": "OutcomePrevious / SourceAnchor",
            "evidence_stages": compact_record_refs([
                str(input_normalization.get("stage", "") or ""),
                str(image_scale.get("stage", "") or ""),
                str(source_upload.get("stage", "") or ""),
            ]),
            "next_action": "Keep source/crop/size fixed while comparing math; otherwise visual drift cannot be attributed cleanly.",
        }

        background_raw = latest_record("EventBackgroundAnchorPreservationCard")
        background_status = str(background_raw.get("status") or "not_recorded")
        background_roi_means = background_raw.get("roi_temporal_means", {}) if isinstance(background_raw.get("roi_temporal_means", {}), dict) else {}
        background_roi_maxes = background_raw.get("roi_temporal_maxes", {}) if isinstance(background_raw.get("roi_temporal_maxes", {}), dict) else {}
        top_band_temporal_mean = safe_float(background_roi_means.get("top_band_background"), 0.0)
        top_band_temporal_max = safe_float(background_roi_maxes.get("top_band_background"), 0.0)
        top_band_pressure = clamp01(((top_band_temporal_mean or 0.0) - 6.5) / 3.0)
        background_pressure = max(
            clamp01(background_raw.get("global_scene_drift_score", 0.0)),
            clamp01(background_raw.get("background_drift_score", 0.0)),
            clamp01(background_raw.get("weak_center_background_separation_score", 0.0)),
            top_band_pressure,
        )
        background_card = {
            "stage": "EventBackgroundAnchorPreservationCard",
            "status": background_status,
            "card_version": "relation_pressure_cards_v2_background_anchor",
            "formula": "Background should remain SourceAnchor evidence while central event carriers move; global background drift means local motion failed to return to the primary Strategy.",
            "control_mode": "REPORT_ONLY",
            "active_control_allowed": False,
            "background_temporal_mean": safe_float(background_raw.get("background_temporal_mean"), None),
            "background_temporal_max": safe_float(background_raw.get("background_temporal_max"), None),
            "center_temporal_mean": safe_float(background_raw.get("center_temporal_mean"), None),
            "full_frame_temporal_mean": safe_float(background_raw.get("full_frame_temporal_mean"), None),
            "center_background_ratio": safe_float(background_raw.get("center_background_ratio"), None),
            "background_drift_score": safe_float(background_raw.get("background_drift_score"), None),
            "weak_center_background_separation_score": safe_float(background_raw.get("weak_center_background_separation_score"), None),
            "global_scene_drift_score": safe_float(background_raw.get("global_scene_drift_score"), None),
            "top_band_temporal_mean": top_band_temporal_mean,
            "top_band_temporal_max": top_band_temporal_max,
            "top_band_pressure": top_band_pressure,
            "roi_temporal_means": background_roi_means,
            "roi_temporal_maxes": background_roi_maxes,
            "background_anchor_pressure": background_pressure,
            "evidence_stages": compact_record_refs([str(background_raw.get("stage", "") or "")]),
            "next_action": (
                "Treat this as global scene drift before raising low_delta_strength."
                if background_status == "global_scene_drift_high"
                else "Use this as the background side of fixed-seed comparisons."
            ),
        }

        spatial_roles = {
            "top_left_background": "SourceAnchor background corner",
            "top_right_background": "SourceAnchor background corner",
            "top_band_background": "SourceAnchor top band",
            "left_side_floor": "SourceAnchor left side / floor",
            "right_side_floor": "SourceAnchor right side / floor",
            "lower_side_floor": "SourceAnchor lower side / floor",
            "center_event_proxy": "intended central ObservedBehavior carrier",
            "full_frame": "global visible Outcome carrier",
        }
        spatial_region_cards = []
        background_region_pressures = []
        center_region_pressure = 0.0
        for roi_name in spatial_roles:
            mean_value = safe_float(background_roi_means.get(roi_name), None)
            max_value = safe_float(background_roi_maxes.get(roi_name), None)
            is_center = roi_name == "center_event_proxy"
            is_full = roi_name == "full_frame"
            if mean_value is None:
                pressure = 0.0
            elif is_center:
                pressure = clamp01((mean_value - 10.0) / 12.0)
                center_region_pressure = pressure
            elif is_full:
                pressure = clamp01((mean_value - 8.0) / 10.0)
            else:
                threshold = 3.5 if "top_left" in roi_name or "top_right" in roi_name else 6.5
                pressure = clamp01((mean_value - threshold) / 4.0)
                background_region_pressures.append(pressure)
            spatial_region_cards.append({
                "roi": roi_name,
                "formula_role": spatial_roles[roi_name],
                "temporal_mean": mean_value,
                "temporal_max": max_value,
                "pressure": pressure,
                "carrier_type": "central_motion" if is_center else ("global_outcome" if is_full else "background_anchor"),
            })
        dominant_region = "none"
        dominant_region_pressure = 0.0
        for item in spatial_region_cards:
            if item.get("carrier_type") == "background_anchor" and float(item.get("pressure", 0.0) or 0.0) >= dominant_region_pressure:
                dominant_region = str(item.get("roi", "") or "none")
                dominant_region_pressure = float(item.get("pressure", 0.0) or 0.0)
        background_region_pressure = max(background_region_pressures) if background_region_pressures else 0.0
        center_background_separation = clamp01(((safe_float(background_card.get("center_background_ratio"), 1.0) or 1.0) - 1.0) / 1.25)
        spatial_anchor_pressure = max(
            background_region_pressure,
            clamp01(background_card.get("background_anchor_pressure", 0.0)),
            clamp01(1.0 - center_background_separation),
        )
        spatial_anchor_map = {
            "stage": "EventSpatialAnchorMap",
            "status": status_from_pressure(spatial_anchor_pressure, "spatial_anchor"),
            "card_version": "spatial_anchor_map_v1_report_only",
            "formula": "SourceAnchor is not one global value: background regions, central motion, seam handoff, and selected tail are separate local carriers that must return to one Strategy.",
            "canonical_formula": "Outcome(t-1) + ObservedBehavior(t-1) = Strategy(t) = ObservedBehavior(t+1) + Outcome(t+1)",
            "control_mode": "REPORT_ONLY",
            "active_control_allowed": False,
            "prompt_text_injection_allowed": False,
            "semantic_math_in_prompt_allowed": False,
            "spatial_anchor_pressure": spatial_anchor_pressure,
            "dominant_background_region": dominant_region,
            "dominant_background_region_pressure": dominant_region_pressure,
            "background_region_pressure": background_region_pressure,
            "center_motion_pressure": center_region_pressure,
            "center_background_separation_score": center_background_separation,
            "region_carriers": spatial_region_cards,
            "seam_handoff_status": str(tail_card.get("status", "") or ""),
            "frame_motion_status": str(frame_card.get("status", "") or ""),
            "source_anchor_status": str(source_card.get("status", "") or ""),
            "background_anchor_status": str(background_card.get("status", "") or ""),
            "next_control_surface": "spatial_anchor_preservation_map",
            "next_action": (
                "Build region-aware evidence before any active background control; do not damp the whole branch for one drifting spatial carrier."
                if spatial_anchor_pressure >= 0.30
                else "Spatial carriers are readable; continue fixed-seed comparison before enabling control."
            ),
        }

        local_cards = [prompt_card, low_card, object_card, tail_card, frame_card, source_card, background_card, spatial_anchor_map]
        sub_strategy_statuses = {
            str(card.get("stage", "") or ""): str(card.get("status", "") or "")
            for card in local_cards
            if isinstance(card, dict)
        }
        divergence_flags = []
        if prompt_status == "changed_runtime_strategy":
            divergence_flags.append("prompt_carrier_did_not_return_to_current_strategy")
        if str(low_card.get("status", "") or "").endswith("_high"):
            divergence_flags.append("low_branch_pressure_high")
        if str(frame_card.get("status", "") or "").endswith("_high"):
            divergence_flags.append("visible_frame_spike_high")
        if str(object_card.get("status", "") or "") == "object_identity_watch":
            divergence_flags.append("object_relation_identity_watch")
        if str(tail_card.get("status", "") or "").endswith("_high"):
            divergence_flags.append("tail_strategy_pressure_high")
        if str(source_card.get("status", "") or "") == "source_anchor_missing_evidence":
            divergence_flags.append("source_anchor_missing")
        if str(background_card.get("status", "") or "") == "global_scene_drift_high":
            divergence_flags.append("background_anchor_global_drift_high")
        if str(spatial_anchor_map.get("status", "") or "").endswith("_high"):
            divergence_flags.append("spatial_anchor_map_high")

        if divergence_flags:
            global_status = "global_strategy_return_watch"
        elif all(str(status or "").endswith("_nominal") or str(status or "").endswith("_stable") or str(status or "") in (
            "clean_same_strategy",
            "protected_negative_payload_drift",
            "object_identity_measured",
            "source_anchor_recorded",
            "source_anchor_crop_disabled",
            "spatial_anchor_nominal",
        ) for status in sub_strategy_statuses.values()):
            global_status = "global_strategy_return_nominal"
        else:
            global_status = "global_strategy_return_measured"

        global_strategy_card = {
            "stage": "EventGlobalStrategyReturnCard",
            "status": global_status,
            "card_version": "relation_pressure_cards_v2_global_return",
            "formula": "All local Strategy points must return to the primary Strategy before any data is passed to the next sampler/node/segment.",
            "control_mode": "REPORT_ONLY",
            "active_control_allowed": False,
            "primary_strategy": "Preserve one event identity, reweight local carriers only as evidence permits, and pass the resulting StrategyCarrier to the next route stage.",
            "primary_strategy_role": "global Strategy(t) / route-level accountability",
            "sub_strategy_return_policy": "Local prompt/source/low/object/tail/frame/spatial strategies are subordinate evidence routes, not independent math controllers.",
            "sub_strategy_statuses": sub_strategy_statuses,
            "divergence_flag_count": len(divergence_flags),
            "divergence_flags": divergence_flags,
            "top_conflict_collision": strategy_matrix.get("top_conflict_collision", ""),
            "top_drift_collision": strategy_matrix.get("top_drift_collision", ""),
            "next_action": (
                "Resolve divergence flags before promoting any local card into active control."
                if divergence_flags
                else "Local cards returned to the global Strategy map; continue fixed-seed evidence collection."
            ),
        }

        return local_cards + [global_strategy_card]

    def _event_topology_strategy_return_map(
        self,
        strategy_matrix=None,
        relation_pressure_cards=None,
        vector_collisions=None,
        object_relation_review=None,
    ):
        """
        Report-only topology synchronizer.

        Local Strategy formulas are allowed to unfold at every carrier collision,
        but this map checks whether each local route returns to the global route
        Strategy before the next sampler/segment receives data. It does not
        change prompts, tensors, sampler state, or cascade routing.
        """
        strategy_matrix = strategy_matrix if isinstance(strategy_matrix, dict) else {}
        relation_pressure_cards = [
            c for c in (relation_pressure_cards or [])
            if isinstance(c, dict)
        ]
        vector_collisions = [
            c for c in (vector_collisions or [])
            if isinstance(c, dict)
        ]
        object_relation_review = object_relation_review if isinstance(object_relation_review, dict) else {}

        def safe_float(value, default=None):
            try:
                out = float(value)
            except Exception:
                return default
            return out if math.isfinite(out) else default

        def clamp01(value):
            value = safe_float(value, 0.0)
            return max(0.0, min(1.0, value))

        def card_by_stage(stage_name):
            for card in relation_pressure_cards:
                if str(card.get("stage", "") or "") == stage_name:
                    return card
            return {}

        def pressure_from_status(status):
            status = str(status or "")
            if status in ("", "not_recorded"):
                return 0.25
            if any(token in status for token in ("missing", "failed", "blocked", "cancelled")):
                return 1.0
            if status.endswith("_high") or status in ("changed_runtime_strategy", "object_identity_watch"):
                return 0.78
            if status.endswith("_watch") or status in ("protected_negative_payload_drift", "object_identity_measured", "source_anchor_crop_disabled"):
                return 0.38
            if status.endswith("_nominal") or status.endswith("_stable") or status in ("clean_same_strategy", "source_anchor_recorded"):
                return 0.12
            return 0.25

        route_contracts = {
            "prompt_image_anchor": {
                "parent_route": "S_global_prompt_source_anchor",
                "return_requirement": "Prompt meaning and source anchor must resolve into one StrategyCandidate before latent seeding.",
                "card_stage": "EventPromptCarrierContinuityCard",
                "secondary_card_stage": "EventSourceAnchorPreservationCard",
            },
            "positive_negative_prompt_polarity": {
                "parent_route": "S_global_prompt_corridor",
                "return_requirement": "Positive and negative carriers must form one semantic corridor, not two competing instructions.",
                "card_stage": "EventPromptCarrierContinuityCard",
            },
            "image_latent_noise_seed": {
                "parent_route": "S_global_source_to_latent_anchor",
                "return_requirement": "SourceAnchor must survive scaling, crop policy, and latent seed initialization.",
                "card_stage": "EventSourceAnchorPreservationCard",
            },
            "high_low_sampler_strategy": {
                "parent_route": "S_global_sampler_route",
                "return_requirement": "OutcomeNext(high) must remain the StrategyCarrier that low sampler refines.",
                "card_stage": "EventLowBranchRelationPressureCard",
            },
            "object_relation_ontology": {
                "parent_route": "S_global_object_relation",
                "return_requirement": "Object identity, contact boundary, and relative motion must stay one relation.",
                "card_stage": "EventObjectCarrierIdentityCard",
            },
            "tail_next_source": {
                "parent_route": "S_global_cascade_continuation",
                "return_requirement": "Selected tail frame must become OutcomePrevious(next segment) without hidden scene reset.",
                "card_stage": "EventTailStrategyContinuityCard",
            },
            "previous_next_frame_motion": {
                "parent_route": "S_global_visible_motion",
                "return_requirement": "Adjacent visible frames must allow motion while preserving event identity.",
                "card_stage": "EventFrameSpikeAttributionCard",
            },
            "visible_video_outcome": {
                "parent_route": "S_global_visible_outcome",
                "return_requirement": "Decoded frames and final save must become the visible Outcome inspected by the user.",
                "card_stage": "EventGlobalStrategyReturnCard",
            },
        }

        collision_by_id = {
            str(item.get("collision_id") or ""): item
            for item in vector_collisions
            if str(item.get("collision_id") or "")
        }

        pressure_terms = object_relation_review.get("pressure_terms", {})
        if not isinstance(pressure_terms, dict):
            pressure_terms = {}
        seam_attribution = str(object_relation_review.get("cascade_post_seam_attribution", "") or "")
        seam_score = clamp01(object_relation_review.get("cascade_post_seam_acceleration_score", 0.0))
        object_drift = clamp01(object_relation_review.get("object_relation_drift_score", 0.0))

        local_routes = []
        for collision_id, contract in route_contracts.items():
            collision = collision_by_id.get(collision_id, {})
            local_formula = collision.get("local_formula", {}) if isinstance(collision.get("local_formula", {}), dict) else {}
            card = card_by_stage(contract.get("card_stage", ""))
            secondary_card = card_by_stage(contract.get("secondary_card_stage", ""))
            card_status = str(card.get("status", "") or "not_recorded")
            secondary_status = str(secondary_card.get("status", "") or "")

            conflict = clamp01(collision.get("conflict_score", 0.0))
            drift = clamp01(collision.get("drift_score", 0.0))
            card_pressure = pressure_from_status(card_status)
            secondary_pressure = pressure_from_status(secondary_status) if secondary_status else 0.0
            extra_pressure = 0.0
            if collision_id == "object_relation_ontology":
                extra_pressure = max(object_drift, clamp01(pressure_terms.get("object_relation_pressure", 0.0)))
            elif collision_id == "high_low_sampler_strategy":
                extra_pressure = clamp01(pressure_terms.get("high_low_pressure", 0.0))
            elif collision_id == "tail_next_source":
                extra_pressure = max(clamp01(pressure_terms.get("tail_pressure", 0.0)), seam_score if seam_attribution else 0.0)
            elif collision_id == "previous_next_frame_motion":
                extra_pressure = max(clamp01(pressure_terms.get("frame_motion_pressure", 0.0)), seam_score if seam_attribution == "late_segment_spike" else 0.0)

            return_pressure = max(conflict, drift, card_pressure, secondary_pressure, extra_pressure)
            return_score = 1.0 - return_pressure
            if return_pressure >= 0.70:
                route_status = "return_watch_high"
            elif return_pressure >= 0.35:
                route_status = "return_watch"
            else:
                route_status = "returned"

            local_routes.append({
                "collision_id": collision_id,
                "local_strategy_id": local_formula.get("local_strategy_id", f"S_collision_{collision_id}"),
                "parent_route": contract.get("parent_route", ""),
                "return_requirement": contract.get("return_requirement", ""),
                "formula_role": (local_formula.get("strategy_point", {}) or {}).get("meaning", ""),
                "carriers": collision.get("carriers", []),
                "collision_status": str(collision.get("status", "") or "not_recorded"),
                "relation_card_stage": contract.get("card_stage", ""),
                "relation_card_status": card_status,
                "secondary_card_status": secondary_status,
                "conflict_score": conflict,
                "drift_score": drift,
                "card_pressure": card_pressure,
                "extra_topology_pressure": extra_pressure,
                "return_pressure": return_pressure,
                "return_score": return_score,
                "return_status": route_status,
                "active_control_allowed": False,
            })

        pressures = [clamp01(route.get("return_pressure", 0.0)) for route in local_routes]
        scores = [clamp01(route.get("return_score", 0.0)) for route in local_routes]
        topology_sync_score = sum(scores) / max(1, len(scores))
        max_pressure = max(pressures) if pressures else 0.0
        unstable_routes = [r for r in local_routes if str(r.get("return_status")) == "return_watch_high"]
        watch_routes = [r for r in local_routes if str(r.get("return_status")).startswith("return_watch")]
        sorted_routes = sorted(local_routes, key=lambda r: clamp01(r.get("return_pressure", 0.0)), reverse=True)
        primary_pressure_axis = [r.get("collision_id", "") for r in sorted_routes[:3]]

        if max_pressure >= 0.70 or len(unstable_routes) > 0:
            status = "topology_return_watch_high"
        elif max_pressure >= 0.35 or len(watch_routes) > 0:
            status = "topology_return_watch"
        else:
            status = "topology_return_nominal"

        next_route = "Keep topology report-only; run fixed-seed comparison before active math."
        if "high_low_sampler_strategy" in primary_pressure_axis and "previous_next_frame_motion" in primary_pressure_axis:
            next_route = (
                "Focus r87/r88 research on sampler-route pressure returning into visible motion; "
                "do not treat this as a seam-stitching fix unless seam pressure rises."
            )
        if seam_attribution == "late_segment_spike":
            next_route = (
                "Late segment spike is the active evidence target: compare high/low sampler pressure, prompt Strategy identity, "
                "and visible motion after the seam before any damping/control."
            )

        return {
            "stage": "EventTopologyStrategyReturnMap",
            "status": status,
            "map_version": "topology_strategy_return_v1_report_only",
            "formula": "Every local collision formula may unfold, but it must return to the global Strategy route before the next sampler/segment receives data.",
            "control_mode": "REPORT_ONLY",
            "active_control_allowed": False,
            "global_strategy_id": "S_global_event_route",
            "global_strategy": {
                "meaning": "prompt meaning = model interpretation = sampler route = latent evolution = visible video outcome",
                "primary_return_law": "Local Strategy points are accountable to the route-level StrategyCarrier, not independent controllers.",
                "canonical_formula": "Outcome(t-1) + ObservedBehavior(t-1) = Strategy(t) = ObservedBehavior(t+1) + Outcome(t+1)",
            },
            "topology_sync_score": topology_sync_score,
            "max_return_pressure": max_pressure,
            "unstable_route_count": len(unstable_routes),
            "watch_route_count": len(watch_routes),
            "primary_pressure_axis": primary_pressure_axis,
            "top_conflict_collision": strategy_matrix.get("top_conflict_collision", ""),
            "top_drift_collision": strategy_matrix.get("top_drift_collision", ""),
            "cascade_post_seam_attribution": seam_attribution,
            "cascade_post_seam_acceleration_score": seam_score,
            "local_strategy_routes": local_routes,
            "strategy_return_sequence": [
                "prompt_image_anchor",
                "positive_negative_prompt_polarity",
                "image_latent_noise_seed",
                "high_low_sampler_strategy",
                "object_relation_ontology",
                "tail_next_source",
                "previous_next_frame_motion",
                "visible_video_outcome",
            ],
            "next_route": next_route,
        }

    def _event_strategy_return_pressure_resolver(
        self,
        topology_strategy_return_map=None,
        relation_pressure_cards=None,
        strategy_matrix=None,
        object_relation_review=None,
    ):
        """
        r88 report-only resolver.

        This is the topology-safe bridge between "we measured pressure" and
        "which non-text control surface should be tested next". It does not
        modify prompts, tensors, samplers, deltas, pause routing, or video
        frames.
        """
        topology_strategy_return_map = topology_strategy_return_map if isinstance(topology_strategy_return_map, dict) else {}
        relation_pressure_cards = [
            c for c in (relation_pressure_cards or [])
            if isinstance(c, dict)
        ]
        strategy_matrix = strategy_matrix if isinstance(strategy_matrix, dict) else {}
        object_relation_review = object_relation_review if isinstance(object_relation_review, dict) else {}

        def safe_float(value, default=None):
            try:
                out = float(value)
            except Exception:
                return default
            return out if math.isfinite(out) else default

        def clamp01(value):
            value = safe_float(value, 0.0)
            return max(0.0, min(1.0, value))

        def card(stage_name):
            for item in relation_pressure_cards:
                if str(item.get("stage", "") or "") == stage_name:
                    return item
            return {}

        def route(collision_id):
            for item in topology_strategy_return_map.get("local_strategy_routes", []) or []:
                if isinstance(item, dict) and str(item.get("collision_id", "") or "") == collision_id:
                    return item
            return {}

        prompt_card = card("EventPromptCarrierContinuityCard")
        low_card = card("EventLowBranchRelationPressureCard")
        frame_card = card("EventFrameSpikeAttributionCard")
        tail_card = card("EventTailStrategyContinuityCard")
        object_card = card("EventObjectCarrierIdentityCard")
        source_card = card("EventSourceAnchorPreservationCard")
        background_card = card("EventBackgroundAnchorPreservationCard")
        spatial_card = card("EventSpatialAnchorMap")
        global_card = card("EventGlobalStrategyReturnCard")

        seam_terms = tail_card.get("seam_pressure_terms", {}) if isinstance(tail_card.get("seam_pressure_terms", {}), dict) else {}
        prompt_status = str(prompt_card.get("status", "") or "not_recorded")
        prompt_carrier_clean = prompt_status not in ("changed_runtime_strategy", "prompt_carrier_did_not_return_to_current_strategy")

        high_low_pressure = max(
            clamp01(low_card.get("high_low_pressure", 0.0)),
            clamp01(low_card.get("low_trace_pressure", 0.0)),
            clamp01(route("high_low_sampler_strategy").get("return_pressure", 0.0)),
        )
        frame_motion_pressure = max(
            clamp01(frame_card.get("frame_pressure", 0.0)),
            clamp01(route("previous_next_frame_motion").get("return_pressure", 0.0)),
        )
        late_segment_spike_pressure = max(
            clamp01(seam_terms.get("max_late_segment_spike_pressure", 0.0)),
            clamp01(topology_strategy_return_map.get("cascade_post_seam_acceleration_score", 0.0))
            if str(topology_strategy_return_map.get("cascade_post_seam_attribution", "") or "") == "late_segment_spike"
            else 0.0,
        )
        seam_boundary_pressure = clamp01(seam_terms.get("max_seam_pressure", 0.0))
        tail_pressure = max(
            clamp01(tail_card.get("tail_pressure", 0.0)),
            clamp01(tail_card.get("post_seam_acceleration_score", 0.0)),
            clamp01(route("tail_next_source").get("return_pressure", 0.0)),
        )
        object_relation_pressure = max(
            clamp01(object_relation_review.get("object_relation_drift_score", 0.0)),
            clamp01(route("object_relation_ontology").get("return_pressure", 0.0)),
        )
        source_anchor_pressure = clamp01(route("prompt_image_anchor").get("return_pressure", 0.0))
        background_anchor_pressure = max(
            clamp01(background_card.get("background_anchor_pressure", 0.0)),
            clamp01(background_card.get("global_scene_drift_score", 0.0)),
        )
        spatial_anchor_pressure = max(
            clamp01(spatial_card.get("spatial_anchor_pressure", 0.0)),
            clamp01(spatial_card.get("background_region_pressure", 0.0)),
        )
        topology_pressure = clamp01(topology_strategy_return_map.get("max_return_pressure", 0.0))

        pressure_vector = {
            "prompt_carrier_pressure": 1.0 if not prompt_carrier_clean else clamp01(route("prompt_image_anchor").get("return_pressure", 0.0)),
            "high_low_sampler_pressure": high_low_pressure,
            "visible_frame_motion_pressure": frame_motion_pressure,
            "late_segment_spike_pressure": late_segment_spike_pressure,
            "seam_boundary_pressure": seam_boundary_pressure,
            "tail_strategy_pressure": tail_pressure,
            "object_relation_pressure": object_relation_pressure,
            "source_anchor_pressure": source_anchor_pressure,
            "background_anchor_pressure": background_anchor_pressure,
            "spatial_anchor_pressure": spatial_anchor_pressure,
            "topology_return_pressure": topology_pressure,
        }
        dominant_pressure = max(pressure_vector.values()) if pressure_vector else 0.0
        dominant_axis = max(pressure_vector, key=pressure_vector.get) if pressure_vector else "none"
        weighted_pressure = clamp01(
            0.24 * high_low_pressure
            + 0.22 * frame_motion_pressure
            + 0.16 * late_segment_spike_pressure
            + 0.11 * tail_pressure
            + 0.09 * object_relation_pressure
            + 0.07 * source_anchor_pressure
            + 0.06 * background_anchor_pressure
            + 0.05 * spatial_anchor_pressure
        )
        strategy_return_pressure = max(weighted_pressure, min(dominant_pressure, 0.82))
        strategy_return_score = 1.0 - strategy_return_pressure

        if not prompt_carrier_clean:
            status = "blocked_until_prompt_carrier_returns"
            primary_attribution = "prompt_carrier_not_returned"
            next_surface = "prompt_payload_identity_guard"
            next_action = "Fix prompt carrier continuity before interpreting sampler or motion math."
        elif high_low_pressure >= 0.50 and frame_motion_pressure >= 0.55 and late_segment_spike_pressure >= 0.45:
            status = "strategy_return_pressure_high"
            primary_attribution = "high_low_visible_late_motion_coupling"
            next_surface = "sampler_to_visible_motion_pressure_window"
            next_action = "Test a bounded sampler-to-visible-motion pressure window; keep prompt text clean and change only one delta/sampler variable."
        elif frame_motion_pressure >= 0.55:
            status = "strategy_return_pressure_high"
            primary_attribution = "visible_frame_motion_pressure"
            next_surface = "visible_motion_stability_review"
            next_action = "Compare fixed-seed visible motion before adding any active damping."
        elif spatial_anchor_pressure >= 0.55:
            status = "strategy_return_pressure_high"
            primary_attribution = "spatial_anchor_region_drift"
            next_surface = "spatial_anchor_preservation_map"
            next_action = "Use region-aware SourceAnchor evidence before any active control; global damping is too broad for this pressure."
        elif background_anchor_pressure >= 0.55:
            status = "strategy_return_pressure_high"
            primary_attribution = "background_anchor_global_drift"
            next_surface = "spatial_anchor_preservation_map"
            next_action = "Read background drift through SpatialAnchorMap first; do not pull the whole branch toward SourceAnchor until the drifting carrier is known."
        elif high_low_pressure >= 0.55:
            status = "strategy_return_pressure_high"
            primary_attribution = "high_low_sampler_strategy_pressure"
            next_surface = "bounded_latent_delta_research"
            next_action = "Investigate high/low delta as a sampler-route carrier, not as a prompt rewrite."
        elif seam_boundary_pressure >= 0.55:
            status = "strategy_return_pressure_high"
            primary_attribution = "cascade_boundary_jump"
            next_surface = "tail_frame_strategy_choice"
            next_action = "Review selected tail frame and MirrorCut boundary before sampler math."
        elif strategy_return_pressure >= 0.35:
            status = "strategy_return_pressure_watch"
            primary_attribution = dominant_axis
            next_surface = "report_only_fixed_seed_comparison"
            next_action = "Run one fixed-seed comparison and keep this resolver report-only."
        else:
            status = "strategy_return_nominal"
            primary_attribution = "no_single_dominant_pressure"
            next_surface = "continue_observation"
            next_action = "Continue collecting fixed-seed evidence; no active math pressure is justified by this report."

        candidate_control_surfaces = [
            {
                "surface": "sampler_to_visible_motion_pressure_window",
                "role": "ObservedBehavior(high/low) -> visible motion Strategy return",
                "when_to_test": "after r87 prompt continuity is clean and high_low + frame pressure are both high",
                "default_active": False,
            },
            {
                "surface": "bounded_latent_delta_research",
                "role": "small high/low delta windows as sampler-route evidence",
                "when_to_test": "one variable per fixed-seed comparison; never public default until visual proof",
                "default_active": False,
            },
            {
                "surface": "tail_frame_strategy_choice",
                "role": "selected tail OutcomePrevious for next segment",
                "when_to_test": "when seam boundary pressure or tail pressure dominates",
                "default_active": False,
            },
            {
                "surface": "spatial_anchor_preservation_map",
                "role": "background ROI / central carrier / seam handoff -> global Strategy return",
                "when_to_test": "when central motion improves but background drift rises across fixed-seed comparisons",
                "default_active": False,
            },
            {
                "surface": "conditioning_route_weight_map",
                "role": "future non-text Strategy density routing",
                "when_to_test": "after report-only density maps are stable across runs",
                "default_active": False,
            },
        ]

        return {
            "stage": "EventStrategyReturnPressureResolver",
            "status": status,
            "resolver_version": "strategy_return_pressure_resolver_v1_report_only",
            "formula": "Local Strategy pressure is folded back into S_global_event_route before any next sampler/control decision is proposed.",
            "canonical_formula": "Outcome(t-1) + ObservedBehavior(t-1) = Strategy(t) = ObservedBehavior(t+1) + Outcome(t+1)",
            "control_mode": "REPORT_ONLY",
            "active_control_allowed": False,
            "model_freedom_policy": "This resolver names pressure and next surfaces; it does not force model physics or inject text.",
            "parent_strategy": "S_global_event_route",
            "sub_strategy_return_policy": "high/low, frame motion, tail, object, source, spatial, and prompt routes are local evidence routes that must return to the parent Strategy.",
            "prompt_carrier_clean": bool(prompt_carrier_clean),
            "pressure_vector": pressure_vector,
            "strategy_return_pressure": strategy_return_pressure,
            "strategy_return_score": strategy_return_score,
            "weighted_pressure": weighted_pressure,
            "dominant_pressure": dominant_pressure,
            "dominant_axis": dominant_axis,
            "primary_attribution": primary_attribution,
            "next_control_surface": next_surface,
            "candidate_control_surfaces": candidate_control_surfaces,
            "do_not_do": [
                "do not inject topology/math prose into the prompt",
                "do not treat negative prompt bans as object physics",
                "do not globally damp motion before high/low and frame-pressure evidence agree",
                "do not tune from a stale-runtime or changed-prompt report",
            ],
            "evidence_stages": [
                stage for stage in [
                    str(prompt_card.get("stage", "") or ""),
                    str(low_card.get("stage", "") or ""),
                    str(frame_card.get("stage", "") or ""),
                    str(tail_card.get("stage", "") or ""),
                    str(object_card.get("stage", "") or ""),
                    str(source_card.get("stage", "") or ""),
                    str(spatial_card.get("stage", "") or ""),
                    str(global_card.get("stage", "") or ""),
                    str(topology_strategy_return_map.get("stage", "") or ""),
                    str(strategy_matrix.get("stage", "") or ""),
                ] if stage
            ],
            "next_action": next_action,
        }

    def _event_visible_motion_strategy_return_gate(
        self,
        strategy_return_pressure_resolver=None,
        relation_pressure_cards=None,
        object_relation_review=None,
        topology_strategy_return_map=None,
    ):
        """
        R139 report-only gate.

        Visible motion is already the completed Outcome(t+1). This gate returns
        it to the next Strategy decision without allowing same-run or blind
        global damping. It exists specifically to avoid treating "the video
        moved" as proof that the model must be constrained.
        """
        resolver = strategy_return_pressure_resolver if isinstance(strategy_return_pressure_resolver, dict) else {}
        relation_pressure_cards = [
            c for c in (relation_pressure_cards or [])
            if isinstance(c, dict)
        ]
        object_relation_review = object_relation_review if isinstance(object_relation_review, dict) else {}
        topology_strategy_return_map = topology_strategy_return_map if isinstance(topology_strategy_return_map, dict) else {}

        def safe_float(value, default=0.0):
            try:
                out = float(value)
            except Exception:
                return default
            return out if math.isfinite(out) else default

        def clamp01(value):
            return max(0.0, min(1.0, safe_float(value, 0.0)))

        def card(stage_name):
            for item in relation_pressure_cards:
                if str(item.get("stage", "") or "") == stage_name:
                    return item
            return {}

        def route(collision_id):
            for item in topology_strategy_return_map.get("local_strategy_routes", []) or []:
                if isinstance(item, dict) and str(item.get("collision_id", "") or "") == collision_id:
                    return item
            return {}

        frame_card = card("EventFrameSpikeAttributionCard")
        tail_card = card("EventTailStrategyContinuityCard")
        object_card = card("EventObjectCarrierIdentityCard")
        background_card = card("EventBackgroundAnchorPreservationCard")
        spatial_card = card("EventSpatialAnchorMap")
        global_card = card("EventGlobalStrategyReturnCard")
        seam_terms = tail_card.get("seam_pressure_terms", {}) if isinstance(tail_card.get("seam_pressure_terms", {}), dict) else {}
        pressure_vector = resolver.get("pressure_vector", {}) if isinstance(resolver.get("pressure_vector", {}), dict) else {}

        visible_frame_motion_pressure = max(
            clamp01(frame_card.get("frame_pressure", 0.0)),
            clamp01(pressure_vector.get("visible_frame_motion_pressure", 0.0)),
            clamp01(route("previous_next_frame_motion").get("return_pressure", 0.0)),
        )
        seam_boundary_pressure = max(
            clamp01(seam_terms.get("max_seam_pressure", 0.0)),
            clamp01(pressure_vector.get("seam_boundary_pressure", 0.0)),
        )
        seam_acceleration_pressure = max(
            clamp01(tail_card.get("post_seam_acceleration_score", 0.0)),
            clamp01(object_relation_review.get("cascade_post_seam_acceleration_score", 0.0)),
            clamp01(topology_strategy_return_map.get("cascade_post_seam_acceleration_score", 0.0)),
        )
        late_segment_spike_pressure = max(
            clamp01(seam_terms.get("max_late_segment_spike_pressure", 0.0)),
            clamp01(pressure_vector.get("late_segment_spike_pressure", 0.0)),
        )
        background_anchor_pressure = max(
            clamp01(background_card.get("background_anchor_pressure", 0.0)),
            clamp01(background_card.get("global_scene_drift_score", 0.0)),
            clamp01(background_card.get("background_drift_score", 0.0)),
            clamp01(pressure_vector.get("background_anchor_pressure", 0.0)),
        )
        top_band_pressure = clamp01(background_card.get("top_band_pressure", 0.0))
        spatial_anchor_pressure = max(
            clamp01(spatial_card.get("spatial_anchor_pressure", 0.0)),
            clamp01(spatial_card.get("background_region_pressure", 0.0)),
            clamp01(pressure_vector.get("spatial_anchor_pressure", 0.0)),
        )
        object_relation_pressure = max(
            clamp01(object_relation_review.get("object_relation_drift_score", 0.0)),
            clamp01(object_card.get("object_identity_pressure", 0.0)),
            clamp01(pressure_vector.get("object_relation_pressure", 0.0)),
        )
        carrier_persistence_score = safe_float(
            object_relation_review.get("carrier_persistence_score", object_card.get("carrier_persistence_score", 1.0)),
            1.0,
        )
        contact_boundary_continuity_score = safe_float(
            object_relation_review.get("contact_boundary_continuity_score", object_card.get("contact_boundary_continuity_score", 1.0)),
            1.0,
        )

        coupled_seam_pressure = max(seam_boundary_pressure, seam_acceleration_pressure, late_segment_spike_pressure)
        coupled_background_pressure = max(background_anchor_pressure, top_band_pressure, spatial_anchor_pressure)
        carrier_loss_pressure = max(
            object_relation_pressure,
            clamp01(1.0 - carrier_persistence_score),
            clamp01(1.0 - contact_boundary_continuity_score),
        )
        visible_motion_coupling_score = max(
            min(visible_frame_motion_pressure, coupled_seam_pressure),
            min(visible_frame_motion_pressure, coupled_background_pressure),
            min(visible_frame_motion_pressure, carrier_loss_pressure),
        )

        coupling_evidence = []
        if coupled_seam_pressure >= 0.35:
            coupling_evidence.append("seam_or_late_motion_pressure")
        if coupled_background_pressure >= 0.45:
            coupling_evidence.append("background_or_spatial_pressure")
        if carrier_loss_pressure >= 0.45:
            coupling_evidence.append("object_or_contact_carrier_loss")

        if visible_frame_motion_pressure < 0.35 and visible_motion_coupling_score < 0.25:
            status = "motion_return_stable"
            severity = "PASS"
            next_surface = "continue_r138_baseline_or_fixed_seed_compare"
            next_action = "Keep collecting fixed-seed evidence; visible motion does not justify active damping."
        elif visible_frame_motion_pressure >= 0.55 and not coupling_evidence:
            status = "motion_return_high_report_only"
            severity = "WARNING"
            next_surface = "visible_motion_stability_review"
            next_action = "Visible motion pressure is high, but no seam/background/object coupling proves a local failure; do not globally damp."
        elif visible_motion_coupling_score >= 0.45:
            status = "motion_return_coupled_pressure_report_only"
            severity = "WARNING"
            if coupled_background_pressure >= max(coupled_seam_pressure, carrier_loss_pressure):
                next_surface = "true_region_topology_evidence_required"
                next_action = "Build/verify real region topology before any background or spatial active control."
            elif coupled_seam_pressure >= max(coupled_background_pressure, carrier_loss_pressure):
                next_surface = "sampler_to_visible_motion_pressure_window"
                next_action = "Compare a bounded sampler-to-visible-motion window only after fixed-seed proof."
            else:
                next_surface = "object_relation_topology_review"
                next_action = "Use object/contact carrier evidence before changing prompt or sampler pressure."
        else:
            status = "motion_return_watch"
            severity = "INFO"
            next_surface = "report_only_fixed_seed_comparison"
            next_action = "Watch visible motion as next-run evidence; no active control is justified yet."

        return {
            "stage": "EventVisibleMotionStrategyReturnGate",
            "status": status,
            "severity": severity,
            "gate_version": "visible_motion_strategy_return_gate_v1",
            "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
            "runtime_name": EVENT_HORIZON_RUNTIME_NAME,
            "canonical_formula": "Outcome(t-1) + ObservedBehavior(t-1) = Strategy(t) = ObservedBehavior(t+1) + Outcome(t+1)",
            "formula": "The generated video is visible Outcome(t+1); its motion evidence can return to the next Strategy, but cannot retroactively justify same-run or global damping.",
            "control_mode": "REPORT_ONLY",
            "active_control_allowed": False,
            "visible_motion_active_control_allowed_next": False,
            "same_run_control_allowed": False,
            "model_freedom_policy": "Motion is not an error by itself. Only coupled seam/background/object evidence can nominate a future bounded route.",
            "parent_strategy": "S_global_event_route",
            "pressure_vector": {
                "visible_frame_motion_pressure": visible_frame_motion_pressure,
                "seam_boundary_pressure": seam_boundary_pressure,
                "seam_acceleration_pressure": seam_acceleration_pressure,
                "late_segment_spike_pressure": late_segment_spike_pressure,
                "background_anchor_pressure": background_anchor_pressure,
                "top_band_pressure": top_band_pressure,
                "spatial_anchor_pressure": spatial_anchor_pressure,
                "object_relation_pressure": object_relation_pressure,
                "carrier_loss_pressure": carrier_loss_pressure,
                "visible_motion_coupling_score": visible_motion_coupling_score,
            },
            "coupling_evidence": coupling_evidence,
            "strategy_return_resolver_status": resolver.get("status", ""),
            "strategy_return_primary_attribution": resolver.get("primary_attribution", ""),
            "strategy_return_next_control_surface": resolver.get("next_control_surface", ""),
            "next_control_surface": next_surface,
            "do_not_do": [
                "do not damp the whole branch from visible motion pressure alone",
                "do not treat saturated scalar pressure as region topology",
                "do not inject report or topology prose into prompt text",
                "do not convert a good motion outcome into a failure just because frame pressure is high",
            ],
            "evidence_stages": [
                stage for stage in [
                    str(frame_card.get("stage", "") or ""),
                    str(tail_card.get("stage", "") or ""),
                    str(object_card.get("stage", "") or ""),
                    str(background_card.get("stage", "") or ""),
                    str(spatial_card.get("stage", "") or ""),
                    str(global_card.get("stage", "") or ""),
                    str(resolver.get("stage", "") or ""),
                    str(topology_strategy_return_map.get("stage", "") or ""),
                ] if stage
            ],
            "next_action": next_action,
        }

    def _event_true_region_topology_evidence(
        self,
        visible_motion_strategy_return_gate=None,
        strategy_return_pressure_resolver=None,
        relation_pressure_cards=None,
        object_relation_review=None,
        topology_strategy_return_map=None,
    ):
        """
        R140 report-only region topology resolver.

        R139 can say that visible motion is coupled to background/spatial
        pressure. R140 decides whether that pressure is a real region map or
        only a saturated scalar. It still does not mutate tensors; it only
        decides whether a future active route has enough region evidence.
        """
        visible_gate = visible_motion_strategy_return_gate if isinstance(visible_motion_strategy_return_gate, dict) else {}
        resolver = strategy_return_pressure_resolver if isinstance(strategy_return_pressure_resolver, dict) else {}
        relation_pressure_cards = [
            c for c in (relation_pressure_cards or [])
            if isinstance(c, dict)
        ]
        object_relation_review = object_relation_review if isinstance(object_relation_review, dict) else {}
        topology_strategy_return_map = topology_strategy_return_map if isinstance(topology_strategy_return_map, dict) else {}

        def safe_float(value, default=0.0):
            try:
                out = float(value)
            except Exception:
                return default
            return out if math.isfinite(out) else default

        def clamp01(value):
            return max(0.0, min(1.0, safe_float(value, 0.0)))

        def card(stage_name):
            for item in relation_pressure_cards:
                if str(item.get("stage", "") or "") == stage_name:
                    return item
            return {}

        def route(collision_id):
            for item in topology_strategy_return_map.get("local_strategy_routes", []) or []:
                if isinstance(item, dict) and str(item.get("collision_id", "") or "") == collision_id:
                    return item
            return {}

        def mean(values):
            values = [safe_float(v, 0.0) for v in values]
            return sum(values) / len(values) if values else 0.0

        def stdev(values):
            values = [safe_float(v, 0.0) for v in values]
            if not values:
                return 0.0
            m = mean(values)
            return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))

        frame_card = card("EventFrameSpikeAttributionCard")
        tail_card = card("EventTailStrategyContinuityCard")
        source_card = card("EventSourceAnchorPreservationCard")
        background_card = card("EventBackgroundAnchorPreservationCard")
        spatial_card = card("EventSpatialAnchorMap")
        object_card = card("EventObjectCarrierIdentityCard")
        seam_terms = tail_card.get("seam_pressure_terms", {}) if isinstance(tail_card.get("seam_pressure_terms", {}), dict) else {}
        visible_pressure_vector = visible_gate.get("pressure_vector", {}) if isinstance(visible_gate.get("pressure_vector", {}), dict) else {}
        resolver_pressure_vector = resolver.get("pressure_vector", {}) if isinstance(resolver.get("pressure_vector", {}), dict) else {}

        visible_frame_motion_pressure = max(
            clamp01(visible_pressure_vector.get("visible_frame_motion_pressure", 0.0)),
            clamp01(resolver_pressure_vector.get("visible_frame_motion_pressure", 0.0)),
            clamp01(frame_card.get("frame_pressure", 0.0)),
            clamp01(route("previous_next_frame_motion").get("return_pressure", 0.0)),
        )
        seam_boundary_pressure = max(
            clamp01(visible_pressure_vector.get("seam_boundary_pressure", 0.0)),
            clamp01(resolver_pressure_vector.get("seam_boundary_pressure", 0.0)),
            clamp01(seam_terms.get("max_seam_pressure", 0.0)),
        )
        late_segment_spike_pressure = max(
            clamp01(visible_pressure_vector.get("late_segment_spike_pressure", 0.0)),
            clamp01(resolver_pressure_vector.get("late_segment_spike_pressure", 0.0)),
            clamp01(seam_terms.get("max_late_segment_spike_pressure", 0.0)),
        )
        background_anchor_pressure = max(
            clamp01(visible_pressure_vector.get("background_anchor_pressure", 0.0)),
            clamp01(resolver_pressure_vector.get("background_anchor_pressure", 0.0)),
            clamp01(background_card.get("background_anchor_pressure", 0.0)),
            clamp01(background_card.get("global_scene_drift_score", 0.0)),
        )
        top_band_pressure = max(
            clamp01(visible_pressure_vector.get("top_band_pressure", 0.0)),
            clamp01(background_card.get("top_band_pressure", 0.0)),
        )
        spatial_anchor_pressure = max(
            clamp01(visible_pressure_vector.get("spatial_anchor_pressure", 0.0)),
            clamp01(resolver_pressure_vector.get("spatial_anchor_pressure", 0.0)),
            clamp01(spatial_card.get("spatial_anchor_pressure", 0.0)),
            clamp01(spatial_card.get("background_region_pressure", 0.0)),
        )
        background_region_pressure = max(
            clamp01(spatial_card.get("background_region_pressure", 0.0)),
            clamp01(spatial_card.get("dominant_background_region_pressure", 0.0)),
        )
        tail_strategy_pressure = max(
            clamp01(tail_card.get("tail_pressure", 0.0)),
            clamp01(tail_card.get("post_seam_acceleration_score", 0.0)),
            clamp01(route("tail_next_source").get("return_pressure", 0.0)),
        )
        source_anchor_pressure = max(
            clamp01(source_card.get("source_anchor_pressure", 0.0)),
            clamp01(route("prompt_image_anchor").get("return_pressure", 0.0)),
        )
        object_relation_pressure = max(
            clamp01(visible_pressure_vector.get("object_relation_pressure", 0.0)),
            clamp01(resolver_pressure_vector.get("object_relation_pressure", 0.0)),
            clamp01(object_relation_review.get("object_relation_drift_score", 0.0)),
            clamp01(object_card.get("object_identity_pressure", 0.0)),
            clamp01(route("object_relation_ontology").get("return_pressure", 0.0)),
        )
        carrier_persistence_score = safe_float(
            object_relation_review.get("carrier_persistence_score", object_card.get("carrier_persistence_score", 1.0)),
            1.0,
        )
        contact_boundary_continuity_score = safe_float(
            object_relation_review.get("contact_boundary_continuity_score", object_card.get("contact_boundary_continuity_score", 1.0)),
            1.0,
        )
        carrier_loss_pressure = max(
            clamp01(visible_pressure_vector.get("carrier_loss_pressure", 0.0)),
            object_relation_pressure,
            clamp01(1.0 - carrier_persistence_score),
            clamp01(1.0 - contact_boundary_continuity_score),
        )

        center_action_carrier = clamp01(
            0.72 * visible_frame_motion_pressure
            + 0.18 * max(object_relation_pressure, carrier_loss_pressure)
            + 0.10 * max(seam_boundary_pressure, late_segment_spike_pressure)
        )
        edge_background_anchor = max(background_anchor_pressure, spatial_anchor_pressure, background_region_pressure)
        top_background_band = top_band_pressure
        selected_tail_source_carrier = max(tail_strategy_pressure, source_anchor_pressure)
        object_contact_carrier = max(object_relation_pressure, carrier_loss_pressure)
        seam_transition_carrier = max(seam_boundary_pressure, late_segment_spike_pressure)

        scalar_pressures = [background_anchor_pressure, spatial_anchor_pressure, top_band_pressure, background_region_pressure]
        scalar_saturation_score = min(1.0, mean([1.0 if p >= 0.95 else p for p in scalar_pressures]))
        saturated_background_scalar = (
            max(scalar_pressures) >= 0.995
            and min(scalar_pressures[:3]) >= 0.995
        )
        role_pressures = [
            center_action_carrier,
            edge_background_anchor,
            top_background_band,
            selected_tail_source_carrier,
            object_contact_carrier,
            seam_transition_carrier,
        ]
        role_count = sum(1 for p in role_pressures if p >= 0.18)
        high_role_count = sum(1 for p in role_pressures if p >= 0.45)
        diversity_score = clamp01(stdev(role_pressures) * 2.25)
        center_background_separation = clamp01(abs(center_action_carrier - edge_background_anchor))
        top_edge_separation = clamp01(abs(top_background_band - edge_background_anchor))
        object_center_separation = clamp01(abs(object_contact_carrier - center_action_carrier))
        tail_center_separation = clamp01(abs(selected_tail_source_carrier - center_action_carrier))
        region_separation_score = clamp01(
            0.34 * diversity_score
            + 0.24 * center_background_separation
            + 0.16 * top_edge_separation
            + 0.14 * object_center_separation
            + 0.12 * tail_center_separation
        )
        coverage_score = clamp01(role_count / 6.0)
        high_coverage_score = clamp01(high_role_count / 4.0)
        saturation_penalty = 0.36 if saturated_background_scalar else clamp01(0.18 * scalar_saturation_score)
        region_readiness_score = clamp01(
            0.46 * region_separation_score
            + 0.32 * coverage_score
            + 0.22 * high_coverage_score
            - saturation_penalty
        )

        region_tiles = [
            {
                "region_id": "center_action_carrier",
                "role": "action/body/motion carrier",
                "normalized_roi_yxhw": [0.18, 0.18, 0.64, 0.64],
                "pressure": float(center_action_carrier),
                "formula_role": "ObservedBehavior motion carrier that must not be confused with background anchor",
            },
            {
                "region_id": "edge_background_anchor",
                "role": "edge/background/source anchor",
                "normalized_roi_yxhw": [0.20, 0.00, 0.80, 1.00],
                "pressure": float(edge_background_anchor),
                "formula_role": "OutcomePrevious/source anchor carrier around the action center",
            },
            {
                "region_id": "top_background_band",
                "role": "top/background color and detail band",
                "normalized_roi_yxhw": [0.00, 0.00, 0.22, 1.00],
                "pressure": float(top_background_band),
                "formula_role": "early warning region for color/noise collapse and background drift",
            },
            {
                "region_id": "selected_tail_source_carrier",
                "role": "selected tail / next source continuity",
                "normalized_roi_yxhw": [0.00, 0.00, 1.00, 1.00],
                "pressure": float(selected_tail_source_carrier),
                "formula_role": "Outcome(t-1) selected by pause/continue before the next sampler",
            },
            {
                "region_id": "object_contact_carrier",
                "role": "object/contact/identity carrier",
                "normalized_roi_yxhw": [0.38, 0.18, 0.54, 0.64],
                "pressure": float(object_contact_carrier),
                "formula_role": "carrier identity/contact boundary evidence, not negative prompt prose",
            },
            {
                "region_id": "seam_transition_carrier",
                "role": "cascade seam / late transition carrier",
                "normalized_roi_yxhw": [0.00, 0.00, 1.00, 1.00],
                "pressure": float(seam_transition_carrier),
                "formula_role": "cascade boundary transition pressure separate from background/spatial pressure",
            },
        ]

        dominant_region = max(region_tiles, key=lambda item: item["pressure"]) if region_tiles else {}
        missing_evidence = []
        if center_action_carrier < 0.18:
            missing_evidence.append("center_action_carrier")
        if edge_background_anchor < 0.18:
            missing_evidence.append("edge_background_anchor")
        if object_contact_carrier < 0.18:
            missing_evidence.append("object_contact_carrier")
        if selected_tail_source_carrier < 0.18:
            missing_evidence.append("selected_tail_source_carrier")
        if saturated_background_scalar:
            missing_evidence.append("non_saturated_background_region_map")

        visible_status = str(visible_gate.get("status", "") or "")
        if not visible_gate:
            status = "true_region_topology_not_recorded"
            severity = "WARNING"
            next_surface = "record_visible_motion_return_first"
            next_action = "Record EventVisibleMotionStrategyReturnGate before interpreting region topology."
        elif saturated_background_scalar and region_readiness_score < 0.55:
            status = "saturated_scalar_not_region_topology"
            severity = "WARNING"
            next_surface = "pixel_region_motion_map_required"
            next_action = "Do not activate spatial/background control; saturated pressure must be converted into real region evidence first."
        elif region_readiness_score >= 0.58 and role_count >= 4:
            status = "true_region_topology_candidate_report_only"
            severity = "INFO"
            next_surface = "bounded_region_low_mid_window_candidate"
            next_action = "Candidate region map exists, but active control remains disabled until fixed-seed A/B proof."
        elif region_readiness_score >= 0.35:
            status = "true_region_topology_watch_report_only"
            severity = "INFO"
            next_surface = "collect_more_region_evidence"
            next_action = "Region roles are partially separable; collect another fixed-seed report before active control."
        else:
            status = "insufficient_true_region_topology"
            severity = "WARNING" if "coupled" in visible_status else "INFO"
            next_surface = "pixel_region_motion_map_required"
            next_action = "Current evidence is scalar pressure, not enough topology. Keep report-only."

        return {
            "stage": "EventTrueRegionTopologyEvidence",
            "status": status,
            "severity": severity,
            "gate_version": "true_region_topology_evidence_v1",
            "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
            "runtime_name": EVENT_HORIZON_RUNTIME_NAME,
            "canonical_formula": "Outcome(t-1) + ObservedBehavior(t-1) = Strategy(t) = ObservedBehavior(t+1) + Outcome(t+1)",
            "formula": "Region topology evidence separates local Strategy carriers before any visible/background/spatial pressure may become active control.",
            "control_mode": "REPORT_ONLY",
            "active_control_allowed": False,
            "true_region_active_control_allowed_next": False,
            "same_run_control_allowed": False,
            "parent_strategy": "S_global_event_route",
            "visible_motion_return_status": visible_status,
            "region_readiness_score": float(region_readiness_score),
            "region_separation_score": float(region_separation_score),
            "region_coverage_score": float(coverage_score),
            "region_high_coverage_score": float(high_coverage_score),
            "scalar_saturation_score": float(scalar_saturation_score),
            "saturated_background_scalar": bool(saturated_background_scalar),
            "role_count": int(role_count),
            "high_role_count": int(high_role_count),
            "dominant_region_id": str(dominant_region.get("region_id", "") or ""),
            "dominant_region_pressure": float(dominant_region.get("pressure", 0.0) or 0.0),
            "region_tiles": region_tiles,
            "pressure_vector": {
                "visible_frame_motion_pressure": float(visible_frame_motion_pressure),
                "seam_boundary_pressure": float(seam_boundary_pressure),
                "late_segment_spike_pressure": float(late_segment_spike_pressure),
                "background_anchor_pressure": float(background_anchor_pressure),
                "top_band_pressure": float(top_band_pressure),
                "spatial_anchor_pressure": float(spatial_anchor_pressure),
                "background_region_pressure": float(background_region_pressure),
                "tail_strategy_pressure": float(tail_strategy_pressure),
                "source_anchor_pressure": float(source_anchor_pressure),
                "object_relation_pressure": float(object_relation_pressure),
                "carrier_loss_pressure": float(carrier_loss_pressure),
            },
            "future_candidate_surfaces": [
                {
                    "surface": "bounded_region_low_mid_window_candidate",
                    "allowed_now": False,
                    "requires": ["region_readiness_score >= 0.58", "role_count >= 4", "fixed_seed_ab_video_proof"],
                },
                {
                    "surface": "pixel_region_motion_map_required",
                    "allowed_now": False,
                    "requires": ["non_saturated_background_region_map", "center/edge/top/object role separation"],
                },
                {
                    "surface": "object_relation_topology_review",
                    "allowed_now": False,
                    "requires": ["object/contact carrier evidence", "carrier persistence improvement proof"],
                },
            ],
            "missing_evidence": missing_evidence,
            "do_not_do": [
                "do not treat one saturated background scalar as a region map",
                "do not activate spatial gain without center/edge/top/object separation",
                "do not inject region explanations into the prompt text",
                "do not freeze useful center motion while trying to preserve background",
            ],
            "evidence_stages": [
                stage for stage in [
                    str(visible_gate.get("stage", "") or ""),
                    str(resolver.get("stage", "") or ""),
                    str(frame_card.get("stage", "") or ""),
                    str(tail_card.get("stage", "") or ""),
                    str(source_card.get("stage", "") or ""),
                    str(background_card.get("stage", "") or ""),
                    str(spatial_card.get("stage", "") or ""),
                    str(object_card.get("stage", "") or ""),
                    str(object_relation_review.get("stage", "") or ""),
                    str(topology_strategy_return_map.get("stage", "") or ""),
                ] if stage
            ],
            "next_control_surface": next_surface,
            "next_action": next_action,
        }

    def _event_fractal_strategy_intersection_map(
        self,
        execution_records=None,
        topology_strategy_return_map=None,
        strategy_return_pressure_resolver=None,
        visible_motion_strategy_return_gate=None,
        true_region_topology_evidence=None,
        relation_pressure_cards=None,
        vector_collisions=None,
        object_relation_review=None,
        depth=7,
    ):
        """
        Report-only recursive topology pass.

        The user-facing law is: every dynamic intersection can unfold the same
        Strategy equality, and any new intersection created by that unfold must
        also return to the parent Strategy. This record maps that idea without
        changing prompt text, tensors, sampler state, cascade routing, or video.
        """
        records = [r for r in (execution_records or []) if isinstance(r, dict)]
        topology_strategy_return_map = topology_strategy_return_map if isinstance(topology_strategy_return_map, dict) else {}
        resolver = strategy_return_pressure_resolver if isinstance(strategy_return_pressure_resolver, dict) else {}
        visible_gate = visible_motion_strategy_return_gate if isinstance(visible_motion_strategy_return_gate, dict) else {}
        true_region = true_region_topology_evidence if isinstance(true_region_topology_evidence, dict) else {}
        relation_pressure_cards = [
            c for c in (relation_pressure_cards or [])
            if isinstance(c, dict)
        ]
        vector_collisions = [
            c for c in (vector_collisions or [])
            if isinstance(c, dict)
        ]
        object_relation_review = object_relation_review if isinstance(object_relation_review, dict) else {}

        def safe_float(value, default=0.0):
            try:
                out = float(value)
            except Exception:
                return default
            return out if math.isfinite(out) else default

        def clamp01(value):
            return max(0.0, min(1.0, safe_float(value, 0.0)))

        def short(text, limit=120):
            text = str(text or "")
            if len(text) <= limit:
                return text
            return text[: limit - 3] + "..."

        def pressure_from_status(status):
            status = str(status or "")
            if status in ("", "not_recorded"):
                return 0.25
            if any(token in status for token in ("missing", "failed", "blocked", "cancelled")):
                return 1.0
            if status.endswith("_high") or "watch_high" in status:
                return 0.78
            if status.endswith("_watch") or "watch" in status:
                return 0.42
            if any(token in status for token in ("candidate", "measured", "recorded", "info")):
                return 0.24
            if status.endswith("_nominal") or status.endswith("_stable") or "pass" in status.lower():
                return 0.10
            return 0.25

        def latest(stage_name):
            for rec in reversed(records):
                if str(rec.get("stage", "") or "") == stage_name:
                    return rec
            return {}

        def latest_prefix(prefix):
            for rec in reversed(records):
                if str(rec.get("stage", "") or "").startswith(prefix):
                    return rec
            return {}

        local_routes = [
            r for r in (topology_strategy_return_map.get("local_strategy_routes", []) or [])
            if isinstance(r, dict)
        ]
        pressure_vector = resolver.get("pressure_vector", {}) if isinstance(resolver.get("pressure_vector", {}), dict) else {}
        region_tiles = [
            t for t in (true_region.get("region_tiles", []) or [])
            if isinstance(t, dict)
        ]
        apply_records = [
            r for r in records
            if str(r.get("stage", "") or "").startswith("EventStrategyControlSurfaceApply_")
        ]

        topology_sync_score = clamp01(topology_strategy_return_map.get("topology_sync_score", 0.0))
        strategy_return_pressure = clamp01(resolver.get("strategy_return_pressure", 1.0 - topology_sync_score))
        region_readiness = clamp01(true_region.get("region_readiness_score", 0.0))
        visible_coupling = clamp01(visible_gate.get("visible_motion_coupling_score", 0.0))
        object_drift = clamp01(object_relation_review.get("object_relation_drift_score", 0.0))
        seam_pressure = max(
            clamp01(pressure_vector.get("seam_boundary_pressure", 0.0)),
            clamp01(pressure_vector.get("late_segment_spike_pressure", 0.0)),
            clamp01(object_relation_review.get("cascade_post_seam_acceleration_score", 0.0)),
        )
        global_pressure = max(
            clamp01(1.0 - topology_sync_score),
            strategy_return_pressure,
            visible_coupling,
            object_drift,
            seam_pressure,
        )
        region_gap = clamp01(1.0 - region_readiness)

        primary_intersections = []
        for route in local_routes:
            collision_id = str(route.get("collision_id", "") or "unknown_collision")
            pressure = max(
                clamp01(route.get("return_pressure", 0.0)),
                pressure_from_status(route.get("return_status")),
            )
            primary_intersections.append({
                "intersection_id": collision_id,
                "source": "topology_strategy_return_map",
                "parent_strategy": str(route.get("parent_route", "S_global_event_route") or "S_global_event_route"),
                "formula_role": short(route.get("return_requirement") or route.get("formula_role") or ""),
                "carriers": route.get("carriers", []) if isinstance(route.get("carriers", []), list) else [],
                "pressure": pressure,
                "alignment_score": clamp01(1.0 - pressure),
                "return_status": str(route.get("return_status", "") or ""),
            })

        for tile in region_tiles:
            pressure = clamp01(tile.get("pressure", 0.0))
            if pressure <= 0.0 and not tile.get("region_id"):
                continue
            primary_intersections.append({
                "intersection_id": f"region_{tile.get('region_id', 'unknown')}",
                "source": "true_region_topology_evidence",
                "parent_strategy": "S_global_region_role_map",
                "formula_role": short(tile.get("formula_role") or tile.get("role") or ""),
                "carriers": [str(tile.get("role", "") or ""), str(tile.get("region_id", "") or "")],
                "pressure": pressure,
                "alignment_score": clamp01(1.0 - pressure),
                "return_status": "region_pressure_measured",
            })

        for rec in apply_records:
            unfold = rec.get("strategy_pressure_unfold", {}) if isinstance(rec.get("strategy_pressure_unfold", {}), dict) else {}
            field = rec.get("strategy_field", {}) if isinstance(rec.get("strategy_field", {}), dict) else {}
            branch = str(rec.get("branch_key", rec.get("branch_name", "")) or "unknown")
            depth_value = unfold.get("recursive_relation_depth", field.get("recursive_relation_depth", 0))
            combined_delta = safe_float(unfold.get("combined_delta", unfold.get("recursive_relation_delta", 0.0)), 0.0)
            max_delta = max(
                abs(safe_float(unfold.get("combined_limit", rec.get("max_delta_window", 0.0)), 0.0)),
                1e-9,
            )
            pressure = clamp01(abs(combined_delta) / max_delta) if str(unfold.get("status", "")) == "active" else pressure_from_status(rec.get("status"))
            primary_intersections.append({
                "intersection_id": f"sampler_recursive_relation_{branch}",
                "source": "strategy_control_surface_apply",
                "parent_strategy": str(rec.get("parent_strategy", "S_global_event_route") or "S_global_event_route"),
                "formula_role": f"{branch} sampler StrategyField recursive relation depth={depth_value}",
                "carriers": [str(rec.get("branch_name", "") or branch), str(rec.get("apply_policy", "") or "")],
                "pressure": pressure,
                "alignment_score": clamp01(1.0 - pressure),
                "return_status": str(unfold.get("status", rec.get("status", "")) or ""),
            })

        if visible_gate:
            primary_intersections.append({
                "intersection_id": "visible_motion_to_next_strategy",
                "source": "visible_motion_strategy_return_gate",
                "parent_strategy": "S_global_visible_outcome",
                "formula_role": "Visible Outcome(t+1) returns as next Strategy evidence before any active control.",
                "carriers": visible_gate.get("coupling_evidence", []) if isinstance(visible_gate.get("coupling_evidence", []), list) else [],
                "pressure": max(visible_coupling, pressure_from_status(visible_gate.get("status"))),
                "alignment_score": clamp01(1.0 - max(visible_coupling, pressure_from_status(visible_gate.get("status")))),
                "return_status": str(visible_gate.get("status", "") or ""),
            })

        # Keep the report compact and deterministic: highest-pressure intersections
        # carry the strongest evidence for recursive Strategy unfold.
        primary_intersections = sorted(
            primary_intersections,
            key=lambda item: (clamp01(item.get("pressure", 0.0)), str(item.get("intersection_id", ""))),
            reverse=True,
        )
        primary_intersections = primary_intersections[:16]

        requested_depth = max(1, min(7, int(depth or 7)))
        layers = []
        previous_nodes = []
        seed_nodes = primary_intersections[:]
        if seed_nodes:
            for layer_index in range(1, requested_depth + 1):
                if layer_index == 1:
                    layer_nodes = []
                    for item in seed_nodes:
                        pressure = clamp01(item.get("pressure", 0.0))
                        layer_nodes.append({
                            "node_id": f"L1::{item.get('intersection_id')}",
                            "source_intersections": [str(item.get("intersection_id", ""))],
                            "parent_strategy": str(item.get("parent_strategy", "S_global_event_route") or "S_global_event_route"),
                            "formula_role": short(item.get("formula_role", "")),
                            "pressure": pressure,
                            "alignment_score": clamp01(1.0 - pressure),
                            "status": (
                                "fractal_return_watch_high"
                                if pressure >= 0.70
                                else "fractal_return_watch"
                                if pressure >= 0.35
                                else "fractal_return_aligned"
                            ),
                        })
                else:
                    layer_nodes = []
                    previous_sorted = sorted(
                        previous_nodes,
                        key=lambda item: clamp01(item.get("pressure", 0.0)),
                        reverse=True,
                    )[:10]
                    for idx, prev in enumerate(previous_sorted):
                        mate = seed_nodes[(idx + layer_index - 2) % len(seed_nodes)]
                        prev_pressure = clamp01(prev.get("pressure", 0.0))
                        mate_pressure = clamp01(mate.get("pressure", 0.0))
                        recursive_pressure = clamp01(
                            (prev_pressure * 0.52)
                            + (mate_pressure * 0.24)
                            + (global_pressure * 0.14)
                            + (region_gap * 0.06)
                            + (visible_coupling * 0.04)
                        )
                        node_id = (
                            f"L{layer_index}::{prev.get('source_intersections', ['unknown'])[0]}"
                            f"->${mate.get('intersection_id', 'unknown')}"
                        )
                        layer_nodes.append({
                            "node_id": node_id.replace("$", ""),
                            "source_intersections": list(dict.fromkeys(
                                [str(x) for x in (prev.get("source_intersections", []) or [])]
                                + [str(mate.get("intersection_id", "") or "unknown")]
                            ))[:6],
                            "parent_strategy": "S_global_event_route",
                            "formula_role": (
                                "Derived Strategy intersection: previous local return is rechecked "
                                f"against {mate.get('intersection_id', 'unknown')} and folded back to the parent Strategy."
                            ),
                            "pressure": recursive_pressure,
                            "alignment_score": clamp01(1.0 - recursive_pressure),
                            "status": (
                                "fractal_return_watch_high"
                                if recursive_pressure >= 0.70
                                else "fractal_return_watch"
                                if recursive_pressure >= 0.35
                                else "fractal_return_aligned"
                            ),
                        })
                layer_pressures = [clamp01(n.get("pressure", 0.0)) for n in layer_nodes]
                mean_pressure = sum(layer_pressures) / max(1, len(layer_pressures))
                max_pressure = max(layer_pressures) if layer_pressures else 0.0
                layers.append({
                    "layer": layer_index,
                    "formula_application": (
                        "Outcome(t-1)+ObservedBehavior(t-1)=Strategy(t)="
                        "ObservedBehavior(t+1)+Outcome(t+1)"
                    ),
                    "intersection_count": len(layer_nodes),
                    "mean_pressure": mean_pressure,
                    "mean_alignment_score": clamp01(1.0 - mean_pressure),
                    "max_pressure": max_pressure,
                    "dominant_intersections": [
                        n.get("node_id", "") for n in sorted(
                            layer_nodes,
                            key=lambda item: clamp01(item.get("pressure", 0.0)),
                            reverse=True,
                        )[:3]
                    ],
                    "nodes": layer_nodes,
                })
                previous_nodes = layer_nodes

        if not primary_intersections:
            status = "fractal_strategy_intersections_not_recorded"
            severity = "WARNING"
            next_surface = "record_strategy_matrix_first"
            next_action = "No readable intersections were present; record Strategy Matrix and relation cards first."
            convergence_state = "unknown"
        else:
            first_pressure = clamp01(layers[0].get("mean_pressure", 0.0)) if layers else 0.0
            final_pressure = clamp01(layers[-1].get("mean_pressure", 0.0)) if layers else 0.0
            final_max = clamp01(layers[-1].get("max_pressure", 0.0)) if layers else 0.0
            alignment_score = clamp01(1.0 - final_pressure)
            if final_pressure < first_pressure - 0.08:
                convergence_state = "fractal_alignment_improving"
            elif final_pressure > first_pressure + 0.08:
                convergence_state = "fractal_alignment_diverging"
            else:
                convergence_state = "fractal_alignment_flat"

            if final_max >= 0.70 or convergence_state == "fractal_alignment_diverging":
                status = "fractal_strategy_alignment_watch_high_report_only"
                severity = "WARNING"
                next_surface = "cascade_local_strategy_partition_report_only"
                next_action = "Topology expands into unstable intersections; keep current body, but split/route cascade-local Strategy evidence before active control."
            elif final_max >= 0.35:
                status = "fractal_strategy_alignment_watch_report_only"
                severity = "INFO"
                next_surface = "same_sampler_strategy_ring_report_only"
                next_action = "Topology is readable but not closed; compare fixed-seed runs before changing active math."
            else:
                status = "fractal_strategy_alignment_candidate_report_only"
                severity = "INFO"
                next_surface = "bounded_model_attractor_candidate"
                next_action = "Seven-layer intersection map is aligned enough for the next report-only candidate surface."

        dominant_axis = [
            item.get("intersection_id", "") for item in primary_intersections[:5]
        ]
        final_layer = layers[-1] if layers else {}

        return {
            "stage": "EventFractalStrategyIntersectionMap",
            "status": status,
            "severity": severity,
            "map_version": "fractal_strategy_intersection_map_v1_depth7_report_only",
            "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
            "runtime_name": EVENT_HORIZON_RUNTIME_NAME,
            "canonical_formula": "Outcome(t-1) + ObservedBehavior(t-1) = Strategy(t) = ObservedBehavior(t+1) + Outcome(t+1)",
            "formula": "Every dynamic intersection unfolds the same Strategy equality; each derived intersection is folded back into S_global_event_route across seven report-only layers.",
            "control_mode": "REPORT_ONLY",
            "active_control_allowed": False,
            "same_run_control_allowed": False,
            "prompt_text_injection_allowed": False,
            "semantic_math_in_prompt_allowed": False,
            "parent_strategy": "S_global_event_route",
            "fractal_depth": requested_depth,
            "primary_intersection_count": len(primary_intersections),
            "formula_inserted_intersection_count": sum(int(layer.get("intersection_count", 0) or 0) for layer in layers),
            "topology_sync_score": topology_sync_score,
            "strategy_return_pressure": strategy_return_pressure,
            "true_region_readiness_score": region_readiness,
            "global_pressure": global_pressure,
            "final_layer_mean_pressure": clamp01(final_layer.get("mean_pressure", 0.0)),
            "final_layer_alignment_score": clamp01(final_layer.get("mean_alignment_score", 0.0)),
            "final_layer_max_pressure": clamp01(final_layer.get("max_pressure", 0.0)),
            "convergence_state": convergence_state,
            "dominant_intersection_axis": dominant_axis,
            "primary_intersections": primary_intersections,
            "fractal_layers": layers,
            "next_control_surface": next_surface,
            "do_not_do": [
                "do not inject formula prose into prompt text",
                "do not treat report-only alignment as visual proof",
                "do not rewrite the sampler loop until this map identifies a stable route",
                "do not let derived local strategies become independent controllers",
            ],
            "next_action": next_action,
        }

    def _event_region_weighted_fractal_strategy_return(
        self,
        fractal_strategy_intersection_map=None,
        true_region_topology_evidence=None,
        visible_motion_strategy_return_gate=None,
        strategy_return_pressure_resolver=None,
        object_relation_review=None,
    ):
        """
        R142 report-only confidence bridge.

        R141 can name a dominant fractal axis, but a loud scalar region can
        still over-dominate the reading. R142 does not change generation. It
        asks whether the fractal axis agrees with visible/video-region evidence
        before any future route treats that axis as useful control evidence.
        """
        fractal = fractal_strategy_intersection_map if isinstance(fractal_strategy_intersection_map, dict) else {}
        true_region = true_region_topology_evidence if isinstance(true_region_topology_evidence, dict) else {}
        visible_gate = visible_motion_strategy_return_gate if isinstance(visible_motion_strategy_return_gate, dict) else {}
        resolver = strategy_return_pressure_resolver if isinstance(strategy_return_pressure_resolver, dict) else {}
        object_relation_review = object_relation_review if isinstance(object_relation_review, dict) else {}

        def safe_float(value, default=0.0):
            try:
                out = float(value)
            except Exception:
                return default
            return out if math.isfinite(out) else default

        def clamp01(value):
            return max(0.0, min(1.0, safe_float(value, 0.0)))

        def axis_class(axis):
            axis = str(axis or "").lower()
            if any(token in axis for token in ("edge_background", "background", "spatial", "top_band", "top_background")):
                return "background"
            if any(token in axis for token in ("object", "contact", "carrier_identity")):
                return "object_contact"
            if any(token in axis for token in ("seam", "tail_next", "selected_tail", "cascade")):
                return "cascade_boundary"
            if any(token in axis for token in ("previous_next_frame", "visible_motion", "center_action", "motion")):
                return "visible_motion"
            if any(token in axis for token in ("high_low", "sampler", "low_branch")):
                return "sampler_handoff"
            if any(token in axis for token in ("prompt", "conditioning", "image_anchor")):
                return "prompt_source"
            return "unknown"

        def strongest_axis(pressure_map):
            if not pressure_map:
                return "unknown"
            return max(pressure_map.items(), key=lambda item: (clamp01(item[1]), str(item[0])))[0]

        def get_tile(region_id):
            for tile in true_region.get("region_tiles", []) or []:
                if isinstance(tile, dict) and str(tile.get("region_id", "") or "") == region_id:
                    return tile
            return {}

        resolver_vector = resolver.get("pressure_vector", {}) if isinstance(resolver.get("pressure_vector", {}), dict) else {}
        visible_vector = visible_gate.get("pressure_vector", {}) if isinstance(visible_gate.get("pressure_vector", {}), dict) else {}
        region_vector = true_region.get("pressure_vector", {}) if isinstance(true_region.get("pressure_vector", {}), dict) else {}

        center_action_pressure = max(
            clamp01(get_tile("center_action_carrier").get("pressure", 0.0)),
            clamp01(visible_vector.get("visible_frame_motion_pressure", 0.0)),
            clamp01(region_vector.get("visible_frame_motion_pressure", 0.0)),
        )
        edge_background_pressure = max(
            clamp01(get_tile("edge_background_anchor").get("pressure", 0.0)),
            clamp01(visible_vector.get("background_anchor_pressure", 0.0)),
            clamp01(visible_vector.get("spatial_anchor_pressure", 0.0)),
            clamp01(resolver_vector.get("background_anchor_pressure", 0.0)),
            clamp01(resolver_vector.get("spatial_anchor_pressure", 0.0)),
            clamp01(region_vector.get("background_anchor_pressure", 0.0)),
            clamp01(region_vector.get("spatial_anchor_pressure", 0.0)),
        )
        top_background_pressure = max(
            clamp01(get_tile("top_background_band").get("pressure", 0.0)),
            clamp01(visible_vector.get("top_band_pressure", 0.0)),
            clamp01(region_vector.get("top_band_pressure", 0.0)),
        )
        object_contact_pressure = max(
            clamp01(get_tile("object_contact_carrier").get("pressure", 0.0)),
            clamp01(visible_vector.get("object_relation_pressure", 0.0)),
            clamp01(visible_vector.get("carrier_loss_pressure", 0.0)),
            clamp01(resolver_vector.get("object_relation_pressure", 0.0)),
            clamp01(object_relation_review.get("object_relation_drift_score", 0.0)),
            clamp01(1.0 - safe_float(object_relation_review.get("carrier_persistence_score", 1.0), 1.0)),
            clamp01(1.0 - safe_float(object_relation_review.get("contact_boundary_continuity_score", 1.0), 1.0)),
        )
        seam_transition_pressure = max(
            clamp01(get_tile("seam_transition_carrier").get("pressure", 0.0)),
            clamp01(visible_vector.get("seam_boundary_pressure", 0.0)),
            clamp01(visible_vector.get("late_segment_spike_pressure", 0.0)),
            clamp01(resolver_vector.get("seam_boundary_pressure", 0.0)),
            clamp01(resolver_vector.get("late_segment_spike_pressure", 0.0)),
            clamp01(object_relation_review.get("cascade_post_seam_acceleration_score", 0.0)),
        )
        tail_source_pressure = clamp01(get_tile("selected_tail_source_carrier").get("pressure", 0.0))
        sampler_handoff_pressure = clamp01(resolver_vector.get("high_low_sampler_pressure", 0.0))
        if sampler_handoff_pressure <= 0.0:
            high_low_candidates = [
                clamp01(item.get("pressure", 0.0))
                for item in (fractal.get("primary_intersections", []) or [])
                if isinstance(item, dict)
                and "high_low" in str(item.get("intersection_id", "") or "")
            ]
            sampler_handoff_pressure = max(high_low_candidates) if high_low_candidates else 0.0

        region_readiness = clamp01(true_region.get("region_readiness_score", 0.0))
        region_separation = clamp01(true_region.get("region_separation_score", 0.0))
        fractal_alignment = clamp01(fractal.get("final_layer_alignment_score", 0.0))
        fractal_final_pressure = clamp01(fractal.get("final_layer_mean_pressure", 0.0))
        saturated_scalar = bool(true_region.get("saturated_background_scalar", False))
        scalar_saturation = clamp01(true_region.get("scalar_saturation_score", 0.0))

        raw_axis_pressure = {
            "background": max(edge_background_pressure, top_background_pressure),
            "visible_motion": center_action_pressure,
            "object_contact": object_contact_pressure,
            "cascade_boundary": max(seam_transition_pressure, tail_source_pressure),
            "sampler_handoff": sampler_handoff_pressure,
        }
        center_edge_ratio = center_action_pressure / max(0.001, edge_background_pressure)
        action_cluster_pressure = max(center_action_pressure, object_contact_pressure, seam_transition_pressure, sampler_handoff_pressure)
        background_overweight_score = clamp01(
            max(0.0, edge_background_pressure - action_cluster_pressure)
            * (1.0 - region_separation)
            + (0.20 if saturated_scalar else 0.0)
            + (0.08 * scalar_saturation if edge_background_pressure >= 0.92 else 0.0)
        )
        background_confidence = clamp01(
            0.25
            + 0.35 * region_separation
            + 0.25 * region_readiness
            + 0.15 * max(0.0, 1.0 - background_overweight_score)
        )
        guarded_axis_pressure = dict(raw_axis_pressure)
        guarded_axis_pressure["background"] = clamp01(raw_axis_pressure["background"] * background_confidence)

        raw_evidence_axis = strongest_axis(raw_axis_pressure)
        guarded_evidence_axis = strongest_axis(guarded_axis_pressure)
        dominant_axis = [
            str(axis or "") for axis in (fractal.get("dominant_intersection_axis", []) or [])
            if str(axis or "")
        ]
        fractal_axis = dominant_axis[0] if dominant_axis else ""
        fractal_axis_class = axis_class(fractal_axis)
        dominant_axis_evidence_match = (
            fractal_axis_class == guarded_evidence_axis
            or guarded_evidence_axis in [axis_class(axis) for axis in dominant_axis[:3]]
        )
        match_score = 1.0 if dominant_axis_evidence_match else 0.35 if fractal_axis_class == raw_evidence_axis else 0.0
        background_overweight_guard = (
            raw_evidence_axis == "background"
            and guarded_evidence_axis != "background"
            and background_overweight_score >= 0.12
        )
        region_axis_confidence = clamp01(
            0.30 * region_readiness
            + 0.24 * region_separation
            + 0.20 * fractal_alignment
            + 0.16 * match_score
            + 0.10 * max(0.0, 1.0 - background_overweight_score)
        )

        if not fractal:
            status = "region_weighted_fractal_not_recorded"
            severity = "WARNING"
            next_surface = "record_fractal_strategy_intersection_map_first"
            next_action = "Record EventFractalStrategyIntersectionMap before weighting the dominant axis."
        elif background_overweight_guard:
            status = "region_weighted_fractal_background_overweight_guard_report_only"
            severity = "WARNING"
            next_surface = "action_background_separation_report_only"
            next_action = "Background/edge pressure is too dominant for its region confidence; refine action/background separation before active control."
        elif region_axis_confidence >= 0.62 and dominant_axis_evidence_match:
            status = "region_weighted_fractal_axis_candidate_report_only"
            severity = "INFO"
            next_surface = (
                "same_sampler_strategy_ring_report_only"
                if guarded_evidence_axis in ("visible_motion", "sampler_handoff", "cascade_boundary")
                else "object_relation_topology_report_only"
                if guarded_evidence_axis == "object_contact"
                else "action_background_separation_report_only"
            )
            next_action = "Dominant fractal axis matches weighted evidence; keep report-only and require fixed-seed A/B before active math."
        else:
            status = "region_weighted_fractal_axis_watch_report_only"
            severity = "INFO"
            next_surface = "collect_region_weighted_fractal_evidence"
            next_action = "Axis confidence is readable but not enough for active control; compare the next fixed-seed run."

        return {
            "stage": "EventRegionWeightedFractalStrategyReturn",
            "status": status,
            "severity": severity,
            "map_version": "region_weighted_fractal_strategy_return_v1_report_only",
            "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
            "runtime_name": EVENT_HORIZON_RUNTIME_NAME,
            "canonical_formula": "Outcome(t-1) + ObservedBehavior(t-1) = Strategy(t) = ObservedBehavior(t+1) + Outcome(t+1)",
            "formula": "Dominant fractal Strategy axes must be weighted by visible region evidence before they may nominate the next control surface.",
            "control_mode": "REPORT_ONLY",
            "active_control_allowed": False,
            "same_run_control_allowed": False,
            "prompt_text_injection_allowed": False,
            "semantic_math_in_prompt_allowed": False,
            "parent_strategy": "S_global_event_route",
            "raw_fractal_dominant_axis": fractal_axis,
            "raw_fractal_axis_class": fractal_axis_class,
            "raw_visible_evidence_axis": raw_evidence_axis,
            "guarded_visible_evidence_axis": guarded_evidence_axis,
            "dominant_axis_evidence_match": bool(dominant_axis_evidence_match),
            "region_axis_confidence": float(region_axis_confidence),
            "center_action_vs_edge_background_ratio": float(center_edge_ratio),
            "background_overweight_guard": bool(background_overweight_guard),
            "background_overweight_score": float(background_overweight_score),
            "background_confidence": float(background_confidence),
            "axis_pressure_vector": {k: float(clamp01(v)) for k, v in raw_axis_pressure.items()},
            "guarded_axis_pressure_vector": {k: float(clamp01(v)) for k, v in guarded_axis_pressure.items()},
            "confidence_terms": {
                "region_readiness": float(region_readiness),
                "region_separation": float(region_separation),
                "fractal_alignment": float(fractal_alignment),
                "fractal_final_pressure": float(fractal_final_pressure),
                "match_score": float(match_score),
                "scalar_saturation": float(scalar_saturation),
            },
            "dominant_intersection_axis": dominant_axis,
            "next_control_surface": next_surface,
            "do_not_do": [
                "do not let edge/background scalar pressure become active control by itself",
                "do not freeze center action to satisfy background stability",
                "do not inject region-weighting explanations into prompt text",
                "do not treat this report-only confidence as visual proof",
            ],
            "next_action": next_action,
        }

    def _event_action_background_separation_evidence(
        self,
        region_weighted_fractal_strategy_return=None,
        true_region_topology_evidence=None,
        visible_motion_strategy_return_gate=None,
        strategy_return_pressure_resolver=None,
        object_relation_review=None,
        pixel_region_motion_map=None,
    ):
        """
        R143 report-only separation bridge.

        R142 can tell when background/edge pressure is too loud. R143 asks
        which part of the pressure belongs to the action carrier and which part
        belongs to the scene/background carrier before any future route can
        safely nominate active math.
        """
        region_weighted = region_weighted_fractal_strategy_return if isinstance(region_weighted_fractal_strategy_return, dict) else {}
        true_region = true_region_topology_evidence if isinstance(true_region_topology_evidence, dict) else {}
        visible_gate = visible_motion_strategy_return_gate if isinstance(visible_motion_strategy_return_gate, dict) else {}
        resolver = strategy_return_pressure_resolver if isinstance(strategy_return_pressure_resolver, dict) else {}
        object_relation_review = object_relation_review if isinstance(object_relation_review, dict) else {}
        pixel_region = pixel_region_motion_map if isinstance(pixel_region_motion_map, dict) else {}

        def safe_float(value, default=0.0):
            try:
                out = float(value)
            except Exception:
                return default
            return out if math.isfinite(out) else default

        def clamp01(value):
            return max(0.0, min(1.0, safe_float(value, 0.0)))

        def ratio(numerator, denominator):
            return safe_float(numerator, 0.0) / max(0.001, safe_float(denominator, 0.0))

        def get_tile(region_id):
            for tile in true_region.get("region_tiles", []) or []:
                if isinstance(tile, dict) and str(tile.get("region_id", "") or "") == region_id:
                    return tile
            return {}

        visible_vector = visible_gate.get("pressure_vector", {}) if isinstance(visible_gate.get("pressure_vector", {}), dict) else {}
        resolver_vector = resolver.get("pressure_vector", {}) if isinstance(resolver.get("pressure_vector", {}), dict) else {}
        region_vector = true_region.get("pressure_vector", {}) if isinstance(true_region.get("pressure_vector", {}), dict) else {}
        raw_axis_vector = region_weighted.get("axis_pressure_vector", {}) if isinstance(region_weighted.get("axis_pressure_vector", {}), dict) else {}
        guarded_axis_vector = region_weighted.get("guarded_axis_pressure_vector", {}) if isinstance(region_weighted.get("guarded_axis_pressure_vector", {}), dict) else {}

        center_action_pressure = max(
            clamp01(get_tile("center_action_carrier").get("pressure", 0.0)),
            clamp01(raw_axis_vector.get("visible_motion", 0.0)),
            clamp01(guarded_axis_vector.get("visible_motion", 0.0)),
            clamp01(visible_vector.get("visible_frame_motion_pressure", 0.0)),
            clamp01(region_vector.get("visible_frame_motion_pressure", 0.0)),
        )
        object_contact_pressure = max(
            clamp01(get_tile("object_contact_carrier").get("pressure", 0.0)),
            clamp01(raw_axis_vector.get("object_contact", 0.0)),
            clamp01(guarded_axis_vector.get("object_contact", 0.0)),
            clamp01(visible_vector.get("object_relation_pressure", 0.0)),
            clamp01(resolver_vector.get("object_relation_pressure", 0.0)),
            clamp01(object_relation_review.get("object_relation_drift_score", 0.0)),
            clamp01(1.0 - safe_float(object_relation_review.get("carrier_persistence_score", 1.0), 1.0)),
            clamp01(1.0 - safe_float(object_relation_review.get("contact_boundary_continuity_score", 1.0), 1.0)),
        )
        seam_transition_pressure = max(
            clamp01(get_tile("seam_transition_carrier").get("pressure", 0.0)),
            clamp01(raw_axis_vector.get("cascade_boundary", 0.0)),
            clamp01(guarded_axis_vector.get("cascade_boundary", 0.0)),
            clamp01(visible_vector.get("seam_boundary_pressure", 0.0)),
            clamp01(visible_vector.get("late_segment_spike_pressure", 0.0)),
            clamp01(resolver_vector.get("seam_boundary_pressure", 0.0)),
            clamp01(resolver_vector.get("late_segment_spike_pressure", 0.0)),
            clamp01(object_relation_review.get("cascade_post_seam_acceleration_score", 0.0)),
        )
        background_pressure = max(
            clamp01(get_tile("edge_background_anchor").get("pressure", 0.0)),
            clamp01(get_tile("top_background_band").get("pressure", 0.0)),
            clamp01(raw_axis_vector.get("background", 0.0)),
            clamp01(visible_vector.get("background_anchor_pressure", 0.0)),
            clamp01(visible_vector.get("spatial_anchor_pressure", 0.0)),
            clamp01(resolver_vector.get("background_anchor_pressure", 0.0)),
            clamp01(resolver_vector.get("spatial_anchor_pressure", 0.0)),
            clamp01(region_vector.get("background_anchor_pressure", 0.0)),
            clamp01(region_vector.get("spatial_anchor_pressure", 0.0)),
        )
        guarded_background_pressure = max(
            clamp01(guarded_axis_vector.get("background", 0.0)),
            clamp01(background_pressure * safe_float(region_weighted.get("background_confidence", 1.0), 1.0)),
        )
        sampler_handoff_pressure = max(
            clamp01(raw_axis_vector.get("sampler_handoff", 0.0)),
            clamp01(guarded_axis_vector.get("sampler_handoff", 0.0)),
            clamp01(resolver_vector.get("high_low_sampler_pressure", 0.0)),
        )

        action_cluster_pressure = max(center_action_pressure, object_contact_pressure, sampler_handoff_pressure)
        scene_cluster_pressure = max(background_pressure, guarded_background_pressure)
        action_to_background_ratio = ratio(action_cluster_pressure, scene_cluster_pressure)
        guarded_action_to_background_ratio = ratio(action_cluster_pressure, guarded_background_pressure)
        center_object_coupling = clamp01(1.0 - abs(center_action_pressure - object_contact_pressure))
        region_separation = clamp01(true_region.get("region_separation_score", 0.0))
        region_readiness = clamp01(true_region.get("region_readiness_score", 0.0))
        region_axis_confidence = clamp01(region_weighted.get("region_axis_confidence", 0.0))
        background_overweight_score = clamp01(region_weighted.get("background_overweight_score", 0.0))
        background_overweight_guard = bool(region_weighted.get("background_overweight_guard", False))

        background_leakage_score = clamp01(
            max(0.0, background_pressure - guarded_background_pressure)
            + (1.0 - region_separation) * 0.30
            + background_overweight_score * 0.40
        )
        pixel_center_edge_ratio = safe_float(pixel_region.get("center_edge_pixel_ratio", 0.0), 0.0)
        pixel_edge_center_ratio = safe_float(pixel_region.get("edge_center_pixel_ratio", 0.0), 0.0)
        pixel_background_leakage = clamp01(pixel_region.get("background_pixel_leakage_score", 0.0))
        pixel_seam_ratio = safe_float(pixel_region.get("estimated_seam_ratio", 0.0), 0.0)
        pressure_pixel_agreement = clamp01(
            1.0
            - min(
                1.0,
                abs(pixel_edge_center_ratio - ratio(guarded_background_pressure, action_cluster_pressure))
            )
        ) if pixel_region else 0.0
        pressure_pixel_disagreement = clamp01(1.0 - pressure_pixel_agreement) if pixel_region else 0.0
        if pixel_region and pixel_center_edge_ratio >= 1.10 and background_leakage_score >= 0.45:
            background_leakage_interpretation = "pressure_overweighted_relative_to_pixels"
        elif pixel_region and pixel_edge_center_ratio >= 0.95:
            background_leakage_interpretation = "pixel_background_leakage_visible"
        elif pixel_region:
            background_leakage_interpretation = "pixel_center_action_dominant"
        else:
            background_leakage_interpretation = "pixel_region_not_recorded"
        seam_interference_score = clamp01(
            0.55 * seam_transition_pressure
            + 0.25 * max(0.0, seam_transition_pressure - action_cluster_pressure)
            + 0.20 * max(0.0, seam_transition_pressure - guarded_background_pressure)
            + (0.12 * max(0.0, pixel_seam_ratio - 1.5) if pixel_region else 0.0)
        )
        separation_confidence = clamp01(
            0.30 * region_readiness
            + 0.28 * region_separation
            + 0.18 * region_axis_confidence
            + 0.14 * max(0.0, 1.0 - background_leakage_score)
            + 0.10 * center_object_coupling
        )
        action_dominance_score = clamp01(action_cluster_pressure - guarded_background_pressure + 0.5)

        if not region_weighted:
            status = "action_background_separation_not_recorded"
            severity = "WARNING"
            next_surface = "record_region_weighted_fractal_first"
            next_action = "Record EventRegionWeightedFractalStrategyReturn before action/background separation."
            recommended_axis = "unknown"
        elif background_overweight_guard or background_leakage_score >= 0.40:
            status = "action_background_separation_needed_report_only"
            severity = "WARNING"
            next_surface = (
                "pixel_pressure_disagreement_review_report_only"
                if background_leakage_interpretation == "pressure_overweighted_relative_to_pixels"
                else "separate_center_action_background_tiles_report_only"
            )
            next_action = "Separate center action/object motion from edge/background pressure before any active control; compare pressure with pixel-region evidence."
            recommended_axis = "center_action_over_background"
        elif seam_interference_score >= 0.52:
            status = "action_background_seam_watch_report_only"
            severity = "WARNING"
            next_surface = "seam_local_action_background_review_report_only"
            next_action = "Seam motion is mixed with action/background evidence; keep the next route local to the cascade boundary."
            recommended_axis = "seam_local"
        elif separation_confidence >= 0.62 and action_to_background_ratio >= 0.75:
            status = "action_background_separation_candidate_report_only"
            severity = "INFO"
            next_surface = "same_sampler_strategy_ring_report_only"
            next_action = "Action/background separation is readable; next report-only route can test same-sampler Strategy ring evidence."
            recommended_axis = "action_cluster"
        else:
            status = "action_background_separation_watch_report_only"
            severity = "INFO"
            next_surface = "collect_action_background_evidence"
            next_action = "Action/background separation is readable but not stable enough; collect another fixed-seed comparison."
            recommended_axis = "watch"

        return {
            "stage": "EventActionBackgroundSeparationEvidence",
            "status": status,
            "severity": severity,
            "map_version": "action_background_separation_evidence_v1_report_only",
            "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
            "runtime_name": EVENT_HORIZON_RUNTIME_NAME,
            "canonical_formula": "Outcome(t-1) + ObservedBehavior(t-1) = Strategy(t) = ObservedBehavior(t+1) + Outcome(t+1)",
            "formula": "Center action, object/contact, seam, and background carriers are separated as local Strategy returns before any next route may nominate active control.",
            "control_mode": "REPORT_ONLY",
            "active_control_allowed": False,
            "same_run_control_allowed": False,
            "prompt_text_injection_allowed": False,
            "semantic_math_in_prompt_allowed": False,
            "parent_strategy": "S_global_event_route",
            "source_surface": "EventRegionWeightedFractalStrategyReturn",
            "center_action_pressure": float(center_action_pressure),
            "object_contact_pressure": float(object_contact_pressure),
            "seam_transition_pressure": float(seam_transition_pressure),
            "sampler_handoff_pressure": float(sampler_handoff_pressure),
            "background_pressure": float(background_pressure),
            "guarded_background_pressure": float(guarded_background_pressure),
            "action_cluster_pressure": float(action_cluster_pressure),
            "scene_cluster_pressure": float(scene_cluster_pressure),
            "center_object_coupling": float(center_object_coupling),
            "action_to_background_ratio": float(action_to_background_ratio),
            "guarded_action_to_background_ratio": float(guarded_action_to_background_ratio),
            "background_leakage_score": float(background_leakage_score),
            "pixel_region_motion_status": str(pixel_region.get("status", "not_recorded") if pixel_region else "not_recorded"),
            "pixel_center_edge_ratio": float(pixel_center_edge_ratio),
            "pixel_edge_center_ratio": float(pixel_edge_center_ratio),
            "pixel_background_leakage_score": float(pixel_background_leakage),
            "pixel_estimated_seam_ratio": float(pixel_seam_ratio),
            "pressure_pixel_agreement": float(pressure_pixel_agreement),
            "pressure_pixel_disagreement": float(pressure_pixel_disagreement),
            "background_leakage_interpretation": background_leakage_interpretation,
            "seam_interference_score": float(seam_interference_score),
            "separation_confidence": float(separation_confidence),
            "action_dominance_score": float(action_dominance_score),
            "background_overweight_guard": bool(background_overweight_guard),
            "recommended_axis": recommended_axis,
            "next_control_surface": next_surface,
            "carrier_roles": {
                "action": ["center_action_carrier", "visible_motion", "sampler_handoff"],
                "object_contact": ["object_contact_carrier", "object_relation_review"],
                "background": ["edge_background_anchor", "top_background_band", "spatial_anchor"],
                "seam": ["seam_transition_carrier", "tail/cascade boundary"],
            },
            "do_not_do": [
                "do not globally freeze the background to fix action drift",
                "do not mix seam jumps with center action pressure",
                "do not move this report into prompt text",
                "do not activate control until fixed-seed visual proof confirms the separated axis",
            ],
            "next_action": next_action,
        }

    def _event_pixel_pressure_disagreement_review(
        self,
        pixel_region_motion_map=None,
        action_background_separation_evidence=None,
        region_weighted_fractal_strategy_return=None,
        true_region_topology_evidence=None,
        visible_motion_strategy_return_gate=None,
    ):
        """
        R146 report-only review.

        Pixel-region motion is visible Outcome evidence. Scalar pressure can be
        useful, but it must not call the edge/background the main problem when
        the decoded pixels show center/action dominance. This review reconciles
        that disagreement before the next route is nominated.
        """
        pixel_region = pixel_region_motion_map if isinstance(pixel_region_motion_map, dict) else {}
        action_background = action_background_separation_evidence if isinstance(action_background_separation_evidence, dict) else {}
        region_weighted = region_weighted_fractal_strategy_return if isinstance(region_weighted_fractal_strategy_return, dict) else {}
        true_region = true_region_topology_evidence if isinstance(true_region_topology_evidence, dict) else {}
        visible_gate = visible_motion_strategy_return_gate if isinstance(visible_motion_strategy_return_gate, dict) else {}

        def safe_float(value, default=0.0):
            try:
                out = float(value)
            except Exception:
                return default
            return out if math.isfinite(out) else default

        def clamp01(value):
            return max(0.0, min(1.0, safe_float(value, 0.0)))

        def ratio(numerator, denominator):
            return safe_float(numerator, 0.0) / max(0.001, safe_float(denominator, 0.0))

        pixel_center_edge_ratio = safe_float(pixel_region.get("center_edge_pixel_ratio", 0.0), 0.0)
        pixel_edge_center_ratio = safe_float(pixel_region.get("edge_center_pixel_ratio", 0.0), 0.0)
        pixel_background_leakage = clamp01(pixel_region.get("background_pixel_leakage_score", 0.0))
        pixel_seam_ratio = safe_float(pixel_region.get("estimated_seam_ratio", 0.0), 0.0)
        pixel_center_dominance = clamp01((pixel_center_edge_ratio - 1.0) / 0.75)
        pixel_edge_visibility = clamp01(pixel_edge_center_ratio)
        pixel_center_action_dominant = bool(pixel_region and pixel_center_edge_ratio >= 1.10 and pixel_edge_center_ratio < 0.95)
        pixel_background_visible = bool(pixel_region and pixel_edge_center_ratio >= 0.95)

        pressure_background_leakage = clamp01(action_background.get("background_leakage_score", 0.0))
        pressure_action_to_background_ratio = safe_float(action_background.get("action_to_background_ratio", 0.0), 0.0)
        pressure_guarded_action_to_background_ratio = safe_float(action_background.get("guarded_action_to_background_ratio", 0.0), 0.0)
        pressure_seam_interference = clamp01(action_background.get("seam_interference_score", 0.0))
        pressure_pixel_agreement = clamp01(action_background.get("pressure_pixel_agreement", 0.0))
        pressure_pixel_disagreement = clamp01(action_background.get("pressure_pixel_disagreement", 1.0 if pixel_region else 0.0))
        background_overweight_score = clamp01(region_weighted.get("background_overweight_score", 0.0))
        background_overweight_guard = bool(region_weighted.get("background_overweight_guard", False))
        region_readiness = clamp01(true_region.get("region_readiness_score", 0.0))
        region_separation = clamp01(true_region.get("region_separation_score", 0.0))
        visible_motion_pressure = clamp01((visible_gate.get("pressure_vector", {}) or {}).get("visible_frame_motion_pressure", 0.0)) if isinstance(visible_gate.get("pressure_vector", {}), dict) else 0.0

        pressure_background_claim = bool(
            background_overweight_guard
            or background_overweight_score >= 0.38
            or pressure_background_leakage >= 0.40
            or str(action_background.get("background_leakage_interpretation", "") or "") == "pressure_overweighted_relative_to_pixels"
        )
        pressure_background_claim_strength = clamp01(
            0.35 * background_overweight_score
            + 0.30 * pressure_background_leakage
            + 0.20 * (1.0 if background_overweight_guard else 0.0)
            + 0.15 * pressure_pixel_disagreement
        )
        scalar_pressure_overweight_score = clamp01(
            0.34 * pressure_background_claim_strength
            + 0.28 * pixel_center_dominance
            + 0.20 * max(0.0, 1.0 - pixel_edge_visibility)
            + 0.18 * max(0.0, pressure_background_leakage - pixel_background_leakage * 0.75)
        ) if pressure_background_claim and pixel_region else 0.0
        seam_locality_score = clamp01(
            0.45 * pressure_seam_interference
            + 0.35 * max(0.0, min(1.0, (pixel_seam_ratio - 1.0) / 1.25))
            + 0.20 * max(0.0, pixel_background_leakage - pixel_center_dominance)
        ) if pixel_region else pressure_seam_interference
        corrected_background_leakage_score = clamp01(
            0.58 * pixel_background_leakage
            + 0.30 * pressure_background_leakage
            + 0.12 * background_overweight_score
            - 0.22 * pixel_center_dominance
        )
        corrected_action_center_confidence = clamp01(
            0.36 * pixel_center_dominance
            + 0.22 * max(0.0, 1.0 - pixel_edge_visibility)
            + 0.18 * pressure_pixel_agreement
            + 0.14 * region_separation
            + 0.10 * visible_motion_pressure
        )
        corrected_action_to_background_ratio = ratio(
            corrected_action_center_confidence + max(0.0, pressure_action_to_background_ratio),
            corrected_background_leakage_score + 0.25,
        )

        if not pixel_region:
            status = "pixel_pressure_disagreement_not_recorded"
            severity = "WARNING"
            recommended_axis = "unknown"
            next_surface = "record_pixel_region_motion_map_first"
            next_action = "Record selected visible pixel-region motion before interpreting pressure disagreement."
        elif not action_background:
            status = "pixel_pressure_disagreement_missing_pressure_evidence"
            severity = "WARNING"
            recommended_axis = "unknown"
            next_surface = "record_action_background_separation_first"
            next_action = "Record action/background pressure evidence before pixel-pressure review."
        elif pixel_center_action_dominant and pressure_background_claim and scalar_pressure_overweight_score >= 0.30:
            status = "pixel_pressure_disagreement_scalar_overweight_report_only"
            severity = "WARNING"
            recommended_axis = "center_action_pixel_outcome"
            next_surface = "pressure_pixel_reweighting_report_only"
            next_action = "Visible pixels show center/action dominance; reduce trust in scalar background pressure before any active route."
        elif seam_locality_score >= 0.52 and pixel_center_action_dominant:
            status = "pixel_pressure_disagreement_seam_local_report_only"
            severity = "INFO"
            recommended_axis = "seam_local_center_action"
            next_surface = "seam_local_action_background_review_report_only"
            next_action = "Treat the disagreement as local seam/transition pressure, not global background leakage."
        elif pixel_background_visible:
            status = "pixel_pressure_disagreement_background_visible_report_only"
            severity = "INFO"
            recommended_axis = "visible_background_motion"
            next_surface = "separate_center_action_background_tiles_report_only"
            next_action = "Background motion is visible in pixels; keep separating action and background before control."
        else:
            status = "pixel_pressure_disagreement_balanced_report_only"
            severity = "INFO"
            recommended_axis = "balanced_pixel_pressure"
            next_surface = "same_sampler_strategy_ring_report_only"
            next_action = "Pixel and pressure evidence are close enough for the next report-only continuity route."

        return {
            "stage": "EventPixelPressureDisagreementReview",
            "status": status,
            "severity": severity,
            "map_version": "pixel_pressure_disagreement_review_v1_report_only",
            "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
            "runtime_name": EVENT_HORIZON_RUNTIME_NAME,
            "canonical_formula": "Outcome(t-1) + ObservedBehavior(t-1) = Strategy(t) = ObservedBehavior(t+1) + Outcome(t+1)",
            "formula": "Selected visible pixel Outcome corrects scalar pressure interpretation before Strategy returns to the next route.",
            "control_mode": "REPORT_ONLY",
            "active_control_allowed": False,
            "same_run_control_allowed": False,
            "prompt_text_injection_allowed": False,
            "semantic_math_in_prompt_allowed": False,
            "parent_strategy": "S_global_event_route",
            "source_surfaces": [
                "EventPixelRegionMotionMapSelection",
                "EventActionBackgroundSeparationEvidence",
                "EventRegionWeightedFractalStrategyReturn",
            ],
            "selected_pixel_source_stage": str(pixel_region.get("source_stage", "") if pixel_region else ""),
            "selected_pixel_selection_reason": str(pixel_region.get("selection_reason", "") if pixel_region else ""),
            "pixel_region_motion_status": str(pixel_region.get("status", "not_recorded") if pixel_region else "not_recorded"),
            "pixel_center_edge_ratio": float(pixel_center_edge_ratio),
            "pixel_edge_center_ratio": float(pixel_edge_center_ratio),
            "pixel_background_leakage_score": float(pixel_background_leakage),
            "pixel_estimated_seam_ratio": float(pixel_seam_ratio),
            "pixel_center_dominance_score": float(pixel_center_dominance),
            "pixel_center_action_dominant": bool(pixel_center_action_dominant),
            "pixel_background_visible": bool(pixel_background_visible),
            "pressure_background_claim": bool(pressure_background_claim),
            "pressure_background_claim_strength": float(pressure_background_claim_strength),
            "pressure_background_leakage_score": float(pressure_background_leakage),
            "pressure_action_to_background_ratio": float(pressure_action_to_background_ratio),
            "pressure_guarded_action_to_background_ratio": float(pressure_guarded_action_to_background_ratio),
            "pressure_pixel_agreement": float(pressure_pixel_agreement),
            "pressure_pixel_disagreement": float(pressure_pixel_disagreement),
            "background_overweight_guard": bool(background_overweight_guard),
            "background_overweight_score": float(background_overweight_score),
            "scalar_pressure_overweight_score": float(scalar_pressure_overweight_score),
            "seam_locality_score": float(seam_locality_score),
            "corrected_background_leakage_score": float(corrected_background_leakage_score),
            "corrected_action_center_confidence": float(corrected_action_center_confidence),
            "corrected_action_to_background_ratio": float(corrected_action_to_background_ratio),
            "region_readiness_score": float(region_readiness),
            "region_separation_score": float(region_separation),
            "visible_motion_pressure": float(visible_motion_pressure),
            "recommended_axis": recommended_axis,
            "next_control_surface": next_surface,
            "do_not_do": [
                "do not let scalar background pressure overrule selected visible pixel Outcome",
                "do not globally freeze the background when the center/action pixels dominate",
                "do not inject pixel-pressure explanations into prompt text",
                "do not activate this route without fixed-seed visual proof",
            ],
            "next_action": next_action,
        }

    def _event_pressure_pixel_reweighting_proposal(
        self,
        pixel_pressure_disagreement_review=None,
        pixel_region_motion_map=None,
        action_background_separation_evidence=None,
        region_weighted_fractal_strategy_return=None,
    ):
        """
        R147 report-only proposal.

        R146 tells us when scalar pressure and visible pixels disagree. R147 does
        not control the sampler. It converts that disagreement into bounded
        future weights so the next A/B can test whether pressure should trust
        visible pixel Outcome more than scalar background/action claims.
        """
        review = pixel_pressure_disagreement_review if isinstance(pixel_pressure_disagreement_review, dict) else {}
        pixel_region = pixel_region_motion_map if isinstance(pixel_region_motion_map, dict) else {}
        action_background = action_background_separation_evidence if isinstance(action_background_separation_evidence, dict) else {}
        region_weighted = region_weighted_fractal_strategy_return if isinstance(region_weighted_fractal_strategy_return, dict) else {}

        def safe_float(value, default=0.0):
            try:
                out = float(value)
            except Exception:
                return default
            return out if math.isfinite(out) else default

        def clamp01(value):
            return max(0.0, min(1.0, safe_float(value, 0.0)))

        def clamp_range(value, minimum, maximum):
            return max(float(minimum), min(float(maximum), safe_float(value, float(minimum))))

        if not review:
            return {
                "stage": "EventPressurePixelReweightingProposal",
                "status": "pressure_pixel_reweighting_not_recorded",
                "severity": "WARNING",
                "map_version": "pressure_pixel_reweighting_proposal_v1_report_only",
                "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
                "runtime_name": EVENT_HORIZON_RUNTIME_NAME,
                "canonical_formula": "Outcome(t-1) + ObservedBehavior(t-1) = Strategy(t) = ObservedBehavior(t+1) + Outcome(t+1)",
                "formula": "A bounded pressure/pixel proposal requires EventPixelPressureDisagreementReview first.",
                "control_mode": "REPORT_ONLY",
                "active_control_allowed": False,
                "same_run_control_allowed": False,
                "prompt_text_injection_allowed": False,
                "semantic_math_in_prompt_allowed": False,
                "generation_side_effect": "none",
                "next_control_surface": "record_pixel_pressure_disagreement_review_first",
                "next_action": "Record R146 pixel-pressure disagreement before building R147 weights.",
            }

        scalar_overweight = clamp01(review.get("scalar_pressure_overweight_score", 0.0))
        corrected_leakage = clamp01(review.get("corrected_background_leakage_score", 0.0))
        corrected_action = clamp01(review.get("corrected_action_center_confidence", 0.0))
        seam_locality = clamp01(review.get("seam_locality_score", 0.0))
        pressure_pixel_agreement = clamp01(review.get("pressure_pixel_agreement", 0.0))
        pressure_pixel_disagreement = clamp01(review.get("pressure_pixel_disagreement", 1.0 - pressure_pixel_agreement))
        pixel_center_dominance = clamp01(review.get("pixel_center_dominance_score", 0.0))
        pixel_center_edge_ratio = safe_float(review.get("pixel_center_edge_ratio", pixel_region.get("center_edge_pixel_ratio", 0.0)), 0.0)
        pixel_center_dominance = max(pixel_center_dominance, clamp01((pixel_center_edge_ratio - 1.0) / 0.75))
        pixel_center_action_dominant = bool(review.get("pixel_center_action_dominant", False))
        background_overweight_score = clamp01(review.get("background_overweight_score", region_weighted.get("background_overweight_score", 0.0)))
        action_background_confidence = clamp01(action_background.get("separation_confidence", 0.0))

        pixel_outcome_trust_weight = clamp01(
            0.45
            + 0.28 * corrected_action
            + 0.18 * pixel_center_dominance
            + 0.09 * pressure_pixel_agreement
        )
        scalar_pressure_trust_weight = clamp01(
            0.72
            - 0.45 * scalar_overweight
            - 0.20 * pixel_center_dominance
            + 0.12 * pressure_pixel_agreement
        )
        bounded_background_pressure_factor = clamp_range(
            1.0
            - 0.32 * scalar_overweight
            - 0.18 * pixel_center_dominance
            + 0.08 * seam_locality,
            0.68,
            1.0,
        )
        action_preservation_factor = clamp_range(
            1.0
            + 0.18 * corrected_action
            + 0.08 * pixel_center_dominance
            - 0.10 * seam_locality,
            0.96,
            1.12,
        )
        seam_protection_weight = clamp01(
            0.35
            + 0.45 * seam_locality
            + 0.20 * pressure_pixel_disagreement
        )
        center_action_priority = clamp01(
            0.42 * corrected_action
            + 0.30 * pixel_center_dominance
            + 0.18 * pressure_pixel_agreement
            + 0.10 * (1.0 - bounded_background_pressure_factor)
        )
        global_background_damping_allowed = False
        bounded_reweighting_candidate = bool(
            pixel_center_action_dominant
            and 0.20 <= scalar_overweight <= 0.60
            and corrected_action >= 0.45
            and seam_locality < 0.55
        )

        if seam_locality >= 0.55:
            status = "pressure_pixel_reweighting_seam_guard_report_only"
            severity = "INFO"
            next_surface = "seam_local_pressure_pixel_reweighting_report_only"
            next_action = "Treat the candidate as seam-local first; do not turn it into global background damping."
        elif bounded_reweighting_candidate:
            status = "pressure_pixel_reweighting_candidate_report_only"
            severity = "INFO"
            next_surface = "fixed_seed_ab_pressure_pixel_reweighting_report_only"
            next_action = "Run a fixed-seed A/B before active low-mid-window pressure/pixel reweighting."
        elif scalar_overweight < 0.20:
            status = "pressure_pixel_reweighting_balanced_watch_report_only"
            severity = "INFO"
            next_surface = "keep_observing_pixel_pressure_balance"
            next_action = "Scalar and pixel evidence are close; keep the route report-only."
        elif scalar_overweight > 0.60 or corrected_action < 0.35:
            status = "pressure_pixel_reweighting_unstable_report_only"
            severity = "WARNING"
            next_surface = "stabilize_pixel_pressure_evidence_before_control"
            next_action = "Evidence is too unstable for active reweighting; collect another same-seed report first."
        else:
            status = "pressure_pixel_reweighting_guarded_report_only"
            severity = "INFO"
            next_surface = "pressure_pixel_reweighting_more_evidence_report_only"
            next_action = "The route is plausible but not strong enough for active control."

        return {
            "stage": "EventPressurePixelReweightingProposal",
            "status": status,
            "severity": severity,
            "map_version": "pressure_pixel_reweighting_proposal_v1_report_only",
            "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
            "runtime_name": EVENT_HORIZON_RUNTIME_NAME,
            "canonical_formula": "Outcome(t-1) + ObservedBehavior(t-1) = Strategy(t) = ObservedBehavior(t+1) + Outcome(t+1)",
            "formula": "Pixel-corrected pressure returns as bounded future weights while Strategy remains centered on the model attractor.",
            "control_mode": "REPORT_ONLY",
            "active_control_allowed": False,
            "same_run_control_allowed": False,
            "prompt_text_injection_allowed": False,
            "semantic_math_in_prompt_allowed": False,
            "generation_side_effect": "none",
            "parent_strategy": "S_global_event_route",
            "source_surfaces": [
                "EventPixelPressureDisagreementReview",
                "EventPixelRegionMotionMapSelection",
                "EventActionBackgroundSeparationEvidence",
            ],
            "r146_reference_status": str(review.get("status", "")),
            "r146_reference_axis": str(review.get("recommended_axis", "")),
            "scalar_pressure_overweight_score": float(scalar_overweight),
            "corrected_background_leakage_score": float(corrected_leakage),
            "corrected_action_center_confidence": float(corrected_action),
            "seam_locality_score": float(seam_locality),
            "pressure_pixel_agreement": float(pressure_pixel_agreement),
            "pressure_pixel_disagreement": float(pressure_pixel_disagreement),
            "pixel_center_dominance_score": float(pixel_center_dominance),
            "pixel_center_action_dominant": bool(pixel_center_action_dominant),
            "background_overweight_score": float(background_overweight_score),
            "action_background_separation_confidence": float(action_background_confidence),
            "pixel_outcome_trust_weight": float(pixel_outcome_trust_weight),
            "scalar_pressure_trust_weight": float(scalar_pressure_trust_weight),
            "bounded_background_pressure_factor": float(bounded_background_pressure_factor),
            "action_preservation_factor": float(action_preservation_factor),
            "seam_protection_weight": float(seam_protection_weight),
            "center_action_priority": float(center_action_priority),
            "global_background_damping_allowed": bool(global_background_damping_allowed),
            "bounded_reweighting_candidate": bool(bounded_reweighting_candidate),
            "future_control_surface": "low_mid_window_pressure_pixel_reweighting_candidate",
            "next_control_surface": next_surface,
            "proposal": {
                "pixel_outcome_trust_weight": float(pixel_outcome_trust_weight),
                "scalar_pressure_trust_weight": float(scalar_pressure_trust_weight),
                "bounded_background_pressure_factor": float(bounded_background_pressure_factor),
                "action_preservation_factor": float(action_preservation_factor),
                "seam_protection_weight": float(seam_protection_weight),
                "center_action_priority": float(center_action_priority),
            },
            "do_not_do": [
                "do not globally damp the background from this report",
                "do not inject pressure/pixel wording into prompt text",
                "do not activate same-run control from this proposal",
                "do not let the proposal become an independent controller outside the model attractor",
            ],
            "next_action": next_action,
        }

    def _select_primary_pixel_region_motion_map(self, execution_records):
        """
        Select the pixel-region record that represents the visible video
        outcome, not the last auxiliary preview/tail record.
        """
        candidates = [
            rec for rec in (execution_records or [])
            if isinstance(rec, dict)
            and str(rec.get("stage", "") or "") == "EventPixelRegionMotionMap"
        ]
        if not candidates:
            return {}

        def with_selection(rec, reason, rank):
            out = dict(rec)
            out["selected_for_strategy_return"] = True
            out["selection_reason"] = str(reason)
            out["selection_rank"] = int(rank)
            out["available_pixel_region_map_count"] = len(candidates)
            out["preferred_outcome_source_order"] = [
                "EventMath_concatenated_frame_motion",
                "EventMath_decoded_frame_motion",
                "EventMath_cascade_*_frame_motion",
                "fallback_latest",
            ]
            return out

        preferred_exact = [
            ("EventMath_concatenated_frame_motion", "final_concatenated_visible_outcome", 1),
            ("EventMath_decoded_frame_motion", "decoded_visible_outcome", 2),
        ]
        for source_stage, reason, rank in preferred_exact:
            for rec in reversed(candidates):
                if str(rec.get("source_stage", "") or "") == source_stage:
                    return with_selection(rec, reason, rank)

        for rec in reversed(candidates):
            source_stage = str(rec.get("source_stage", "") or "")
            if source_stage.startswith("EventMath_cascade_") and source_stage.endswith("_frame_motion"):
                return with_selection(rec, "latest_cascade_visible_outcome", 3)

        return with_selection(candidates[-1], "fallback_latest_pixel_region_record", 9)

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
        relation_pressure_cards = [
            item for item in (body.get("relation_pressure_cards", []) or [])
            if isinstance(item, dict)
        ]

        def relation_card_status(stage_name):
            for item in relation_pressure_cards:
                if str(item.get("stage", "") or "") == stage_name:
                    return str(item.get("status", "") or "")
            return "not_recorded"

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
            "strategy_control_surface_status": (body.get("strategy_control_surface_plan", {}) or {}).get("status", "not_recorded") if isinstance(body.get("strategy_control_surface_plan", {}), dict) else "not_recorded",
            "strategy_control_surface_policy": (body.get("strategy_control_surface_plan", {}) or {}).get("policy", "") if isinstance(body.get("strategy_control_surface_plan", {}), dict) else "",
            "strategy_control_surface_path": (body.get("strategy_control_surface_plan", {}) or {}).get("active_generation_math_path", "") if isinstance(body.get("strategy_control_surface_plan", {}), dict) else "",
            "strategy_control_surface_apply_count": len(body.get("strategy_control_surface_apply_records", []) or []),
            "relation_pressure_card_count": len(relation_pressure_cards),
            "prompt_carrier_continuity_status": relation_card_status("EventPromptCarrierContinuityCard"),
            "low_branch_relation_pressure_status": relation_card_status("EventLowBranchRelationPressureCard"),
            "object_carrier_identity_status": relation_card_status("EventObjectCarrierIdentityCard"),
            "tail_strategy_continuity_status": relation_card_status("EventTailStrategyContinuityCard"),
            "frame_spike_attribution_status": relation_card_status("EventFrameSpikeAttributionCard"),
            "source_anchor_preservation_status": relation_card_status("EventSourceAnchorPreservationCard"),
            "background_anchor_preservation_status": relation_card_status("EventBackgroundAnchorPreservationCard"),
            "spatial_anchor_map_status": relation_card_status("EventSpatialAnchorMap"),
            "global_strategy_return_status": relation_card_status("EventGlobalStrategyReturnCard"),
            "topology_strategy_return_status": (body.get("topology_strategy_return_map", {}) or {}).get("status", "not_recorded") if isinstance(body.get("topology_strategy_return_map", {}), dict) else "not_recorded",
            "topology_sync_score": (body.get("topology_strategy_return_map", {}) or {}).get("topology_sync_score", "") if isinstance(body.get("topology_strategy_return_map", {}), dict) else "",
            "topology_unstable_route_count": (body.get("topology_strategy_return_map", {}) or {}).get("unstable_route_count", "") if isinstance(body.get("topology_strategy_return_map", {}), dict) else "",
            "topology_watch_route_count": (body.get("topology_strategy_return_map", {}) or {}).get("watch_route_count", "") if isinstance(body.get("topology_strategy_return_map", {}), dict) else "",
            "topology_primary_pressure_axis": (body.get("topology_strategy_return_map", {}) or {}).get("primary_pressure_axis", []) if isinstance(body.get("topology_strategy_return_map", {}), dict) else [],
            "topology_next_route": (body.get("topology_strategy_return_map", {}) or {}).get("next_route", "") if isinstance(body.get("topology_strategy_return_map", {}), dict) else "",
            "strategy_return_resolver_status": (body.get("strategy_return_pressure_resolver", {}) or {}).get("status", "not_recorded") if isinstance(body.get("strategy_return_pressure_resolver", {}), dict) else "not_recorded",
            "strategy_return_pressure": (body.get("strategy_return_pressure_resolver", {}) or {}).get("strategy_return_pressure", "") if isinstance(body.get("strategy_return_pressure_resolver", {}), dict) else "",
            "strategy_return_primary_attribution": (body.get("strategy_return_pressure_resolver", {}) or {}).get("primary_attribution", "") if isinstance(body.get("strategy_return_pressure_resolver", {}), dict) else "",
            "strategy_return_next_control_surface": (body.get("strategy_return_pressure_resolver", {}) or {}).get("next_control_surface", "") if isinstance(body.get("strategy_return_pressure_resolver", {}), dict) else "",
            "strategy_return_active_control_allowed": (body.get("strategy_return_pressure_resolver", {}) or {}).get("active_control_allowed", False) if isinstance(body.get("strategy_return_pressure_resolver", {}), dict) else False,
            "visible_motion_strategy_return_status": (body.get("visible_motion_strategy_return_gate", {}) or {}).get("status", "not_recorded") if isinstance(body.get("visible_motion_strategy_return_gate", {}), dict) else "not_recorded",
            "visible_motion_strategy_return_severity": (body.get("visible_motion_strategy_return_gate", {}) or {}).get("severity", "") if isinstance(body.get("visible_motion_strategy_return_gate", {}), dict) else "",
            "visible_motion_next_control_surface": (body.get("visible_motion_strategy_return_gate", {}) or {}).get("next_control_surface", "") if isinstance(body.get("visible_motion_strategy_return_gate", {}), dict) else "",
            "visible_motion_active_control_allowed_next": (body.get("visible_motion_strategy_return_gate", {}) or {}).get("visible_motion_active_control_allowed_next", False) if isinstance(body.get("visible_motion_strategy_return_gate", {}), dict) else False,
            "true_region_topology_status": (body.get("true_region_topology_evidence", {}) or {}).get("status", "not_recorded") if isinstance(body.get("true_region_topology_evidence", {}), dict) else "not_recorded",
            "true_region_topology_severity": (body.get("true_region_topology_evidence", {}) or {}).get("severity", "") if isinstance(body.get("true_region_topology_evidence", {}), dict) else "",
            "true_region_readiness_score": (body.get("true_region_topology_evidence", {}) or {}).get("region_readiness_score", "") if isinstance(body.get("true_region_topology_evidence", {}), dict) else "",
            "true_region_separation_score": (body.get("true_region_topology_evidence", {}) or {}).get("region_separation_score", "") if isinstance(body.get("true_region_topology_evidence", {}), dict) else "",
            "true_region_dominant_region_id": (body.get("true_region_topology_evidence", {}) or {}).get("dominant_region_id", "") if isinstance(body.get("true_region_topology_evidence", {}), dict) else "",
            "true_region_next_control_surface": (body.get("true_region_topology_evidence", {}) or {}).get("next_control_surface", "") if isinstance(body.get("true_region_topology_evidence", {}), dict) else "",
            "true_region_active_control_allowed_next": (body.get("true_region_topology_evidence", {}) or {}).get("true_region_active_control_allowed_next", False) if isinstance(body.get("true_region_topology_evidence", {}), dict) else False,
            "fractal_strategy_intersection_status": (body.get("fractal_strategy_intersection_map", {}) or {}).get("status", "not_recorded") if isinstance(body.get("fractal_strategy_intersection_map", {}), dict) else "not_recorded",
            "fractal_strategy_intersection_severity": (body.get("fractal_strategy_intersection_map", {}) or {}).get("severity", "") if isinstance(body.get("fractal_strategy_intersection_map", {}), dict) else "",
            "fractal_strategy_depth": (body.get("fractal_strategy_intersection_map", {}) or {}).get("fractal_depth", "") if isinstance(body.get("fractal_strategy_intersection_map", {}), dict) else "",
            "fractal_strategy_primary_intersection_count": (body.get("fractal_strategy_intersection_map", {}) or {}).get("primary_intersection_count", 0) if isinstance(body.get("fractal_strategy_intersection_map", {}), dict) else 0,
            "fractal_strategy_final_alignment_score": (body.get("fractal_strategy_intersection_map", {}) or {}).get("final_layer_alignment_score", "") if isinstance(body.get("fractal_strategy_intersection_map", {}), dict) else "",
            "fractal_strategy_convergence_state": (body.get("fractal_strategy_intersection_map", {}) or {}).get("convergence_state", "") if isinstance(body.get("fractal_strategy_intersection_map", {}), dict) else "",
            "fractal_strategy_dominant_axis": (body.get("fractal_strategy_intersection_map", {}) or {}).get("dominant_intersection_axis", []) if isinstance(body.get("fractal_strategy_intersection_map", {}), dict) else [],
            "fractal_strategy_next_control_surface": (body.get("fractal_strategy_intersection_map", {}) or {}).get("next_control_surface", "") if isinstance(body.get("fractal_strategy_intersection_map", {}), dict) else "",
            "region_weighted_fractal_status": (body.get("region_weighted_fractal_strategy_return", {}) or {}).get("status", "not_recorded") if isinstance(body.get("region_weighted_fractal_strategy_return", {}), dict) else "not_recorded",
            "region_weighted_fractal_severity": (body.get("region_weighted_fractal_strategy_return", {}) or {}).get("severity", "") if isinstance(body.get("region_weighted_fractal_strategy_return", {}), dict) else "",
            "region_axis_confidence": (body.get("region_weighted_fractal_strategy_return", {}) or {}).get("region_axis_confidence", "") if isinstance(body.get("region_weighted_fractal_strategy_return", {}), dict) else "",
            "dominant_axis_evidence_match": (body.get("region_weighted_fractal_strategy_return", {}) or {}).get("dominant_axis_evidence_match", False) if isinstance(body.get("region_weighted_fractal_strategy_return", {}), dict) else False,
            "background_overweight_guard": (body.get("region_weighted_fractal_strategy_return", {}) or {}).get("background_overweight_guard", False) if isinstance(body.get("region_weighted_fractal_strategy_return", {}), dict) else False,
            "guarded_visible_evidence_axis": (body.get("region_weighted_fractal_strategy_return", {}) or {}).get("guarded_visible_evidence_axis", "") if isinstance(body.get("region_weighted_fractal_strategy_return", {}), dict) else "",
            "region_weighted_fractal_next_control_surface": (body.get("region_weighted_fractal_strategy_return", {}) or {}).get("next_control_surface", "") if isinstance(body.get("region_weighted_fractal_strategy_return", {}), dict) else "",
            "pixel_region_motion_status": (body.get("pixel_region_motion_map", {}) or {}).get("status", "not_recorded") if isinstance(body.get("pixel_region_motion_map", {}), dict) else "not_recorded",
            "pixel_region_motion_severity": (body.get("pixel_region_motion_map", {}) or {}).get("severity", "") if isinstance(body.get("pixel_region_motion_map", {}), dict) else "",
            "pixel_center_edge_ratio": (body.get("pixel_region_motion_map", {}) or {}).get("center_edge_pixel_ratio", "") if isinstance(body.get("pixel_region_motion_map", {}), dict) else "",
            "pixel_edge_center_ratio": (body.get("pixel_region_motion_map", {}) or {}).get("edge_center_pixel_ratio", "") if isinstance(body.get("pixel_region_motion_map", {}), dict) else "",
            "pixel_estimated_seam_ratio": (body.get("pixel_region_motion_map", {}) or {}).get("estimated_seam_ratio", "") if isinstance(body.get("pixel_region_motion_map", {}), dict) else "",
            "pixel_background_leakage_score": (body.get("pixel_region_motion_map", {}) or {}).get("background_pixel_leakage_score", "") if isinstance(body.get("pixel_region_motion_map", {}), dict) else "",
            "pixel_region_next_control_surface": (body.get("pixel_region_motion_map", {}) or {}).get("next_control_surface", "") if isinstance(body.get("pixel_region_motion_map", {}), dict) else "",
            "cascade_seam_impulse_status": (body.get("cascade_seam_impulse_review", {}) or {}).get("status", "not_recorded") if isinstance(body.get("cascade_seam_impulse_review", {}), dict) else "not_recorded",
            "cascade_seam_impulse_severity": (body.get("cascade_seam_impulse_review", {}) or {}).get("severity", "") if isinstance(body.get("cascade_seam_impulse_review", {}), dict) else "",
            "cascade_seam_impulse_score": (body.get("cascade_seam_impulse_review", {}) or {}).get("seam_impulse_score", "") if isinstance(body.get("cascade_seam_impulse_review", {}), dict) else "",
            "cascade_seam_vector_score": (body.get("cascade_seam_impulse_review", {}) or {}).get("vector_seam_impulse_score", "") if isinstance(body.get("cascade_seam_impulse_review", {}), dict) else "",
            "cascade_seam_visible_score": (body.get("cascade_seam_impulse_review", {}) or {}).get("visible_seam_delta_score", "") if isinstance(body.get("cascade_seam_impulse_review", {}), dict) else "",
            "cascade_seam_visible_over_median": (body.get("cascade_seam_impulse_review", {}) or {}).get("visible_seam_over_median_ratio", "") if isinstance(body.get("cascade_seam_impulse_review", {}), dict) else "",
            "cascade_seam_visible_transition_index": (body.get("cascade_seam_impulse_review", {}) or {}).get("visible_boundary_transition_index", "") if isinstance(body.get("cascade_seam_impulse_review", {}), dict) else "",
            "cascade_seam_visible_top_rank": (body.get("cascade_seam_impulse_review", {}) or {}).get("visible_window_top_transition_rank", "") if isinstance(body.get("cascade_seam_impulse_review", {}), dict) else "",
            "cascade_seam_worst_boundary_index": (body.get("cascade_seam_impulse_review", {}) or {}).get("worst_boundary_index", "") if isinstance(body.get("cascade_seam_impulse_review", {}), dict) else "",
            "cascade_seam_boundary_jump_score": (body.get("cascade_seam_impulse_review", {}) or {}).get("boundary_jump_score", "") if isinstance(body.get("cascade_seam_impulse_review", {}), dict) else "",
            "cascade_seam_entry_acceleration_score": (body.get("cascade_seam_impulse_review", {}) or {}).get("entry_acceleration_score", "") if isinstance(body.get("cascade_seam_impulse_review", {}), dict) else "",
            "cascade_seam_direction_switch_score": (body.get("cascade_seam_impulse_review", {}) or {}).get("direction_switch_score", "") if isinstance(body.get("cascade_seam_impulse_review", {}), dict) else "",
            "cascade_seam_tail_entry_direction_cosine": (body.get("cascade_seam_impulse_review", {}) or {}).get("tail_entry_direction_cosine", "") if isinstance(body.get("cascade_seam_impulse_review", {}), dict) else "",
            "cascade_seam_next_control_surface": (body.get("cascade_seam_impulse_review", {}) or {}).get("next_control_surface", "") if isinstance(body.get("cascade_seam_impulse_review", {}), dict) else "",
            "tail_next_source_continuity_status": (body.get("tail_next_source_continuity_proposal", {}) or {}).get("status", "not_recorded") if isinstance(body.get("tail_next_source_continuity_proposal", {}), dict) else "not_recorded",
            "tail_next_source_continuity_severity": (body.get("tail_next_source_continuity_proposal", {}) or {}).get("severity", "") if isinstance(body.get("tail_next_source_continuity_proposal", {}), dict) else "",
            "tail_next_source_continuity_pressure": (body.get("tail_next_source_continuity_proposal", {}) or {}).get("continuity_pressure_score", "") if isinstance(body.get("tail_next_source_continuity_proposal", {}), dict) else "",
            "tail_next_source_source_gap_score": (body.get("tail_next_source_continuity_proposal", {}) or {}).get("source_gap_score", "") if isinstance(body.get("tail_next_source_continuity_proposal", {}), dict) else "",
            "tail_next_source_entry_ratio_score": (body.get("tail_next_source_continuity_proposal", {}) or {}).get("entry_ratio_score", "") if isinstance(body.get("tail_next_source_continuity_proposal", {}), dict) else "",
            "tail_next_source_source_delta_over_median": (body.get("tail_next_source_continuity_proposal", {}) or {}).get("source_delta_over_global_median", "") if isinstance(body.get("tail_next_source_continuity_proposal", {}), dict) else "",
            "tail_next_source_next_surface": (body.get("tail_next_source_continuity_proposal", {}) or {}).get("next_control_surface", "") if isinstance(body.get("tail_next_source_continuity_proposal", {}), dict) else "",
            "cascade_seam_phase_status": (body.get("cascade_seam_phase_classifier", {}) or {}).get("status", "not_recorded") if isinstance(body.get("cascade_seam_phase_classifier", {}), dict) else "not_recorded",
            "cascade_seam_phase_severity": (body.get("cascade_seam_phase_classifier", {}) or {}).get("severity", "") if isinstance(body.get("cascade_seam_phase_classifier", {}), dict) else "",
            "cascade_seam_phase_dominant_axis": (body.get("cascade_seam_phase_classifier", {}) or {}).get("dominant_axis", "") if isinstance(body.get("cascade_seam_phase_classifier", {}), dict) else "",
            "cascade_seam_phase_dominant_score": (body.get("cascade_seam_phase_classifier", {}) or {}).get("dominant_score", "") if isinstance(body.get("cascade_seam_phase_classifier", {}), dict) else "",
            "cascade_seam_phase_semantic_score": ((body.get("cascade_seam_phase_classifier", {}) or {}).get("axis_scores", {}) or {}).get("semantic_phase_reentry", "") if isinstance((body.get("cascade_seam_phase_classifier", {}) or {}).get("axis_scores", {}) if isinstance(body.get("cascade_seam_phase_classifier", {}), dict) else {}, dict) else "",
            "cascade_seam_phase_prompt_text_change_score": ((body.get("cascade_seam_phase_classifier", {}) or {}).get("axis_scores", {}) or {}).get("prompt_text_change", "") if isinstance((body.get("cascade_seam_phase_classifier", {}) or {}).get("axis_scores", {}) if isinstance(body.get("cascade_seam_phase_classifier", {}), dict) else {}, dict) else "",
            "cascade_seam_phase_prompt_score": ((body.get("cascade_seam_phase_classifier", {}) or {}).get("axis_scores", {}) or {}).get("prompt_phase_reentry", "") if isinstance((body.get("cascade_seam_phase_classifier", {}) or {}).get("axis_scores", {}) if isinstance(body.get("cascade_seam_phase_classifier", {}), dict) else {}, dict) else "",
            "cascade_seam_phase_latent_score": ((body.get("cascade_seam_phase_classifier", {}) or {}).get("axis_scores", {}) or {}).get("latent_carrier_mismatch", "") if isinstance((body.get("cascade_seam_phase_classifier", {}) or {}).get("axis_scores", {}) if isinstance(body.get("cascade_seam_phase_classifier", {}), dict) else {}, dict) else "",
            "cascade_seam_phase_background_score": ((body.get("cascade_seam_phase_classifier", {}) or {}).get("axis_scores", {}) or {}).get("background_anchor_conflict", "") if isinstance((body.get("cascade_seam_phase_classifier", {}) or {}).get("axis_scores", {}) if isinstance(body.get("cascade_seam_phase_classifier", {}), dict) else {}, dict) else "",
            "cascade_seam_phase_center_score": ((body.get("cascade_seam_phase_classifier", {}) or {}).get("axis_scores", {}) or {}).get("center_action_overdrive", "") if isinstance((body.get("cascade_seam_phase_classifier", {}) or {}).get("axis_scores", {}) if isinstance(body.get("cascade_seam_phase_classifier", {}), dict) else {}, dict) else "",
            "cascade_seam_phase_sampler_score": ((body.get("cascade_seam_phase_classifier", {}) or {}).get("axis_scores", {}) or {}).get("sampler_handoff_reset", "") if isinstance((body.get("cascade_seam_phase_classifier", {}) or {}).get("axis_scores", {}) if isinstance(body.get("cascade_seam_phase_classifier", {}), dict) else {}, dict) else "",
            "cascade_seam_phase_next_surface": (body.get("cascade_seam_phase_classifier", {}) or {}).get("next_control_surface", "") if isinstance(body.get("cascade_seam_phase_classifier", {}), dict) else "",
            "semantic_phase_schedule_status": (body.get("semantic_phase_schedule_proposal", {}) or {}).get("status", "not_recorded") if isinstance(body.get("semantic_phase_schedule_proposal", {}), dict) else "not_recorded",
            "semantic_phase_schedule_severity": (body.get("semantic_phase_schedule_proposal", {}) or {}).get("severity", "") if isinstance(body.get("semantic_phase_schedule_proposal", {}), dict) else "",
            "semantic_phase_schedule_score": (body.get("semantic_phase_schedule_proposal", {}) or {}).get("semantic_phase_reentry_score", "") if isinstance(body.get("semantic_phase_schedule_proposal", {}), dict) else "",
            "semantic_phase_schedule_prompt_clean": (body.get("semantic_phase_schedule_proposal", {}) or {}).get("prompt_carrier_clean", "") if isinstance(body.get("semantic_phase_schedule_proposal", {}), dict) else "",
            "semantic_phase_schedule_next_surface": (body.get("semantic_phase_schedule_proposal", {}) or {}).get("next_control_surface", "") if isinstance(body.get("semantic_phase_schedule_proposal", {}), dict) else "",
            "action_background_separation_status": (body.get("action_background_separation_evidence", {}) or {}).get("status", "not_recorded") if isinstance(body.get("action_background_separation_evidence", {}), dict) else "not_recorded",
            "action_background_separation_severity": (body.get("action_background_separation_evidence", {}) or {}).get("severity", "") if isinstance(body.get("action_background_separation_evidence", {}), dict) else "",
            "action_background_separation_confidence": (body.get("action_background_separation_evidence", {}) or {}).get("separation_confidence", "") if isinstance(body.get("action_background_separation_evidence", {}), dict) else "",
            "action_to_background_ratio": (body.get("action_background_separation_evidence", {}) or {}).get("action_to_background_ratio", "") if isinstance(body.get("action_background_separation_evidence", {}), dict) else "",
            "background_leakage_score": (body.get("action_background_separation_evidence", {}) or {}).get("background_leakage_score", "") if isinstance(body.get("action_background_separation_evidence", {}), dict) else "",
            "seam_interference_score": (body.get("action_background_separation_evidence", {}) or {}).get("seam_interference_score", "") if isinstance(body.get("action_background_separation_evidence", {}), dict) else "",
            "action_background_recommended_axis": (body.get("action_background_separation_evidence", {}) or {}).get("recommended_axis", "") if isinstance(body.get("action_background_separation_evidence", {}), dict) else "",
            "action_background_next_control_surface": (body.get("action_background_separation_evidence", {}) or {}).get("next_control_surface", "") if isinstance(body.get("action_background_separation_evidence", {}), dict) else "",
            "pixel_pressure_disagreement_status": (body.get("pixel_pressure_disagreement_review", {}) or {}).get("status", "not_recorded") if isinstance(body.get("pixel_pressure_disagreement_review", {}), dict) else "not_recorded",
            "pixel_pressure_disagreement_severity": (body.get("pixel_pressure_disagreement_review", {}) or {}).get("severity", "") if isinstance(body.get("pixel_pressure_disagreement_review", {}), dict) else "",
            "pixel_pressure_scalar_overweight_score": (body.get("pixel_pressure_disagreement_review", {}) or {}).get("scalar_pressure_overweight_score", "") if isinstance(body.get("pixel_pressure_disagreement_review", {}), dict) else "",
            "pixel_pressure_corrected_background_leakage_score": (body.get("pixel_pressure_disagreement_review", {}) or {}).get("corrected_background_leakage_score", "") if isinstance(body.get("pixel_pressure_disagreement_review", {}), dict) else "",
            "pixel_pressure_corrected_action_center_confidence": (body.get("pixel_pressure_disagreement_review", {}) or {}).get("corrected_action_center_confidence", "") if isinstance(body.get("pixel_pressure_disagreement_review", {}), dict) else "",
            "pixel_pressure_recommended_axis": (body.get("pixel_pressure_disagreement_review", {}) or {}).get("recommended_axis", "") if isinstance(body.get("pixel_pressure_disagreement_review", {}), dict) else "",
            "pixel_pressure_next_control_surface": (body.get("pixel_pressure_disagreement_review", {}) or {}).get("next_control_surface", "") if isinstance(body.get("pixel_pressure_disagreement_review", {}), dict) else "",
            "pressure_pixel_reweighting_status": (body.get("pressure_pixel_reweighting_proposal", {}) or {}).get("status", "not_recorded") if isinstance(body.get("pressure_pixel_reweighting_proposal", {}), dict) else "not_recorded",
            "pressure_pixel_reweighting_severity": (body.get("pressure_pixel_reweighting_proposal", {}) or {}).get("severity", "") if isinstance(body.get("pressure_pixel_reweighting_proposal", {}), dict) else "",
            "pressure_pixel_outcome_trust_weight": (body.get("pressure_pixel_reweighting_proposal", {}) or {}).get("pixel_outcome_trust_weight", "") if isinstance(body.get("pressure_pixel_reweighting_proposal", {}), dict) else "",
            "pressure_pixel_scalar_trust_weight": (body.get("pressure_pixel_reweighting_proposal", {}) or {}).get("scalar_pressure_trust_weight", "") if isinstance(body.get("pressure_pixel_reweighting_proposal", {}), dict) else "",
            "pressure_pixel_background_factor": (body.get("pressure_pixel_reweighting_proposal", {}) or {}).get("bounded_background_pressure_factor", "") if isinstance(body.get("pressure_pixel_reweighting_proposal", {}), dict) else "",
            "pressure_pixel_action_factor": (body.get("pressure_pixel_reweighting_proposal", {}) or {}).get("action_preservation_factor", "") if isinstance(body.get("pressure_pixel_reweighting_proposal", {}), dict) else "",
            "pressure_pixel_seam_protection_weight": (body.get("pressure_pixel_reweighting_proposal", {}) or {}).get("seam_protection_weight", "") if isinstance(body.get("pressure_pixel_reweighting_proposal", {}), dict) else "",
            "pressure_pixel_next_control_surface": (body.get("pressure_pixel_reweighting_proposal", {}) or {}).get("next_control_surface", "") if isinstance(body.get("pressure_pixel_reweighting_proposal", {}), dict) else "",
            "public_release_readiness_status": (body.get("public_release_readiness_gate", {}) or {}).get("status", "not_recorded") if isinstance(body.get("public_release_readiness_gate", {}), dict) else "not_recorded",
            "public_release_readiness_severity": (body.get("public_release_readiness_gate", {}) or {}).get("severity", "") if isinstance(body.get("public_release_readiness_gate", {}), dict) else "",
            "public_release_readiness_blocker_count": len((body.get("public_release_readiness_gate", {}) or {}).get("blockers", []) or []) if isinstance(body.get("public_release_readiness_gate", {}), dict) else 0,
            "public_release_readiness_warning_count": len((body.get("public_release_readiness_gate", {}) or {}).get("warnings", []) or []) if isinstance(body.get("public_release_readiness_gate", {}), dict) else 0,
            "public_release_readiness_research_flag_count": len((body.get("public_release_readiness_gate", {}) or {}).get("research_flags", []) or []) if isinstance(body.get("public_release_readiness_gate", {}), dict) else 0,
            "public_release_readiness_next_action": (body.get("public_release_readiness_gate", {}) or {}).get("next_action", "") if isinstance(body.get("public_release_readiness_gate", {}), dict) else "",
            "public_surface_contract_status": (body.get("public_surface_contract", {}) or {}).get("status", "not_recorded") if isinstance(body.get("public_surface_contract", {}), dict) else "not_recorded",
            "public_surface_contract_severity": (body.get("public_surface_contract", {}) or {}).get("severity", "") if isinstance(body.get("public_surface_contract", {}), dict) else "",
            "public_surface_contract_warning_count": len((body.get("public_surface_contract", {}) or {}).get("warnings", []) or []) if isinstance(body.get("public_surface_contract", {}), dict) else 0,
            "public_surface_contract_research_flag_count": len((body.get("public_surface_contract", {}) or {}).get("research_flags", []) or []) if isinstance(body.get("public_surface_contract", {}), dict) else 0,
            "public_package_static_scan_status": (body.get("public_package_static_scan", {}) or {}).get("status", "not_recorded") if isinstance(body.get("public_package_static_scan", {}), dict) else "not_recorded",
            "public_package_static_scan_severity": (body.get("public_package_static_scan", {}) or {}).get("severity", "") if isinstance(body.get("public_package_static_scan", {}), dict) else "",
            "public_package_static_scan_forbidden_dir_count": (body.get("public_package_static_scan", {}) or {}).get("forbidden_dir_count", 0) if isinstance(body.get("public_package_static_scan", {}), dict) else 0,
            "public_package_static_scan_forbidden_file_count": (body.get("public_package_static_scan", {}) or {}).get("forbidden_file_count", 0) if isinstance(body.get("public_package_static_scan", {}), dict) else 0,
            "math_topology_ledger_status": (body.get("math_topology_ledger", {}) or {}).get("status", "not_recorded") if isinstance(body.get("math_topology_ledger", {}), dict) else "not_recorded",
            "math_topology_ledger_severity": (body.get("math_topology_ledger", {}) or {}).get("severity", "") if isinstance(body.get("math_topology_ledger", {}), dict) else "",
            "math_topology_surface_count": (body.get("math_topology_ledger", {}) or {}).get("surface_count", 0) if isinstance(body.get("math_topology_ledger", {}), dict) else 0,
            "math_topology_present_surface_count": (body.get("math_topology_ledger", {}) or {}).get("present_surface_count", 0) if isinstance(body.get("math_topology_ledger", {}), dict) else 0,
            "math_topology_active_generation_surface_count": (body.get("math_topology_ledger", {}) or {}).get("active_generation_surface_count", 0) if isinstance(body.get("math_topology_ledger", {}), dict) else 0,
            "math_topology_research_surface_count": (body.get("math_topology_ledger", {}) or {}).get("research_surface_count", 0) if isinstance(body.get("math_topology_ledger", {}), dict) else 0,
            "math_topology_graph_status": (body.get("math_topology_dependency_graph", {}) or {}).get("status", "not_recorded") if isinstance(body.get("math_topology_dependency_graph", {}), dict) else "not_recorded",
            "math_topology_graph_severity": (body.get("math_topology_dependency_graph", {}) or {}).get("severity", "") if isinstance(body.get("math_topology_dependency_graph", {}), dict) else "",
            "math_topology_graph_node_count": (body.get("math_topology_dependency_graph", {}) or {}).get("node_count", 0) if isinstance(body.get("math_topology_dependency_graph", {}), dict) else 0,
            "math_topology_graph_edge_count": (body.get("math_topology_dependency_graph", {}) or {}).get("edge_count", 0) if isinstance(body.get("math_topology_dependency_graph", {}), dict) else 0,
            "math_topology_graph_active_generation_edge_count": (body.get("math_topology_dependency_graph", {}) or {}).get("active_generation_edge_count", 0) if isinstance(body.get("math_topology_dependency_graph", {}), dict) else 0,
            "math_topology_graph_research_edge_count": (body.get("math_topology_dependency_graph", {}) or {}).get("research_edge_count", 0) if isinstance(body.get("math_topology_dependency_graph", {}), dict) else 0,
            "top_conflict_collision": (body.get("strategy_matrix", {}) or {}).get("top_conflict_collision", "") if isinstance(body.get("strategy_matrix", {}), dict) else "",
            "top_drift_collision": (body.get("strategy_matrix", {}) or {}).get("top_drift_collision", "") if isinstance(body.get("strategy_matrix", {}), dict) else "",
            "object_relation_review_status": (body.get("object_relation_review", {}) or {}).get("status", "not_recorded") if isinstance(body.get("object_relation_review", {}), dict) else "not_recorded",
            "object_relation_drift_score": (body.get("object_relation_review", {}) or {}).get("object_relation_drift_score", "") if isinstance(body.get("object_relation_review", {}), dict) else "",
            "carrier_persistence_score": (body.get("object_relation_review", {}) or {}).get("carrier_persistence_score", "") if isinstance(body.get("object_relation_review", {}), dict) else "",
            "contact_boundary_continuity_score": (body.get("object_relation_review", {}) or {}).get("contact_boundary_continuity_score", "") if isinstance(body.get("object_relation_review", {}), dict) else "",
            "cascade_post_seam_acceleration_score": (body.get("object_relation_review", {}) or {}).get("cascade_post_seam_acceleration_score", "") if isinstance(body.get("object_relation_review", {}), dict) else "",
            "cascade_post_seam_attribution": (body.get("object_relation_review", {}) or {}).get("cascade_post_seam_attribution", "") if isinstance(body.get("object_relation_review", {}), dict) else "",
            "required_exact_missing": checks.get("required_exact_missing", []),
            "required_prefix_missing": checks.get("required_prefix_missing", []),
            "video_stage_missing": checks.get("video_stage_missing", []),
            "order_violations": order_audit.get("violations", []) if isinstance(order_audit, dict) else [],
            "next_action": gate.get("next_action", "") if isinstance(gate, dict) else "",
        }

    def _event_public_release_readiness_gate(self, packet, execution_records, gate=None, body=None):
        gate = gate if isinstance(gate, dict) else {}
        body = body if isinstance(body, dict) else {}
        metadata = packet.get("metadata", {}) if isinstance(packet, dict) else {}

        def latest_record(stage_name):
            for rec in reversed(execution_records or []):
                if isinstance(rec, dict) and str(rec.get("stage", "") or "") == stage_name:
                    return rec
            return {}

        def latest_prefix(prefix):
            for rec in reversed(execution_records or []):
                if isinstance(rec, dict) and str(rec.get("stage", "") or "").startswith(prefix):
                    return rec
            return {}

        def as_bool(value, default=False):
            if isinstance(value, bool):
                return value
            if value is None:
                return bool(default)
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)

        def issue(reason, evidence=None):
            out = {"reason": str(reason or "unknown")}
            if evidence is not None:
                out["evidence"] = evidence
            return out

        blockers = []
        warnings = []
        research_flags = []

        completion_status = str(gate.get("status", "UNKNOWN") or "UNKNOWN")
        result_status = str(body.get("result_status", "") or "")
        final_output_ok = as_bool(gate.get("final_output_ok", False))
        if completion_status != "PASS":
            blockers.append(issue("completion_gate_not_pass", {"completion_gate": completion_status}))
        if result_status and result_status != "VIDEO":
            blockers.append(issue("result_status_is_not_video", {"result_status": result_status}))
        if not final_output_ok:
            blockers.append(issue("final_output_not_confirmed", {"final_output_ok": final_output_ok}))

        input_normalization = latest_record("EventInputNormalization")
        input_adjustments = latest_record("EventInputNormalizationAdjustments")
        adjustment_count = 0
        if isinstance(input_normalization, dict):
            try:
                adjustment_count = int(input_normalization.get("adjustment_count", 0) or 0)
            except Exception:
                adjustment_count = 0
        adjustment_reasons = (
            input_normalization.get("adjustment_reason_histogram", {})
            if isinstance(input_normalization.get("adjustment_reason_histogram", {}), dict)
            else {}
        )
        if adjustment_count > 0:
            warnings.append(issue(
                "input_normalization_adjusted_workflow_values",
                {
                    "adjustment_count": adjustment_count,
                    "adjustment_reason_histogram": adjustment_reasons,
                    "adjustments_stage_present": bool(input_adjustments),
                },
            ))

        prompt_lock = latest_record("EventPromptPurityLock")
        transcode_apply = latest_record("EventPromptStrategyTranscodeApply")
        prompt_lock_present = bool(prompt_lock)
        if not prompt_lock_present:
            warnings.append(issue("prompt_purity_lock_not_recorded"))
        else:
            prompt_purity_lock = as_bool(prompt_lock.get("prompt_purity_lock", False))
            prompt_text_injection_allowed = as_bool(prompt_lock.get("prompt_text_injection_allowed", True), True)
            semantic_math_in_prompt_allowed = as_bool(prompt_lock.get("semantic_math_in_prompt_allowed", True), True)
            if not prompt_purity_lock:
                blockers.append(issue("prompt_purity_lock_false"))
            if prompt_text_injection_allowed or semantic_math_in_prompt_allowed:
                blockers.append(issue(
                    "prompt_text_or_semantic_math_injection_allowed",
                    {
                        "prompt_text_injection_allowed": prompt_text_injection_allowed,
                        "semantic_math_in_prompt_allowed": semantic_math_in_prompt_allowed,
                    },
                ))

        prompt_transcode_mode = str(
            transcode_apply.get("prompt_transcode_mode")
            or prompt_lock.get("prompt_transcode_mode")
            or "REPORT_ONLY"
        ).upper()
        if prompt_transcode_mode not in ("REPORT_ONLY", "OFF"):
            warnings.append(issue(
                "prompt_semantic_map_mode_enabled",
                {
                    "prompt_transcode_mode": prompt_transcode_mode,
                    "policy": transcode_apply.get("transcode_policy", "") if isinstance(transcode_apply, dict) else "",
                },
            ))

        math_summary = latest_record("EventMathControlSummary")
        math_mode = str(math_summary.get("math_control_mode", "OBSERVE_ONLY") or "OBSERVE_ONLY").upper()
        active_path = str(math_summary.get("active_generation_math_path", "") or "")
        active_allowed = as_bool(math_summary.get("strategy_control_surface_active_allowed", False))
        sampler_trace_mode = str(math_summary.get("sampler_trace_mode", "OFF") or "OFF").upper()
        safe_public_modes = {"OBSERVE_ONLY", "LATENT_DELTA_SCALE", "SELECTED_TAIL_SOURCE_RECONSTRUCTION"}
        research_modes = {"STRATEGY_PRESSURE_WINDOW", "LATENT_MEMORY_BRIDGE", "PRESSURE_PIXEL_REWEIGHTING", "SOURCE_NOISE_FIELD_SHAPING", "MAX_RISK_STRATEGY_RING", "DEEP_STEP_DELTA_CONTROL"}
        if math_mode == "MAX_RISK_STRATEGY_RING":
            research_flags.append(issue(
                "max_risk_strategy_ring_is_explicit_research_only",
                {"math_control_mode": math_mode, "active_generation_math_path": active_path},
            ))
        if math_mode == "DEEP_STEP_DELTA_CONTROL":
            research_flags.append(issue(
                "deep_step_delta_control_is_research_only",
                {"math_control_mode": math_mode, "active_generation_math_path": active_path},
            ))
        elif math_mode in research_modes:
            research_flags.append(issue(
                "research_math_mode_active",
                {"math_control_mode": math_mode, "active_generation_math_path": active_path},
            ))
        elif math_mode not in safe_public_modes:
            warnings.append(issue(
                "unknown_or_unclassified_math_mode",
                {"math_control_mode": math_mode, "active_generation_math_path": active_path},
            ))
        if active_allowed and math_mode not in safe_public_modes:
            research_flags.append(issue(
                "active_generation_math_surface_enabled",
                {"math_control_mode": math_mode, "active_generation_math_path": active_path},
            ))
        if sampler_trace_mode not in ("", "OFF"):
            warnings.append(issue(
                "sampler_trace_diagnostic_mode_enabled",
                {
                    "sampler_trace_mode": sampler_trace_mode,
                    "sampler_trace_max_steps": math_summary.get("sampler_trace_max_steps", ""),
                },
            ))

        public_surface_contract = latest_record("EventPublicSurfaceContract")
        if public_surface_contract:
            surface_status = str(public_surface_contract.get("status", "") or "").lower()
            surface_evidence = {
                "status": public_surface_contract.get("status", ""),
                "severity": public_surface_contract.get("severity", ""),
                "blocker_count": len(public_surface_contract.get("blockers", []) or []),
                "warning_count": len(public_surface_contract.get("warnings", []) or []),
                "research_flag_count": len(public_surface_contract.get("research_flags", []) or []),
                "default_deviation_count": len(public_surface_contract.get("default_deviations", []) or []),
            }
            if surface_status == "invalid_public_surface":
                blockers.append(issue("public_surface_contract_blocked", surface_evidence))
            elif surface_status == "research_mode":
                research_flags.append(issue("public_surface_contract_research_mode", surface_evidence))
            elif surface_status == "public_warning":
                warnings.append(issue("public_surface_contract_warning", surface_evidence))
        else:
            warnings.append(issue("public_surface_contract_not_recorded"))

        public_package_static = latest_record("EventPublicPackageStaticScan")
        if public_package_static:
            package_static_status = str(public_package_static.get("status", "") or "").lower()
            package_static_evidence = {
                "status": public_package_static.get("status", ""),
                "severity": public_package_static.get("severity", ""),
                "package_root": public_package_static.get("package_root", ""),
                "missing_required_files": public_package_static.get("missing_required_files", []),
                "missing_required_dirs": public_package_static.get("missing_required_dirs", []),
                "forbidden_dir_count": public_package_static.get("forbidden_dir_count", 0),
                "forbidden_file_count": public_package_static.get("forbidden_file_count", 0),
                "blocker_count": len(public_package_static.get("blockers", []) or []),
                "warning_count": len(public_package_static.get("warnings", []) or []),
            }
            if package_static_status == "public_package_static_blocked":
                blockers.append(issue("public_package_static_scan_blocked", package_static_evidence))
            elif package_static_status == "public_package_static_warning":
                warnings.append(issue("public_package_static_scan_warning", package_static_evidence))
            elif package_static_status != "public_package_static_clean":
                warnings.append(issue("public_package_static_scan_unknown_status", package_static_evidence))
        else:
            warnings.append(issue("public_package_static_scan_not_recorded"))

        math_topology_ledger = latest_record("EventMathTopologyLedger")
        if math_topology_ledger:
            ledger_status = str(math_topology_ledger.get("status", "") or "").lower()
            ledger_evidence = {
                "status": math_topology_ledger.get("status", ""),
                "severity": math_topology_ledger.get("severity", ""),
                "surface_count": math_topology_ledger.get("surface_count", 0),
                "present_surface_count": math_topology_ledger.get("present_surface_count", 0),
                "missing_required_surface_ids": math_topology_ledger.get("missing_required_surface_ids", []),
                "active_generation_surface_count": math_topology_ledger.get("active_generation_surface_count", 0),
                "active_generation_surface_ids": math_topology_ledger.get("active_generation_surface_ids", []),
                "research_surface_count": math_topology_ledger.get("research_surface_count", 0),
                "research_surface_ids": math_topology_ledger.get("research_surface_ids", []),
            }
            if ledger_status == "incomplete_math_topology":
                warnings.append(issue("math_topology_ledger_incomplete", ledger_evidence))
            elif int(math_topology_ledger.get("active_generation_surface_count", 0) or 0) > 0:
                research_flags.append(issue("math_topology_active_generation_surfaces_present", ledger_evidence))
            elif int(math_topology_ledger.get("research_surface_count", 0) or 0) > 0:
                warnings.append(issue("math_topology_research_or_diagnostic_surfaces_present", ledger_evidence))
        else:
            warnings.append(issue("math_topology_ledger_not_recorded"))

        math_topology_graph = latest_record("EventMathTopologyDependencyGraph")
        if math_topology_graph:
            graph_status = str(math_topology_graph.get("status", "") or "").lower()
            graph_evidence = {
                "status": math_topology_graph.get("status", ""),
                "severity": math_topology_graph.get("severity", ""),
                "node_count": math_topology_graph.get("node_count", 0),
                "edge_count": math_topology_graph.get("edge_count", 0),
                "required_missing_edge_ids": math_topology_graph.get("required_missing_edge_ids", []),
                "active_generation_edge_count": math_topology_graph.get("active_generation_edge_count", 0),
                "active_generation_edge_ids": math_topology_graph.get("active_generation_edge_ids", []),
                "research_edge_count": math_topology_graph.get("research_edge_count", 0),
                "research_edge_ids": math_topology_graph.get("research_edge_ids", []),
            }
            if graph_status in ("missing_math_topology_ledger", "incomplete_math_topology_graph"):
                warnings.append(issue("math_topology_dependency_graph_incomplete", graph_evidence))
            elif int(math_topology_graph.get("active_generation_edge_count", 0) or 0) > 0:
                research_flags.append(issue("math_topology_dependency_graph_active_edges_present", graph_evidence))
            elif int(math_topology_graph.get("research_edge_count", 0) or 0) > 0:
                warnings.append(issue("math_topology_dependency_graph_research_edges_present", graph_evidence))
        else:
            warnings.append(issue("math_topology_dependency_graph_not_recorded"))

        r126_route = latest_prefix("EventR126LowMidWindowSpatialControlRoute_")
        if r126_route and str(r126_route.get("status", "") or "") == "active":
            research_flags.append(issue(
                "r126_low_mid_window_spatial_route_active",
                {
                    "stage": r126_route.get("stage", ""),
                    "route_key": r126_route.get("route_key", ""),
                    "additional_sampler_calls": r126_route.get("additional_sampler_calls", ""),
                },
            ))

        tail_summary = metadata.get("tail_5_formula_summary", {})
        if not isinstance(tail_summary, dict):
            tail_summary = metadata.get("tail_3_formula_summary", {})
        if isinstance(tail_summary, dict):
            selection_policy = str(tail_summary.get("formula_recommendation_selection_policy", "") or "")
            if selection_policy.startswith("system_best_for_continuation_when_enabled"):
                warnings.append(issue(
                    "formula_tail_recommendation_selected_initial_tail",
                    {
                        "initial_selected_tail_index": tail_summary.get("initial_selected_tail_index", ""),
                        "system_best_for_continuation": tail_summary.get("system_best_for_continuation", ""),
                    },
                ))

        if blockers:
            status = "not_public_ready"
            severity = "BLOCKED"
            next_action = "Fix blockers first; do not package this run as public-ready evidence."
        elif research_flags:
            status = "research_mode"
            severity = "RESEARCH"
            next_action = "Use this run for internal math evidence; package only after a safe-mode comparison passes."
        elif warnings:
            status = "public_warning"
            severity = "WARNING"
            next_action = "Public route is structurally safe but review warnings and the actual mp4 before release."
        else:
            status = "public_ready"
            severity = "PASS"
            next_action = "This run is public-safe evidence structurally; still inspect the video for visual quality."

        return {
            "stage": "EventPublicReleaseReadinessGate",
            "status": status,
            "severity": severity,
            "gate_version": "public_release_readiness_gate_v1",
            "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
            "runtime_name": EVENT_HORIZON_RUNTIME_NAME,
            "formula": "Public readiness reads the whole finalized event route: clean carriers, completion, math mode, and research flags return to one Strategy safety verdict.",
            "does_not_measure_visual_quality": True,
            "completion_gate": completion_status,
            "result_status": result_status,
            "final_output_ok": final_output_ok,
            "math_control_mode": math_mode,
            "active_generation_math_path": active_path,
            "strategy_control_surface_active_allowed": active_allowed,
            "prompt_purity_lock_present": prompt_lock_present,
            "prompt_transcode_mode": prompt_transcode_mode,
            "input_adjustment_count": adjustment_count,
            "sampler_trace_mode": sampler_trace_mode,
            "math_topology_ledger_status": math_topology_ledger.get("status", "not_recorded") if isinstance(math_topology_ledger, dict) else "not_recorded",
            "math_topology_active_generation_surface_count": math_topology_ledger.get("active_generation_surface_count", 0) if isinstance(math_topology_ledger, dict) else 0,
            "math_topology_research_surface_count": math_topology_ledger.get("research_surface_count", 0) if isinstance(math_topology_ledger, dict) else 0,
            "math_topology_graph_status": math_topology_graph.get("status", "not_recorded") if isinstance(math_topology_graph, dict) else "not_recorded",
            "math_topology_graph_active_generation_edge_count": math_topology_graph.get("active_generation_edge_count", 0) if isinstance(math_topology_graph, dict) else 0,
            "math_topology_graph_research_edge_count": math_topology_graph.get("research_edge_count", 0) if isinstance(math_topology_graph, dict) else 0,
            "public_package_static_scan_status": public_package_static.get("status", "not_recorded") if isinstance(public_package_static, dict) else "not_recorded",
            "public_package_static_scan_severity": public_package_static.get("severity", "") if isinstance(public_package_static, dict) else "",
            "public_package_static_scan_forbidden_dir_count": public_package_static.get("forbidden_dir_count", 0) if isinstance(public_package_static, dict) else 0,
            "public_package_static_scan_forbidden_file_count": public_package_static.get("forbidden_file_count", 0) if isinstance(public_package_static, dict) else 0,
            "blockers": blockers,
            "warnings": warnings,
            "research_flags": research_flags,
            "public_safe_modes": sorted(safe_public_modes),
            "research_modes": sorted(research_modes),
            "next_action": next_action,
        }

    def _event_public_release_candidate_manifest(self, packet, execution_records, saved_report_path="", saved_video_path=""):
        metadata = packet.get("metadata", {}) if isinstance(packet, dict) else {}
        program_status = metadata.get("event_program_status", {}) if isinstance(metadata, dict) else {}
        if not isinstance(program_status, dict):
            program_status = {}

        def latest_record(stage_name):
            for rec in reversed(execution_records or []):
                if isinstance(rec, dict) and str(rec.get("stage", "") or "") == stage_name:
                    return rec
            return {}

        def as_bool(value, default=False):
            if isinstance(value, bool):
                return value
            if value is None:
                return bool(default)
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)

        def issue(reason, evidence=None):
            out = {"reason": str(reason or "unknown")}
            if evidence is not None:
                out["evidence"] = evidence
            return out

        gate = latest_record("EventCoreBodyCompletionGate")
        release_gate = latest_record("EventPublicReleaseReadinessGate")
        public_surface = latest_record("EventPublicSurfaceContract")
        public_package_static = latest_record("EventPublicPackageStaticScan")
        math_ledger = latest_record("EventMathTopologyLedger")
        math_graph = latest_record("EventMathTopologyDependencyGraph")
        save_report_record = latest_record("EventSaveReport")
        sidecars = latest_record("EventRuntimeMonitorSidecars")

        result_status = str(
            release_gate.get("result_status")
            or program_status.get("result_status")
            or metadata.get("result_status")
            or ""
        )
        final_output_ok = as_bool(gate.get("final_output_ok", False))
        completion_status = str(gate.get("status", "UNKNOWN") or "UNKNOWN")
        video_path = str(saved_video_path or program_status.get("saved_video_path") or metadata.get("saved_video_path") or "")
        report_path = str(saved_report_path or save_report_record.get("path", "") or "")

        blockers = []
        warnings = []
        research_flags = []

        if str(EVENT_HORIZON_RUNTIME_VERSION).lower().endswith("-dev"):
            blockers.append(issue(
                "dev_runtime_version_cannot_be_public_archive",
                {"runtime_version": EVENT_HORIZON_RUNTIME_VERSION, "runtime_name": EVENT_HORIZON_RUNTIME_NAME},
            ))
        if completion_status != "PASS":
            blockers.append(issue("completion_gate_not_pass", {"completion_gate": completion_status}))
        if result_status != "VIDEO":
            blockers.append(issue("result_status_is_not_video", {"result_status": result_status}))
        if not final_output_ok:
            blockers.append(issue("final_output_not_confirmed", {"final_output_ok": final_output_ok}))
        if not video_path:
            blockers.append(issue("saved_video_path_missing"))
        elif Path(video_path).is_absolute() and not Path(video_path).exists():
            warnings.append(issue("saved_video_path_not_found_on_disk", {"saved_video_path": video_path}))
        if not report_path:
            blockers.append(issue("saved_report_path_missing"))
        elif Path(report_path).is_absolute() and not Path(report_path).exists():
            blockers.append(issue("saved_report_path_not_found_on_disk", {"saved_report_path": report_path}))

        report_status = str(save_report_record.get("status", "not_recorded") or "not_recorded")
        if report_status != "standard_comfy_output_ok":
            blockers.append(issue("report_save_not_confirmed", {"event_save_report_status": report_status}))
        elif not as_bool(save_report_record.get("nonempty", False)):
            blockers.append(issue("saved_report_empty", {"bytes": save_report_record.get("bytes", "")}))

        release_status = str(release_gate.get("status", "not_recorded") or "not_recorded")
        if release_status == "not_public_ready":
            blockers.append(issue("public_release_readiness_blocked", {"status": release_status}))
        elif release_status == "research_mode":
            research_flags.append(issue("public_release_readiness_research_mode", {"status": release_status}))
        elif release_status == "public_warning":
            warnings.append(issue("public_release_readiness_warning", {"status": release_status}))
        elif release_status != "public_ready":
            warnings.append(issue("public_release_readiness_not_recorded_or_unknown", {"status": release_status}))

        public_surface_status = str(public_surface.get("status", "not_recorded") or "not_recorded")
        if public_surface_status == "invalid_public_surface":
            blockers.append(issue("public_surface_invalid", {"status": public_surface_status}))
        elif public_surface_status == "research_mode":
            research_flags.append(issue("public_surface_research_mode", {"status": public_surface_status}))
        elif public_surface_status == "public_warning":
            warnings.append(issue("public_surface_warning", {"status": public_surface_status}))
        elif public_surface_status not in ("public_safe", "not_recorded"):
            warnings.append(issue("public_surface_unknown_status", {"status": public_surface_status}))

        package_static_status = str(public_package_static.get("status", "not_recorded") or "not_recorded")
        if package_static_status == "public_package_static_blocked":
            blockers.append(issue(
                "public_package_static_scan_blocked",
                {
                    "status": package_static_status,
                    "package_root": public_package_static.get("package_root", ""),
                    "missing_required_files": public_package_static.get("missing_required_files", []),
                    "missing_required_dirs": public_package_static.get("missing_required_dirs", []),
                    "forbidden_dir_count": public_package_static.get("forbidden_dir_count", 0),
                    "forbidden_file_count": public_package_static.get("forbidden_file_count", 0),
                },
            ))
        elif package_static_status == "public_package_static_warning":
            warnings.append(issue(
                "public_package_static_scan_warning",
                {
                    "status": package_static_status,
                    "package_root": public_package_static.get("package_root", ""),
                    "warning_count": len(public_package_static.get("warnings", []) or []),
                },
            ))
        elif package_static_status != "public_package_static_clean":
            warnings.append(issue("public_package_static_scan_not_recorded_or_unknown", {"status": package_static_status}))

        graph_status = str(math_graph.get("status", "not_recorded") or "not_recorded")
        if graph_status in ("missing_math_topology_ledger", "incomplete_math_topology_graph"):
            warnings.append(issue("math_dependency_graph_incomplete", {"status": graph_status}))
        graph_nodes = math_graph.get("nodes", []) if isinstance(math_graph.get("nodes", []), list) else []
        graph_edges = math_graph.get("edges", []) if isinstance(math_graph.get("edges", []), list) else []
        graph_node_ids = {
            str(item.get("node_id", "") or "")
            for item in graph_nodes
            if isinstance(item, dict)
        }
        graph_edge_ids = {
            str(item.get("edge_id", "") or "")
            for item in graph_edges
            if isinstance(item, dict)
        }
        graph_contains_release_manifest = "public_release_candidate_manifest" in graph_node_ids
        graph_contains_package_verdict = "public_package_verdict" in graph_node_ids
        graph_contains_manifest_to_package_edge = "release_manifest_to_package_verdict" in graph_edge_ids
        graph_contains_package_static_scan = "public_package_static_scan" in graph_node_ids
        graph_contains_package_static_to_manifest_edge = "package_static_scan_to_release_manifest" in graph_edge_ids
        if math_graph and not graph_contains_package_static_scan:
            warnings.append(issue(
                "math_dependency_graph_missing_public_package_static_scan_node",
                {"required_node_id": "public_package_static_scan"},
            ))
        if math_graph and not graph_contains_package_static_to_manifest_edge:
            warnings.append(issue(
                "math_dependency_graph_missing_package_static_to_manifest_edge",
                {"required_edge_id": "package_static_scan_to_release_manifest"},
            ))
        if math_graph and not graph_contains_release_manifest:
            warnings.append(issue(
                "math_dependency_graph_missing_release_manifest_node",
                {"required_node_id": "public_release_candidate_manifest"},
            ))
        if math_graph and not graph_contains_package_verdict:
            warnings.append(issue(
                "math_dependency_graph_missing_public_package_verdict_node",
                {"required_node_id": "public_package_verdict"},
            ))
        if math_graph and not graph_contains_manifest_to_package_edge:
            warnings.append(issue(
                "math_dependency_graph_missing_manifest_to_package_edge",
                {"required_edge_id": "release_manifest_to_package_verdict"},
            ))
        if int(math_graph.get("active_generation_edge_count", 0) or 0) > 0:
            research_flags.append(issue(
                "math_dependency_graph_has_active_generation_edges",
                {"active_generation_edge_ids": math_graph.get("active_generation_edge_ids", [])},
            ))
        if int(math_ledger.get("active_generation_surface_count", 0) or 0) > 0:
            research_flags.append(issue(
                "math_topology_has_active_generation_surfaces",
                {"active_generation_surface_ids": math_ledger.get("active_generation_surface_ids", [])},
            ))

        if sidecars:
            if str(sidecars.get("status", "") or "") != "ok":
                warnings.append(issue("runtime_monitor_sidecars_not_ok", {"status": sidecars.get("status", "")}))
        else:
            warnings.append(issue("runtime_monitor_sidecars_not_recorded"))

        if blockers:
            status = "not_release_candidate"
            severity = "BLOCKED"
            can_package = False
            next_action = "Do not package this exact run; fix blockers or promote a non-dev release build first."
        elif research_flags:
            status = "research_candidate_only"
            severity = "RESEARCH"
            can_package = False
            next_action = "Use as internal math evidence only; create a safe public run before packaging."
        elif warnings:
            status = "release_candidate_with_warnings"
            severity = "WARNING"
            can_package = True
            next_action = "Candidate can support release notes after human video review and warning triage."
        else:
            status = "release_candidate"
            severity = "PASS"
            can_package = True
            next_action = "Candidate is structurally packageable after human video review."

        return {
            "stage": "EventPublicReleaseCandidateManifest",
            "status": status,
            "severity": severity,
            "manifest_version": "public_release_candidate_manifest_v1",
            "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
            "runtime_name": EVENT_HORIZON_RUNTIME_NAME,
            "formula": "Public release is a Strategy Return over artifacts: runtime label, VIDEO outcome, saved report, public/research gates, math topology, and human video review must converge before packaging.",
            "control_mode": "REPORT_ONLY",
            "does_not_change_generation": True,
            "does_not_measure_visual_quality": True,
            "requires_human_video_review": True,
            "can_package_public_archive": bool(can_package),
            "completion_gate": completion_status,
            "result_status": result_status,
            "final_output_ok": final_output_ok,
            "saved_video_path": video_path,
            "saved_report_path": report_path,
            "event_save_report_status": report_status,
            "public_release_readiness_status": release_status,
            "public_surface_contract_status": public_surface_status,
            "public_package_static_scan_status": package_static_status,
            "math_topology_ledger_status": math_ledger.get("status", "not_recorded") if isinstance(math_ledger, dict) else "not_recorded",
            "math_topology_graph_status": graph_status,
            "math_topology_graph_contains_release_manifest": bool(graph_contains_release_manifest),
            "math_topology_graph_contains_public_package_verdict": bool(graph_contains_package_verdict),
            "math_topology_graph_contains_manifest_to_package_edge": bool(graph_contains_manifest_to_package_edge),
            "math_topology_graph_contains_public_package_static_scan": bool(graph_contains_package_static_scan),
            "math_topology_graph_contains_package_static_to_manifest_edge": bool(graph_contains_package_static_to_manifest_edge),
            "math_active_surface_count": math_ledger.get("active_generation_surface_count", 0) if isinstance(math_ledger, dict) else 0,
            "math_active_edge_count": math_graph.get("active_generation_edge_count", 0) if isinstance(math_graph, dict) else 0,
            "blockers": blockers,
            "warnings": warnings,
            "research_flags": research_flags,
            "public_package_requirements": [
                "non-dev runtime version",
                "VIDEO result",
                "CompletionGate PASS",
                "saved nonempty Markdown report",
                "EventPublicReleaseReadinessGate not research/not_public_ready",
                "EventMathTopologyDependencyGraph not active research",
                "human mp4 visual inspection",
                "English-only public package scan",
                "zip contents scan with no pycache/internal clutter",
            ],
            "next_action": next_action,
        }

    def _event_core_body_report_card(self, audit, gate=None, public_release_gate=None, math_topology_ledger=None, math_topology_graph=None):
        checks = audit.get("checks", {}) if isinstance(audit, dict) else {}
        gate = gate if isinstance(gate, dict) else {}
        public_release_gate = public_release_gate if isinstance(public_release_gate, dict) else {}
        math_topology_ledger = math_topology_ledger if isinstance(math_topology_ledger, dict) else {}
        math_topology_graph = math_topology_graph if isinstance(math_topology_graph, dict) else {}
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
            "public_release_readiness_status": public_release_gate.get("status", "not_recorded"),
            "public_release_readiness_severity": public_release_gate.get("severity", ""),
            "public_release_readiness_blockers": public_release_gate.get("blockers", []),
            "public_release_readiness_warnings": public_release_gate.get("warnings", []),
            "public_release_readiness_research_flags": public_release_gate.get("research_flags", []),
            "public_surface_contract_status": (audit.get("public_surface_contract_status", "not_recorded") if isinstance(audit, dict) else "not_recorded"),
            "public_surface_contract_severity": (audit.get("public_surface_contract_severity", "") if isinstance(audit, dict) else ""),
            "public_surface_contract_warning_count": (audit.get("public_surface_contract_warning_count", 0) if isinstance(audit, dict) else 0),
            "public_surface_contract_research_flag_count": (audit.get("public_surface_contract_research_flag_count", 0) if isinstance(audit, dict) else 0),
            "public_package_static_scan_status": (audit.get("public_package_static_scan_status", "not_recorded") if isinstance(audit, dict) else "not_recorded"),
            "public_package_static_scan_severity": (audit.get("public_package_static_scan_severity", "") if isinstance(audit, dict) else ""),
            "public_package_static_scan_forbidden_dir_count": (audit.get("public_package_static_scan_forbidden_dir_count", 0) if isinstance(audit, dict) else 0),
            "public_package_static_scan_forbidden_file_count": (audit.get("public_package_static_scan_forbidden_file_count", 0) if isinstance(audit, dict) else 0),
            "math_topology_ledger_status": math_topology_ledger.get("status", "not_recorded"),
            "math_topology_ledger_severity": math_topology_ledger.get("severity", ""),
            "math_topology_surface_count": math_topology_ledger.get("surface_count", 0),
            "math_topology_present_surface_count": math_topology_ledger.get("present_surface_count", 0),
            "math_topology_active_generation_surface_count": math_topology_ledger.get("active_generation_surface_count", 0),
            "math_topology_research_surface_count": math_topology_ledger.get("research_surface_count", 0),
            "math_topology_active_generation_surface_ids": math_topology_ledger.get("active_generation_surface_ids", []),
            "math_topology_graph_status": math_topology_graph.get("status", "not_recorded"),
            "math_topology_graph_severity": math_topology_graph.get("severity", ""),
            "math_topology_graph_node_count": math_topology_graph.get("node_count", 0),
            "math_topology_graph_edge_count": math_topology_graph.get("edge_count", 0),
            "math_topology_graph_active_generation_edge_count": math_topology_graph.get("active_generation_edge_count", 0),
            "math_topology_graph_research_edge_count": math_topology_graph.get("research_edge_count", 0),
            "math_topology_graph_active_generation_edge_ids": math_topology_graph.get("active_generation_edge_ids", []),
            "math_topology_graph_research_edge_ids": math_topology_graph.get("research_edge_ids", []),
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
        object_relation_review = self._event_object_relation_review_from_records(
            execution_records,
            strategy_matrix=strategy_matrix,
            vector_collisions=vector_collisions,
        )
        strategy_matrix, vector_collisions = self._event_apply_object_relation_review(
            object_relation_review,
            strategy_matrix,
            vector_collisions,
        )
        body["strategy_matrix"] = strategy_matrix
        body["vector_collision_records"] = vector_collisions
        body["object_relation_review"] = object_relation_review

        def latest_execution_record(stage_name):
            for rec in reversed(execution_records or []):
                if isinstance(rec, dict) and str(rec.get("stage", "") or "") == str(stage_name):
                    return rec
            return {}

        cascade_seam_impulse_review = latest_execution_record("EventCascadeSeamImpulseReview")
        body["cascade_seam_impulse_review"] = cascade_seam_impulse_review
        strategy_matrix["cascade_seam_impulse_status"] = cascade_seam_impulse_review.get("status", "not_recorded") if isinstance(cascade_seam_impulse_review, dict) else "not_recorded"
        strategy_matrix["cascade_seam_impulse_score"] = cascade_seam_impulse_review.get("seam_impulse_score", "") if isinstance(cascade_seam_impulse_review, dict) else ""
        strategy_matrix["cascade_seam_vector_score"] = cascade_seam_impulse_review.get("vector_seam_impulse_score", "") if isinstance(cascade_seam_impulse_review, dict) else ""
        strategy_matrix["cascade_seam_visible_score"] = cascade_seam_impulse_review.get("visible_seam_delta_score", "") if isinstance(cascade_seam_impulse_review, dict) else ""
        strategy_matrix["cascade_seam_visible_over_median"] = cascade_seam_impulse_review.get("visible_seam_over_median_ratio", "") if isinstance(cascade_seam_impulse_review, dict) else ""
        strategy_matrix["cascade_seam_impulse_next_surface"] = cascade_seam_impulse_review.get("next_control_surface", "") if isinstance(cascade_seam_impulse_review, dict) else ""
        tail_next_source_continuity_proposal = latest_execution_record("EventTailNextSourceStrategyContinuityProposal")
        body["tail_next_source_continuity_proposal"] = tail_next_source_continuity_proposal
        strategy_matrix["tail_next_source_continuity_status"] = tail_next_source_continuity_proposal.get("status", "not_recorded") if isinstance(tail_next_source_continuity_proposal, dict) else "not_recorded"
        strategy_matrix["tail_next_source_continuity_pressure"] = tail_next_source_continuity_proposal.get("continuity_pressure_score", "") if isinstance(tail_next_source_continuity_proposal, dict) else ""
        strategy_matrix["tail_next_source_next_surface"] = tail_next_source_continuity_proposal.get("next_control_surface", "") if isinstance(tail_next_source_continuity_proposal, dict) else ""
        cascade_seam_phase_classifier = latest_execution_record("EventCascadeSeamPhaseClassifier")
        body["cascade_seam_phase_classifier"] = cascade_seam_phase_classifier
        phase_scores = cascade_seam_phase_classifier.get("axis_scores", {}) if isinstance(cascade_seam_phase_classifier, dict) else {}
        if not isinstance(phase_scores, dict):
            phase_scores = {}
        strategy_matrix["cascade_seam_phase_status"] = cascade_seam_phase_classifier.get("status", "not_recorded") if isinstance(cascade_seam_phase_classifier, dict) else "not_recorded"
        strategy_matrix["cascade_seam_phase_dominant_axis"] = cascade_seam_phase_classifier.get("dominant_axis", "") if isinstance(cascade_seam_phase_classifier, dict) else ""
        strategy_matrix["cascade_seam_phase_dominant_score"] = cascade_seam_phase_classifier.get("dominant_score", "") if isinstance(cascade_seam_phase_classifier, dict) else ""
        strategy_matrix["cascade_seam_phase_semantic_score"] = phase_scores.get("semantic_phase_reentry", "")
        strategy_matrix["cascade_seam_phase_prompt_text_change_score"] = phase_scores.get("prompt_text_change", "")
        strategy_matrix["cascade_seam_phase_prompt_score"] = phase_scores.get("prompt_phase_reentry", "")
        strategy_matrix["cascade_seam_phase_latent_score"] = phase_scores.get("latent_carrier_mismatch", "")
        strategy_matrix["cascade_seam_phase_background_score"] = phase_scores.get("background_anchor_conflict", "")
        strategy_matrix["cascade_seam_phase_center_score"] = phase_scores.get("center_action_overdrive", "")
        strategy_matrix["cascade_seam_phase_sampler_score"] = phase_scores.get("sampler_handoff_reset", "")
        strategy_matrix["cascade_seam_phase_next_surface"] = cascade_seam_phase_classifier.get("next_control_surface", "") if isinstance(cascade_seam_phase_classifier, dict) else ""
        semantic_phase_schedule_proposal = latest_execution_record("EventCascadeSemanticPhaseScheduleProposal")
        body["semantic_phase_schedule_proposal"] = semantic_phase_schedule_proposal
        strategy_matrix["semantic_phase_schedule_status"] = semantic_phase_schedule_proposal.get("status", "not_recorded") if isinstance(semantic_phase_schedule_proposal, dict) else "not_recorded"
        strategy_matrix["semantic_phase_schedule_score"] = semantic_phase_schedule_proposal.get("semantic_phase_reentry_score", "") if isinstance(semantic_phase_schedule_proposal, dict) else ""
        strategy_matrix["semantic_phase_schedule_prompt_clean"] = semantic_phase_schedule_proposal.get("prompt_carrier_clean", "") if isinstance(semantic_phase_schedule_proposal, dict) else ""
        strategy_matrix["semantic_phase_schedule_next_surface"] = semantic_phase_schedule_proposal.get("next_control_surface", "") if isinstance(semantic_phase_schedule_proposal, dict) else ""
        selected_tail_source_package = latest_execution_record("EventSelectedTailSourceReconstructionPackage")
        max_risk_strategy_package = latest_execution_record("EventMaxRiskStrategyRingPackage")
        body["selected_tail_source_reconstruction_package"] = selected_tail_source_package
        body["max_risk_strategy_ring_package"] = max_risk_strategy_package
        strategy_matrix["safe_math_package_status"] = selected_tail_source_package.get("status", "not_recorded") if isinstance(selected_tail_source_package, dict) else "not_recorded"
        strategy_matrix["safe_math_package_rebirth_risk"] = selected_tail_source_package.get("rebirth_risk_score", "") if isinstance(selected_tail_source_package, dict) else ""
        strategy_matrix["safe_math_package_next_surface"] = selected_tail_source_package.get("next_control_surface", "") if isinstance(selected_tail_source_package, dict) else ""
        strategy_matrix["max_risk_math_package_status"] = max_risk_strategy_package.get("status", "not_recorded") if isinstance(max_risk_strategy_package, dict) else "not_recorded"
        strategy_matrix["max_risk_math_package_ring_pressure"] = max_risk_strategy_package.get("strategy_ring_pressure_score", "") if isinstance(max_risk_strategy_package, dict) else ""
        strategy_matrix["max_risk_math_package_active"] = max_risk_strategy_package.get("active_control_allowed", "") if isinstance(max_risk_strategy_package, dict) else ""
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
        relation_pressure_cards = self._event_relation_pressure_cards_from_records(
            execution_records,
            strategy_matrix=strategy_matrix,
            object_relation_review=object_relation_review,
            vector_collisions=vector_collisions,
        )
        strategy_matrix["relation_pressure_card_count"] = len(relation_pressure_cards)
        strategy_matrix["relation_pressure_card_statuses"] = {
            str(card.get("stage", "") or ""): str(card.get("status", "") or "")
            for card in relation_pressure_cards
            if isinstance(card, dict)
        }
        body["relation_pressure_cards"] = relation_pressure_cards
        topology_strategy_return_map = self._event_topology_strategy_return_map(
            strategy_matrix=strategy_matrix,
            relation_pressure_cards=relation_pressure_cards,
            vector_collisions=vector_collisions,
            object_relation_review=object_relation_review,
        )
        body["topology_strategy_return_map"] = topology_strategy_return_map
        strategy_matrix["topology_strategy_return_status"] = topology_strategy_return_map.get("status", "")
        strategy_matrix["topology_sync_score"] = topology_strategy_return_map.get("topology_sync_score", "")
        strategy_matrix["topology_primary_pressure_axis"] = topology_strategy_return_map.get("primary_pressure_axis", [])
        strategy_return_pressure_resolver = self._event_strategy_return_pressure_resolver(
            topology_strategy_return_map=topology_strategy_return_map,
            relation_pressure_cards=relation_pressure_cards,
            strategy_matrix=strategy_matrix,
            object_relation_review=object_relation_review,
        )
        body["strategy_return_pressure_resolver"] = strategy_return_pressure_resolver
        strategy_matrix["strategy_return_resolver_status"] = strategy_return_pressure_resolver.get("status", "")
        strategy_matrix["strategy_return_pressure"] = strategy_return_pressure_resolver.get("strategy_return_pressure", "")
        strategy_matrix["strategy_return_primary_attribution"] = strategy_return_pressure_resolver.get("primary_attribution", "")
        strategy_matrix["strategy_return_next_control_surface"] = strategy_return_pressure_resolver.get("next_control_surface", "")
        visible_motion_strategy_return_gate = self._event_visible_motion_strategy_return_gate(
            strategy_return_pressure_resolver=strategy_return_pressure_resolver,
            relation_pressure_cards=relation_pressure_cards,
            object_relation_review=object_relation_review,
            topology_strategy_return_map=topology_strategy_return_map,
        )
        body["visible_motion_strategy_return_gate"] = visible_motion_strategy_return_gate
        strategy_matrix["visible_motion_strategy_return_status"] = visible_motion_strategy_return_gate.get("status", "")
        strategy_matrix["visible_motion_next_control_surface"] = visible_motion_strategy_return_gate.get("next_control_surface", "")
        strategy_matrix["visible_motion_active_control_allowed_next"] = visible_motion_strategy_return_gate.get("visible_motion_active_control_allowed_next", False)
        true_region_topology_evidence = self._event_true_region_topology_evidence(
            visible_motion_strategy_return_gate=visible_motion_strategy_return_gate,
            strategy_return_pressure_resolver=strategy_return_pressure_resolver,
            relation_pressure_cards=relation_pressure_cards,
            object_relation_review=object_relation_review,
            topology_strategy_return_map=topology_strategy_return_map,
        )
        body["true_region_topology_evidence"] = true_region_topology_evidence
        strategy_matrix["true_region_topology_status"] = true_region_topology_evidence.get("status", "")
        strategy_matrix["true_region_readiness_score"] = true_region_topology_evidence.get("region_readiness_score", "")
        strategy_matrix["true_region_next_control_surface"] = true_region_topology_evidence.get("next_control_surface", "")
        strategy_matrix["true_region_active_control_allowed_next"] = true_region_topology_evidence.get("true_region_active_control_allowed_next", False)
        fractal_strategy_intersection_map = self._event_fractal_strategy_intersection_map(
            execution_records=execution_records,
            topology_strategy_return_map=topology_strategy_return_map,
            strategy_return_pressure_resolver=strategy_return_pressure_resolver,
            visible_motion_strategy_return_gate=visible_motion_strategy_return_gate,
            true_region_topology_evidence=true_region_topology_evidence,
            relation_pressure_cards=relation_pressure_cards,
            vector_collisions=vector_collisions,
            object_relation_review=object_relation_review,
            depth=7,
        )
        body["fractal_strategy_intersection_map"] = fractal_strategy_intersection_map
        strategy_matrix["fractal_strategy_intersection_status"] = fractal_strategy_intersection_map.get("status", "")
        strategy_matrix["fractal_strategy_intersection_depth"] = fractal_strategy_intersection_map.get("fractal_depth", "")
        strategy_matrix["fractal_strategy_alignment_score"] = fractal_strategy_intersection_map.get("final_layer_alignment_score", "")
        strategy_matrix["fractal_strategy_next_control_surface"] = fractal_strategy_intersection_map.get("next_control_surface", "")
        region_weighted_fractal_strategy_return = self._event_region_weighted_fractal_strategy_return(
            fractal_strategy_intersection_map=fractal_strategy_intersection_map,
            true_region_topology_evidence=true_region_topology_evidence,
            visible_motion_strategy_return_gate=visible_motion_strategy_return_gate,
            strategy_return_pressure_resolver=strategy_return_pressure_resolver,
            object_relation_review=object_relation_review,
        )
        body["region_weighted_fractal_strategy_return"] = region_weighted_fractal_strategy_return
        strategy_matrix["region_weighted_fractal_status"] = region_weighted_fractal_strategy_return.get("status", "")
        strategy_matrix["region_axis_confidence"] = region_weighted_fractal_strategy_return.get("region_axis_confidence", "")
        strategy_matrix["dominant_axis_evidence_match"] = region_weighted_fractal_strategy_return.get("dominant_axis_evidence_match", False)
        strategy_matrix["background_overweight_guard"] = region_weighted_fractal_strategy_return.get("background_overweight_guard", False)
        strategy_matrix["region_weighted_fractal_next_control_surface"] = region_weighted_fractal_strategy_return.get("next_control_surface", "")
        pixel_region_motion_map = self._select_primary_pixel_region_motion_map(execution_records)
        body["pixel_region_motion_map"] = pixel_region_motion_map
        strategy_matrix["pixel_region_motion_status"] = pixel_region_motion_map.get("status", "not_recorded") if isinstance(pixel_region_motion_map, dict) else "not_recorded"
        strategy_matrix["pixel_center_edge_ratio"] = pixel_region_motion_map.get("center_edge_pixel_ratio", "") if isinstance(pixel_region_motion_map, dict) else ""
        strategy_matrix["pixel_edge_center_ratio"] = pixel_region_motion_map.get("edge_center_pixel_ratio", "") if isinstance(pixel_region_motion_map, dict) else ""
        strategy_matrix["pixel_estimated_seam_ratio"] = pixel_region_motion_map.get("estimated_seam_ratio", "") if isinstance(pixel_region_motion_map, dict) else ""
        strategy_matrix["pixel_background_leakage_score"] = pixel_region_motion_map.get("background_pixel_leakage_score", "") if isinstance(pixel_region_motion_map, dict) else ""
        strategy_matrix["pixel_region_next_control_surface"] = pixel_region_motion_map.get("next_control_surface", "") if isinstance(pixel_region_motion_map, dict) else ""
        pixel_region_motion_selection = {
            "stage": "EventPixelRegionMotionMapSelection",
            "status": "selected_primary_visible_outcome" if pixel_region_motion_map else "not_recorded",
            "severity": "INFO" if pixel_region_motion_map else "WARNING",
            "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
            "runtime_name": EVENT_HORIZON_RUNTIME_NAME,
            "selected_source_stage": str(pixel_region_motion_map.get("source_stage", "") if isinstance(pixel_region_motion_map, dict) else ""),
            "selected_status": str(pixel_region_motion_map.get("status", "") if isinstance(pixel_region_motion_map, dict) else ""),
            "selection_reason": str(pixel_region_motion_map.get("selection_reason", "") if isinstance(pixel_region_motion_map, dict) else ""),
            "available_pixel_region_map_count": int(pixel_region_motion_map.get("available_pixel_region_map_count", 0) if isinstance(pixel_region_motion_map, dict) else 0),
            "center_edge_pixel_ratio": pixel_region_motion_map.get("center_edge_pixel_ratio", "") if isinstance(pixel_region_motion_map, dict) else "",
            "edge_center_pixel_ratio": pixel_region_motion_map.get("edge_center_pixel_ratio", "") if isinstance(pixel_region_motion_map, dict) else "",
            "estimated_seam_ratio": pixel_region_motion_map.get("estimated_seam_ratio", "") if isinstance(pixel_region_motion_map, dict) else "",
            "background_pixel_leakage_score": pixel_region_motion_map.get("background_pixel_leakage_score", "") if isinstance(pixel_region_motion_map, dict) else "",
            "control_mode": "REPORT_ONLY",
            "active_control_allowed": False,
            "same_run_control_allowed": False,
            "formula": "The primary pixel-region record must represent the visible video Outcome, not an auxiliary tail preview record.",
        }
        body["pixel_region_motion_map_selection"] = pixel_region_motion_selection
        action_background_separation_evidence = self._event_action_background_separation_evidence(
            region_weighted_fractal_strategy_return=region_weighted_fractal_strategy_return,
            true_region_topology_evidence=true_region_topology_evidence,
            visible_motion_strategy_return_gate=visible_motion_strategy_return_gate,
            strategy_return_pressure_resolver=strategy_return_pressure_resolver,
            object_relation_review=object_relation_review,
            pixel_region_motion_map=pixel_region_motion_map,
        )
        body["action_background_separation_evidence"] = action_background_separation_evidence
        strategy_matrix["action_background_separation_status"] = action_background_separation_evidence.get("status", "")
        strategy_matrix["action_background_separation_confidence"] = action_background_separation_evidence.get("separation_confidence", "")
        strategy_matrix["action_background_next_control_surface"] = action_background_separation_evidence.get("next_control_surface", "")
        pixel_pressure_disagreement_review = self._event_pixel_pressure_disagreement_review(
            pixel_region_motion_map=pixel_region_motion_map,
            action_background_separation_evidence=action_background_separation_evidence,
            region_weighted_fractal_strategy_return=region_weighted_fractal_strategy_return,
            true_region_topology_evidence=true_region_topology_evidence,
            visible_motion_strategy_return_gate=visible_motion_strategy_return_gate,
        )
        body["pixel_pressure_disagreement_review"] = pixel_pressure_disagreement_review
        strategy_matrix["pixel_pressure_disagreement_status"] = pixel_pressure_disagreement_review.get("status", "")
        strategy_matrix["pixel_pressure_scalar_overweight_score"] = pixel_pressure_disagreement_review.get("scalar_pressure_overweight_score", "")
        strategy_matrix["pixel_pressure_corrected_background_leakage_score"] = pixel_pressure_disagreement_review.get("corrected_background_leakage_score", "")
        strategy_matrix["pixel_pressure_corrected_action_center_confidence"] = pixel_pressure_disagreement_review.get("corrected_action_center_confidence", "")
        strategy_matrix["pixel_pressure_recommended_axis"] = pixel_pressure_disagreement_review.get("recommended_axis", "")
        strategy_matrix["pixel_pressure_next_control_surface"] = pixel_pressure_disagreement_review.get("next_control_surface", "")
        pressure_pixel_reweighting_proposal = self._event_pressure_pixel_reweighting_proposal(
            pixel_pressure_disagreement_review=pixel_pressure_disagreement_review,
            pixel_region_motion_map=pixel_region_motion_map,
            action_background_separation_evidence=action_background_separation_evidence,
            region_weighted_fractal_strategy_return=region_weighted_fractal_strategy_return,
        )
        body["pressure_pixel_reweighting_proposal"] = pressure_pixel_reweighting_proposal
        strategy_matrix["pressure_pixel_reweighting_status"] = pressure_pixel_reweighting_proposal.get("status", "")
        strategy_matrix["pressure_pixel_outcome_trust_weight"] = pressure_pixel_reweighting_proposal.get("pixel_outcome_trust_weight", "")
        strategy_matrix["pressure_pixel_scalar_trust_weight"] = pressure_pixel_reweighting_proposal.get("scalar_pressure_trust_weight", "")
        strategy_matrix["pressure_pixel_background_factor"] = pressure_pixel_reweighting_proposal.get("bounded_background_pressure_factor", "")
        strategy_matrix["pressure_pixel_action_factor"] = pressure_pixel_reweighting_proposal.get("action_preservation_factor", "")
        strategy_matrix["pressure_pixel_next_control_surface"] = pressure_pixel_reweighting_proposal.get("next_control_surface", "")
        execution_records.append(object_relation_review)
        execution_records.extend(vector_collisions)
        execution_records.extend(local_micro_formula_records)
        execution_records.append(strategy_guidance)
        execution_records.extend(relation_pressure_cards)
        execution_records.append(topology_strategy_return_map)
        execution_records.append(strategy_return_pressure_resolver)
        execution_records.append(visible_motion_strategy_return_gate)
        execution_records.append(true_region_topology_evidence)
        execution_records.append(fractal_strategy_intersection_map)
        execution_records.append(region_weighted_fractal_strategy_return)
        execution_records.append(pixel_region_motion_selection)
        execution_records.append(action_background_separation_evidence)
        execution_records.append(pixel_pressure_disagreement_review)
        execution_records.append(pressure_pixel_reweighting_proposal)
        execution_records.append(strategy_matrix)
        math_topology_ledger = self._event_math_topology_ledger_from_records(execution_records)
        body["math_topology_ledger"] = math_topology_ledger
        execution_records.append(math_topology_ledger)
        math_topology_graph = self._event_math_topology_dependency_graph(math_topology_ledger)
        body["math_topology_dependency_graph"] = math_topology_graph
        execution_records.append(math_topology_graph)
        if body.get("runtime_monitor_summary"):
            execution_records.append({
                "stage": "EventRuntimeMonitorSummary",
                "status": "recorded",
                **body.get("runtime_monitor_summary", {}),
                "formula": "Runtime timing and memory are observer-only ObservedBehavior extensions.",
            })
        public_release_gate = self._event_public_release_readiness_gate(
            packet,
            execution_records,
            gate=gate,
            body=body,
        )
        body["public_release_readiness_gate"] = public_release_gate
        execution_records.append(public_release_gate)
        summary = self._event_core_body_summary_record(audit, order_audit, gate, body=body)
        body["summary"] = summary
        execution_records.append(summary)
        execution_records.append(self._event_core_body_report_card(
            audit,
            gate,
            public_release_gate=public_release_gate,
            math_topology_ledger=math_topology_ledger,
            math_topology_graph=math_topology_graph,
        ))
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





