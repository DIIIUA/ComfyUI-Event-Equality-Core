from ...core.enums import (
    TECH_TEXT,
    TECH_LATENT,
    TECH_IMAGE,
    TECH_NOISE,
    TECH_CONDITIONING,
    REL_GUIDES,
    SEV_INFO,
    SEV_LOW,
    CONFLICT_WAN_HIGH_LOW_ROUTE_MISSING,
    CONFLICT_WAN_TEMPORAL_ROUTE_UNKNOWN,
    CONFLICT_WAN_PROMPT_ROUTE_WEAK,
    CONFLICT_WAN_SOURCE_ROUTE_MISSING,
    CONFLICT_WAN_DECODE_ROUTE_MISSING,
)
from ...core.packet import ensure_packet, add_conflict, add_relation, get_projection_ids_for_signal
from ...core.conflict import make_conflict
from ...core.relation import make_event_relation


def _signals_by_type(packet, technical_type):
    return [
        sig for sig in packet.get("signals", {}).values()
        if sig.get("technical_type") == technical_type
    ]


def _has_stage_or_route(signal, needles):
    source_stage = str(signal.get("source_stage", "")).lower()
    route_id = str(signal.get("route_id", "")).lower()
    position = str(signal.get("position_context", {}).get("route_position", "")).lower()
    hay = " ".join([source_stage, route_id, position])
    return any(str(n).lower() in hay for n in needles)


def analyze_wan_routes(packet, hints=None):
    packet = ensure_packet(packet)
    hints = hints or {}

    text_signals = _signals_by_type(packet, TECH_TEXT)
    latent_signals = _signals_by_type(packet, TECH_LATENT)
    image_signals = _signals_by_type(packet, TECH_IMAGE)
    noise_signals = _signals_by_type(packet, TECH_NOISE)
    conditioning_signals = _signals_by_type(packet, TECH_CONDITIONING)

    high_candidates = [s for s in latent_signals if _has_stage_or_route(s, ["high", "high_sampler"])]
    low_candidates = [s for s in latent_signals if _has_stage_or_route(s, ["low", "low_sampler"])]
    before_candidates = [s for s in latent_signals if _has_stage_or_route(s, ["before_sampler", "before"])]
    after_candidates = [s for s in latent_signals if _has_stage_or_route(s, ["after_sampler", "after"])]

    route_status = {
        "source_route": "present" if image_signals or any(_has_stage_or_route(s, ["source"]) for s in latent_signals) else "missing",
        "prompt_umt5_route": "present" if text_signals or conditioning_signals else "missing",
        "conditioning_route": "present" if conditioning_signals else "missing",
        "noise_route": "present" if noise_signals else "missing",
        "high_sampler_route": "present" if high_candidates or hints.get("has_high_sampler") else "unknown",
        "low_sampler_route": "present" if low_candidates or hints.get("has_low_sampler") else "unknown",
        "temporal_route": "present" if hints.get("has_temporal_module") else "unknown",
        "decode_route": "present" if any(_has_stage_or_route(s, ["decode", "decoded", "output"]) for s in image_signals) else ("present" if image_signals else "missing"),
        "output_video_route": "present" if image_signals else "missing",
    }

    route_labels = {
        "wan.source_image_route": [s["id"] for s in image_signals],
        "wan.prompt_umt5_route": [s["id"] for s in text_signals],
        "wan.conditioning_route": [s["id"] for s in conditioning_signals],
        "wan.noise_route": [s["id"] for s in noise_signals],
        "wan.high_sampler_route": [s["id"] for s in high_candidates],
        "wan.low_sampler_route": [s["id"] for s in low_candidates],
        "wan.sampler_before_route": [s["id"] for s in before_candidates],
        "wan.sampler_after_route": [s["id"] for s in after_candidates],
        "wan.decode_route": [s["id"] for s in image_signals if _has_stage_or_route(s, ["decode", "decoded", "output"])],
    }

    diagnostics = {
        "text_signals": len(text_signals),
        "latent_signals": len(latent_signals),
        "image_signals": len(image_signals),
        "noise_signals": len(noise_signals),
        "conditioning_signals": len(conditioning_signals),
        "has_high_low_hint": bool(hints.get("has_high_sampler") or hints.get("has_low_sampler")),
        "workflow_hint": hints.get("workflow_hint", ""),
        "has_lora_stack": bool(hints.get("has_lora_stack")),
        "has_accvid": bool(hints.get("has_accvid")),
        "has_lightx2v": bool(hints.get("has_lightx2v")),
    }

    return {
        "route_status": route_status,
        "route_labels": route_labels,
        "diagnostics": diagnostics,
        "high_candidates": high_candidates,
        "low_candidates": low_candidates,
    }


def apply_wan_adapter(packet, mode="BASIC", hints=None):
    packet = ensure_packet(packet)
    hints = hints or {}
    analysis = analyze_wan_routes(packet, hints=hints)
    conflicts = []
    created_relations = []

    route_status = analysis["route_status"]

    def add_wan_conflict(conflict_type, severity, symptom, suggestion, route):
        conflict = make_conflict(
            conflict_type,
            severity=severity,
            stage_position="EventWanAdapter",
            suspected_cause=f"Wan route status warning for {route}.",
            observed_symptom=symptom,
            suggested_response=suggestion,
            metadata={"route": route, "wan_mode": mode},
        )
        nonlocal packet
        packet = add_conflict(packet, conflict)
        conflicts.append(conflict["id"])

    if route_status.get("source_route") == "missing":
        add_wan_conflict(
            CONFLICT_WAN_SOURCE_ROUTE_MISSING,
            SEV_LOW,
            "Wan source route is missing.",
            "Connect source image or source latent if this workflow is image/video anchored.",
            "source_route",
        )

    if route_status.get("prompt_umt5_route") == "missing":
        add_wan_conflict(
            CONFLICT_WAN_PROMPT_ROUTE_WEAK,
            SEV_LOW,
            "Wan prompt/UMT5 route is missing or weak.",
            "Connect text/conditioning route if prompt diagnostics are needed.",
            "prompt_umt5_route",
        )

    if route_status.get("high_sampler_route") == "unknown" or route_status.get("low_sampler_route") == "unknown":
        add_wan_conflict(
            CONFLICT_WAN_HIGH_LOW_ROUTE_MISSING,
            SEV_INFO,
            "Wan high/low sampler route is unknown.",
            "Provide high/low hints or label stages/routes with high_sampler and low_sampler.",
            "high_low",
        )

    if route_status.get("temporal_route") == "unknown":
        add_wan_conflict(
            CONFLICT_WAN_TEMPORAL_ROUTE_UNKNOWN,
            SEV_INFO,
            "Wan temporal route is unknown.",
            "Enable has_temporal_module if the workflow has a temporal consistency component.",
            "temporal_route",
        )

    if route_status.get("decode_route") == "missing":
        add_wan_conflict(
            CONFLICT_WAN_DECODE_ROUTE_MISSING,
            SEV_LOW,
            "Wan decode route is missing.",
            "Connect decoded image/video output if decode diagnostics are needed.",
            "decode_route",
        )

    # If clear high and low candidates exist, create a labeling relation.
    high_candidates = analysis.get("high_candidates", [])
    low_candidates = analysis.get("low_candidates", [])
    if high_candidates and low_candidates:
        high_id = high_candidates[-1]["id"]
        low_id = low_candidates[-1]["id"]
        relation = make_event_relation(
            relation_type=REL_GUIDES,
            source_signal_ids=[high_id],
            target_signal_ids=[low_id],
            source_projection_ids=get_projection_ids_for_signal(packet, high_id),
            target_projection_ids=get_projection_ids_for_signal(packet, low_id),
            operator_name="EventWanAdapter",
            formula_meaning="Wan high route guides low route refinement",
            local_strategy_id="S0.wan.high_low",
            metadata={
                "wan_relation": "HIGH_GUIDES_LOW",
                "adapter_mode": mode,
            },
        )
        packet = add_relation(packet, relation)
        created_relations.append(relation["id"])

    packet.setdefault("metadata", {}).setdefault("wan_adapter", {})
    packet["metadata"]["wan_adapter"].update({
        "enabled": True,
        "mode": mode,
        "route_status": route_status,
        "route_labels": analysis["route_labels"],
        "diagnostics": analysis["diagnostics"],
        "conflict_ids": conflicts,
        "created_relation_ids": created_relations,
    })

    return packet, {
        "route_status": route_status,
        "route_labels": analysis["route_labels"],
        "diagnostics": analysis["diagnostics"],
        "conflict_ids": conflicts,
        "created_relation_ids": created_relations,
    }
