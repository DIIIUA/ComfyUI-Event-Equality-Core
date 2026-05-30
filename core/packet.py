from .memory import make_route_memory, summarize_memory, add_stage_record


def make_event_packet(metadata=None) -> dict:
    """Create a minimal EVENT_PACKET container."""
    return {
        "signals": {},
        "projections": {},
        "relations": {},
        "sstates": {},
        "equality_links": {},
        "conflicts": {},
        "route_memory": make_route_memory(),
        "current_sstate_id": None,
        "current_stage": None,
        "metadata": metadata or {},
    }


def ensure_packet(packet=None) -> dict:
    """Ensure an object is a usable EVENT_PACKET dict."""
    if packet is None or not isinstance(packet, dict):
        return make_event_packet()

    packet.setdefault("signals", {})
    packet.setdefault("projections", {})
    packet.setdefault("relations", {})
    packet.setdefault("sstates", {})
    packet.setdefault("equality_links", {})
    packet.setdefault("conflicts", {})
    packet.setdefault("route_memory", make_route_memory())
    packet.setdefault("current_sstate_id", None)
    packet.setdefault("current_stage", None)
    packet.setdefault("metadata", {})
    return packet


def _append_unique(target_list, value):
    if value and value not in target_list:
        target_list.append(value)


def add_signal(packet: dict, signal: dict) -> dict:
    packet = ensure_packet(packet)
    packet["signals"][signal["id"]] = signal
    _append_unique(packet["route_memory"].setdefault("signal_ids", []), signal["id"])
    return packet


def add_projection(packet: dict, projection: dict) -> dict:
    packet = ensure_packet(packet)
    packet["projections"][projection["id"]] = projection
    _append_unique(packet["route_memory"].setdefault("projection_ids", []), projection["id"])
    return packet


def add_relation(packet: dict, relation: dict) -> dict:
    packet = ensure_packet(packet)
    packet["relations"][relation["id"]] = relation
    _append_unique(packet["route_memory"].setdefault("relation_ids", []), relation["id"])
    return packet


def add_sstate(packet: dict, sstate: dict) -> dict:
    packet = ensure_packet(packet)
    packet["sstates"][sstate["id"]] = sstate
    packet["current_sstate_id"] = sstate["id"]
    _append_unique(packet["route_memory"].setdefault("sstate_ids", []), sstate["id"])
    return packet


def add_conflict(packet: dict, conflict: dict) -> dict:
    packet = ensure_packet(packet)
    packet["conflicts"][conflict["id"]] = conflict
    _append_unique(packet["route_memory"].setdefault("conflict_ids", []), conflict["id"])

    # Attach conflict id to involved relations/sstates if present.
    for rel_id in conflict.get("involved_relation_ids", []):
        rel = packet.get("relations", {}).get(rel_id)
        if rel is not None:
            _append_unique(rel.setdefault("conflict_ids", []), conflict["id"])

    for s_id in conflict.get("involved_sstate_ids", []):
        sstate = packet.get("sstates", {}).get(s_id)
        if sstate is not None:
            _append_unique(sstate.setdefault("conflict_ids", []), conflict["id"])

    return packet


def get_latest_signals(packet: dict, n: int = 1) -> list:
    packet = ensure_packet(packet)
    values = list(packet.get("signals", {}).values())
    return values[-n:]


def get_projection_ids_for_signal(packet: dict, signal_id: str) -> list:
    packet = ensure_packet(packet)
    return [
        proj_id for proj_id, proj in packet.get("projections", {}).items()
        if proj.get("source_signal_id") == signal_id
    ]


def signal_exists(packet: dict, signal_id: str) -> bool:
    packet = ensure_packet(packet)
    return signal_id in packet.get("signals", {})


def packet_summary(packet=None) -> dict:
    """Return compact EVENT_PACKET summary."""
    packet = ensure_packet(packet)
    memory = packet.get("route_memory", {})
    memory_summary = summarize_memory(memory)

    return {
        "signals": len(packet.get("signals", {})),
        "projections": len(packet.get("projections", {})),
        "relations": len(packet.get("relations", {})),
        "sstates": len(packet.get("sstates", {})),
        "conflicts": len(packet.get("conflicts", {})),
        "current_sstate_id": packet.get("current_sstate_id"),
        "route_memory_id": memory.get("id"),
        "stage_records": memory_summary.get("stage_records", 0),
        "sampler_step_records": memory_summary.get("sampler_step_records", 0),
    }


def record_stage(
    packet: dict,
    stage_name: str,
    action: str,
    observed_behavior: str = "",
    metadata=None,
    input_signal_ids=None,
    output_signal_ids=None,
    projection_ids=None,
    relation_ids=None,
    sstate_ids=None,
    conflict_ids=None,
    formula_note: str = "",
) -> dict:
    """Attach a stage record to RouteMemory."""
    packet = ensure_packet(packet)
    record = {
        "stage_name": stage_name,
        "node_name": stage_name,
        "action": action,
        "input_signal_ids": input_signal_ids or [],
        "output_signal_ids": output_signal_ids or [],
        "projection_ids": projection_ids or [],
        "relation_ids": relation_ids or [],
        "sstate_ids": sstate_ids or [],
        "conflict_ids": conflict_ids or [],
        "observed_behavior": observed_behavior,
        "formula_note": formula_note,
        "metadata": metadata or {},
    }
    packet["route_memory"] = add_stage_record(packet.get("route_memory", {}), record)
    packet["current_stage"] = stage_name
    return packet
