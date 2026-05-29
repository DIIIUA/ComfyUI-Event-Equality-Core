from ..utils.ids import new_id


def make_route_memory() -> dict:
    """Create minimal RouteMemory.

    RouteMemory is required even in Code Pass 1 because it is the memory
    carrier for future EventSignal/EventRelation/SState records.
    """
    return {
        "id": new_id("route_memory"),
        "route_ids": [],
        "signal_ids": [],
        "relation_ids": [],
        "projection_ids": [],
        "sstate_ids": [],
        "equality_link_ids": [],
        "conflict_ids": [],
        "stage_records": [],
        "sampler_step_records": [],
        "summary": {},
        "metadata": {},
    }


def summarize_memory(memory: dict) -> dict:
    """Return compact counts for RouteMemory."""
    if not isinstance(memory, dict):
        return {
            "signals": 0,
            "relations": 0,
            "projections": 0,
            "sstates": 0,
            "conflicts": 0,
            "stage_records": 0,
            "sampler_step_records": 0,
        }

    summary = {
        "signals": len(memory.get("signal_ids", [])),
        "relations": len(memory.get("relation_ids", [])),
        "projections": len(memory.get("projection_ids", [])),
        "sstates": len(memory.get("sstate_ids", [])),
        "conflicts": len(memory.get("conflict_ids", [])),
        "stage_records": len(memory.get("stage_records", [])),
        "sampler_step_records": len(memory.get("sampler_step_records", [])),
    }
    memory["summary"] = summary
    return summary


def add_stage_record(memory: dict, record: dict) -> dict:
    """Append a stage record to RouteMemory."""
    if not isinstance(memory, dict):
        memory = make_route_memory()
    memory.setdefault("stage_records", []).append(record or {})
    summarize_memory(memory)
    return memory
