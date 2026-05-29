from ..utils.ids import new_sstate_id


def make_sstate(
    position="S0",
    active_signal_ids=None,
    active_relation_ids=None,
    active_projection_ids=None,
    parent_id=None,
    relation_graph=None,
    local_strategies=None,
    equality_links=None,
    conflict_ids=None,
    route_memory_id=None,
    next_requirement_id=None,
    confidence=1.0,
    metadata=None,
):
    """Create an SState.

    SState is relation-based, not signal-based.
    """
    return {
        "id": new_sstate_id(str(position).lower()),
        "position": position,
        "parent_id": parent_id,
        "active_signal_ids": active_signal_ids or [],
        "active_relation_ids": active_relation_ids or [],
        "active_projection_ids": active_projection_ids or [],
        "relation_graph": relation_graph or {},
        "local_strategies": local_strategies or {},
        "equality_center": position,
        "equality_links": equality_links or [],
        "conflict_ids": conflict_ids or [],
        "route_memory_id": route_memory_id,
        "next_requirement_id": next_requirement_id,
        "confidence": confidence,
        "metadata": metadata or {},
    }
