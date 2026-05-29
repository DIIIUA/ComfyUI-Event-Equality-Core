from ..core.sstate import make_sstate
from ..core.packet import add_sstate


def build_sstate_from_packet(packet: dict, position="S0", active_relation_ids=None):
    """Build SState from active relations in packet."""
    relations = packet.get("relations", {})
    if active_relation_ids is None:
        active_relation_ids = list(relations.keys())

    active_signal_ids = set()
    active_projection_ids = set()
    local_strategies = {}
    relation_graph = {}

    for rel_id in active_relation_ids:
        rel = relations.get(rel_id)
        if not rel:
            continue

        for sid in rel.get("source_signal_ids", []):
            active_signal_ids.add(sid)
        for sid in rel.get("target_signal_ids", []):
            active_signal_ids.add(sid)

        for pid in rel.get("source_projection_ids", []):
            active_projection_ids.add(pid)
        for pid in rel.get("target_projection_ids", []):
            active_projection_ids.add(pid)

        local_id = rel.get("local_strategy_id")
        if not local_id:
            local_id = f"{position}.{str(rel.get('relation_type', 'relation')).lower()}"

        local_strategies[local_id] = {
            "relation_id": rel_id,
            "relation_type": rel.get("relation_type"),
            "source_signal_ids": rel.get("source_signal_ids", []),
            "target_signal_ids": rel.get("target_signal_ids", []),
            "formula_meaning": rel.get("formula_meaning", ""),
        }

        relation_graph[rel_id] = {
            "sources": rel.get("source_signal_ids", []),
            "targets": rel.get("target_signal_ids", []),
            "type": rel.get("relation_type"),
            "local_strategy_id": local_id,
        }

    sstate = make_sstate(
        position=position,
        active_signal_ids=list(active_signal_ids),
        active_relation_ids=active_relation_ids,
        active_projection_ids=list(active_projection_ids),
        relation_graph=relation_graph,
        local_strategies=local_strategies,
        route_memory_id=packet.get("route_memory", {}).get("id"),
        metadata={"created_by": "SResolver"},
    )

    packet = add_sstate(packet, sstate)
    return packet, sstate
