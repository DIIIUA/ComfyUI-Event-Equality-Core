from .enums import REL_UNKNOWN, EQ_UNKNOWN
from ..utils.ids import new_relation_id


def make_event_relation(
    relation_type=REL_UNKNOWN,
    source_signal_ids=None,
    target_signal_ids=None,
    source_projection_ids=None,
    target_projection_ids=None,
    operator_name="RelationBuilder",
    formula_meaning="",
    parent_strategy_id=None,
    local_strategy_id=None,
    equality_status=EQ_UNKNOWN,
    confidence=1.0,
    metadata=None,
):
    """Create an EventRelation.

    Relations may be one-to-one, one-to-many, many-to-one, or many-to-many.
    """
    return {
        "id": new_relation_id(str(relation_type).lower()),
        "relation_type": relation_type,
        "source_signal_ids": source_signal_ids or [],
        "target_signal_ids": target_signal_ids or [],
        "source_projection_ids": source_projection_ids or [],
        "target_projection_ids": target_projection_ids or [],
        "parent_strategy_id": parent_strategy_id,
        "local_strategy_id": local_strategy_id,
        "operator_name": operator_name,
        "formula_meaning": formula_meaning,
        "equality_status": equality_status,
        "confidence": confidence,
        "conflict_ids": [],
        "metadata": metadata or {},
    }
