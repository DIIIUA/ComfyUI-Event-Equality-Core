from .enums import SPACE_EVENT
from ..utils.ids import new_projection_id


def make_event_projection(
    source_signal_id,
    operator_name="unknown",
    role_vector=None,
    numeric_summary=None,
    semantic_summary=None,
    stability_score=None,
    change_score=None,
    freedom_score=None,
    constraint_score=None,
    confidence=1.0,
    metadata=None,
):
    """Create an EventProjection."""
    return {
        "id": new_projection_id(str(operator_name).lower()),
        "source_signal_id": source_signal_id,
        "projection_space": SPACE_EVENT,
        "operator_name": operator_name,
        "role_vector": role_vector or {},
        "stability_score": stability_score,
        "change_score": change_score,
        "freedom_score": freedom_score,
        "constraint_score": constraint_score,
        "route_signature": "",
        "confidence": confidence,
        "numeric_summary": numeric_summary or {},
        "semantic_summary": semantic_summary or {},
        "metadata": metadata or {},
    }
