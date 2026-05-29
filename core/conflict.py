from .enums import SEV_INFO
from ..utils.ids import new_conflict_id


def make_conflict(
    conflict_type,
    severity=SEV_INFO,
    involved_signal_ids=None,
    involved_relation_ids=None,
    involved_projection_ids=None,
    involved_sstate_ids=None,
    stage_position="unknown",
    suspected_cause="",
    observed_symptom="",
    suggested_response="",
    metadata=None,
):
    """Create an EventConflict."""
    return {
        "id": new_conflict_id(str(conflict_type).lower()),
        "conflict_type": conflict_type,
        "severity": severity,
        "involved_signal_ids": involved_signal_ids or [],
        "involved_relation_ids": involved_relation_ids or [],
        "involved_projection_ids": involved_projection_ids or [],
        "involved_sstate_ids": involved_sstate_ids or [],
        "stage_position": stage_position,
        "suspected_cause": suspected_cause,
        "observed_symptom": observed_symptom,
        "suggested_response": suggested_response,
        "metadata": metadata or {},
    }
