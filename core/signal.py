from .enums import TECH_UNKNOWN, ROLE_UNKNOWN, SPACE_UNKNOWN
from ..utils.ids import new_signal_id
from ..utils.tensor_stats import safe_shape, safe_dtype, safe_device


def make_event_signal(
    technical_type=TECH_UNKNOWN,
    formula_role=ROLE_UNKNOWN,
    representation_space=SPACE_UNKNOWN,
    raw_ref=None,
    source_stage="unknown",
    created_by="unknown",
    reader_operator="unknown",
    route_id="default",
    position_context=None,
    metadata=None,
):
    """Create an EventSignal.

    raw_ref may remain in packet for downstream readers, but reports must
    use signal_public_summary and never print raw_ref.
    """
    kind = str(technical_type).lower()
    return {
        "id": new_signal_id(kind),
        "technical_type": technical_type,
        "formula_role": formula_role,
        "representation_space": representation_space,
        "raw_ref": raw_ref,
        "source_stage": source_stage,
        "created_by": created_by,
        "position_context": position_context or {},
        "shape": safe_shape(raw_ref),
        "dtype": safe_dtype(raw_ref),
        "device": safe_device(raw_ref),
        "reader_operator": reader_operator,
        "route_id": route_id,
        "parent_signal_ids": [],
        "child_signal_ids": [],
        "metadata": metadata or {},
    }


def signal_public_summary(signal: dict) -> dict:
    """Return a report-safe signal summary without raw_ref."""
    if not isinstance(signal, dict):
        return {}
    return {k: v for k, v in signal.items() if k != "raw_ref"}
