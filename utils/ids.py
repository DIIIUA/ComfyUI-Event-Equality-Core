# Simple readable ID utilities for Event Equality Core.

_COUNTERS = {}


def reset_ids():
    """Reset in-memory counters. Mostly useful for tests."""
    global _COUNTERS
    _COUNTERS = {}


def new_id(prefix: str) -> str:
    """Create a readable process-local ID."""
    safe_prefix = str(prefix).strip().replace(" ", "_").replace("/", "_") or "id"
    value = _COUNTERS.get(safe_prefix, 0) + 1
    _COUNTERS[safe_prefix] = value
    return f"{safe_prefix}_{value:06d}"


def new_signal_id(kind: str = "signal") -> str:
    return new_id(f"sig_{kind}")


def new_projection_id(kind: str = "projection") -> str:
    return new_id(f"proj_{kind}")


def new_relation_id(kind: str = "relation") -> str:
    return new_id(f"rel_{kind}")


def new_sstate_id(position: str = "s") -> str:
    return new_id(f"s_{position}")


def new_conflict_id(kind: str = "conflict") -> str:
    return new_id(f"conflict_{kind}")
