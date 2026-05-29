from ..core.enums import (
    TECH_TEXT,
    TECH_LATENT,
    TECH_IMAGE,
    TECH_NOISE,
    TECH_CONDITIONING,
    TECH_DELTA,
    ROLE_STRATEGY_CURRENT,
    ROLE_STRATEGY_CANDIDATE,
    ROLE_STRATEGY_CARRIER,
    ROLE_OUTCOME_PREVIOUS,
    ROLE_OUTCOME_NEXT,
    ROLE_OBSERVED_BEHAVIOR,
    ROLE_UNKNOWN,
)


def resolve_role(technical_type, position_context=None, manual_role=None):
    """Resolve FormulaRole for Code Pass 6."""
    if manual_role and manual_role not in ("AUTO", ROLE_UNKNOWN):
        return manual_role

    ctx = position_context or {}
    route_position = str(ctx.get("route_position") or "").lower()
    stage_name = str(ctx.get("stage_name") or "").lower()

    if technical_type == TECH_TEXT:
        return ROLE_STRATEGY_CURRENT

    if technical_type == TECH_LATENT:
        if "after_sampler" in route_position or "after" in route_position or "after_sampler" in stage_name:
            return ROLE_OUTCOME_NEXT
        if "before_sampler" in route_position or "before" in route_position or "before_sampler" in stage_name:
            return ROLE_OUTCOME_PREVIOUS
        return ROLE_STRATEGY_CARRIER

    if technical_type == TECH_IMAGE:
        if "after_decode" in route_position or "output" in route_position or "decode" in stage_name:
            return ROLE_OUTCOME_NEXT
        return ROLE_OUTCOME_PREVIOUS

    if technical_type == TECH_NOISE:
        return ROLE_STRATEGY_CANDIDATE

    if technical_type == TECH_CONDITIONING:
        return ROLE_STRATEGY_CARRIER

    if technical_type == TECH_DELTA:
        return ROLE_OBSERVED_BEHAVIOR

    return ROLE_UNKNOWN
