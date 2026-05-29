from .base import ReaderOperator
from ..core.enums import TECH_CONDITIONING
from ..core.projection import make_event_projection
from ..utils.tensor_stats import summarize_conditioning


class ConditioningStrategyReader(ReaderOperator):
    name = "ConditioningStrategyReader"
    accepted_technical_types = [TECH_CONDITIONING]

    def read(self, signal: dict) -> dict:
        cond = signal.get("raw_ref")
        numeric_summary = summarize_conditioning(cond)
        role_vector = {
            "is_conditioning": True,
            "is_strategy_carrier": True,
        }
        confidence = 0.8 if numeric_summary.get("length") is not None else 0.4
        return make_event_projection(
            source_signal_id=signal["id"],
            operator_name=self.name,
            role_vector=role_vector,
            numeric_summary=numeric_summary,
            confidence=confidence,
        )
