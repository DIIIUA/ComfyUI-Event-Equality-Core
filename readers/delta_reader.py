from .base import ReaderOperator
from ..core.enums import TECH_DELTA
from ..core.projection import make_event_projection
from ..utils.tensor_stats import summarize_delta


class DeltaReader(ReaderOperator):
    name = "DeltaReader"
    accepted_technical_types = [TECH_DELTA]

    def read(self, signal: dict) -> dict:
        delta = signal.get("raw_ref")
        before_ref = signal.get("metadata", {}).get("before_ref")
        numeric_summary = summarize_delta(delta, before=before_ref)

        role_vector = {
            "is_delta": True,
            "is_observed_behavior": True,
            "has_shape": bool(numeric_summary.get("shape")),
        }
        confidence = 1.0 if numeric_summary.get("shape") else 0.4

        return make_event_projection(
            source_signal_id=signal["id"],
            operator_name=self.name,
            role_vector=role_vector,
            numeric_summary=numeric_summary,
            confidence=confidence,
        )
