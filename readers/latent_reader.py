from .base import ReaderOperator
from ..core.enums import TECH_LATENT
from ..core.projection import make_event_projection
from ..utils.tensor_stats import summarize_latent


class LatentEventReader(ReaderOperator):
    name = "LatentEventReader"
    accepted_technical_types = [TECH_LATENT]

    def read(self, signal: dict) -> dict:
        latent = signal.get("raw_ref")
        numeric_summary = summarize_latent(latent)
        role_vector = {
            "is_latent": True,
            "is_hidden_event_carrier": True,
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
