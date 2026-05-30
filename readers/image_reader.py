from .base import ReaderOperator
from ..core.enums import TECH_IMAGE
from ..core.projection import make_event_projection
from ..utils.tensor_stats import summarize_image


class ImageOutcomeReader(ReaderOperator):
    name = "ImageOutcomeReader"
    accepted_technical_types = [TECH_IMAGE]

    def read(self, signal: dict) -> dict:
        image = signal.get("raw_ref")
        numeric_summary = summarize_image(image)
        role_vector = {
            "is_image": True,
            "is_visible_outcome": True,
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
