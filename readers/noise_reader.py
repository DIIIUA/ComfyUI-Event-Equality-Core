from .base import ReaderOperator
from ..core.enums import TECH_NOISE
from ..core.projection import make_event_projection
from ..utils.tensor_stats import summarize_noise


class NoisePossibilityReader(ReaderOperator):
    name = "NoisePossibilityReader"
    accepted_technical_types = [TECH_NOISE]

    def read(self, signal: dict) -> dict:
        noise = signal.get("raw_ref")
        numeric_summary = summarize_noise(noise)
        metadata = signal.get("metadata", {}) or {}
        for key in ["seed", "noise_strength", "noise_mode", "generated_from_latent_shape", "input_socket"]:
            if key in metadata:
                numeric_summary[key] = metadata[key]

        role_vector = {
            "is_noise": True,
            "is_possibility_field": True,
            "is_strategy_candidate": True,
            "has_shape": bool(numeric_summary.get("shape")),
        }
        confidence = 1.0 if numeric_summary.get("shape") else 0.4
        return make_event_projection(
            source_signal_id=signal["id"],
            operator_name=self.name,
            role_vector=role_vector,
            numeric_summary=numeric_summary,
            confidence=confidence,
            metadata={
                "noise_route": True,
                "noise_mode": metadata.get("noise_mode", ""),
                "seed": metadata.get("seed"),
                "noise_strength": metadata.get("noise_strength"),
            },
        )
