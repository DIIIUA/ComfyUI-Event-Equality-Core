from .base import ReaderOperator
from ..core.enums import TECH_TEXT
from ..core.projection import make_event_projection


def _present(value):
    return bool(str(value or "").strip())


def build_priority_map(structured: dict) -> dict:
    return {
        "P0_geometry_anchors": structured.get("stable_anchors", ""),
        "P1_primary_objects_roles": structured.get("main_strategy", ""),
        "P2_motion_force_direction": " | ".join([x for x in [
            structured.get("motion_axis", ""),
            structured.get("active_changes", ""),
        ] if _present(x)]),
        "P3_contact_zone": structured.get("contact_rule", ""),
        "P4_reciprocal_reaction": structured.get("reciprocal_reaction", ""),
        "P5_deformation_material_support": " | ".join([x for x in [
            structured.get("deformation_rule", ""),
            structured.get("material_rule", ""),
            structured.get("support_rule", ""),
        ] if _present(x)]),
        "P6_endpoint_equilibrium": structured.get("endpoint", ""),
        "P7_loop_closure": structured.get("loop_rule", ""),
        "P8_negative_constraints": " | ".join([x for x in [
            structured.get("forbidden_drift", ""),
            structured.get("negative_additions", ""),
        ] if _present(x)]),
        "P9_detail_low_priority": structured.get("priority_notes", ""),
    }


def build_route_hints(structured: dict) -> dict:
    return {
        "source_route_should_preserve": structured.get("stable_anchors", ""),
        "sampler_route_should_change": " | ".join([x for x in [
            structured.get("active_changes", ""),
            structured.get("motion_axis", ""),
        ] if _present(x)]),
        "contact_route_should_focus": " | ".join([x for x in [
            structured.get("contact_rule", ""),
            structured.get("reciprocal_reaction", ""),
        ] if _present(x)]),
        "decode_route_should_reveal": structured.get("endpoint", ""),
        "noise_route_should_suppress": structured.get("forbidden_drift", ""),
        "output_route_should_match": " | ".join([x for x in [
            structured.get("endpoint", ""),
            structured.get("loop_rule", ""),
        ] if _present(x)]),
    }


def build_strategy_expectations(structured: dict) -> dict:
    return {
        "expected_stable": structured.get("stable_anchors", ""),
        "expected_change": structured.get("active_changes", ""),
        "expected_direction": structured.get("motion_axis", ""),
        "expected_contact": structured.get("contact_rule", ""),
        "expected_reaction": structured.get("reciprocal_reaction", ""),
        "expected_deformation": structured.get("deformation_rule", ""),
        "expected_endpoint": structured.get("endpoint", ""),
        "forbidden_changes": structured.get("forbidden_drift", ""),
    }


class TextStrategyReader(ReaderOperator):
    name = "TextStrategyReader"
    accepted_technical_types = [TECH_TEXT]

    def read(self, signal: dict) -> dict:
        metadata = signal.get("metadata", {})
        structured = metadata.get("structured_strategy")
        text = signal.get("raw_ref")
        if text is None:
            text = metadata.get("text", "")
        text = str(text or "")

        if isinstance(structured, dict):
            semantic_summary = {
                "is_structured_strategy": True,
                "main_strategy": structured.get("main_strategy", ""),
                "stable_anchors": structured.get("stable_anchors", ""),
                "active_changes": structured.get("active_changes", ""),
                "motion_axis": structured.get("motion_axis", ""),
                "contact_rule": structured.get("contact_rule", ""),
                "reciprocal_reaction": structured.get("reciprocal_reaction", ""),
                "deformation_rule": structured.get("deformation_rule", ""),
                "material_rule": structured.get("material_rule", ""),
                "support_rule": structured.get("support_rule", ""),
                "endpoint": structured.get("endpoint", ""),
                "loop_rule": structured.get("loop_rule", ""),
                "forbidden_drift": structured.get("forbidden_drift", ""),
                "negative_additions": structured.get("negative_additions", ""),
                "priority_notes": structured.get("priority_notes", ""),
                "char_count": len(text),
                "word_count": len(text.split()),
                "line_count": len(text.splitlines()),
                "has_text": bool(text.strip()),
            }
            role_vector = {
                "is_text": True,
                "is_strategy_candidate": True,
                "is_structured_strategy": True,
                "has_main_strategy": _present(structured.get("main_strategy")),
                "has_anchors": _present(structured.get("stable_anchors")),
                "has_active_changes": _present(structured.get("active_changes")),
                "has_motion_axis": _present(structured.get("motion_axis")),
                "has_contact_rule": _present(structured.get("contact_rule")),
                "has_reciprocal_reaction": _present(structured.get("reciprocal_reaction")),
                "has_endpoint": _present(structured.get("endpoint")),
                "has_forbidden_drift": _present(structured.get("forbidden_drift")),
            }
            projection_metadata = {
                "priority_map": build_priority_map(structured),
                "route_hints": build_route_hints(structured),
                "strategy_expectations": build_strategy_expectations(structured),
                "strategy_strength": structured.get("strategy_strength", 1.0),
            }
            confidence = 1.0 if _present(structured.get("main_strategy")) else 0.6
            return make_event_projection(
                source_signal_id=signal["id"],
                operator_name=self.name,
                role_vector=role_vector,
                semantic_summary=semantic_summary,
                confidence=confidence,
                metadata=projection_metadata,
            )

        words = text.split()
        lines = text.splitlines()

        semantic_summary = {
            "is_structured_strategy": False,
            "char_count": len(text),
            "word_count": len(words),
            "line_count": len(lines),
            "has_text": bool(text.strip()),
            "anchor_terms": [],
            "active_terms": [],
            "contact_terms": [],
            "endpoint_terms": [],
            "negative_terms": [],
        }

        role_vector = {
            "is_text": True,
            "is_strategy_candidate": True,
            "has_text": bool(text.strip()),
        }

        return make_event_projection(
            source_signal_id=signal["id"],
            operator_name=self.name,
            role_vector=role_vector,
            semantic_summary=semantic_summary,
            confidence=1.0 if text.strip() else 0.2,
        )
