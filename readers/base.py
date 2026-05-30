from ..core.projection import make_event_projection


class ReaderOperator:
    name = "BaseReader"
    accepted_technical_types = []
    accepted_formula_roles = []

    def can_read(self, signal: dict) -> bool:
        tech_ok = (
            not self.accepted_technical_types
            or signal.get("technical_type") in self.accepted_technical_types
        )
        role_ok = (
            not self.accepted_formula_roles
            or signal.get("formula_role") in self.accepted_formula_roles
        )
        return tech_ok and role_ok

    def read(self, signal: dict) -> dict:
        return make_event_projection(
            source_signal_id=signal["id"],
            operator_name=self.name,
            confidence=0.0,
            metadata={"warning": "BaseReader does not read signal content"},
        )
