from ..core.projection import make_event_projection
from ..readers.text_reader import TextStrategyReader
from ..readers.latent_reader import LatentEventReader
from ..readers.image_reader import ImageOutcomeReader
from ..readers.noise_reader import NoisePossibilityReader
from ..readers.conditioning_reader import ConditioningStrategyReader
from ..readers.delta_reader import DeltaReader


class OperatorRegistry:
    def __init__(self):
        self.readers = []
        self.register(TextStrategyReader())
        self.register(LatentEventReader())
        self.register(ImageOutcomeReader())
        self.register(NoisePossibilityReader())
        self.register(ConditioningStrategyReader())
        self.register(DeltaReader())

    def register(self, reader):
        self.readers.append(reader)

    def select_reader(self, signal):
        for reader in self.readers:
            if reader.can_read(signal):
                return reader
        return None

    def read(self, signal):
        reader = self.select_reader(signal)
        if reader is None:
            return make_event_projection(
                source_signal_id=signal["id"],
                operator_name="UnknownReader",
                confidence=0.0,
                metadata={"warning": "NO_READER_FOUND"},
            )
        return reader.read(signal)
