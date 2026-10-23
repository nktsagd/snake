"""Runtime settings shared by the Pygame UI and audio worker."""
from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class VoiceSettings:
    threshold: float = .55
    energy: float = .004
    confirmations: int = 2


class Settings:
    def __init__(self, threshold=.55, energy=.004, confirmations=2):
        self._lock = threading.Lock()
        self._voice = VoiceSettings(threshold, energy, confirmations)

    def voice(self):
        with self._lock:
            return self._voice

    def update_voice(self, *, threshold=None, energy=None, confirmations=None):
        with self._lock:
            self._voice = VoiceSettings(
                threshold=self._voice.threshold if threshold is None else min(1., max(.1, threshold)),
                energy=self._voice.energy if energy is None else min(.03, max(.0005, energy)),
                confirmations=self._voice.confirmations if confirmations is None else min(4, max(1, confirmations)),
            )
            return self._voice
