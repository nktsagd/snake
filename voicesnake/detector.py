from dataclasses import dataclass
from typing import Optional

DIRECTIONS = {'up', 'down', 'left', 'right'}


@dataclass(frozen=True)
class Command:
    direction: str
    confidence: float
    detected_at: float
    onset_at: Optional[float]


class Detector:
    """Confirm consecutive full-vocabulary predictions; rearm on quiet audio."""
    def __init__(self, threshold=.55, confirmations=2, release=.3):
        self.threshold = threshold
        self.confirmations = confirmations
        self.release = release
        self.reset()

    def reset(self):
        self.candidate = None
        self.count = 0
        self.latched = False
        self.quiet_since = None
        self.last_label = None
        self.last_emitted_at = -float("inf")
        self.reason = "Ready"

    def update(self, label, confidence, active, now, onset=None):
        if not active:
            self.reason = "Quiet / waiting for voice"
            self.candidate, self.count = None, 0
            if self.quiet_since is None:
                self.quiet_since = now
            if now - self.quiet_since >= self.release:
                self.latched = False
            return None
        self.quiet_since = None
        if self.latched and label == self.last_label:
            self.candidate, self.count = None, 0
            self.reason = "Same command held / say a new direction"
            return None
        if now - self.last_emitted_at < .35:
            self.reason = "Short command cooldown"
            return None
        if label not in DIRECTIONS or confidence < self.threshold:
            self.candidate, self.count = None, 0
            self.reason = "Below confidence threshold"
            return None
        self.count = self.count + 1 if self.candidate == label else 1
        self.candidate = label
        self.reason = f"Confirming {label}: {self.count}/{self.confirmations}"
        if self.count < self.confirmations:
            return None
        self.latched = True
        self.last_label, self.last_emitted_at = label, now
        self.candidate, self.count = None, 0
        self.reason = "Sent " + label.upper()
        return Command(label, confidence, now, onset)
