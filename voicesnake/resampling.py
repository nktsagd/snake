"""Continuous mono resampling for microphone rates, without optional dependencies."""
import numpy as np


class StreamingResampler:
    def __init__(self, source_rate, target_rate=16000):
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError('Sample rates must be positive')
        self.source_rate = float(source_rate)
        self.target_rate = float(target_rate)
        self.consumed = 0
        self.produced = 0
        self.previous = 0.
        if source_rate > target_rate:
            positions = np.arange(63) - 31
            cutoff = .45 * target_rate / source_rate
            taps = 2 * cutoff * np.sinc(2 * cutoff * positions) * np.hamming(63)
            self.taps = taps / taps.sum()
        else:
            self.taps = np.ones(1)
        self.tail = np.zeros(len(self.taps) - 1)

    def process(self, audio):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not len(audio):
            return audio.copy()
        if self.source_rate == self.target_rate:
            return audio.copy()
        combined = np.concatenate((self.tail, audio))
        filtered = np.convolve(combined, self.taps, mode='valid')
        if len(self.tail):
            self.tail = combined[-len(self.tail):].copy()
        end = self.consumed + len(audio) - 1
        # Integer output index avoids accumulating a fractional phase rounding error.
        limit = int(np.floor(end * self.target_rate / self.source_rate)) + 1
        indices = np.arange(self.produced, limit, dtype=np.float64)
        positions = indices * self.source_rate / self.target_rate - self.consumed
        extended = np.concatenate(([self.previous], filtered))
        output = np.interp(positions + 1, np.arange(len(extended)), extended).astype(np.float32)
        self.previous = float(filtered[-1])
        self.consumed += len(audio)
        self.produced = limit
        return output
