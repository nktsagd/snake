import queue
import threading
import time
from dataclasses import dataclass
import numpy as np
from .detector import Detector
from .resampling import StreamingResampler


@dataclass(frozen=True)
class Status:
    microphone: str = 'Loading model...'
    label: str = '-'
    confidence: float = 0.
    inference_ms: float = 0.
    rms: float = 0.
    onset_ms: float = 0.
    decision: str = "Waiting"


class AudioWorker:
    """Callback resamples and buffers audio; a separate worker owns model inference."""
    def __init__(self, threshold=.55, confirmations=2, energy=.004, interval=.08, device=None, settings=None):
        self.commands = queue.Queue(maxsize=8)
        self.status = Status()
        self.settings = settings
        self.detector = Detector(threshold, confirmations)
        self.energy, self.interval, self.device = energy, interval, device
        self.stop_event = threading.Event()
        self.enabled = threading.Event()
        self.enabled.set()
        self.lock = threading.Lock()
        self.buffer = np.zeros(16000, dtype=np.float32)
        self.filled = 0
        self.latest_at = 0.
        self.audio_warning = ''
        self.thread = None
        self.generation = 0
        self.resampler = StreamingResampler(16000)

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True, name='voice-inference')
        self.thread.start()

    def set_enabled(self, enabled):
        with self.lock:
            if enabled != self.enabled.is_set():
                self.generation += 1
                self.enabled.set() if enabled else self.enabled.clear()

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)

    def callback(self, data, frames, timing, status):
        if frames <= 0:
            return
        samples = self.resampler.process(data[:, 0])
        if not len(samples):
            return
        with self.lock:
            n = min(len(samples), len(self.buffer))
            self.buffer[:-n] = self.buffer[n:]
            self.buffer[-n:] = samples[-n:]
            self.filled = min(16000, self.filled+n)
            self.latest_at = time.perf_counter()
            self.audio_warning = str(status) if status else ''

    def run(self):
        try:
            import sounddevice as sd
            from .model import Recognizer
            recognizer = Recognizer()
            latest_ms = 0.
            onset = None
            last_voice = -float("inf")
            previously_active = False
            last_generation = self.generation
            self.status = Status(microphone='Opening microphone...')
            rate = 16000
            try:
                sd.check_input_settings(device=self.device, channels=1, dtype='float32', samplerate=rate)
            except (sd.PortAudioError, ValueError):
                rate = float(sd.query_devices(self.device, kind='input')['default_samplerate'])
                sd.check_input_settings(device=self.device, channels=1, dtype='float32', samplerate=rate)
            self.resampler = StreamingResampler(rate)
            with sd.InputStream(samplerate=rate, channels=1, dtype='float32',
                                blocksize=round(rate * .02), device=self.device, callback=self.callback):
                self.status = Status(microphone='Buffering')
                while not self.stop_event.is_set():
                    start = time.perf_counter()
                    with self.lock:
                        x = self.buffer.copy()
                        generation = self.generation
                        filled, captured, warning = self.filled, self.latest_at, self.audio_warning
                    if filled < 16000 or start-captured > .25:
                        self.status = Status(microphone='Buffering' if filled < 16000 else 'Audio stalled')
                        self.stop_event.wait(self.interval)
                        continue
                    if generation != last_generation:
                        self.detector.reset()
                        last_generation = generation
                    rms = float(np.sqrt(np.mean(x[-1920:]**2)))
                    voice_settings = self.settings.voice() if self.settings else None
                    energy_threshold = voice_settings.energy if voice_settings else self.energy
                    if rms >= energy_threshold:
                        last_voice = captured
                    active = captured-last_voice < .24
                    if active and not previously_active:
                        onset = captured - .12
                    previously_active = active
                    label, confidence, ms = '-', 0., 0.
                    enabled = self.enabled.is_set()
                    if not enabled:
                        self.detector.reset()
                    elif active:
                        label, confidence, ms = recognizer.predict(x)
                        latest_ms = ms
                    now = time.perf_counter()
                    if generation != self.generation:
                        self.detector.reset()
                        continue
                    if self.settings:
                        voice_settings = self.settings.voice()
                        self.detector.threshold = voice_settings.threshold
                        self.detector.confirmations = voice_settings.confirmations
                    command = self.detector.update(label, confidence, active, now, onset) if enabled else None
                    if command and self.enabled.is_set():
                        try:
                            self.commands.put_nowait(command)
                        except queue.Full:
                            pass
                    self.status = Status('Audio warning: '+warning if warning else 'Listening',
                                         label, confidence, latest_ms, rms,
                                         (now-onset)*1000 if command and onset is not None else 0.,
                                         self.detector.reason if enabled else "Voice paused / disabled")
                    self.stop_event.wait(max(0., self.interval-(time.perf_counter()-start)))
        except Exception as exc:
            self.status = Status(microphone=f'Voice unavailable: {exc}')
