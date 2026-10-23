"""Four-direction GRU trained from scratch; frontend is embedded in TorchScript."""
from pathlib import Path
import json
import hashlib
import time
import numpy as np
import torch

ASSETS = Path(__file__).resolve().parent / 'assets'
CHECKPOINT = ASSETS / 'voicesnake_gru4.pt'
LABELS = ASSETS / 'labels.json'
CLASSES = ['up', 'right', 'left', 'down']


def verify_assets():
    manifest = json.loads((ASSETS / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        path = ASSETS / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Model asset missing or checksum mismatch: {name}. Reinstall the complete VoiceSnake package.')
    return manifest


class Recognizer:
    def __init__(self, checkpoint=CHECKPOINT, threads=2, labels=LABELS):
        if not Path(checkpoint).is_file() or not Path(labels).is_file():
            raise FileNotFoundError('Missing bundled model assets. Reinstall VoiceSnake including voicesnake/assets/.')
        if Path(checkpoint) == CHECKPOINT and Path(labels) == LABELS:
            verify_assets()
        metadata = json.loads(Path(labels).read_text(encoding='utf-8'))
        if metadata['classes'] != CLASSES or metadata['sample_rate'] != 16000 or metadata['samples'] != 16000:
            raise ValueError('Expected up/right/left/down model with mono 16 kHz, 16000-sample input')
        self.classes = metadata['classes']
        torch.set_num_threads(threads)
        self.network = torch.jit.load(str(checkpoint), map_location='cpu').eval()
        self.predict(np.zeros(16000, dtype=np.float32))

    @torch.inference_mode()
    def predict(self, samples):
        samples = np.asarray(samples, dtype=np.float32)
        if samples.shape != (16000,) or not np.isfinite(samples).all():
            raise ValueError('Expected one second of finite mono 16 kHz audio')
        x = torch.from_numpy(samples.copy()).unsqueeze(0)
        start = time.perf_counter()
        logits = self.network(x)
        if logits.shape != (1, 4):
            raise ValueError('Expected four logits from the direction model')
        probabilities = logits[0].softmax(dim=-1)
        ms = (time.perf_counter() - start) * 1000
        index = int(probabilities.argmax())
        return self.classes[index], float(probabilities[index]), ms
