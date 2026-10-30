"""Export the selected DS-CNN, including its waveform frontend, and check parity."""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as F
from scipy.io import wavfile

from compare import DSCNN, Frontend, OUT, audio, make_manifest

HERE = Path(__file__).resolve().parent
MODEL = HERE / 'voicesnake_dscnn_medium.onnx'
LABELS = ['up', 'right', 'left', 'down', 'unknown', 'background']


class RealSpectrogram(nn.Module):
    """STFT expressed as real convolutions, avoiding complex ONNX tensors."""
    def __init__(self):
        super().__init__()
        n = torch.arange(400, dtype=torch.float64)
        k = torch.arange(200, dtype=torch.float64)[:, None]
        phase = 2 * torch.pi * k * n / 400
        window = torch.hann_window(400, dtype=torch.float64)
        self.register_buffer('real', (phase.cos() * window).float().unsqueeze(1))
        self.register_buffer('imag', (-phase.sin() * window).float().unsqueeze(1))

    def forward(self, x):
        x = F.pad(x.unsqueeze(1), (200, 200), mode='reflect')
        real = F.conv1d(x, self.real, stride=160)
        imag = F.conv1d(x, self.imag, stride=160)
        power = (real.square()+imag.square()).reshape(x.size(0), 100, 2, -1).mean(dim=2)
        features = power.clamp_min(1e-6).log()
        mean = features.mean(dim=(1, 2), keepdim=True)
        std = features.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
        return ((features-mean)/std).transpose(1, 2)


class WaveformModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.frontend = RealSpectrogram()
        self.backbone = backbone

    def forward(self, waveform):
        return self.backbone(self.frontend(waveform))


def main():
    torch.set_num_threads(4)
    checkpoint = torch.load(HERE / 'checkpoints/dscnn_medium.pt', map_location='cpu', weights_only=True)
    backbone = DSCNN(64, 4).eval()
    backbone.load_state_dict(checkpoint['state_dict'])
    model = WaveformModel(backbone).eval()
    original = Frontend().eval()
    dummy = torch.zeros(1, 16000)
    torch.onnx.export(model, dummy, str(MODEL), input_names=['waveform'], output_names=['logits'],
                      dynamic_axes={'waveform': {0: 'batch'}, 'logits': {0: 'batch'}},
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(str(MODEL), full_check=True)
    options = ort.SessionOptions(); options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(MODEL), sess_options=options, providers=['CPUExecutionProvider'])

    manifest = OUT/'manifest.json'
    rows = json.loads(manifest.read_text())['rows'] if manifest.exists() else make_manifest()
    rng = np.random.default_rng(42)
    selected = []
    for kind in ['target', 'seen_unknown', 'unseen_unknown', 'background']:
        pool = [r for r in rows if r['split'] == 'test' and r['kind'] == kind]
        selected += [pool[i] for i in rng.choice(len(pool), min(32, len(pool)), replace=False)]
    errors, feature_errors, disagreements, command_disagreements = [], [], 0, 0
    threshold = json.loads((HERE/'results.json').read_text())['dscnn_medium']['test']['threshold']
    for row in selected:
        if row['kind'] == 'background':
            sr, wav = wavfile.read(row['path'])
            assert sr == 16000
            x = wav[row['offset']:row['offset']+16000].astype(np.float32) / 32768
        else:
            x = audio(row['path'])
        tensor = torch.from_numpy(x).unsqueeze(0)
        with torch.inference_mode():
            ref_features = original(tensor)
            export_features = model.frontend(tensor)
            ref = backbone(ref_features).numpy()
        actual = session.run(['logits'], {'waveform': tensor.numpy()})[0]
        feature_errors.append(float((ref_features-export_features).abs().max()))
        errors.append(float(np.max(np.abs(ref-actual))))
        ref_i, out_i = int(ref.argmax()), int(actual.argmax())
        disagreements += ref_i != out_i
        def decision(logits):
            z = logits[0] - logits[0].max(); p = np.exp(z) / np.exp(z).sum()
            i = int(p.argmax())
            return i if i < 4 and p[i] >= threshold else -1
        command_disagreements += decision(ref) != decision(actual)
    assert max(errors) < .01 and disagreements == 0 and command_disagreements == 0, (max(errors), disagreements, command_disagreements)
    # Also check dynamic batch output and zero waveform are finite.
    zeros = session.run(None, {'waveform': np.zeros((2, 16000), dtype=np.float32)})[0]
    assert zeros.shape == (2, 6) and np.isfinite(zeros).all()
    report = {'model':MODEL.name, 'checkpoint_epoch':checkpoint['epoch'],
              'model_sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),
              'input':'float32 [batch, 16000], mono 16 kHz', 'output':'float32 [batch, 6] logits',
              'labels':LABELS, 'threshold_validation':threshold, 'opset':17,
              'test_clips':len(selected), 'per_kind':{k:sum(r['kind']==k for r in selected) for k in ['target','seen_unknown','unseen_unknown','background']},
              'frontend_max_abs_error':max(feature_errors), 'logit_max_abs_error':max(errors),
              'argmax_disagreements':int(disagreements), 'decision_disagreements':int(command_disagreements),
              'runtime':'ONNX Runtime CPU', 'onnx_version':onnx.__version__, 'onnxruntime_version':ort.__version__}
    (HERE/'onnx_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
