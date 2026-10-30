"""Train four keyword spotters on the same Speech Commands v2 features.

Run from the repository root: .venv/bin/python training/compare.py
Outputs are isolated under training_outputs/model_comparison.
"""
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import wavfile
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path('archive')
OUT = Path('training_outputs/model_comparison')
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
LABELS = ['up', 'right', 'left', 'down', 'unknown', 'background']
HELD_VAL = {'cat', 'dog'}
HELD_TEST = {'bed', 'bird'}
NOISE = {'train': ['doing_the_dishes', 'exercise_bike', 'pink_noise', 'white_noise'],
         'val': ['running_tap'], 'test': ['dude_miaowing']}
torch.set_num_threads(4)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def stable_split(name):
    speaker = name.split('_nohash_')[0]
    bucket = int(hashlib.sha1(speaker.encode()).hexdigest(), 16) % 100
    return 'train' if bucket < 70 else 'val' if bucket < 85 else 'test'


class Frontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('window', torch.hann_window(400))

    def forward(self, x):
        p = torch.stft(x, 400, 160, 400, self.window, center=True, return_complex=True).abs().square()[:, :200]
        p = p.reshape(len(x), 100, 2, -1).mean(dim=2).clamp_min(1e-6).log()
        return ((p-p.mean((1, 2), keepdim=True))/p.std((1, 2), keepdim=True).clamp_min(1e-5)).transpose(1, 2)


class GRU(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.gru = nn.GRU(100, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(hidden*4, 64), nn.ReLU(), nn.Dropout(.25), nn.Linear(64, 6))

    def forward(self, x):
        x, _ = self.gru(x)
        return self.head(torch.cat((x.mean(1), x.max(1).values), 1))


class Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                                 nn.BatchNorm2d(channels), nn.ReLU(),
                                 nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels))
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class DSCNN(nn.Module):
    def __init__(self, width, blocks):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, width, 5, stride=4, padding=2, bias=False),
                                 nn.BatchNorm2d(width), nn.ReLU(),
                                 nn.MaxPool2d(2), *(Block(width) for _ in range(blocks)),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(.25), nn.Linear(width, 6))

    def forward(self, x):
        return self.net(x.unsqueeze(1))


def audio(path):
    sr, x = wavfile.read(path)
    if sr != 16000 or x.ndim != 1:
        raise ValueError(f'Expected mono 16kHz: {path} ({sr}, {x.shape})')
    x = x.astype(np.float32) / 32768 if x.dtype == np.int16 else x.astype(np.float32)
    return np.pad(x[:16000], (0, max(0, 16000-len(x))))


def make_manifest():
    rng = random.Random(SEED)
    rows = []
    for folder in sorted(ROOT.iterdir()):
        if not folder.is_dir() or folder.name.startswith('_'):
            continue
        word = folder.name
        target = LABELS.index(word) if word in LABELS[:4] else 4
        groups = {'train': [], 'val': [], 'test': []}
        for path in sorted(folder.glob('*.wav')):
            split = stable_split(path.name)
            if word in HELD_VAL and split != 'val' or word in HELD_TEST and split != 'test':
                continue
            groups[split].append(path)
        for split, paths in groups.items():
            rng.shuffle(paths)
            # Cap unknown per word, so the negative class does not dominate storage.
            if target == 4:
                paths = paths[:600 if split == 'train' else 160]
            for path in paths:
                kind = 'target' if target < 4 else 'unseen_unknown' if word in HELD_VAL | HELD_TEST else 'seen_unknown'
                rows.append(dict(path=str(path), split=split, word=word, label=target, kind=kind, offset=0))
    for split, names in NOISE.items():
        for name in names:
            path = ROOT / '_background_noise_' / (name+'.wav')
            sr, wav = wavfile.read(path, mmap=True)
            assert sr == 16000 and wav.ndim == 1
            offsets = list(range(0, len(wav)-16000+1, 16000))
            for offset in offsets:
                rows.append(dict(path=str(path), split=split, word=name, label=5, kind='background', offset=offset))
    return rows


def load_features(rows):
    cache = OUT/'features.npy'
    manifest = OUT/'manifest.json'
    serialized = json.dumps(rows, sort_keys=True)
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    if cache.exists() and manifest.exists() and json.loads(manifest.read_text()).get('complete') and json.loads(manifest.read_text())['digest'] == digest:
        return np.load(cache, mmap_mode='r')
    manifest.write_text(json.dumps({'digest': digest, 'rows': rows}, indent=2))
    features = np.lib.format.open_memmap(cache, mode='w+', dtype='float32', shape=(len(rows), 101, 100))
    frontend = Frontend().eval()
    noise_cache = {}
    for start in range(0, len(rows), 128):
        batch = []
        for r in rows[start:start+128]:
            if r['kind'] == 'background':
                if r['path'] not in noise_cache:
                    _, y = wavfile.read(r['path'])
                    noise_cache[r['path']] = y.astype(np.float32)/32768
                x = noise_cache[r['path']][r['offset']:r['offset']+16000]
            else:
                x = audio(r['path'])
            batch.append(x)
        with torch.inference_mode():
            features[start:start+len(batch)] = frontend(torch.from_numpy(np.stack(batch))).numpy()
        if start % 5120 == 0:
            print(f'features {start}/{len(rows)}', flush=True)
    features.flush()
    manifest.write_text(json.dumps({'digest': digest, 'complete': True, 'rows': rows}, indent=2))
    return features


class Features(Dataset):
    def __init__(self, rows, features, indices, augment=False):
        self.rows, self.features, self.indices, self.augment = rows, features, indices, augment
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        index = self.indices[i]
        x = torch.from_numpy(np.array(self.features[index], copy=True))
        if self.augment:
            t, f = random.randrange(96), random.randrange(96)
            x[t:t+6] = 0; x[:, f:f+5] = 0
        return x, self.rows[index]['label']


def evaluate(model, loader, indices, rows):
    model.eval(); logits, labels = [], []
    with torch.inference_mode():
        for x, y in loader:
            logits.append(model(x)); labels.append(y)
    p = torch.cat(logits).softmax(-1).numpy(); y = torch.cat(labels).numpy()
    return p, y, [rows[i]['kind'] for i in indices]


def metrics(p, y, kinds, threshold):
    pred = p.argmax(1); command = np.argmax(p[:, :4], axis=1)
    accepted = (pred < 4) & (p[np.arange(len(p)), command] >= threshold)
    kinds = np.asarray(kinds)
    result = {'accuracy_6class': float(np.mean(pred == y)), 'threshold': float(threshold),
              'target_correct_accepted': float(np.mean((accepted & (command == y))[y < 4])),
              'target_accepted_wrong': float(np.mean((accepted & (command != y))[y < 4]))}
    for kind in ['seen_unknown', 'unseen_unknown', 'background']:
        mask = kinds == kind
        result[kind+'_false_accept'] = float(accepted[mask].mean()) if mask.any() else None
        result[kind+'_count'] = int(mask.sum())
    result['confusion_matrix'] = [[int(np.sum((y == i) & (pred == j))) for j in range(6)] for i in range(6)]
    return result


def main():
    rows = make_manifest()
    print('Dataset:', len(rows), dict(Counter((r['split'], r['kind']) for r in rows)), flush=True)
    speakers = {s: {Path(r['path']).name.split('_nohash_')[0] for r in rows if r['split'] == s and r['kind'] != 'background'} for s in ['train', 'val', 'test']}
    assert all(not speakers[a] & speakers[b] for a, b in [('train','val'), ('train','test'), ('val','test')])
    X = load_features(rows)
    indices = {s: [i for i,r in enumerate(rows) if r['split'] == s] for s in ['train','val','test']}
    datasets = {s: Features(rows, X, indices[s], s == 'train') for s in indices}
    counts = Counter(rows[i]['label'] for i in indices['train'])
    weights = torch.tensor([1/counts[rows[i]['label']] for i in indices['train']], dtype=torch.double)
    loaders = {s: DataLoader(datasets[s], batch_size=64, num_workers=0,
                             sampler=WeightedRandomSampler(weights, 12000, replacement=True) if s == 'train' else None)
               for s in indices}
    factories = {'gru64': lambda: GRU(64), 'gru128': lambda: GRU(128),
                 'dscnn_small': lambda: DSCNN(32, 3), 'dscnn_medium': lambda: DSCNN(64, 4)}
    result_path = OUT/'results.json'
    results = json.loads(result_path.read_text()) if result_path.exists() else {}
    for name, factory in factories.items():
        if name in results:
            print(f'Skipping completed {name}', flush=True)
            continue
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        model = factory(); optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=.5)
        criterion = nn.CrossEntropyLoss(); best = float('inf'); stale = 0; history = []
        print(f'\n{name}: {sum(p.numel() for p in model.parameters())} parameters', flush=True)
        for epoch in range(1, 21):
            start = time.perf_counter(); model.train(); total = 0
            for x, y in loaders['train']:
                optimizer.zero_grad(); loss = criterion(model(x), y); loss.backward(); optimizer.step()
                total += loss.item()*len(y)
            model.eval(); val_loss = 0
            with torch.inference_mode():
                for x, y in loaders['val']:
                    val_loss += criterion(model(x), y).item()*len(y)
            val_loss /= len(datasets['val']); scheduler.step(val_loss)
            history.append({'epoch':epoch,'train_loss':total/12000,'val_loss':val_loss,'seconds':time.perf_counter()-start})
            print(f"{name} epoch {epoch}: train={total/12000:.4f} val={val_loss:.4f} time={history[-1]['seconds']:.1f}s", flush=True)
            if val_loss < best - .0001:
                best=val_loss; stale=0
                torch.save({'state_dict':model.state_dict(), 'epoch':epoch, 'model':name}, OUT/f'{name}.pt')
            else:
                stale += 1
                if stale >= 5: break
        checkpoint = torch.load(OUT/f'{name}.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        vp, vy, vk = evaluate(model, loaders['val'], indices['val'], rows)
        # Choose minimum threshold meeting <=5% false accepts on validation unseen words.
        mask = np.array(vk) == 'unseen_unknown'; top = vp.argmax(1); scores = vp[:, :4].max(1)
        options = np.r_[0., np.unique(scores[mask & (top < 4)]), 1.000001]
        threshold = next(float(t) for t in options if np.mean((top[mask] < 4) & (scores[mask] >= t)) <= .05)
        tp, ty, tk = evaluate(model, loaders['test'], indices['test'], rows)
        model.eval(); sample = torch.from_numpy(np.array(X[indices['test'][0]:indices['test'][0]+1], copy=True))
        with torch.inference_mode():
            for _ in range(20): model(sample)
            times=[]
            for _ in range(200):
                start=time.perf_counter(); model(sample); times.append((time.perf_counter()-start)*1000)
        results[name] = {'parameters':sum(p.numel() for p in model.parameters()), 'best_epoch':checkpoint['epoch'],
                         'validation_loss':best, 'validation':metrics(vp,vy,vk,threshold), 'test':metrics(tp,ty,tk,threshold),
                         'cpu_feature_inference_ms_p50':float(np.percentile(times,50)),
                         'cpu_feature_inference_ms_p95':float(np.percentile(times,95)), 'history':history}
        (OUT/'results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__': main()
