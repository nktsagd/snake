"""Evaluate the shipped ONNX model at multiple confidence thresholds.

Run from the repository root: .venv/bin/python training/evaluate_thresholds.py
Uses the original speaker-disjoint manifest and local Speech Commands v0.02.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import onnxruntime as ort
from scipy.io import wavfile
from compare import OUT, audio

MODEL = Path('public/voicesnake.onnx')
THRESHOLDS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.919165, 0.95, 0.98]
LABELS = ['up', 'right', 'left', 'down', 'unknown', 'background']
options = ort.SessionOptions(); options.intra_op_num_threads = 4
session = ort.InferenceSession(str(MODEL), options, providers=['CPUExecutionProvider'])
manifest = json.loads((OUT/'manifest.json').read_text())
assert manifest.get('complete'), 'Feature manifest is not marked complete'
rows = manifest['rows']
noise = {}


def read(r):
    if r['kind'] != 'background': return audio(r['path'])
    if r['path'] not in noise:
        rate, data = wavfile.read(r['path']); assert rate == 16000
        noise[r['path']] = data.astype(np.float32) / 32768
    return noise[r['path']][r['offset']:r['offset']+16000]


def probs(waves):
    logits = session.run(['logits'], {'waveform': np.asarray(waves, dtype=np.float32)})[0]
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def clip_scores(split):
    selected = [r for r in rows if r['split'] == split]
    outputs = []
    for i in range(0, len(selected), 32):
        outputs.append(probs(np.stack([read(r) for r in selected[i:i+32]])))
        if i % 1024 == 0: print(f'{split}: {i}/{len(selected)}', flush=True)
    p = np.concatenate(outputs); pred = p.argmax(1); conf = p.max(1)
    labels = np.array([r['label'] for r in selected]); kinds = np.array([r['kind'] for r in selected]); words = np.array([r['word'] for r in selected])
    results = []
    for t in THRESHOLDS:
        accepted = (pred < 4) & (conf >= t)
        entry = {'threshold': t, 'target_correct': int(((pred == labels) & accepted & (labels < 4)).sum()),
                 'target_wrong': int(((pred != labels) & accepted & (labels < 4)).sum()),
                 'target_total': int((labels < 4).sum())}
        for k in ['seen_unknown', 'unseen_unknown', 'background']:
            mask = kinds == k
            entry[k] = {'accepted': int((accepted & mask).sum()), 'total': int(mask.sum())}
        entry['per_direction'] = {word: {'correct':int(((pred==labels) & accepted & (words==word)).sum()), 'total':int((words==word).sum())} for word in LABELS[:4]}
        entry['unseen_by_word'] = {word: {'accepted':int((accepted & (words==word)).sum()), 'total':int((words==word).sum())} for word in sorted(set(words[kinds == 'unseen_unknown']))}
        results.append(entry)
    return results


class Detector:
    def __init__(self, threshold):
        self.threshold=threshold; self.candidate=''; self.count=0; self.last=''; self.quiet=None; self.emitted=-float('inf')
    def update(self, label, confidence, active, now):
        if not active:
            self.candidate=''; self.count=0
            if self.quiet is None: self.quiet=now
            if now-self.quiet >= 300: self.last=''
            return None
        self.quiet=None
        if label==self.last or now-self.emitted<350 or label not in LABELS[:4] or confidence<self.threshold:
            self.candidate=''; self.count=0; return None
        self.count = self.count+1 if self.candidate==label else 1; self.candidate=label
        if self.count<2: return None
        self.last=label; self.emitted=now; self.count=0; return label


def streaming(split):
    # Fixed, representative-sized sample; use the same clips at every threshold.
    # Each isolated clip is surrounded by silence. This is not real continuous audio.
    selected=[]
    for word in LABELS[:4]: selected.extend([r for r in rows if r['split']==split and r['word']==word][:25])
    for kind in ['seen_unknown','unseen_unknown','background']:
        candidates = [r for r in rows if r['split']==split and r['kind']==kind]
        rng=np.random.default_rng(42)
        selected.extend([candidates[i] for i in rng.choice(len(candidates), min(50, len(candidates)), replace=False)])
    counts={str(t):Counter() for t in THRESHOLDS}
    for index,r in enumerate(selected):
        wave=np.zeros(40000,dtype=np.float32); wave[16000:32000]=read(r)
        windows=[]; events=[]; last_voice=-float('inf')
        for end in range(1600,len(wave)+1,1600):
            chunk=wave[end-1600:end]; now=end/16
            if np.sqrt(np.mean(chunk**2)) >= .004: last_voice=now
            active=now-last_voice<240
            if active and end>=16000:
                events.append((now, True, len(windows))); windows.append(wave[end-16000:end])
            elif not active: events.append((now, False, None))
        p=probs(np.stack(windows)) if windows else np.empty((0,6))
        for threshold in THRESHOLDS:
            detector=Detector(threshold); emitted=[]
            for now,active,idx in events:
                if active:
                    best=int(p[idx].argmax()); command=detector.update(LABELS[best],float(p[idx,best]),True,now)
                else: command=detector.update('background',0,False,now)
                if command: emitted.append(command)
            c=counts[str(threshold)]; c[r['kind']+'_total']+=1
            if r['kind']=='target':
                c['target_correct']+=int(r['word'] in emitted)
                c['target_wrong']+=int(any(x!=r['word'] for x in emitted))
            else: c[r['kind']+'_false']+=int(bool(emitted))
        if index%50==0: print(f'stream {split}: {index}/{len(selected)}',flush=True)
    return {k:dict(v) for k,v in counts.items()}


def main():
    result={'model_sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(), 'method':'ONNX Runtime CPU, full isolated clips; streaming simulation uses 100ms hops, RMS .004, 2 confirmations and 350ms cooldown',
            'clips':{}, 'streaming':{}}
    for split in ['val','test']:
        result['clips'][split]=clip_scores(split)
        result['streaming'][split]=streaming(split)
    path=OUT/'threshold_sweep.json'; path.write_text(json.dumps(result,indent=2)+'\n')
    print(f'Wrote {path}',flush=True)


if __name__ == '__main__': main()
