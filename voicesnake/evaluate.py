"""Evaluate labeled WAV folders or one WAV; replay the real-time decision gate."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import soundfile as sf
from .model import Recognizer
from .detector import Detector, DIRECTIONS


def replay(model, samples, threshold, energy, confirmations):
    detector=Detector(threshold,confirmations)
    # Initial quiet and trailing quiet make utterance boundaries explicit.
    stream=np.concatenate([np.zeros(16000,dtype=np.float32),samples,np.zeros(16000,dtype=np.float32)])
    commands=[]; inference=[]; last_voice=-float("inf")
    for end in range(16000,len(stream)+1,1280):
        window=stream[end-16000:end]
        now=end/16000
        if float(np.sqrt(np.mean(window[-1920:]**2)))>=energy: last_voice=now
        active=now-last_voice<.24
        label,confidence,ms=model.predict(window) if active else ('-',0.,0.)
        if active: inference.append(ms)
        command=detector.update(label,confidence,active,end/16000)
        if command: commands.append({'direction':command.direction,'confidence':confidence,'audio_time_s':end/16000-1})
    return commands,inference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('path',type=Path,help='16 kHz WAV or directory with label/*.wav')
    p.add_argument('--limit',type=int,default=5,help='Files per class')
    p.add_argument('--threshold',type=float,default=.55)
    p.add_argument('--energy',type=float,default=.004)
    p.add_argument('--confirmations',type=int,default=2)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.output is None:
        from .paths import data_home
        args.output=data_home()/"evaluation.json"
    if args.path.is_file(): files=[args.path]
    else:
        files=[]
        for folder in sorted(args.path.iterdir()):
            if folder.is_dir(): files.extend(sorted(folder.glob('*.wav'))[:args.limit])
    if not files: p.error('No WAV files found')
    model=Recognizer(); rows=[]; timings=[]
    for path in files:
        samples,sr=sf.read(path,dtype='float32')
        if sr!=16000 or samples.ndim!=1 or not np.isfinite(samples).all():
            raise ValueError(f'{path}: expected finite mono 16 kHz WAV')
        window=np.pad(samples[:16000],(0,max(0,16000-len(samples))))
        label,confidence,ms=model.predict(window)
        events,latencies=replay(model,samples,args.threshold,args.energy,args.confirmations)
        timings.extend(latencies)
        expected=path.parent.name
        target=[expected] if expected in DIRECTIONS else []
        rows.append(dict(file=str(path),expected=expected,predicted=label,confidence=confidence,
                         inference_ms=ms,commands=events,exact_commands=[e['direction'] for e in events]==target))
    result=dict(parameters=dict(threshold=args.threshold, energy=args.energy, confirmations=args.confirmations, window_s=1, hop_s=.08, energy_hangover_s=.24), files=len(rows),top1_correct=sum(r['predicted']==r['expected'] for r in rows),
                exact_stream_commands=sum(r['exact_commands'] for r in rows),
                inference_p50_ms=float(np.median(timings)) if timings else None,
                inference_p95_ms=float(np.percentile(timings,95)) if timings else None,
                note='Selected prerecorded clips; not a held-out benchmark or live-microphone accuracy. Replay timing excludes compute and device latency.',results=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='results'},indent=2))

if __name__=='__main__': main()
