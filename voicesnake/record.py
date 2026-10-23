"""Record labeled local microphone clips for evaluation (no training)."""
import argparse
from pathlib import Path
import sounddevice as sd
import soundfile as sf


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('label',choices=['up','down','left','right','noise','silence','other'])
    p.add_argument('--seconds',type=float,default=2)
    p.add_argument('--device',type=int)
    args=p.parse_args()
    if args.seconds<=0: p.error('seconds must be positive')
    from .paths import data_home
    directory=data_home()/'recordings'/args.label
    directory.mkdir(parents=True,exist_ok=True)
    import time
    path=directory/f'{time.time_ns()}.wav'
    print(f'Recording {args.seconds}s: {args.label}. Speak once after this message.')
    audio=sd.rec(int(args.seconds*16000),samplerate=16000,channels=1,dtype='float32',device=args.device)
    sd.wait(); sf.write(path,audio,16000)
    print(path)

if __name__=='__main__': main()
