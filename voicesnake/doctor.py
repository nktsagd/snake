"""Portable dependency, asset, inference and input-device diagnostics."""
import importlib.metadata
import json
import platform
import sys
from .paths import data_home

VERSIONS = {'torch':'2.8.0','numpy':'1.26.4','pygame':'2.6.1','sounddevice':'0.5.6','soundfile':'0.13.1'}


def inspect_environment():
    checks=[]
    def add(name, status, detail): checks.append(dict(check=name,status=status,detail=str(detail)))
    add('Python', 'OK' if (3,9)<=sys.version_info[:2]<(3,13) else 'FAIL', sys.version.split()[0])
    add('Platform','OK',f'{platform.system()} {platform.machine()} ({platform.architecture()[0]})')
    for name, expected in VERSIONS.items():
        try:
            actual=importlib.metadata.version(name)
            add(name,'OK' if actual.split('+')[0]==expected else 'WARN', f'{actual}; pinned {expected}')
        except importlib.metadata.PackageNotFoundError: add(name,'FAIL','Not installed. Run scripts/bootstrap.py.')
    try:
        from .model import Recognizer,verify_assets
        manifest=verify_assets()
        import numpy as np
        r=Recognizer()
        _,_,ms=r.predict(np.zeros(16000,dtype=np.float32))
        add('Bundled GRU','OK',f"SHA-256 verified; labels={manifest['classes']}; CPU forward={ms:.2f} ms")
    except Exception as exc: add('Bundled GRU','FAIL',exc)
    try:
        import sounddevice as sd
        devices=sd.query_devices()
        inputs=[(i,d) for i,d in enumerate(devices) if d['max_input_channels']>0]
        if not inputs: add('Microphone','WARN','No input devices. Check OS microphone permission; keyboard mode is available.')
        else:
            add('Microphone','OK','; '.join(f"{i}: {d['name']} ({d['default_samplerate']:g} Hz)" for i,d in inputs))
            try:
                sd.query_devices(kind='input')
                add('Default input','OK','Use --device INDEX to select a different input')
            except Exception as exc: add('Default input','WARN',exc)
    except Exception as exc: add('PortAudio','WARN',f'{exc}. Linux may require libportaudio2; check audio service and permissions.')
    try:
        directory=data_home();directory.mkdir(parents=True,exist_ok=True)
        import tempfile
        with tempfile.TemporaryFile(dir=directory) as file: file.write(b'check')
        add('Writable output','OK',directory)
    except OSError as exc: add('Writable output','FAIL',f'{exc}; set VOICESNAKE_HOME to a writable directory')
    return {'checks':checks,'passed':not any(c['status']=='FAIL' for c in checks)}


def main():
    report=inspect_environment()
    for check in report['checks']:
        print(f"[{check['status']}] {check['check']}: {check['detail']}")
    if not report['passed']: raise SystemExit(1)

if __name__=='__main__': main()
