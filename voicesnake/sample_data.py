"""Download official TensorFlow mini Speech Commands and extract selected WAVs."""
import argparse
from pathlib import Path
import tempfile
import urllib.request
import zipfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('data/samples'))
    p.add_argument('--limit',type=int,default=5)
    args=p.parse_args()
    if args.limit<1: p.error('limit must be positive')
    url='https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip'
    with tempfile.TemporaryDirectory() as tmp:
        archive=Path(tmp)/'samples.zip'
        print('Downloading mini Speech Commands (~174 MiB)...')
        urllib.request.urlretrieve(url,archive)
        with zipfile.ZipFile(archive) as z:
            for label in ['up','down','left','right','yes','no','stop','go']:
                names=sorted(n for n in z.namelist() if n.startswith(f'mini_speech_commands/{label}/') and n.endswith('.wav'))[:args.limit]
                folder=args.output/label; folder.mkdir(parents=True,exist_ok=True)
                for name in names:
                    (folder/Path(name).name).write_bytes(z.read(name))
            # Keep dataset attribution with downloaded samples.
            for name in z.namelist():
                if Path(name).name.lower() in ('readme.md','readme.txt','license') and not name.startswith('__MACOSX'):
                    (args.output/Path(name).name).write_bytes(z.read(name))
    print('Saved samples to',args.output)

if __name__=='__main__': main()
