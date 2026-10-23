"""Build a portable source/model archive with a member checksum manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def collect_files():
    paths = set()
    for name in ('voicesnake', 'tests', 'docs'):
        folder = ROOT / name
        if folder.is_dir():
            paths.update(p for p in folder.rglob('*') if p.is_file() and not p.is_symlink()
                         and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo')
                         and not any(part.startswith('.') for part in p.relative_to(ROOT).parts))
    for pattern in ('README*', 'requirements*.txt', '*.ipynb'):
        paths.update(p for p in ROOT.glob(pattern) if p.is_file() and not p.is_symlink())
    for name in ('pyproject.toml', 'constraints-runtime.txt', 'run_game.py', 'scripts/bootstrap.py',
                 'scripts/package_release.py', 'MODEL_PROVENANCE.md', 'LICENSE'):
        path = ROOT / name
        if path.is_file() and not path.is_symlink():
            paths.add(path)
    for name in ('best_state.pt', 'metrics.json', 'history.csv', 'split_manifest.csv',
                 'labels.json', 'learning_curves.png', 'confusion_matrix.png'):
        path = ROOT / 'training_outputs' / 'voicesnake_gru4' / name
        if path.is_file() and not path.is_symlink():
            paths.add(path)
    return sorted(paths, key=lambda p: p.relative_to(ROOT).as_posix())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'release' / 'VoiceSnake-desktop.zip')
    args = parser.parse_args()
    files = collect_files()
    if not files:
        parser.error('No release files found')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        def add(name, data):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
        for path in files:
            name = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            hashes[name] = hashlib.sha256(data).hexdigest()
            add(name, data)
        add('SHA256SUMS.json', (json.dumps(hashes, indent=2, sort_keys=True) + '\n').encode())
    print(f'Created {args.output.resolve()} ({len(files)} files)')


if __name__ == '__main__':
    main()
