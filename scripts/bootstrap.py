"""Create a machine-local environment; never copy .venv between computers."""
import argparse
import os
import platform
from pathlib import Path
import struct
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true', help='Also install notebook training dependencies')
    args = parser.parse_args()
    if not (3, 9) <= sys.version_info[:2] < (3, 13) or struct.calcsize('P') != 8:
        parser.error('Use 64-bit Python 3.9–3.12 (Python 3.11 recommended).')
    architecture=platform.machine().lower()
    if sys.platform=='darwin' and architecture not in ('arm64','aarch64'):
        parser.error('Pinned PyTorch 2.8.0 macOS wheels require Apple Silicon. Intel macOS is not supported by this release.')
    environment = ROOT / '.venv'
    executable = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not executable.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    # Verify an existing environment before using it, without deleting user files.
    subprocess.run([str(executable), '-c', "import sys,struct; assert (3,9)<=sys.version_info[:2]<(3,13) and struct.calcsize('P')==8, 'Recreate .venv using 64-bit Python 3.9-3.12'"], check=True)
    command = [str(executable), '-m', 'pip', 'install', '-c', str(ROOT/'constraints-runtime.txt'), 'torch==2.8.0']
    if sys.platform != 'darwin':
        command += ['--index-url', 'https://download.pytorch.org/whl/cpu']
    subprocess.run(command, cwd=ROOT, check=True)
    subprocess.run([str(executable), '-m', 'pip', 'install', '-c', str(ROOT/'constraints-runtime.txt'), str(ROOT)], cwd=ROOT, check=True)
    if args.train:
        subprocess.run([str(executable), '-m', 'pip', 'install', '-r', str(ROOT / 'requirements-train.txt')], cwd=ROOT, check=True)
    print('Ready. Run: python run_game.py')


if __name__ == '__main__':
    main()
