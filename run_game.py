"""Launch VoiceSnake with its local environment, without shell activation."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    executable = ROOT / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not executable.is_file():
        print('Environment missing. First run: python scripts/bootstrap.py', file=sys.stderr)
        return 1
    return subprocess.call([str(executable), '-m', 'voicesnake', *sys.argv[1:]], cwd=ROOT)


if __name__ == '__main__':
    sys.exit(main())
