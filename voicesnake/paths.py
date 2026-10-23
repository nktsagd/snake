"""Writable session data is independent of the checkout and install directory."""
import os
from pathlib import Path
import sys


def data_home():
    override = os.environ.get('VOICESNAKE_HOME')
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == 'win32':
        return Path(os.environ.get('LOCALAPPDATA', str(Path.home() / 'AppData' / 'Local'))) / 'VoiceSnake'
    return Path.home() / '.voicesnake'
