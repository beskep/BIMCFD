import sys
from pathlib import Path

_PARENT = Path(__file__).parent.resolve()

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
  # pyinstaller
  PRJ_DIR = _PARENT
  SRC_DIR = _PARENT.joinpath('src')
else:
  PRJ_DIR = _PARENT.parent
  SRC_DIR = _PARENT

RESOURCE_DIR = SRC_DIR.joinpath('resource')
TEMPLATE_DIR = SRC_DIR.joinpath('template')
