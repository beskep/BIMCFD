import sys
from pathlib import Path

_PARENT = Path(__file__).parent.resolve()

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
  # pyinstaller
  # FIXME: 안돼잖아?
  PRJ_DIR = _PARENT
  SRC_DIR = _PARENT.joinpath('src')
else:
  PRJ_DIR = _PARENT.parent
  SRC_DIR = _PARENT

RESOURCE_DIR = SRC_DIR.joinpath('resource')
TEMPLATE_DIR = SRC_DIR.joinpath('template')

_SRC_DIR = SRC_DIR.as_posix()
if SRC_DIR not in sys.path:
  import logging

  logger = logging.getLogger(__name__)
  logger.info('Source dir: %s', _SRC_DIR)
  sys.path.insert(0, _SRC_DIR)
