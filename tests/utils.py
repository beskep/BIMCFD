import sys
from pathlib import Path

_PARENT = Path(__file__).parent.resolve()

PRJ_DIR = _PARENT.parent
SRC_DIR = PRJ_DIR.joinpath('src')

RESOURCE_DIR = SRC_DIR.joinpath('resource')
TEMPLATE_DIR = SRC_DIR.joinpath('template')

_SRC_DIR = SRC_DIR.as_posix()
if SRC_DIR not in sys.path:
  import logging

  logger = logging.getLogger(__name__)
  logger.info('Source dir: %s', _SRC_DIR)
  sys.path.insert(0, _SRC_DIR)
