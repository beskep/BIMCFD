import logging
import logging.config
import sys
from pathlib import Path

import yaml
from rich.logging import RichHandler

TTA = True

_righ_handler = RichHandler(level=logging.INFO, show_time=False)

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
  # pyinstaller
  PRJ_DIR = Path(getattr(sys, '_MEIPASS'))
  SRC_DIR = PRJ_DIR.joinpath('src')
else:
  SRC_DIR = Path(__file__).parent.resolve()
  PRJ_DIR = SRC_DIR.parent

RESOURCE_DIR = SRC_DIR.joinpath('resource')
TEMPLATE_DIR = SRC_DIR.joinpath('template')

_SRC_DIR = SRC_DIR.as_posix()
if _SRC_DIR not in sys.path:
  sys.path.insert(0, _SRC_DIR)

if 'utils' not in sys.modules:
  msgs = []

  config_path = RESOURCE_DIR.joinpath('logging.yaml')
  if not config_path.exists():
    msgs.append('{} not found'.format(config_path))
  else:
    with open(config_path, 'r', encoding='utf-8') as f:
      config = yaml.load(f, Loader=yaml.FullLoader)

    logging.config.dictConfig(config)
    logging.getLogger('BIMCFD').addHandler(_righ_handler)

  try:
    from kivy.logger import Logger as kvlogger

    kvlogger.handlers = [
        x for x in kvlogger.handlers if not isinstance(x, logging.StreamHandler)
    ]
    kvlogger.addHandler(_righ_handler)
  except ModuleNotFoundError as e:
    msgs.append(str(e))

else:
  msgs = []

logger = logging.getLogger('BIMCFD')

if 'utils' not in sys.modules:
  logger.info('project dir: %s', PRJ_DIR)
  logger.info('src dir: %s', SRC_DIR)

  for msg in msgs:
    logger.error(msg)
