import logging
import logging.config
import sys
from pathlib import Path

import yaml
from rich.logging import RichHandler

TTA = True
# todo: 로거 설정 ThermalImage 참조해서 바꾸기

_rich_handler = RichHandler(level=logging.INFO, show_time=False)

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

config_path = SRC_DIR.joinpath('config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
  config = yaml.load(f, Loader=yaml.FullLoader)

if 'utils' not in sys.modules:
  msgs = []

  logging.config.dictConfig(config['logging'])
  logging.getLogger('BIMCFD').addHandler(_rich_handler)

  try:
    from kivy.logger import Logger as kvlogger

    kvlogger.handlers = [
        x for x in kvlogger.handlers if not isinstance(x, logging.StreamHandler)
    ]
    kvlogger.addHandler(_rich_handler)
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
