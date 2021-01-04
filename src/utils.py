import logging
import logging.config
import sys
from pathlib import Path

import yaml
from rich.logging import RichHandler

LOGGER_NAME = 'BIMCFD'

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


def get_config():
  config_path = SRC_DIR.joinpath('config.yaml')
  with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

  return config


def get_logger() -> logging.Logger:
  logger = logging.getLogger(LOGGER_NAME)

  return logger


if not logging.getLogger().handlers:
  config = get_config()
  logging.config.dictConfig(config=config['logging'])

  rich_handler = RichHandler(level=logging.INFO, show_time=False)
  logger = get_logger()
  logger.addHandler(rich_handler)

  try:
    from kivy.logger import Logger as kvlogger

    kvlogger.handlers = [
        handler for handler in kvlogger.handlers
        if not isinstance(handler, logging.StreamHandler)
    ]
    kvlogger.addHandler(rich_handler)
  except ImportError:
    logger.error('kivy logger setting error', exc_info=True)

  logger.info('Initialized utils')
  logger.debug('prj dir: %s', PRJ_DIR)
  logger.debug('src dir: %s', SRC_DIR)
