import logging
import logging.config
import sys
from pathlib import Path
from typing import Union

import yaml
from loguru import logger
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


# def get_logger() -> logging.Logger:
#   logger = logging.getLogger(LOGGER_NAME)

#   return logger

# if not logging.getLogger().handlers:
#   config = get_config()
#   logging.config.dictConfig(config=config['logging'])

#   rich_handler = RichHandler(level=logging.INFO, show_time=False)
#   logger = get_logger()
#   logger.addHandler(rich_handler)

#   try:
#     from kivy.logger import Logger as kvlogger

#     kvlogger.handlers = [
#         handler for handler in kvlogger.handlers
#         if not isinstance(handler, logging.StreamHandler)
#     ]
#     kvlogger.addHandler(rich_handler)
#   except ImportError:
#     logger.error('kivy logger setting error', exc_info=True)

#   logger.info('Initialized utils')
#   logger.debug('prj dir: %s', PRJ_DIR)
#   logger.debug('src dir: %s', SRC_DIR)


def set_logger(level: Union[int, str] = 20, handle_kivy_logger=False):
  if isinstance(level, str):
    levels = {
        'TRACE': 5,
        'DEBUG': 10,
        'INFO': 20,
        'SUCCESS': 25,
        'WARNING': 30,
        'ERROR': 40,
        'CRITICAL': 50
    }
    try:
      level = levels[level.upper()]
    except KeyError as e:
      raise KeyError('`{}` not in {}'.format(level, set(levels.keys()))) from e

  rich_handler = RichHandler(log_time_format='[%X]')

  if getattr(logger, 'lvl', -1) != level:
    logger.remove()

    logger.add(rich_handler,
               level=level,
               format='{message}',
               backtrace=False,
               enqueue=True)
    logger.add('BIMCFD.log',
               level='DEBUG',
               rotation='1 week',
               retention='1 month',
               encoding='UTF-8-SIG',
               enqueue=True)

    setattr(logger, 'lvl', level)

    if not handle_kivy_logger:
      return

    try:
      from kivy.logger import Logger as kvlogger

      kvlogger.handlers = [
          handler for handler in kvlogger.handlers
          if not isinstance(handler, logging.StreamHandler)
      ]
      kvlogger.addHandler(rich_handler)
    except ImportError:
      logger.error('kivy logger setting error')
