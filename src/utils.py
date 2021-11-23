import logging
import sys
from pathlib import Path
from typing import Union

import yaml
from loguru import logger
from rich.logging import RichHandler

IS_FROZEN = getattr(sys, 'frozen', False)


class DIR:
  if IS_FROZEN:
    if hasattr(sys, '_MEIPASS'):
      PRJ = Path(getattr(sys, '_MEIPASS'))
    else:
      PRJ = Path(sys.executable).parent
  else:
    PRJ = Path(__file__).parents[1]

  SRC = PRJ.joinpath('src')
  RESOURCE = PRJ.joinpath('resource')
  TEMPLATE = RESOURCE.joinpath('template')


_SRC_DIR = DIR.SRC.as_posix()
if _SRC_DIR not in sys.path:
  sys.path.insert(0, _SRC_DIR)


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
