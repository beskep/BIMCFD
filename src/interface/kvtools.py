import os

from kivy.config import Config
from kivy.core.text import LabelBase
from kivy.lang.builder import Builder


def config():
  Config.set(section='kivy', option='log_enable', value=0)
  Config.set(section='kivy', option='exit_on_escape', value=0)
  Config.set(section='graphics', option='multisamples', value=4)
  Config.set(section='input',
             option='mouse',
             value='mouse,multitouch_on_demand')


def load_kv(path, encoding='UTF-8'):
  if not os.path.exists(path):
    raise FileNotFoundError(path)

  with open(path, 'r', encoding=encoding) as f:
    kv = Builder.load_string(f.read(), filname=str(path))

  return kv


def register_font(name,
                  fn_regular,
                  fn_italic=None,
                  fn_bold=None,
                  fn_bolditalic=None):
  LabelBase.register(name=name,
                     fn_regular=fn_regular,
                     fn_italic=fn_italic,
                     fn_bold=fn_bold,
                     fn_bolditalic=fn_bolditalic)


def set_window_size(size: tuple):
  """kivy window의 크기 설정
  *설정 시 window가 생성됨*

  Parameters
  ----------
  size : tuple
      (width, height)
  """
  from kivy.core.window import Window

  Window.size = size

  try:
    # Window를 중앙에 정렬
    import ctypes

    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)

    if (size[0] <= screen_width) and (size[1] <= screen_height):
      Window.left = int((screen_width - size[0]) / 2.0)
      Window.top = int((screen_height - size[1]) / 2.0)

  except Exception:
    pass
