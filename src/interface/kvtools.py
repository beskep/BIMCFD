import os

from kivy.lang.builder import Builder
from kivy.core.text import LabelBase


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
