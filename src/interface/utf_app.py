import os

from kivy.app import App
from kivy.lang.builder import Builder
from kivymd.app import MDApp


def load_kv(path, encoding='UTF-8'):
  if not os.path.exists(path):
    raise FileNotFoundError(path)

  with open(path, 'r', encoding=encoding) as f:
    kv = Builder.load_string(f.read())

  return kv


class UtfApp(App):

  def run(self, kv_path=None, encoding='UTF-8'):
    if kv_path is not None:
      kv = load_kv(path=kv_path, encoding=encoding)

      if kv is not None:
        self.built = True
        self.root = kv

    super(__class__, self).run()


class UtfMDApp(MDApp):

  def run(self, kv_path=None, encoding='UTF-8'):
    if kv_path is not None:
      kv = load_kv(path=kv_path, encoding=encoding)

      if kv is not None:
        self.built = True
        self.root = kv

    super(__class__, self).run()
