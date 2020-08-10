import os

from kivy.app import App
from kivy.lang.builder import Builder


def load_kv(path, encoding='utf-8'):
  if not os.path.exists(path):
    raise FileNotFoundError(path)

  with open(path, 'r', encoding=encoding) as f:
    kv = Builder.load_string(f.read())

  return kv


class UtfApp(App):

  def run(self, file_path=None):
    if file_path is not None:
      kv = load_kv(file_path)

      if kv is not None:
        self.built = True
        self.root = kv

    super(__class__, self).run()
