import os

from kivy.app import App
from kivymd.app import MDApp
from interface.kvtools import load_kv


class UtfApp(App):
  """
  쓸모 없음
  app 이름과 같은 kv 파일 이름만 피하면 됨
  """

  def run(self, kv_path=None, encoding='UTF-8'):
    if kv_path is not None:
      kv = load_kv(path=kv_path, encoding=encoding)

      if kv is not None:
        self.built = True
        self.root = kv

    super(__class__, self).run()


class UtfMDApp(MDApp):
  """
  쓸모 없음
  app 이름과 같은 kv 파일 이름만 피하면 됨
  """

  def run(self, kv_path=None, encoding='UTF-8'):
    if kv_path is not None:
      kv = load_kv(path=kv_path, encoding=encoding)

      if kv is not None:
        self.built = True
        self.root = kv

    super(__class__, self).run()
