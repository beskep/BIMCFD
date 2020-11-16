from kivy.lang.builder import Builder
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDFlatButton, MDRaisedButton
from kivymd.uix.dialog import MDDialog

from interface.widgets.text_field import TextFieldNumeric, TextFieldUnit


class CfdSettingContent(MDBoxLayout):
  flag_kv_loaded = False
  kv = '''
<CfdSettingContent>
  orientation: 'vertical'
  spacing: dp(10)
  MDBoxLayout:
    MDBoxLayout:
      MDCheckbox:
        active: False
        size_hint_x: None
      MDLabel:
        text: 'Energy 해석'
    MDBoxLayout:
      MDCheckbox:
        active: False
        size_hint_x: None
      MDLabel:
        text: 'Friction 해석'
  
  MDBoxLayout:
    MDCheckbox:
      active: True
      size_hint_x: None
    MDLabel:
      text: '격자 해상도'
    MDCheckbox:
      active: False
      size_hint_x: None
    MDLabel:
      text: '격자 크기'
  
  MDBoxLayout:
    spacing: dp(10)
    TextFieldNumeric:
      hint_text: '격자 해상도'
    TextFieldUnit:
      hint_text: '격자 크기'
      unit: 'm'
  '''

  def __init__(self, *args, **kwargs):
    self.load_kv()
    super(CfdSettingContent, self).__init__(*args, **kwargs)

  @classmethod
  def load_kv(cls):
    if not cls.flag_kv_loaded:
      Builder.load_string(cls.kv)
      cls.flag_kv_loaded = True


class CfdSettingDialog(MDDialog):

  def __init__(self, title='CFD 설정', **kwargs):
    kwargs['type'] = 'custom'
    kwargs['content_cls'] = CfdSettingContent()
    kwargs['buttons'] = [
        MDFlatButton(text='취소'),
        MDRaisedButton(text='설정'),
    ]

    super(CfdSettingDialog, self).__init__(title=title, **kwargs)
