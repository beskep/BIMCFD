import os.path

from kivy.properties import BooleanProperty, ObjectProperty, StringProperty
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.textfield import MDTextField


class TextFieldFont(MDTextField):

  def __init__(self, **kwargs):
    self.has_had_text = False
    self.error = None
    self.helper_text = None
    super().__init__(**kwargs)

  def on_font_name(self, instance, value):
    self._hint_lbl.font_name = value
    self._msg_lbl.font_name = value


class TextFieldPath(TextFieldFont):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.helper_text_mode = 'on_error'

  def on_text(self, instance, text):
    path = os.path.normpath(self.text)
    if not os.path.exists(path):
      self.error = True
      self.helper_text = '유효하지 않은 경로입니다'
    else:
      self.error = False

    super().on_text(instance, text)


class TextFieldNumeric(TextFieldFont):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.helper_text_mode = 'on_error'

  def on_text(self, instance, text: str):
    if text and not text.replace('.', '', 1).isdigit():
      self.error = True
      self.helper_text = '숫자를 입력해주세요'
    else:
      self.error = False

    super().on_text(instance, text)


class TextFieldUnit(MDBoxLayout):
  text = StringProperty('')
  hint_text = StringProperty('')
  unit = StringProperty('')

  main_text = ObjectProperty('')
  unit_text = ObjectProperty('')

  show_unit = BooleanProperty(True)
  main_text_disabled = BooleanProperty(False)

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self._main_text = TextFieldNumeric()
    self._main_text.size_hint_x = 0.75

    self._unit_text = TextFieldFont()
    self._unit_text.text = self.unit
    self._unit_text.disabled = True
    self._unit_text.halign = 'right'
    self._unit_text.size_hint_x = 0.25

    self.add_widget(self._main_text)
    self.add_widget(self._unit_text)

  def get_main_text(self):
    return self._main_text.text

  def on_text(self, instance, value):
    self._main_text.text = value

  def on_hint_text(self, instance, value):
    self._main_text.hint_text = value

  def on_show_unit(self, instance, value):
    if value:
      self._unit_text.text = self.unit
    else:
      self._unit_text.text = ''

  def on_unit(self, instance, value):
    if self.show_unit:
      self._unit_text.text = value

  def on_main_text(self, instance, option: dict):
    for key, value in option.items():
      setattr(self._main_text, key, value)

  def on_unit_text(self, instance, option: dict):
    for key, value in option.items():
      setattr(self._main_text, key, value)

  def on_main_text_disabled(self, instance, value):
    self._main_text.disabled = bool(value)
