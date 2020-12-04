from kivy.lang.builder import Builder
from kivy.metrics import dp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDFlatButton, MDRaisedButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.selectioncontrol import MDCheckbox

from interface.widgets.text_field import TextFieldUnit


class SpacingBox(MDBoxLayout):
  spacing = dp(20)


class CheckOnlyBox(MDCheckbox):

  def on_touch_down(self, touch):
    if self.state == 'normal':
      return super(CheckOnlyBox, self).on_touch_down(touch)


class CfdSettingContent(MDBoxLayout):
  flag_kv_loaded = False
  kv = '''
<CfdSettingContent>
  orientation: 'vertical'
  height: dp(360)
  # size_hint_y: 1

  SpacingBox:
    orientation: 'vertical'
    size_hint_y: 0.5
    padding: dp(10)

    SpacingBox:
      size_hint_y: 0.6
      MDBoxLayout:
        MDCheckbox:
          active: False
          size_hint_x: None
          id: flag_energy
        MDLabel:
          text: 'Energy 해석'

      MDBoxLayout:
        MDCheckbox:
          active: False
          size_hint_x: None
          id: flag_friction
        MDLabel:
          text: 'Friction 해석'
      
    SpacingBox:
      TextFieldUnit:
        hint_text: '외부 온도'
        text: '300'
        unit: 'K'
        id: text_external_temperature

      TextFieldUnit:
        hint_text: '외부 열전도율'
        unit: 'W/m²K'
        id: text_external_htc

  MDBoxLayout:
    orientation: 'vertical'
    padding: dp(10)

    SpacingBox:
      size_hint_y: 0.6
      MDBoxLayout:
        CheckOnlyBox:
          active: True
          group: 'mesh'
          size_hint_x: None
          id: flag_mesh_resolution
        MDLabel:
          text: '격자 해상도 설정'

      MDBoxLayout:
        CheckOnlyBox:
          active: False
          group: 'mesh'
          size_hint_x: None
          id: flag_mesh_size
        MDLabel:
          text: '격자 크기 설정'
    
    SpacingBox:
      TextFieldUnit:
        hint_text: '격자 해상도'
        text: '24'
        show_unit: False
        id: text_mesh_resolution
      TextFieldUnit:
        hint_text: '격자 크기'
        unit: 'm'
        id: text_mesh_size

    SpacingBox:
      TextFieldUnit:
        hint_text: '최소 격자 크기'
        text: '0'
        unit: 'm'
        id: text_mesh_min_size
      TextFieldUnit:
        hint_text: '외부 영역 크기'
        text: '5'
        unit: '배'
        id: text_external_size
    
    SpacingBox:
      TextFieldUnit:
        hint_text: 'Boundary Cell Size'
        text: '0'
        unit: 'm'
        id: text_boundary_cell_size
      TextFieldUnit:
        hint_text: 'No. of Boundary Layers'
        text: '0'
        id: text_boundary_layers_count
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
  _option_ids = (
      'flag_energy',
      'flag_friction',
      'text_external_temperature',
      'text_external_htc',
      'flag_mesh_resolution',
      'flag_mesh_size',
      'text_mesh_resolution',
      'text_mesh_size',
      'text_mesh_min_size',
      'text_external_size',
      'text_boundary_cell_size',
      'text_boundary_layers_count',
  )

  def __init__(self, title='CFD 세부 설정', **kwargs):
    cancel_button = MDFlatButton(text='취소')
    set_button = MDRaisedButton(text='설정')

    cancel_button.on_release = self.dismiss
    set_button.on_release = self._get_options_and_dismiss

    content = CfdSettingContent()
    self.content_height = content.height

    kwargs['type'] = 'custom'
    kwargs['content_cls'] = content
    kwargs['buttons'] = [cancel_button, set_button]

    super(CfdSettingDialog, self).__init__(title=title, **kwargs)

    self._options = None
    # todo: 모델 특성길이에 따라 해상도 변경

  @property
  def options(self):
    if self._options is None:
      self._get_options()

    return self._options

  def update_height(self, *args):
    """대부분은 버그입니다."""
    self.content_cls.height = self.content_height
    self._spacer_top = self.content_height + dp(24)

  @staticmethod
  def _get_option(key, ids):
    widget = getattr(ids, key)
    if key.startswith('flag'):
      option = (widget.state == 'down')
    elif key.startswith('text'):
      text = widget.main_text

      try:
        option = float(text)
      except ValueError:
        option = None
    else:
      raise ValueError

    return option

  def _get_options(self):
    ids = self.content_cls.ids
    self._options = {x: self._get_option(x, ids) for x in self._option_ids}

  def _get_options_and_dismiss(self):
    self._get_options()
    self.dismiss()
