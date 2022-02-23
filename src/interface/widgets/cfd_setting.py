import os
from typing import Type, Union

from utils import DIR

from kivy.lang.builder import Builder
from kivy.metrics import dp

from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDFlatButton, MDRaisedButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.selectioncontrol import MDCheckbox

from interface.widgets.text_field import TextFieldUnit

ROUGHNESS_DB_PATH = DIR.RESOURCE.joinpath('misc/풍속 고도 분포 계수.xlsx')


class SpacingBox(MDBoxLayout):
  spacing = dp(20)


class CheckOnlyBox(MDCheckbox):

  def on_touch_down(self, touch):
    if self.state == 'normal':
      super().on_touch_down(touch)


class DialogContent(MDBoxLayout):
  __loaded = False
  _kv = ''
  option_ids = tuple()

  def __init__(self, *args, **kwargs):
    self.load_kv()
    super().__init__(*args, **kwargs)

  @classmethod
  def load_kv(cls):
    if not cls.__loaded:
      Builder.load_string(cls._kv)
      cls.__loaded = True

  def _option(self, key: str) -> Union[bool, float, str]:
    widget = getattr(self.ids, key)

    if key.startswith('flag'):
      option = (widget.state == 'down')
    elif key.startswith('text'):
      option = widget.get_main_text()

      try:
        option = float(option)
      except ValueError:
        pass
    else:
      raise ValueError('option id error')

    return option

  def options(self):
    return {x: self._option(x) for x in self.option_ids}


class CfdSettingContent(DialogContent):
  option_ids = (
      'flag_heat_flux',
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
      'text_num_of_subdomains',
      'text_vertical_dimension',
      'text_inner_buffer',
  )
  _kv = '''
<CfdSettingContent>
  orientation: 'vertical'
  height: dp(400)

  SpacingBox:
    orientation: 'vertical'
    size_hint_y: 0.3
    padding: dp(10)

    SpacingBox:
      size_hint_y: 0.6
      MDBoxLayout:
        MDCheckbox:
          active: False
          size_hint_x: None
          id: flag_heat_flux
        MDLabel:
          text: '열유속 해석'

      MDBoxLayout:
        MDCheckbox:
          active: False
          size_hint_x: None
          id: flag_friction
        MDLabel:
          text: '벽면 마찰 해석'

    SpacingBox:
      TextFieldUnit:
        hint_text: '외부 온도'
        text: '293.15'
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
          on_state: root.select_mesh_size_method()
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
        hint_text: '수직 방향 차원'
        text: '2'
        id: text_vertical_dimension
      TextFieldUnit:
        hint_text: '버퍼 크기 (격자 설계 보조 공간)'
        text: '0.2'
        unit: '배'
        id: text_inner_buffer

    SpacingBox:
      TextFieldUnit:
        hint_text: 'No. of Boundary Layers'
        text: '0'
        id: text_boundary_layers_count
      TextFieldUnit:
        hint_text: 'Boundary Cell Size'
        text: '0'
        unit: 'm'
        id: text_boundary_cell_size

    SpacingBox:
      TextFieldUnit:
        hint_text: 'No. of subdomains'
        text: ''
        id: text_num_of_subdomains
  '''

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.ids.text_mesh_size.main_text_disabled = True

  def select_mesh_size_method(self):
    flag_resolution = (self.ids.flag_mesh_resolution.state == 'down')
    resolution_field: TextFieldUnit = self.ids.text_mesh_resolution
    mesh_size_field: TextFieldUnit = self.ids.text_mesh_size

    resolution_field.main_text_disabled = not flag_resolution
    mesh_size_field.main_text_disabled = flag_resolution

  def options(self):
    opt = super().options()
    assert opt['text_mesh_resolution'] or opt['text_mesh_size']

    return opt


class ExternalSettingContent(DialogContent):
  option_ids = ('text_wind_profile_roughness',)
  _kv = '''
<ExternalSettingContent>
  orientation: 'vertical'
  height: dp(100)

  MDBoxLayout:
    orientation: 'vertical'
    padding: dp(10)

    TextFieldUnit:
      hint_text: '풍속 고도 분포 계수'
      text: '0.20'
      id: text_wind_profile_roughness

    MDBoxLayout:
      MDLabel:
        text: '지역별 풍속고도분포계수 DB'

      MDRaisedButton:
        text: '확인'
        on_release: root.open_roughness_db()
  '''

  @staticmethod
  def open_roughness_db():
    os.startfile(ROUGHNESS_DB_PATH)


class CfdSettingDialog(MDDialog):

  def __init__(self,
               title='CFD 세부 설정',
               content_cls: Type[DialogContent] = CfdSettingContent,
               **kwargs):
    cancel_button = MDFlatButton(text='취소')
    set_button = MDRaisedButton(text='설정')

    cancel_button.on_release = self.dismiss
    set_button.on_release = self._get_options_and_dismiss

    self.content_cls: DialogContent
    content = content_cls()
    self.content_height = content.height
    self._spacer_top = 0
    self._options = None

    kwargs['type'] = 'custom'
    kwargs['content_cls'] = content
    kwargs['buttons'] = [cancel_button, set_button]

    super().__init__(title=title, **kwargs)

  def _update_options(self):
    self._options = self.content_cls.options()

  @property
  def options(self):
    if self._options is None:
      self._update_options()

    return self._options.copy()

  def update_height(self, *args):
    """대부분은 버그입니다."""
    self.content_cls.height = self.content_height
    self._spacer_top = self.content_height + dp(24)

  def _get_options_and_dismiss(self):
    self._update_options()
    self.dismiss()

  def set_grid_resolution(self, resolution: float):
    widget: TextFieldUnit = self.content_cls.ids.text_mesh_resolution
    widget.text = str(resolution)
