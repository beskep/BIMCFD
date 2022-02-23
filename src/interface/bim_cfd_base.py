import threading
from pathlib import Path

from kivy.clock import mainthread
from kivy.metrics import dp
from kivy.uix.widget import Widget, WidgetException
from loguru import logger

from kivymd.app import MDApp
from kivymd.uix.bottomnavigation import MDBottomNavigation
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.datatables import MDDataTable
from kivymd.uix.dialog import MDDialog
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.label import MDLabel
from kivymd.uix.list import ThreeLineIconListItem
from kivymd.uix.snackbar import Snackbar
from kivymd.uix.spinner import MDSpinner

from .widgets.cfd_setting import (CfdSettingContent, CfdSettingDialog,
                                  ExternalSettingContent)
from .widgets.drop_down import DropDownMenu
from .widgets.panel_title import PanelTitle
from .widgets.text_field import TextFieldPath, TextFieldUnit

_FONT_STYLES = ('H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'Subtitle1', 'Subtitle2',
                'Body1', 'Body2', 'Button', 'Caption', 'Overline')


def with_spinner(fn):
  """BimCfdApp의 method 실행 중 spinner 애니메이션을 보여줌

  Parameters
  ----------
  fn : function
  """

  def fn_and_deactivate(self, *args, **kwargs):
    with logger.catch(reraise=True):
      fn(self, *args, **kwargs)
    self.activate_spinner(False)

  def wrapper(self, *args, **kwargs):
    self.activate_spinner(True)
    th = threading.Thread(target=fn_and_deactivate,
                          args=((self,) + args),
                          kwargs=kwargs)
    th.daemon = True
    th.start()
    # th.join()
    return th

  return wrapper


class BimCfdWidget(MDBoxLayout):
  pass


class SimplificationContent(MDBoxLayout):
  pass


class BimCfdAppBase(MDApp):

  def __init__(self, **kwargs):
    self.root = None
    self.built = None

    super().__init__(**kwargs)

    # GUI setting
    self.theme_cls.theme_style = 'Light'
    self.theme_cls.primary_palette = 'Gray'
    self.theme_cls.primary_hue = '600'

    for fs in _FONT_STYLES:
      self.theme_cls.font_styles[fs][0] = 'NotoSansKR'

    self.title = 'CFD Environment Variables'

    self.file_manager = MDFileManager()
    self.file_manager.exit_manager = self.exit_file_manager
    self.file_manager.select_path = self.select_path

    self._bim_path = '\\'
    self._save_dir = '\\'

    self._space_menu: DropDownMenu = None

    self._spinner: MDSpinner = None

    self._snackbar = Snackbar()
    self._snackbar.duration = 2
    self._snackbar.font_size = dp(16)

    self._cfd_dialog = None
    self._external_dialog = None
    self._simpl_dialog = None

  def build(self):
    return BimCfdWidget()

  def manual_build(self):
    """run() 실행 전 build

    Raises
    ------
    WidgetException
        widget 생성 실패
    """
    root = self.build()

    if not root:
      msg = '{} failed to build'.format(__class__.__name__)
      logger.error(msg)
      raise WidgetException(msg)

    self.root = root
    self.built = True

  def on_start(self):
    super().on_start()

    # snackbar
    gap = dp(10)
    self._snackbar.snackbar_x = gap
    self._snackbar.snackbar_y = gap
    self._snackbar.size_hint_x = (self.root.width - 2 * gap) / self.root.width

  @property
  def bim_path_field(self) -> TextFieldPath:
    return self.root.ids.file_panel.ids.bim_path

  @property
  def save_dir_field(self) -> TextFieldPath:
    return self.root.ids.output_panel.ids.save_dir

  @property
  def spaces_menu(self) -> DropDownMenu:
    if self._space_menu is None:
      self._space_menu = self.root.ids.file_panel.ids.space

      def callback(*args, **kwargs):
        self._space_menu.select_item(*args, **kwargs)

      self._space_menu.menu.on_release = callback

    return self._space_menu

  @property
  def spinner(self) -> MDSpinner:
    if self._spinner is None:
      spinner_layout: Widget = self.root.ids.view_panel.ids.spinner_layout

      spinner = MDSpinner()
      spinner.size_hint = (None, None)
      spinner.size = (dp(100), dp(100))
      spinner.active = False
      spinner_layout.add_widget(spinner)

      self._spinner = spinner

    return self._spinner

  @property
  def status_bar(self) -> MDLabel:
    return self.root.ids.status_bar

  @property
  def vis_navigation(self) -> MDBottomNavigation:
    return self.root.ids.view_panel.ids.navigation

  @property
  def cfd_dialog(self) -> CfdSettingDialog:
    if self._cfd_dialog is None:
      self._cfd_dialog = CfdSettingDialog(title='CFD 세부 설정',
                                          content_cls=CfdSettingContent)

    return self._cfd_dialog

  @property
  def external_dialog(self) -> CfdSettingDialog:
    if self._external_dialog is None:
      self._external_dialog = CfdSettingDialog(
          title='풍환경 설정', content_cls=ExternalSettingContent)

    return self._external_dialog

  @property
  def vis_layout(self) -> MDBoxLayout:
    return self.root.ids.view_panel.ids.vis_layout

  @property
  def geom_table_layout(self) -> MDBoxLayout:
    return self.root.ids.view_panel.ids.geom_table

  @property
  def material_table_layout(self) -> MDBoxLayout:
    return self.root.ids.view_panel.ids.material_table

  def open_file_manager(self, mode):
    if mode == 'bim':
      cur_dir = self._bim_path
      ext = ['.ifc', '.IFC', '.stl', '.STL', '.stp', '.STP', '.step', '.STEP']
      search = 'all'
    elif mode == 'save':
      cur_dir = self._save_dir
      ext = []
      search = 'dirs'
    else:
      msg = 'file_manager.mode ({}) not in ["bim", "save"]'.format(mode)
      logger.error(msg)
      raise ValueError(msg)

    self.file_manager.mode = mode
    self.file_manager.ext = ext
    self.file_manager.search = search
    self.file_manager.show(cur_dir)

    if not self.file_manager._window_manager_open:
      self.file_manager._window_manager.open()
      self.file_manager._window_manager_open = True

  def exit_file_manager(self, *args, **kwargs):
    self.file_manager.close()

  def select_path(self, path):
    path = Path(path).resolve()
    mode = self.file_manager.mode

    if mode == 'bim':
      self.check_bim_path(path)
      self._bim_path = path.as_posix()
      self.bim_path_field.text = path.as_posix()

    elif mode == 'save':
      if not path.is_dir():
        path = path.parent

      self._save_dir = path.as_posix()
      self.save_dir_field.text = path.as_posix()

    else:
      msg = 'file_manager.mode ({}) not in ["bim", "save"]'.format(mode)
      logger.error(msg)
      raise ValueError(msg)

    self.exit_file_manager()

  def check_bim_path(self, path):
    if Path(path).suffix.lower() == '.ifc':
      self.bim_path_field.error = False
    else:
      self.bim_path_field.error = True
      self.bim_path_field.helper_text = 'IFC 파일이 아닙니다'

    self.bim_path_field.focus = True

  def check_simplify(self, state):
    """단순화 여부에 따라 형상 단순화 옵션 활성화/비활성화

    Parameters
    ----------
    state : str
        단순화 여부 선택 버튼의 state ({'down', 'normal'})
    """
    assert state in ['down', 'normal'], 'state is {}'.format(state)

    options = ['dist_threshold', 'vol_threshold', 'angle_threshold']
    simplification_ids = self.root.ids.simplification_panel.ids

    for option in options:
      field: TextFieldUnit = simplification_ids[option]
      field.main_text_disabled = (state == 'normal')

  def check_relative_threshold(self, state):
    """단순화에 상대적 기준 설정 여부에 따라 필드의 단위 표시/비표시

    Parameters
    ----------
    state : str
        상대적 기준 적용 버튼의 state ({'down', 'normal'})
    """
    assert state in ['down', 'normal'], 'state is {}'.format(state)

    options = ['dist_threshold', 'vol_threshold']
    simplification_ids = self.root.ids.simplification_panel.ids

    for option in options:
      field: TextFieldUnit = simplification_ids[option]
      field.show_unit = state == 'normal'

  def get_simplification_options(self):
    default_options = (
        ('flag_simplify', False, '건물 형상 최적화 적용 여부'),
        ('flag_relative_threshold', False, '상대적 거리 기준 적용'),
        ('flag_preserve_openings', True, '개구부 형상 유지'),
        ('flag_opening_volume', True, '개구부 부피 유지'),
        ('flag_internal_faces', False, '내부 표면 추출'),
        ('precision', 0.1, '정밀도'),
        ('dist_threshold', 0.5, '단순화 거리 기준'),
        ('vol_threshold', 0.0, '단순화 부피 기준'),
        ('angle_threshold', 90.0, '단순화 각도 기준'),
    )

    option_ids: dict = self.root.ids.simplification_panel.ids.copy()
    if self._simpl_dialog is not None:
      option_ids.update(self._simpl_dialog.content_cls.ids.copy())
    options = dict()

    for key, default, label in default_options:
      if key not in option_ids:
        logger.debug('기본 설정 적용 {}: {}', label, default)
        value = default
      elif isinstance(default, bool):
        state = option_ids[key].state
        assert state in {'down', 'normal'}
        value = (state == 'down')
      else:
        field: TextFieldUnit = option_ids[key]

        try:
          value = float(field.get_main_text())
        except ValueError:
          msg = '[{}] 설정값이 올바르지 않습니다.'.format(label)
          logger.warning(msg)
          self.show_snackbar(msg)
          return None

      options[key] = value

    return options

  def get_openfoam_options(self):
    sopt = self.get_simplification_options()
    copt = self.cfd_dialog.options
    copt.update(self.external_dialog.options)

    ofopt = {
        'solver': 'simpleFoam',
        'flag_external_zone':
            (self.root.ids.cfd_panel.ids.flag_external_zone.state == 'down'),
        'flag_interior_faces': sopt['flag_internal_faces'],
        'flag_opening_volume': sopt['flag_opening_volume'],
        # 'flag_heat_flux': copt['flag_heat_flux'],
        # 'flag_friction': copt['flag_friction'],
        # 'external_temperature': copt['text_external_temperature'],
        # 'heat_transfer_coefficient': copt['text_external_htc'],
        # 'external_zone_size': copt['text_external_size'],
        # 'z0': copt['text_wind_profile_roughness']
    }

    if copt['flag_mesh_resolution']:
      ofopt['grid_resolution'] = copt['text_mesh_resolution']
    elif copt['flag_mesh_size']:
      ofopt['max_cell_size'] = copt['text_mesh_size']
    else:
      raise ValueError

    keys = {
        'flag_friction': 'flag_friction',
        'flag_heat_flux': 'flag_heat_flux',
        'flag_internal_faces': 'flag_internal_faces',
        'flag_opening_volume': 'flag_opening_volume',
        'text_boundary_cell_size': 'boundary_cell_size',
        'text_external_htc': 'heat_transfer_coefficient',
        'text_external_size': 'external_zone_size',
        'text_external_temperature': 'external_temperature',
        'text_inner_buffer': 'inner_buffer',
        'text_mesh_min_size': 'min_cell_size',
        'text_num_of_subdomains': 'num_of_subdomains',
        'text_vertical_dimension': 'vertical_dimension',
        'text_wind_profile_roughness': 'z0',
    }
    for dkey, okey in keys.items():
      ofopt[okey] = copt.get(dkey, None) or None

    if copt['text_boundary_layers_count']:
      ofopt['boundary_layers'] = {'nLayers': copt['text_boundary_layers_count']}

    return ofopt

  @mainthread
  def _set_grid_resolution(self, resolution):
    self.cfd_dialog.set_grid_resolution(resolution=resolution)

  def show_snackbar(self, message, duration=None):
    self._snackbar.text = message

    if duration:
      self._snackbar.duration = duration

    try:
      if hasattr(self._snackbar, 'open'):
        self._snackbar.open()
      else:
        self._snackbar.show()
    except WidgetException:
      # 이미 보이는 (animation 중인) snackbar가 존재함
      # 이번 요청은 무시
      pass

    self.status_bar.text = message
    logger.info('[Snackbar] {}', message)

  @mainthread
  def activate_spinner(self, active=True):
    if active:
      self.vis_navigation.switch_tab('visualization')

    self.spinner.active = active

  def add_geom_table(self, column_data, row_data):
    data_table = MDDataTable(column_data=column_data,
                             row_data=row_data,
                             rows_num=len(row_data))
    data_table.pos_hint = {'center_x': 0.5, 'center_y': 0.5}

    self.geom_table_layout.add_widget(data_table)

  def add_material_table(self, column_data, row_data):
    data_table = MDDataTable(column_data=column_data,
                             row_data=row_data,
                             rows_num=len(row_data))
    data_table.pos_hint = {'center_x': 0.5, 'center_y': 0.5}

    self.material_table_layout.add_widget(data_table)

  def open_simplification_dialog(self):
    # XXX messy
    if self._simpl_dialog is None:
      ok_btn = MDRaisedButton(text='확인')
      self._simpl_dialog = MDDialog(title='건물 형상 최적화 설정',
                                    type='custom',
                                    content_cls=SimplificationContent(),
                                    buttons=[ok_btn])
      ok_btn.on_release = self._simpl_dialog.dismiss
      self._simpl_dialog.width = 450

    self._simpl_dialog.open()
