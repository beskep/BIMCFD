import threading
from pathlib import Path

import utils

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

from .widgets.cfd_setting import CfdSettingDialog
from .widgets.drop_down import DropDownMenu
from .widgets.panel_title import PanelTitle
from .widgets.text_field import TextFieldPath, TextFieldUnit

_FONT_STYLES = ('H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'Subtitle1', 'Subtitle2',
                'Body1', 'Body2', 'Button', 'Caption', 'Overline')

_MATERIALS = (
    ('Brick (Fired Clay)', 0.675, 790),
    ('Concrete Block', 0.5624888, 626.8238),
    ('Gypsum Board', 0.16, 1090),
    ('Plaster Board', 0.58, 1090),
    ('Plywood', 0.12, 1210),
    ('Mineral Fiber Insulation', 0.05, 960),
    ('Asbestos-cement Board', 0.58, 1000),
    ('Hardboard (High Density)', 0.82, 1340),
    ('Heavyweight Concrete', 1.95, 900),
    ('Lightweight Concrete', 0.53, 840),
)


def with_spinner(fn):
  """BimCfdApp의 method 실행 중 spinner 애니메이션을 보여줌

  Parameters
  ----------
  fn : function
  """

  def fn_and_deactivate(self, *args, **kwargs):
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


class MaterialListItem(ThreeLineIconListItem):
  divider = None


def _material_list_item(values):
  return MaterialListItem(
      text=values[0],
      secondary_text=f'Conductivity: {values[1]:.3e} W/mK',
      tertiary_text=f'Specific heat: {values[2]:.3e} J/kgK',
  )


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
    self._solver_menu: DropDownMenu = None

    self._spinner: MDSpinner = None

    self._snackbar = Snackbar()
    self._snackbar.duration = 2
    self._snackbar.font_size = dp(16)

    self._cfd_dialog = None
    self._cfd_options = None
    self._simpl_dialog = None
    self._material_dialog = None

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

  # @property
  # def visualize_button(self) -> MDRaisedButton:
  #   return self.root.ids.file_panel.ids.visualize

  # @property
  # def simplify_button(self) -> MDRaisedButton:
  #   return self.root.ids.simplification_panel.ids.simplify

  # @property
  # def execute_button(self) -> MDRaisedButton:
  #   return self.root.ids.cfd_panel.ids.execute

  @property
  def spaces_menu(self) -> DropDownMenu:
    if self._space_menu is None:
      self._space_menu = self.root.ids.file_panel.ids.space

      def callback(*args, **kwargs):
        self._space_menu.select_item(*args, **kwargs)
        # self.visualize_button.disabled = False
        # self.simplify_button.disabled = False

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
      self._cfd_dialog = CfdSettingDialog()

    return self._cfd_dialog

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
      ext = ['.ifc', '.IFC']
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
        ('flag_external_zone', False, '외부 영역 추출'),
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
    simplfication_opt = self.get_simplification_options()
    try:
      solver = self.solver_menu.selected_item_text()
    except AttributeError:
      solver = 'buoyantSimpleFoam'

    try:
      dialog_opt = self.cfd_dialog.options
    except AttributeError:
      dialog_opt = {
          'flag_friction': False,
          'flag_heat_flux': True,
          'flag_mesh_resolution': True,
          'flag_mesh_size': False,
          'text_boundary_cell_size': 0,
          'text_boundary_layers_count': 0,
          'text_external_htc': None,
          'text_external_size': 5,
          'text_external_temperature': 293.15,
          'text_mesh_min_size': 0,
          'text_mesh_resolution': 24,
          'text_mesh_size': None,
          'text_num_of_subdomains': None,
      }

    ofopt = {
        'solver': solver,
        'flag_external_zone': simplfication_opt['flag_external_zone'],
        'flag_interior_faces': simplfication_opt['flag_internal_faces'],
        'flag_heat_flux': dialog_opt['flag_heat_flux'],
        'flag_friction': dialog_opt['flag_friction'],
        'external_temperature': dialog_opt['text_external_temperature'],
        'heat_transfer_coefficient': dialog_opt['text_external_htc'],
        'external_zone_size': dialog_opt['text_external_size']
    }

    if dialog_opt['flag_mesh_resolution']:
      ofopt['grid_resolution'] = dialog_opt['text_mesh_resolution']
    elif dialog_opt['flag_mesh_size']:
      ofopt['max_cell_size'] = dialog_opt['text_mesh_size']
    else:
      raise ValueError

    keys = {
        'text_mesh_min_size': 'min_cell_size',
        'text_boundary_cell_size': 'boundary_cell_size',
        'text_num_of_subdomains': 'num_of_subdomains',
    }
    for dkey, okey in keys.items():
      option = dialog_opt[dkey]
      ofopt[okey] = option if option else None

    if dialog_opt['text_boundary_layers_count']:
      ofopt['boundary_layers'] = {
          'nLayers': dialog_opt['text_boundary_layers_count']
      }

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
    if self._simpl_dialog is None:
      ok_btn = MDRaisedButton(text='확인')
      self._simpl_dialog = MDDialog(title='건물 형상 최적화 설정',
                                    type='custom',
                                    content_cls=SimplificationContent(),
                                    buttons=[ok_btn])
      ok_btn.on_release = self._simpl_dialog.dismiss
      self._simpl_dialog.width = 450

    self._simpl_dialog.open()
