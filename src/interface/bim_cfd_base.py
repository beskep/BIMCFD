import threading
from pathlib import Path

import utils

from kivy.clock import mainthread
from kivy.metrics import dp
from kivy.uix.widget import Widget, WidgetException
from kivymd.app import MDApp
from kivymd.uix.bottomnavigation import MDBottomNavigation
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.datatables import MDDataTable
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.snackbar import Snackbar
from kivymd.uix.spinner import MDSpinner

from interface.widgets.cfd_setting import CfdSettingDialog
from interface.widgets.drop_down import DropDownMenu
from interface.widgets.panel_title import PanelTitle
from interface.widgets.text_field import TextFieldPath, TextFieldUnit

_FONT_STYLES = ('H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'Subtitle1', 'Subtitle2',
                'Body1', 'Body2', 'Button', 'Caption', 'Overline')


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


class BimCfdAppBase(MDApp):

  def __init__(self, **kwargs):
    self.root = None
    self.built = None

    super().__init__(**kwargs)

    self._logger = utils.get_logger()

    # GUI setting
    self.theme_cls.theme_style = 'Light'
    self.theme_cls.primary_palette = 'Green'

    for fs in _FONT_STYLES:
      self.theme_cls.font_styles[fs][0] = 'NotoSansKR'

    self.title = 'BIM-CFD Pre-processing App ver 1.0'

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
      self._logger.error(msg)
      raise WidgetException(msg)

    self.root = root
    self.built = True

  def on_start(self):
    super().on_start()

    # simplification option
    self.root.ids.simplification_panel.ids.flag_simplify.active = False

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
    return self.root.ids.file_panel.ids.save_dir

  @property
  def visualize_button(self) -> MDRaisedButton:
    return self.root.ids.file_panel.ids.visualize

  @property
  def simplify_button(self) -> MDRaisedButton:
    return self.root.ids.simplification_panel.ids.simplify

  @property
  def execute_button(self) -> MDRaisedButton:
    return self.root.ids.cfd_panel.ids.execute

  @property
  def spaces_menu(self) -> DropDownMenu:
    if self._space_menu is None:
      self._space_menu = self.root.ids.file_panel.ids.space

      def callback(*args, **kwargs):
        self._space_menu.select_item(*args, **kwargs)
        self.visualize_button.disabled = False
        self.simplify_button.disabled = False

      self._space_menu.menu.on_release = callback

    return self._space_menu

  @property
  def solver_menu(self) -> DropDownMenu:
    return self.root.ids.cfd_panel.ids.solver

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
      self._logger.error(msg)
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
      self._logger.error(msg)
      raise ValueError(msg)

    self.exit_file_manager()

  def check_bim_path(self, path: Path):
    if path.suffix.lower() == '.ifc':
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
    boolean_options = [
        'flag_simplify',
        'flag_relative_threshold',
        'flag_preserve_openings',
        'flag_opening_volume',
        'flag_internal_faces',
        'flag_external_zone',
    ]
    numerical_options = [
        'precision',
        'dist_threshold',
        'vol_threshold',
        'angle_threshold',
    ]
    numerical_option_lables = [
        '정밀도',
        '단순화 거리 기준',
        '단순화 부피 기준',
        '단순화 각도 기준',
    ]

    option_ids = self.root.ids.simplification_panel.ids
    res = dict()

    for option in boolean_options:
      state = option_ids[option].state
      assert state in ['down', 'normal']
      res[option] = (state == 'down')

    for option, label in zip(numerical_options, numerical_option_lables):
      field: TextFieldUnit = option_ids[option]

      try:
        value = float(field.get_main_text())
      except ValueError:
        msg = '[{}] 설정값이 올바르지 않습니다.'.format(label)
        self._logger.warning(msg)
        self.show_snackbar(msg)
        return None

      res[option] = value

    return res

  def get_openfoam_options(self):
    solver = self.solver_menu.selected_item_text()
    simplfication_opt = self.get_simplification_options()
    dialog_opt = self.cfd_dialog.options

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
        'text_external_htc': 'heat_transfer_coefficient',
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
      self._snackbar.open()
    except WidgetException:
      # 이미 보이는 (animation 중인) snackbar가 존재함
      # 이번 요청은 무시
      pass

    self._logger.info('[Snackbar] %s', message)

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
