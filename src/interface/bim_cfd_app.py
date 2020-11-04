import logging
import sys
from pathlib import Path

from kivy.core.text import LabelBase
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.uix.widget import Widget
from kivymd import hooks_path
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.datatables import MDDataTable
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.snackbar import Snackbar
from kivymd.uix.spinner import MDSpinner

_SRC_DIR = Path(__file__).parents[1]
if str(_SRC_DIR) not in sys.path:
  sys.path.append(str(_SRC_DIR))

import utils
from converter import ifc_converter as ifccnv

from interface.utf_app import UtfMDApp, load_kv
from interface.widgets import topo_widget as topo
from interface.widgets.drop_down import DropDownMenu
from interface.widgets.panel_title import PanelTitle
from interface.widgets.text_field import TextFieldPath, TextFieldUnit

_FONT_STYLES = [
    'H1',
    'H2',
    'H3',
    'H4',
    'H5',
    'H6',
    'Subtitle1',
    'Subtitle2',
    'Body1',
    'Body2',
    'Button',
    'Caption',
    'Overline',
]


class BimCfdWidget(MDBoxLayout):
  pass


class BimCfdApp(UtfMDApp):

  def __init__(self, **kwargs):
    super(BimCfdApp, self).__init__(**kwargs)

    self._logger = logging.getLogger(self.__class__.__name__)

    # GUI setting
    self.theme_cls.theme_style = 'Light'
    self.theme_cls.primary_palette = 'Green'

    for fs in _FONT_STYLES:
      self.theme_cls.font_styles[fs][0] = 'NotoSansKR'

    self.title = 'BIM-CFD Pre-processing App ver1.0'

    self.file_manager = MDFileManager()
    self.file_manager.exit_manager = self.exit_file_manager
    self.file_manager.select_path = self.select_path

    self._bim_path = '/'
    self._save_dir = '/'

    self._bim_path_field: TextFieldPath = None
    self._save_dir_field: TextFieldPath = None
    self._space_menu: DropDownMenu = None
    self._solver_menu: DropDownMenu = None

    self._vis_layout: MDBoxLayout = None
    self._spinner: MDSpinner = None
    self._geom_table_layout: MDBoxLayout = None
    self._material_table_layout: MDBoxLayout = None

    self._snackbar = Snackbar()
    self._snackbar.duration = 2

    # IFC Converter setting
    self._converter: ifccnv.IfcConverter = None
    # TODO: 정밀도 설정 converter에 전달
    self._spaces: list = None
    self._target_space_id = None
    self._vis_deflection = (0.1, 0.5)

  def build(self):
    return BimCfdWidget()

  @property
  def bim_path_field(self):
    if self._bim_path_field is None:
      self._bim_path_field = self.root.ids.file_panel.ids.bim_path
    return self._bim_path_field

  @property
  def save_dir_field(self):
    if self._save_dir_field is None:
      self._save_dir_field = self.root.ids.file_panel.ids.save_dir
    return self._save_dir_field

  @property
  def spaces_menu(self):
    if self._space_menu is None:
      self._space_menu = self.root.ids.file_panel.ids.space
    return self._space_menu

  @property
  def solver_menu(self):
    if self._solver_menu is None:
      self._solver_menu = self.root.ids.cfd_panel.ids.solver
    return self._solver_menu

  @property
  def spinner(self):
    if self._spinner is None:
      spinner_layout: Widget = self.root.ids.view_panel.ids.spinner_layout
      spinner = MDSpinner()
      spinner.size_hint = (None, None)
      spinner.size = (dp(50), dp(50))
      spinner.active = False
      spinner_layout.add_widget(spinner)
      self._spinner = spinner
    return self._spinner

  @property
  def vis_layout(self):
    if self._vis_layout is None:
      self._vis_layout = self.root.ids.view_panel.ids.vis_layout
    return self._vis_layout

  @property
  def geom_table_layout(self):
    if self._geom_table_layout is None:
      self._geom_table_layout = self.root.ids.view_panel.ids.geom_table
    return self._geom_table_layout

  @property
  def material_table_layout(self):
    if self._material_table_layout is None:
      self._material_table_layout = self.root.ids.view_panel.ids.material_table
    return self._material_table_layout

  def open_file_manager(self, mode):
    if mode == 'bim':
      cur_dir = self._bim_path
      ext = ['.ifc', '.IFC']
    elif mode == 'save':
      cur_dir = self._save_dir
      ext = []
    else:
      msg = 'file_manager.mode ({}) not in {"bim", "save"}'.format(
          self.file_manager.mode)
      self._logger.error(msg)
      raise ValueError(msg)

    self.file_manager.mode = mode
    self.file_manager.ext = ext
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
      self._save_dir = path.as_posix()
      self.save_dir_field.text = path.as_posix()

    else:
      msg = 'file_manager.mode ({}) not in {"bim", "save"}'.format(mode)
      self._logger.error(msg)
      raise ValueError(msg)

    self.file_manager.mode = None
    self.exit_file_manager()

    if mode == 'bim' and path.suffix.lower() == '.ifc':
      self.activate_spinner(True)
      self.load_ifc(path)
      self.update_ifc_spaces()
      self.activate_spinner(False)

  def check_bim_path(self, path: Path):
    if path.suffix.lower() == '.ifc':
      self.bim_path_field.error = False
    else:
      self.bim_path_field.error = True
      self.bim_path_field.helper_text = 'IFC 파일이 아닙니다'

    self.bim_path_field.focus = True

  def load_ifc(self, path: Path):
    try:
      self._converter = ifccnv.IfcConverter(path=path.as_posix())
    except Exception as e:
      self._logger.error(e)
      self.show_snackbar('IFC 로드 실패')

  def update_ifc_spaces(self):
    if self._converter is not None:
      spaces = self._converter.ifc.by_type('IfcSpace')
      width = len(str(len(spaces) + 1))

      names = [
          'Space {:0{}d}.  {}'.format(i + 1, width, ifccnv.entity_name(entity))
          for i, entity in enumerate(spaces)
      ]
      ids = [str(x.id()) for x in spaces]

      self.spaces_menu.set_items(
          text=names,
          right_text=ids,
          icon='floor-plan',  # TODO: Material Icon Font 업데이트
          right_icon='identifier')

      self._spaces = spaces
    else:
      self._logger.error('IFC가 업데이트 되지 않음.')

  def selected_space_entity(self):
    if self.spaces_menu.selected_item is None:
      self.show_snackbar('공간을 선택해주세요')
      space_entity = None
    else:
      selected_text = self.spaces_menu.selected_item_text()
      assert '.' in selected_text

      space_idx = selected_text[:selected_text.find('.')][6:]
      space_entity = self._spaces[int(space_idx) - 1]

    return space_entity

  def visualize_topology(self, spaces, openings):
    space_mesh = topo.TopoDsMesh(shapes=spaces,
                                 linear_deflection=self._vis_deflection[0],
                                 angular_deflection=self._vis_deflection[1],
                                 color=(1.0, 1.0, 1.0, 0.5))
    mesh = [space_mesh]

    if openings:
      openings_mesh = topo.TopoDsMesh(
          shapes=openings,
          linear_deflection=self._vis_deflection[0],
          angular_deflection=self._vis_deflection[1],
          color=(0.216, 0.494, 0.722, 0.5))
      mesh.append(openings_mesh)

    self.vis_layout.clear_widgets()
    self.vis_layout.add_widget(topo.TopoRenderer(shapes=mesh))

  def visualize_selected_space(self):
    space_entity = self.selected_space_entity()
    if space_entity is None:
      return

    _, space, _, openings = self._converter.convert_space(space_entity)
    self.visualize_topology(spaces=[space], openings=openings)

  def check_simplify(self, state):
    """단순화 여부에 따라 형상 단순화 옵션 활성화/비활성화

    Parameters
    ----------
    state : str
        단순화 여부 선택 버튼의 state ({'down', 'normal'})
    """
    options = ['dist_threshold', 'vol_threshold', 'angle_threshold']
    simplification_ids = self.root.ids.simplification_panel.ids

    for option in options:
      field: TextFieldUnit = simplification_ids[option]

      if state == 'down':
        field.activate()
      elif state == 'normal':
        field.deactivate()
      else:
        msg = 'Unexpected button state: {}'.format(state)
        self._logger.error(msg)

  def check_relative_threshold(self, state):
    """단순화에 상대적 기준 설정 여부에 따라 필드의 단위 표시/비표시

    Parameters
    ----------
    state : str
        상대적 기준 적용 버튼의 state ({'down', 'normal'})
    """
    options = ['dist_threshold', 'vol_threshold', 'angle_threshold']
    simplification_ids = self.root.ids.simplification_panel.ids

    for option in options:
      field: TextFieldUnit = simplification_ids[option]

      if state == 'down':
        field.deactivate_unit()
      elif state == 'normal':
        field.activate_unit()
      else:
        msg = 'Unexpected button state: {}'.format(state)
        self._logger.error(msg)

  def show_snackbar(self, message, duration=None):
    self._snackbar.text = message

    if duration:
      self._snackbar.duration = duration

    self._snackbar.show()

  def activate_spinner(self, active=True):
    self.spinner.active = active

  def test_add_tables(self):
    self.add_geom_table(
        column_data=[('변수', dp(30)), ('단순화 전', dp(30)), ('단순화 후', dp(30))],
        row_data=[('부피', 1, 2), ('표면적', 3, 4), ('특성길이', 5, 6)],
    )
    self.add_material_table(
        column_data=[('재료명', dp(30)), ('두께', dp(30)), ('열전도율', dp(30)),
                     ('매칭 결과', dp(30))],
        row_data=[('a', 'b', 'c'), (1, 2, 3), (4, 5, 6), ('A', 'B', 'C')],
    )

  def add_geom_table(self, column_data, row_data):
    """test용
    """

    data_table = MDDataTable(column_data=column_data, row_data=row_data)
    self.geom_table_layout.add_widget(data_table)

  def add_material_table(self, column_data, row_data):
    """test용
    """

    data_table = MDDataTable(column_data=column_data, row_data=row_data)
    self.material_table_layout.add_widget(data_table)


if __name__ == "__main__":
  font_regular = utils.RESOURCE_DIR.joinpath('NotoSansCJKkr-Medium.otf')
  font_bold = utils.RESOURCE_DIR.joinpath('NotoSansCJKkr-Bold.otf')

  LabelBase.register(name='NotoSansKR',
                     fn_regular=font_regular.as_posix(),
                     fn_bold=font_bold.as_posix())

  # font_icon = utils.RESOURCE_DIR.joinpath('materialdesignicons-webfont.ttf')
  # font_icon = Path(
  #     r'C:\Miniconda3\envs\bcmd36\Lib\site-packages\kivymd\fonts\materialdesignicons-webfont.ttf'
  # )
  # LabelBase.register(name='Icons', fn_regular=font_icon.as_posix())

  Window.size = (1280, 720)

  kv_dir = utils.SRC_DIR.joinpath('./interface/kvs')
  kvs = [
      kv_dir.joinpath(x).with_suffix('.kv') for x in
      ['cfd_panel', 'file_panel', 'simplification_panel', 'view_panel']
  ]
  for kv in kvs:
    load_kv(kv)

  app = BimCfdApp()
  app.run(utils.SRC_DIR.joinpath('interface/kvs/bim_cfd.kv'))
