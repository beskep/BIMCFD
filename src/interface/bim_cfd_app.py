import re
from pathlib import Path

from kivy.clock import mainthread
from kivy.metrics import dp

try:
  import utils
except Exception:
  raise

from converter import ifc_converter as ifccnv
from converter import openfoam

from interface import kvtools
from interface.bim_cfd_base import BimCfdAppBase, with_spinner
from interface.widgets import topo_widget as topo
from interface.widgets.drop_down import DropDownMenu
from interface.widgets.panel_title import PanelTitle
from interface.widgets.text_field import TextFieldPath, TextFieldUnit


class IfcEntityText:
  text_format = '[{:{fmt}d}] {}'
  pattern = re.compile('^\[(\d+)\] (.*)$')

  @classmethod
  def menu_text(cls, index: int, name, width: int = None):
    fmt = '' if width is None else '0{}'.format(width)
    text = cls.text_format.format(index, name, fmt=fmt)

    return text

  @classmethod
  def index(cls, text: str):
    match = cls.pattern.match(text)
    index = int(match.group(1))

    return index


class BimCfdApp(BimCfdAppBase):

  def __init__(self, **kwargs):
    super(BimCfdApp, self).__init__(**kwargs)

    self._converter: ifccnv.IfcConverter = None
    self._spaces: list = None
    self._target_space_id = None

    self._simplified: dict = None

  def build_and_initialize(self):
    self.manual_build()

    self.solver_menu.set_items(text=openfoam.supported_solvers())

  def select_path(self, path):
    super(BimCfdApp, self).select_path(path)

    path = Path(path)
    if self.file_manager.mode == 'bim' and path.suffix.lower() == '.ifc':
      self.load_ifc(path)

  @with_spinner
  def load_ifc(self, path: Path):
    try:
      self._converter = ifccnv.IfcConverter(path=path.as_posix())
    except Exception as e:
      self._logger.error(e)
      self.show_snackbar('IFC 로드 실패')
      return

    options = self.get_simplification_options()
    if options is not None:
      self._converter.brep_deflection = options['precision']
      self.update_ifc_spaces()

  @mainthread
  def update_ifc_spaces(self):
    if self._converter is not None:
      spaces = self._converter.ifc.by_type('IfcSpace')
      width = len(str(len(spaces) + 1))

      names = [
          IfcEntityText.menu_text(index=(i + 1),
                                  name=ifccnv.entity_name(entity),
                                  width=width)
          for i, entity in enumerate(spaces)
      ]
      ids = [str(x.id()) for x in spaces]

      self.spaces_menu.set_items(
          text=names,
          right_text=ids,
          icon='folder-outline',
          # icon='floor-plan',  # TODO: Material Icon Font 업데이트
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
      index = IfcEntityText.index(selected_text)
      space_entity = self._spaces[index - 1]

    return space_entity

  @mainthread
  def visualize_topology(self, spaces, openings=None):
    space_mesh = topo.TopoDsMesh(
        shapes=spaces,
        linear_deflection=self._converter.brep_deflection[0],
        angular_deflection=self._converter.brep_deflection[1],
        color=(1.0, 1.0, 1.0, 0.5))
    mesh = [space_mesh]

    if openings:
      openings_mesh = topo.TopoDsMesh(
          shapes=openings,
          linear_deflection=self._converter.brep_deflection[0],
          angular_deflection=self._converter.brep_deflection[1],
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

  @with_spinner
  def simplify_space(self):
    space = self.selected_space_entity()
    if space is None:
      return

    options = self.get_simplification_options()
    if options is None:
      return

    if options['flag_simplify']:
      # degree to rad
      options['angle_threshold'] *= (3.141592 / 180)
    else:
      options['dist_threshold'] = 0.0
      options['vol_threshold'] = 0.0
      options['angle_threshold'] = 0.0

    simplified = self._converter.simplify_space(
        spaces=space,
        save_dir=None,
        case_name=None,
        threshold_volume=options['vol_threshold'],
        threshold_dist=options['dist_threshold'],
        threshold_angle=options['angle_threshold'],
        relative_threshold=options['flag_relative_threshold'],
        preserve_opening=options['flag_preserve_openings'],
        opening_volume=options['flag_opening_volume'])
    assert simplified is not None

    self._simplified = simplified

    self.show_simplification_results()
    self.execute_button.disabled = False
    self.show_snackbar('형상 전처리 완료', duration=1)

  @mainthread
  def show_simplification_results(self):
    simplified = self._simplified
    if not simplified:
      self.show_snackbar('형상 전처리 필요')
      return

    # TODO: 표 다듬기
    geom_cols = [('변수', dp(50)), ('전처리 전', dp(50)), ('전처리 후', dp(50))]
    geom_orig: dict = simplified['info']['original_geometry']
    geom_simp: dict = simplified['info']['simplified_geometry']

    if geom_simp is None:
      geom_simp = dict()

    geom_vars = list(geom_orig.keys())
    geom_rows = [(x, geom_orig[x], geom_simp.get(x, 'NA')) for x in geom_vars]
    self.geom_table_layout.clear_widgets()
    self.add_geom_table(column_data=geom_cols, row_data=geom_rows)

    geom = simplified['simplified']
    if geom is None:
      geom = simplified['shape']

    self.visualize_topology(spaces=[geom])

  @with_spinner
  def _execute_helper(self, simplified, save_dir, openfoam_options):
    assert simplified is not None
    self._converter.openfoam_case(simplified=simplified,
                                  save_dir=save_dir,
                                  case_name='BIMCFD',
                                  openfoam_options=openfoam_options)

  def execute(self):
    if not self.save_dir_field.text:
      self.show_snackbar('저장 경로를 설정해주세요')
      return

    save_dir = Path(self.save_dir_field.text).resolve()
    if not save_dir.exists():
      self.show_snackbar('저장 경로가 존재하지 않습니다')
      return

    if self._simplified is None:
      self.show_snackbar('형상 전처리가 완료되지 않았습니다')
      return

    # solver = self.solver_menu.selected_item_text()
    # ofoptions = {'solver': solver}
    ofoptions = None

    # TODO: internal face 설정 적용
    # TODO: external zone 설정 적용
    self._execute_helper(simplified=self._simplified,
                         save_dir=save_dir,
                         openfoam_options=ofoptions)

  def test_add_tables(self):
    self.add_geom_table(
        column_data=[('변수', dp(30)), ('전처리 전', dp(30)), ('전처리 후', dp(30))],
        row_data=[('부피', 1, 2), ('표면적', 3, 4), ('특성길이', 5, 6)],
    )
    self.add_material_table(
        column_data=[('재료명', dp(30)), ('두께', dp(30)), ('열전도율', dp(30)),
                     ('매칭 결과', dp(30))],
        row_data=[('a', 'b', 'c'), (1, 2, 3), (4, 5, 6), ('A', 'B', 'C')],
    )


if __name__ == "__main__":
  font_regular = utils.RESOURCE_DIR.joinpath('NotoSansCJKkr-Medium.otf')
  font_bold = utils.RESOURCE_DIR.joinpath('NotoSansCJKkr-Bold.otf')

  kvtools.register_font(name='NotoSansKR',
                        fn_regular=font_regular.as_posix(),
                        fn_bold=font_bold.as_posix())
  kvtools.set_window_size(size=(1280, 720))

  kv_dir = utils.SRC_DIR.joinpath('./interface/kvs')
  kvs = [
      'bim_cfd',
      'file_panel',
      'simplification_panel',
      'cfd_panel',
      'view_panel',
  ]
  for kv in kvs:
    kvpath = kv_dir.joinpath(kv).with_suffix('.kv')
    kvtools.load_kv(kvpath)

  app = BimCfdApp()
  app.build_and_initialize()

  # test
  # app.file_manager.mode = 'bim'
  # app.exit_file_manager = lambda: print()
  # app.select_path((r'D:\repo\IFC\National Institute of Building Sciences'
  #                  r'\Project 3. Medical Clinic\2011-09-14-Clinic-IFC'
  #                  r'\Clinic_A_20110906_optimized.ifc'))

  app.run()
