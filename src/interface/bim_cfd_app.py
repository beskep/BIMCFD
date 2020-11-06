import sys
import threading
from pathlib import Path

_SRC_DIR = Path(__file__).parents[1]
if str(_SRC_DIR) not in sys.path:
  sys.path.append(str(_SRC_DIR))

import utils
from converter import ifc_converter as ifccnv
from converter import openfoam

from interface import kvtools
from interface.bim_cfd_base import BimCfdAppBase
from interface.widgets import topo_widget as topo
from interface.widgets.drop_down import DropDownMenu
from interface.widgets.panel_title import PanelTitle
from interface.widgets.text_field import TextFieldPath, TextFieldUnit


class BimCfdApp(BimCfdAppBase):

  def __init__(self, **kwargs):
    super(BimCfdApp, self).__init__(**kwargs)

    self._converter: ifccnv.IfcConverter = None
    self._spaces: list = None
    self._target_space_id = None

    self._simplified_result: dict = None

  def build_and_initialize(self):
    self.manual_build()

    self.solver_menu.set_items(text=openfoam.supported_solvers())

  def select_path(self, path):
    super(BimCfdApp, self).select_path(path)

    path = Path(path)

    if self.file_manager.mode == 'bim' and path.suffix.lower() == '.ifc':
      self.activate_spinner(True)

      self.load_ifc(path)
      self.update_ifc_spaces()

      self.activate_spinner(False)

  def load_ifc(self, path: Path):
    try:
      self._converter = ifccnv.IfcConverter(path=path.as_posix())
    except Exception as e:
      self._logger.error(e)
      self.show_snackbar('IFC 로드 실패')

    options = self.get_simplification_options()
    self._converter.brep_deflection = options['precision']

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
          icon='identifier',
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
      assert '.' in selected_text

      space_idx = selected_text[:selected_text.find('.')][6:]
      space_entity = self._spaces[int(space_idx) - 1]

    return space_entity

  def visualize_topology(self, spaces, openings):
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

  def execute_helper(self):
    space = self.selected_space_entity()
    options = self.get_simplification_options()
    options['angle_threshold'] *= (3.141592 / 180)

    simplified = self._converter.simplify_space(
        spaces=space,
        save_dir=None,
        case_name=None,
        threshold_volume=options['vol_threshold'],
        threshold_dist=options['dist_threshold'],
        threshold_angle=options['angle_threshold'],
        relative_threshold=options['relative_threshold'],
        preserve_opening=options['preserve_openings'],
        opening_volume=options['opening_volume'])

    self.activate_spinner(False)

  def execute(self):
    space = self.selected_space_entity()
    if space is None:
      return

    self.activate_spinner(True)

    thread = threading.Thread(target=self.execute_helper)
    thread.start()
    # thread.join()

    return

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
  # app.select_path(
  #     r'D:\repo\IFC\National Institute of Building Sciences\Project 3. Medical Clinic\2011-09-14-Clinic-IFC\Clinic_A_20110906_optimized.ifc'
  # )

  app.run()
