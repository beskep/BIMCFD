import re
from pathlib import Path

import utils

from kivy.clock import mainthread
from kivy.metrics import dp

from converter import ifc_converter as ifccnv
from converter import openfoam
from interface import kvtools
from interface.bim_cfd_base import BimCfdAppBase, with_spinner
from interface.widgets import topo_widget as topo


def _itc(geom_info: dict):
  itc = (3.0 / geom_info['face_count'] + 6.0 / geom_info['edge_count'])
  itc = min(1.0, itc)
  itc = '{:.2e}'.format(itc)

  return itc


def _fmt(value, fmt='.2f'):
  if not value:
    res = 'NA'
  else:
    res = '{:{}}'.format(value, fmt)

  return res


class IfcEntityText:
  text_format = '[{:{fmt}d}] {}'
  pattern = re.compile(r'^\[(\d+)\] (.*)$')

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
  _geom_vars_kor = {
      'volume': '부피 (m³)',
      'area': '면적 (m²)',
      'characteristic_length': '길이 (m)',
      'face_count': 'Face 개수',
      'edge_count': 'Edge 개수',
      'vertex_count': 'Vertex 개수'
  }

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self._converter: ifccnv.IfcConverter = None
    self._spaces: list = None
    self._target_space_id = None
    self._simplified: dict = None

    # todo: 그래픽 관련 method 모두 mainthread로

  def on_start(self):
    super().on_start()

    self.solver_menu.set_items(text=openfoam.supported_solvers())

  def select_path(self, path):
    super().select_path(path)

    path = Path(path)
    if self.file_manager.mode == 'bim' and path.suffix.lower() == '.ifc':
      self.load_ifc(path)

  @with_spinner
  def load_ifc(self, path: Path):
    try:
      self._converter = ifccnv.IfcConverter(path=path.as_posix())
    except Exception:
      self.show_snackbar('IFC 로드 실패')
      self._logger.error('IFC 로드 실패', exc_info=True)
      return

    self.update_ifc_spaces()

    options = self.get_simplification_options()
    if options is not None:
      self._converter.brep_deflection = options['precision']

    self.show_snackbar('IFC 로드 완료', duration=1.0)

  @mainthread
  def update_ifc_spaces(self):
    if self._converter is not None:
      spaces = self._converter.ifc.by_type('IfcSpace')
      spaces.sort(key=lambda x: x.Name)

      width = len(str(len(spaces) + 1))
      names = [
          IfcEntityText.menu_text(index=(i + 1),
                                  name=ifccnv.entity_name(entity),
                                  width=width)
          for i, entity in enumerate(spaces)
      ]
      ids = [str(x.id()) for x in spaces]

      self.spaces_menu.set_items(text=names,
                                 right_text=ids,
                                 icon='floor-plan',
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
      self._logger.debug('Selected space entity: %s', space_entity)

    return space_entity

  @mainthread
  def visualize_topology(self, spaces, openings=None):
    space_mesh = topo.TopoDsMesh(
        shapes=spaces,
        linear_deflection=self._converter.brep_deflection[0],
        angular_deflection=self._converter.brep_deflection[1],
        color=(1.0, 1.0, 1.0, 0.5))
    meshes = [space_mesh]

    if openings:
      openings_mesh = topo.TopoDsMesh(
          shapes=openings,
          linear_deflection=self._converter.brep_deflection[0],
          angular_deflection=self._converter.brep_deflection[1],
          color=(0.216, 0.494, 0.722, 0.5))
      meshes.append(openings_mesh)

    topo_renderer = topo.TopoRenderer(shapes=meshes,
                                      default_scale=0.8,
                                      near=0.01,
                                      perspective=0.01)
    self.vis_layout.clear_widgets()
    self.vis_layout.add_widget(topo_renderer)

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
      options['angle_threshold'] *= (3.141592 / 180)  # degree to rad
    else:
      msg = '형상 단순화를 시행하지 않습니다. 전처리에 오랜 시간이 소요될 수 있습니다.'
      self._logger.warning(msg)
      options['dist_threshold'] = 0.001  # 작은 기준으로 전처리
      options['vol_threshold'] = 0.0
      options['angle_threshold'] = 0.0
      options['flag_relative_threshold'] = False

    # 단순화
    simplified = self._converter.simplify_space(
        spaces=space,
        threshold_volume=options['vol_threshold'],
        threshold_dist=options['dist_threshold'],
        threshold_angle=options['angle_threshold'],
        relative_threshold=options['flag_relative_threshold'],
        preserve_opening=options['flag_preserve_openings'],
        opening_volume=options['flag_opening_volume'])
    assert simplified is not None
    self._simplified = simplified

    # 해상도 설정
    cl = simplified['info']['original_geometry']['characteristic_length']
    if cl < 0.5:
      grid_resolution = 8
    elif cl < 0.7:
      grid_resolution = 16
    else:
      grid_resolution = 24
    self._set_grid_resolution(grid_resolution)

    self.show_simplification_results()
    self.execute_button.disabled = False
    self.show_snackbar('형상 전처리 완료', duration=1.0)

  def show_geom_info(self, simplified):
    # TODO: 표 다듬기 - 중앙 정렬
    # todo: 단순화된 형상의 face 개수 증가하는 문제 해결 (un-brep/tessellation?)
    geom_cols = [
        ('변수', dp(50)),
        ('전처리 전', dp(50)),
        ('전처리 후', dp(50)),
    ]
    geom_orig: dict = simplified['info']['original_geometry']
    geom_simp: dict = simplified['info']['fused_geometry']
    if geom_simp is None:
      geom_simp = simplified['info']['simplified_geometry']
      if geom_simp is None:
        geom_simp = dict()

    geom_vars = list(geom_orig.keys())
    geom_rows = [(
        self._geom_vars_kor[x],
        _fmt(geom_orig[x]),
        _fmt(geom_simp.get(x, None)),
    ) for x in geom_vars if x in self._geom_vars_kor]
    geom_rows.append(
        ('Inverse Topology Count', _itc(geom_orig), _itc(geom_simp)))

    self.geom_table_layout.clear_widgets()
    self.add_geom_table(column_data=geom_cols, row_data=geom_rows)

  def show_material_info(self, simplified):
    walls = simplified['walls']
    _, match_dict = self._converter.match_thermal_propes(walls=walls)

    materials = sorted(list(match_dict.keys()))

    cols = [
        ('재료\n(원본)', dp(40)),
        ('재료\n(매치)', dp(40)),
        ('열전도율\n(W/mK)', dp(20)),
        ('밀도\n(kg/m³)', dp(20)),
        ('비열\n(J/kg·K)', dp(20)),
    ]

    def _row(material):
      props = match_dict[material][0]
      matched_name = match_dict[material][2]
      row = (
          material,
          matched_name,
          _fmt(props['k'], '.3e'),
          _fmt(props['rho'], '.3e'),
          _fmt(props['Cp'], '.3e'),
      )

      return row

    rows = [_row(material) for material in materials]

    self.material_table_layout.clear_widgets()
    self.add_material_table(column_data=cols, row_data=rows)

  @mainthread
  def show_simplification_results(self):
    simplified = self._simplified
    if not simplified:
      self.show_snackbar('형상 전처리 필요')
      return

    # todo: outer faces, edges 개수만 표시하기
    self.show_geom_info(simplified)
    self.show_material_info(simplified)

    geom = simplified['fused']
    if geom is None:
      geom = simplified['original']

    self.visualize_topology(spaces=[geom])

  @with_spinner
  def _execute_helper(self, simplified, save_dir: Path):
    assert simplified is not None
    openfoam_options = self.get_openfoam_options()

    geom_dir = save_dir.joinpath('BIMCFD', 'geometry')
    if not geom_dir.exists():
      geom_dir.mkdir(parents=True)

    self._converter.save_simplified_space(simplified=simplified,
                                          path=geom_dir.as_posix())

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

    if save_dir.joinpath('BIMCFD').exists():
      self.show_snackbar('대상 경로에 이미 `BIMCFD` 폴더가 존재합니다.')
      return

    if self._simplified is None:
      self.show_snackbar('형상 전처리가 완료되지 않았습니다')
      return

    self._execute_helper(simplified=self._simplified, save_dir=save_dir)

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


def main():
  font_regular = utils.RESOURCE_DIR.joinpath('NotoSansCJKkr-Medium.otf')
  font_bold = utils.RESOURCE_DIR.joinpath('NotoSansCJKkr-Bold.otf')

  kvtools.register_font(name='NotoSansKR',
                        fn_regular=font_regular.as_posix(),
                        fn_bold=font_bold.as_posix())
  kvtools.config()
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
  app.manual_build()

  # test
  # app.file_manager.mode = 'bim'
  # app.exit_file_manager = lambda: print()
  # app.select_path((r'D:\repo\IFC\National Institute of Building Sciences'
  #                  r'\Project 3. Medical Clinic\2011-09-14-Clinic-IFC'
  #                  r'\Clinic_A_20110906_optimized.ifc'))

  app.run()


if __name__ == "__main__":
  main()
