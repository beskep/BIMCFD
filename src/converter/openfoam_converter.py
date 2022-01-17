from collections import OrderedDict, namedtuple
from dataclasses import dataclass, fields
from itertools import chain
from pathlib import Path
from typing import Optional

from utils import DIR

from ifcopenshell import entity_instance as IfcEntity
from loguru import logger

from butterfly.foamfile import FoamFile

from converter.ifc_utils import get_storey
from converter.material_match import Friction
from converter.material_match import MaterialMatch as _MaterialMatch
from converter.openfoam import BoundaryFieldDict, OpenFoamCase

PATH_MATERIAL_LAYER = DIR.TEMPLATE.joinpath('material_layer.txt')
PATH_TEMPERATURE = DIR.TEMPLATE.joinpath('temperature.txt')
PATH_MATERIAL_LAYER.stat()
PATH_TEMPERATURE.stat()


def to_openfoam_vector(values):
  return '( ' + ' '.join([str(x) for x in values]) + ' )'


ThermalProps = namedtuple(
    'ThermalProps',
    ['name', 'thickness', 'matched_name', 'conductivity', 'score'])


@dataclass
class OpenFoamOption:
  solver: str = 'simpleFoam'
  flag_energy: bool = True
  flag_heat_flux: bool = False
  flag_friction: bool = False
  flag_interior_faces: bool = False
  flag_external_zone: bool = False

  external_zone_size: float = 5.0
  external_temperature: float = 300
  heat_transfer_coefficient: str = '1e8'
  roughness_constant: float = 0.5
  roughness_factor: float = 1.0

  max_cell_size: Optional[float] = None
  min_cell_size: Optional[float] = None
  grid_resolution: float = 24.0
  boundary_cell_size: Optional[float] = None
  boundary_layers: Optional[dict] = None
  num_of_subdomains: int = 1

  def __post_init__(self):
    if not OpenFoamCase.is_supported_solver(self.solver):
      raise ValueError(f'지원하지 않는 solver입니다: {self.solver}')

  @classmethod
  def from_dict(cls, d: dict):
    class_fields = set(x.name for x in fields(cls))
    input_fields = set(d.keys())

    defaults = sorted(class_fields - input_fields)
    unused = sorted(input_fields - class_fields)

    if defaults:
      logger.info('Default options: {}', defaults)
    if unused:
      logger.info('Unused options: {}', unused)

    return cls(**{k: v for k, v in d.items() if k in class_fields})


class MaterialMatch(_MaterialMatch):

  @staticmethod
  def layers_info(entity: IfcEntity):
    association = [
        x for x in entity.HasAssociations if x.is_a('IfcRelAssociatesMaterial')
    ]
    assert len(association) < 2

    if len(association) == 0:
      return None

    try:
      relating_material = association[0].RelatingMaterial
      if hasattr(relating_material, 'MaterialLayers'):
        layer = relating_material.MaterialLayers
      else:
        layer = relating_material.ForLayerSet.MaterialLayers
    except AttributeError:
      return None

    layers_info = [(x.Material.Name, x.LayerThickness / 1000.0) for x in layer]

    return layers_info

  def _match_thermal_helper(self, layer, match_dict):
    matched = [match_dict[x[0].lower()] for x in layer]

    matched_name = [x[2] for x in matched]
    conductivity = [x[0]['k'] for x in matched]
    score = [self.scorer(x[1], x[2]) for x in matched]

    return matched_name, conductivity, score

  def match_thermal_prop(self, walls, remove_na=True):
    layer_info = [self.layers_info(x) for x in walls]
    layer_info_wo_na = [x for x in layer_info if x is not None]
    if remove_na:
      layer_info = layer_info_wo_na

    unique_names = set(
        x[0].lower() for x in chain.from_iterable(layer_info_wo_na))
    match_dict = {x: self.thermal_properties(x) for x in unique_names}

    return layer_info, match_dict

  def match_thermal_props_by_layer(self, walls, remove_na=True):
    layer_info, match_dict = self.match_thermal_prop(walls=walls,
                                                     remove_na=remove_na)

    def _prop(li):
      if not li:
        prop = ThermalProps(None, None, None, None, None)
      else:
        matched = self._match_thermal_helper(li, match_dict)
        prop = ThermalProps(name=[x[0] for x in li],
                            thickness=[x[1] for x in li],
                            matched_name=matched[0],
                            conductivity=matched[1],
                            score=matched[2])

      return prop

    props = [_prop(x) for x in layer_info]

    return props


class OpenFoamConverter:
  _TW = 15  # thermal width
  _FW = 17  # friction width
  _MW = 27  # mesh width

  def __init__(self,
               options: Optional[dict] = None,
               min_thermal_score=0,
               min_friction_score=0):
    self._mm = MaterialMatch()

    self._min_thermal_score = min_thermal_score
    self._min_friction_score = min_friction_score

    self._opt = OpenFoamOption.from_dict(options or {})

  def _temperature_wall(self, srf, prop: ThermalProps, flag_heat_flux: bool,
                        heat_transfer_coefficient: str):
    bf = BoundaryFieldDict(width=self._TW)
    is_extracted = (bool(prop.conductivity) and
                    (not self._min_thermal_score or
                     all(self._min_thermal_score <= x for x in prop.score)))

    # surface name
    bf.add_comment(f'Surface name: "{srf.Name}"')
    if hasattr(srf, 'LongName') and srf.LongName:
      bf.add_comment(f'Surface long name: "{srf.LongName}"')

    # surface type
    bf.add_comment(f'Surface type: "{srf.is_a()}"')

    # global id
    bf.add_comment(f'Global id: "{srf.GlobalId}"')

    # storey info
    storey = get_storey(srf)
    if storey is not None:
      bf.add_comment(f'Storey: "{storey.Name}"')
      bf.add_empty_line()

    if prop.name:
      # BIM material named
      bf.add_comment('Material names:')
      for mat_idx, name in enumerate(prop.name):
        bf.add_comment(f'Material {mat_idx + 1}: "{name}"')
      bf.add_empty_line()

      # matched material name
      if is_extracted:
        bf.add_comment('Matched material names:')
        for mat_idx, name in enumerate(prop.matched_name):
          bf.add_comment(f'Material {mat_idx + 1}: "{name}"')
        bf.add_empty_line()

    # boundary conditions
    if flag_heat_flux:
      if is_extracted:
        bf.add_value('type', 'externalWallHeatFluxTemperature')
        bf.add_value('mode', 'coefficient')
        bf.add_value('Ta', f'uniform {self._opt.external_temperature}')
        bf.add_value('h', f'uniform {heat_transfer_coefficient}')
        bf.add_value('thicknessLayers', to_openfoam_vector(prop.thickness))
        bf.add_value('kappaLayers', to_openfoam_vector(prop.conductivity))
        bf.add_value('kappaMethod', 'solidThermo')
      else:
        bf.add_comment('Material information not specified')
      bf.add_value('type', 'fixedValue')

    # TODO external_temperature 맞는지 확인
    bf.add_value('value', f'uniform {self._opt.external_temperature}')

    return bf

  def temperature_dict(self,
                       surface,
                       surface_name=None,
                       opening_names=None) -> OrderedDict:
    """오픈폼 0/T에 입력하기 위한 boundaryField 항목을 생성
    벽 heat flux 해석이 가능한 경우 externalWallHeatFluxTemperature 설정
    https://www.openfoam.com/documentation/guides/latest/api/externalWallHeatFluxTemperatureFvPatchScalarField_8H_source.html

    Parameters
    ----------
    surface : List[IfcEntity]
        경계면을 구성하는 IfcWall, IfcSlab
    surface_name : List[str], optional
        surface 이름. 미지정 시 순서대로 'Surface_n'으로 저장
    opening_names : list[str], optional
        opening surface 이름. 미지정 시 경계조건 설정 안함

    Returns
    -------
    OrderedDict
        T.boundaryField
    """

    if surface_name is None:
      surface_name = ['Surface_' + str(x) for x in range(len(surface))]
    else:
      if len(surface) != len(surface_name):
        raise ValueError

      surface_name = [
          x if x.startswith('Surface_') else 'Surface_' + x
          for x in surface_name
      ]

    htc = self._opt.heat_transfer_coefficient
    try:
      htc_ = float(self._opt.heat_transfer_coefficient)
      if htc_ >= 1e4 or htc_ <= 0.01:
        htc = format(htc_, '.3e')
    except (ValueError, TypeError):
      pass

    props = self._mm.match_thermal_props_by_layer(surface, remove_na=False)
    assert len(surface) == len(props)

    flag_heat_flux = (self._opt.flag_heat_flux and
                      OpenFoamCase.is_conductivity_available(self._opt.solver))
    bfs = BoundaryFieldDict()

    # opening
    if opening_names:
      opening_bf = BoundaryFieldDict(width=self._TW)
      opening_bf.add_value('type', 'fixedValue')
      opening_bf.add_value('value', f'uniform {self._opt.external_temperature}')
      for op in opening_names:
        bfs[op] = opening_bf
        bfs.add_empty_line()

    # wall
    for srf, srf_name, prop in zip(surface, surface_name, props):
      bfs[srf_name] = self._temperature_wall(srf=srf,
                                             prop=prop,
                                             flag_heat_flux=flag_heat_flux,
                                             heat_transfer_coefficient=htc)
      bfs.add_empty_line()

    return bfs

  def _rough_wall_nut_surface(self, srf, friction: Friction, turbulence: bool):
    bf = BoundaryFieldDict(width=self._FW)

    # surface name
    bf.add_comment(f'Surface name: "{srf.Name}"')
    if hasattr(srf, 'LongName') and srf.LongName:
      bf.add_comment(f'Surface long name: "{srf.LongName}"')

    # surface type
    bf.add_comment(f'Surface type: "{srf.is_a()}"')
    bf.add_empty_line()

    bf.add_comment(f'Material names: {friction.eng_name}')

    is_extracted = (not self._min_friction_score or
                    (self._min_friction_score <= friction.score))
    if turbulence and is_extracted:
      bf.add_comment(f'Matched material names: {friction.nearest_material}')
      bf.add_empty_line()

      bf.add_value('type', 'nutURoughWallFunction')
      bf.add_value('roughnessHeight', friction.roughness)
      bf.add_value('roughnessConstant', self._opt.roughness_constant)
      bf.add_value('roughnessFactor', self._opt.roughness_factor)
    else:
      if turbulence and not is_extracted:
        bf.add_comment('Material not matched')
        bf.add_empty_line()

      bf.add_value('type', 'nutkWallFunction')
      bf.add_value('value', '$internalField')

    bf.add_empty_line()

    return bf

  def rough_wall_nut_dict(self, surface, surface_name=None, opening_names=None):
    """오픈폼 0/nut에 입력하기 위한 boundaryField 항목 생성
    난류 해석이 가능한 경우 nutURoughWallFunction 설정
    https://www.openfoam.com/documentation/guides/latest/api/classFoam_1_1nutURoughWallFunctionFvPatchScalarField.html

    Parameters
    ----------
    surface : List[IfcEntity]
        경계면을 구성하는 IfcWall, IfcSlab
    surface_name : List[str], optional
        surface 이름. 미지정 시 순서대로 'Surface_n'으로 저장
    opening_names : list[str], optional
        opening surface 이름. 미지정 시 경계조건 설정 안함

    Returns
    -------
    OrderedDict
        T.boundaryField
    """
    if surface_name is None:
      surface_name = ['Surface_' + str(x) for x in range(len(surface))]
    else:
      if len(surface) != len(surface_name):
        raise ValueError

      surface_name = [
          x if x.startswith('Surface_') else 'Surface_' + x
          for x in surface_name
      ]

    # 가장 첫번째 재료 이름, 조도 추정 추출함
    first_material = [
        x[0][0] if x else None
        for x in [self._mm.layers_info(x) for x in surface]
    ]
    frictions = [self._mm.friction(x) for x in first_material]

    turbulence = OpenFoamCase.is_turbulence_available(self._opt.solver)
    bfs = BoundaryFieldDict()

    # opening
    if opening_names:
      opening_bf = OrderedDict([('type'.ljust(self._FW), 'nutkWallFunction'),
                                ('value'.ljust(self._FW), 'uniform 0')])
      for op in opening_names:
        bfs[op] = opening_bf
        bfs.add_empty_line()

    # wall
    for srf, srf_name, friction in zip(surface, surface_name, frictions):
      bfs[srf_name] = self._rough_wall_nut_surface(srf=srf,
                                                   friction=friction,
                                                   turbulence=turbulence)
      bfs.add_empty_line()

    return bfs

  @staticmethod
  def _zero_boundary_fields(case: OpenFoamCase,
                            target_fields: Optional[list] = None,
                            drop_fields: Optional[list] = None):
    ffs = [x for x in case.foam_files if x.location == '"0"']

    if target_fields:
      invalid_fields = [
          x for x in target_fields if x not in [x.name for x in ffs]
      ]
      if invalid_fields:
        raise ValueError(f'Invalid target boundary fields: {invalid_fields}')

      ffs = [x for x in ffs if x.name in target_fields]

    if drop_fields:
      ffs = [x for x in ffs if x.name not in drop_fields]

    return ffs

  @staticmethod
  def _zero_boundary_field_dict(foam_file: FoamFile, openings, inlets, outlets,
                                walls):
    bf_template = foam_file.values['boundaryField']
    opening_field = bf_template['opening']
    fields = (
        bf_template['opening'],
        bf_template.get('inlet', opening_field),
        bf_template.get('outlet', opening_field),
        bf_template['wall'],
    )

    bf = BoundaryFieldDict()
    for names, field in zip([openings, inlets, outlets, walls], fields):
      for surface in names:
        bf[surface] = field
        bf.add_empty_line()

    return bf

  def zero_boundary_field(self,
                          case: OpenFoamCase,
                          wall_names: list,
                          opening_names=None,
                          inlet_names=None,
                          outlet_names=None,
                          target_fields: Optional[list] = None,
                          drop_fields: Optional[list] = None):
    """수동 생성 안한 나머지 field 설정"""
    opening_names = opening_names or []
    inlet_names = inlet_names or []
    outlet_names = outlet_names or []
    wall_names = [
        (x if x.startswith('Surface_') else 'Surface_' + x) for x in wall_names
    ]

    ffs = self._zero_boundary_fields(case=case,
                                     target_fields=target_fields,
                                     drop_fields=drop_fields)
    for ff in ffs:
      bf = self._zero_boundary_field_dict(foam_file=ff,
                                          openings=opening_names,
                                          inlets=inlet_names,
                                          outlets=outlet_names,
                                          walls=wall_names)
      case.change_boundary_field(variable=ff.name, boundary_field=bf)

  def cf_mesh_dict(self, max_cell_size: float):
    mesh_dict = BoundaryFieldDict(width=self._MW)

    mesh_dict.add_comment('Path to the surface mesh')
    mesh_dict.add_value('surfaceFile', '"geometry.fms"')
    mesh_dict.add_empty_line()

    mesh_dict.add_comment('Maximum cell size in the mesh (mandatory)')
    mesh_dict.add_value('maxCellSize', max_cell_size)
    mesh_dict.add_empty_line()

    mesh_dict.add_comment('Minimum cell size in the mesh (optional)')
    if self._opt.min_cell_size is None:
      mesh_dict.add_comment('minCellSize  null;')
    else:
      mesh_dict.add_value('minCellSize', self._opt.min_cell_size)
    mesh_dict.add_empty_line()

    mesh_dict.add_comment('Maximum cell size at the boundary (optional)')
    if self._opt.boundary_cell_size is None:
      mesh_dict.add_comment('boundaryCellSize  null;')
    else:
      mesh_dict.add_value('boundaryCellSize', self._opt.boundary_cell_size)
    mesh_dict.add_empty_line()

    mesh_dict.add_comment('Stops the meshing process when it is not possible')
    mesh_dict.add_comment('to capture all geometric features (optional)')
    mesh_dict.add_value('enforceGeometryConstraints', 1)
    mesh_dict.add_empty_line()

    mesh_dict.add_comment('Keep cells in the mesh template which')
    mesh_dict.add_comment('intersect selected patches/subsets (optional)')
    keep_cells = {'keepCells': '1'}
    mesh_dict['keepCellsIntersectingPatches'] = OrderedDict([
        (f'{x}.*', keep_cells) for x in
        ['Opening', 'Surface', 'Interior', 'Ceiling', 'External', 'Ground']
    ])
    mesh_dict.add_empty_line()

    if self._opt.boundary_layers is not None:
      boundary_layers = {
          'patchBoundaryLayers': {
              '"(Surface|Opening).*"': {
                  'nLayers       ':
                      str(self._opt.boundary_layers.get('nLayers', 5)),
                  'thicknessRatio':
                      str(self._opt.boundary_layers.get('thicknessRatio', 1.1))
              }
          }
      }
      mesh_dict['boundaryLayers'] = boundary_layers

    return mesh_dict

  def write_openfoam_case(self,
                          simplified: dict,
                          save_dir: Path,
                          name='BIMCFD'):
    case = OpenFoamCase.from_template(solver=self._opt.solver,
                                      save_dir=save_dir,
                                      name=name)

    wall_names = ['Surface_' + x for x in simplified['wall_names']]
    opening_names = simplified['opening_names']

    drop_fields = []
    if (OpenFoamCase.is_energy_available(self._opt.solver) and
        self._opt.flag_energy):
      bf_t = self.temperature_dict(surface=simplified['walls'],
                                   surface_name=wall_names,
                                   opening_names=opening_names)
      case.change_boundary_field(variable='T', boundary_field=bf_t)
      drop_fields.append('T')

    if (OpenFoamCase.is_turbulence_available(self._opt.solver) and
        self._opt.flag_friction):
      bf_nut = self.rough_wall_nut_dict(surface=simplified['walls'],
                                        surface_name=wall_names,
                                        opening_names=opening_names)
      case.change_boundary_field(variable='nut', boundary_field=bf_nut)
      drop_fields.append('nut')

    self.zero_boundary_field(case=case,
                             wall_names=wall_names,
                             opening_names=opening_names,
                             drop_fields=drop_fields)

    max_cell_size = self._opt.max_cell_size
    if not max_cell_size:
      try:
        cl = simplified['info']['fused_geometry']['characteristic_length']
      except KeyError:
        cl = simplified['info']['original_geometry']['characteristic_length']

      max_cell_size = cl / self._opt.grid_resolution

    mesh_dict = self.cf_mesh_dict(max_cell_size=max_cell_size)
    case.change_foam_file_value('meshDict', mesh_dict)

    case.save(overwrite=False, minimum=False)
    case.save_shell(solver=self._opt.solver,
                    num_of_subdomains=self._opt.num_of_subdomains)
