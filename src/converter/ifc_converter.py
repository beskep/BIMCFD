import json
import logging
import os
from collections import OrderedDict, defaultdict
from itertools import chain, combinations
from shutil import copy2
from typing import Iterable, List, Tuple, Union

from utils import TEMPLATE_DIR

import ifcopenshell
import ifcopenshell.geom
from ifcopenshell import entity_instance as IfcEntity
from OCC.Core.TopoDS import TopoDS_Builder, TopoDS_Compound, TopoDS_Shape
from OCC.Extend import DataExchange
from OCC.Extend.TopologyUtils import TopologyExplorer

from converter.material_match import MaterialMatch
from converter.obj_convert import write_obj
from converter.openfoam import BoundaryFieldDict, OpenFoamCase
from converter.simplify import (compare_shapes, fuse_compound,
                                geometric_features, make_external_zone,
                                shapes_distance, simplify_space)

PATH_MATERIAL_LAYER = TEMPLATE_DIR.joinpath('material_layer.txt')
PATH_TEMPERATURE = TEMPLATE_DIR.joinpath('temperature.txt')
assert PATH_MATERIAL_LAYER.exists(), PATH_MATERIAL_LAYER
assert PATH_TEMPERATURE.exists(), PATH_MATERIAL_LAYER


def to_openfoam_vector(values):
  return '( ' + ' '.join([str(x) for x in values]) + ' )'


class IfcConverter:

  _default_openfoam_options = {
      'solver': 'simpleFoam',
      'flag_energy': True,
      'flag_friction': False,
      'flag_interior_faces': False,
      'flag_external_zone': False,
      'external_zone_size': 5.0,
      'external_temperature': 300,
      'heat_transfer_coefficient': '1e8',
      'max_cell_size': None,
      'min_cell_size': None,
      'grid_resolution': 24.0,
      'boundary_cell_size': None,
      'boundary_layers': None
  }

  def __init__(self,
               path: str,
               wall_types=('IfcWall',),
               slab_types=('IfcSlab',),
               covering_types=('IfcCovering',)):
    self._logger = logging.getLogger(self.__class__.__name__)

    self._file_path = os.path.normpath(path)
    self._ifc = ifcopenshell.open(self._file_path)
    self._settings = ifcopenshell.geom.settings()
    self._settings.set(self._settings.USE_PYTHON_OPENCASCADE, True)

    self._ifc_types_wall = wall_types
    self._ifc_types_slab = slab_types
    self._ifc_types_covering = covering_types
    self._spaces_walls = None  # TODO: 삭제
    self._storeys = None

    self._brep_deflection = (0.9, 0.5)

    # Simplification options
    self.threshold_internal_volume = 0.0
    self.threshold_internal_length = 0.0
    self.threshold_surface_dist = 0.0
    self.threshold_surface_angle = 0.0
    self.threshold_surface_vol_ratio = 0.5
    self.relative_threshold = True
    self.preserve_opening = True
    self.extract_opening_volume = True
    self.fuzzy = 0.0

    # OpenFOAM options
    self._of_options = self._default_openfoam_options.copy()

    self.tol_bbox = 1e-8
    self.tol_cut = 0.0

    self._material_match = None
    self._min_match_score = None

  @property
  def ifc(self) -> ifcopenshell.file:
    return self._ifc

  @property
  def settings(self):
    return self._settings

  @property
  def file_path(self):
    return self._file_path

  @property
  def brep_deflection(self):
    return self._brep_deflection

  @brep_deflection.setter
  def brep_deflection(self, value):
    if isinstance(value, (int, float)):
      self._brep_deflection = (value, 0.5)
    elif isinstance(value, Iterable):
      self._brep_deflection = value
    else:
      raise ValueError

  @property
  def default_openfoam_options(self):
    return self._default_openfoam_options.copy()

  @property
  def openfoam_options(self):
    return self._of_options

  def set_openfoam_options(self, **kwargs):
    essential = {
        key: kwargs.pop(key, value)
        for key, value in self.default_openfoam_options.items()
    }
    self._of_options.update(essential)
    self._of_options.update(kwargs)

  @property
  def materiam_match(self) -> MaterialMatch:
    if self._material_match is None:
      self._material_match = MaterialMatch()
    return self._material_match

  @property
  def minimum_match_score(self):
    return self._min_match_score

  @minimum_match_score.setter
  def minimum_match_score(self, value):
    value = int(value)
    if value <= 0 or 100 < value:
      raise ValueError('Score out of range')
    self._min_match_score = value

  def create_geometry(self, entity: IfcEntity):
    return ifcopenshell.geom.create_shape(self.settings, entity).geometry

  def get_relating_space(self, target: Union[int, IfcEntity]):
    """
    :param target: int일 경우 entity의 id
    :return: List[IfcEntity]
    """
    if isinstance(target, int):
      target = self._ifc.by_id(target)

    provides_boundaries = target.ProvidesBoundaries
    relating_space = [pb.RelatingSpace for pb in provides_boundaries]

    return relating_space

  def get_space_wall_dict(self):
    """
    :return: defaultdict, {space: List[wall]}
    """
    spaces_walls = defaultdict(set)

    walls = list(
        chain.from_iterable(
            [self._ifc.by_type(x) for x in self._ifc_types_wall]))

    for wall in walls:
      spaces = self.get_relating_space(wall)
      for space in spaces:
        spaces_walls[space].add(wall)

    return spaces_walls

  def get_storey_info(self):
    storeys = self._ifc.by_type('IfcBuildingStorey')
    storeys_dict = defaultdict(set)

    for storey in storeys:
      for decom in storey.IsDecomposedBy:
        storeys_dict[storey].update(decom.RelatedObjects)

    return storeys_dict

  @staticmethod
  def get_intersection_walls(spaces_walls: dict):
    intersection = dict()
    for sp1, sp2 in combinations(spaces_walls.keys(), 2):
      inter = spaces_walls[sp1] & spaces_walls[sp2]
      if inter:
        intersection[(sp1, sp2)] = inter
    return intersection

  def get_openings(self, wall: IfcEntity):
    assert any([wall.is_a(x) for x in self._ifc_types_wall])

    shapes = []
    for opening in wall.HasOpenings:
      try:
        shape = self.create_geometry(opening.RelatedOpeningElement)
        shapes.append(shape)
      except RuntimeError as e:
        msg = ('Opening 추출 실패:\n{}\n{}\n{}'.format(
            self.file_path, opening.RelatedOpeningElement, e))
        self._logger.warning(msg)

    return shapes

  @staticmethod
  def _layers_info(entity: IfcEntity):
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
    score = [self.materiam_match.scorer(x[1], x[2]) for x in matched]

    return matched_name, conductivity, score

  def match_thermal_props(self, walls, remove_na=True):
    layer_info = [self._layers_info(x) for x in walls]
    layer_info_without_na = [x for x in layer_info if x is not None]
    if remove_na:
      layer_info = layer_info_without_na

    unique_names = set(
        [x[0].lower() for x in chain.from_iterable(layer_info_without_na)])
    match_dict = {
        x: self.materiam_match.thermal_properties(x) for x in unique_names
    }

    matched = [
        self._match_thermal_helper(x, match_dict) if x else x
        for x in layer_info
    ]

    name = [[x[0] for x in y] if y else None for y in layer_info]
    thickness = [[x[1] for x in y] if y else None for y in layer_info]
    matched_name = [x[0] if x else None for x in matched]
    conductivity = [x[1] if x else None for x in matched]
    score = [x[2] if x else None for x in matched]

    return name, thickness, matched_name, conductivity, score

  def openfoam_temperature_dict(self,
                                solver,
                                surface,
                                surface_name=None,
                                opening_names=None,
                                min_score=None,
                                temperature=300,
                                heat_transfer_coefficient=1e8) -> OrderedDict:
    """오픈폼 0/T에 입력하기 위한 boundaryField 항목을 생성
    벽 heat flux 해석이 가능한 경우 externalWallHeatFluxTemperature 설정
    https://www.openfoam.com/documentation/guides/latest/api/externalWallHeatFluxTemperatureFvPatchScalarField_8H_source.html

    Parameters
    ----------
    solver : str
        OpenFOAM 솔버 종류
    surface : List[IfcEntity]
        경계면을 구성하는 IfcWall, IfcSlab
    surface_name : List[str], optional
        surface 이름. 미지정 시 순서대로 'Surface_n'으로 저장
    opening_names : list[str], optional
        opening surface 이름. 미지정 시 경계조건 설정 안함
    min_score : Union[int, float], optional
        재료 매칭의 최소 점수 (0-100)
    temperature : float
        외부/벽체 온도 [K]
    heat_transfer_coefficient : float
        외부/벽 간 열전도율 [W/m^2/K]

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

    try:
      if heat_transfer_coefficient >= 1e4 or heat_transfer_coefficient <= 0.01:
        heat_transfer_coefficient = format(heat_transfer_coefficient, '.3e')
    except TypeError:
      pass

    names, thickness, matched_names, conductivity, score = self.match_thermal_props(
        surface, remove_na=False)
    assert len(surface) == len(names)

    wall_heat_flux = OpenFoamCase.is_conductivity_available(solver)
    bfs = BoundaryFieldDict()
    width = 15

    # opening
    if opening_names:
      opening_bf = OrderedDict([('type'.ljust(width), 'fixedValue'),
                                ('value'.ljust(width), temperature)])
      for op in opening_names:
        bfs[op] = opening_bf
        bfs.add_empty_line()

    # wall
    for idx, (srf, srf_name) in enumerate(zip(surface, surface_name)):
      bf = BoundaryFieldDict()
      bf.set_width(width)

      if min_score:
        is_extracted = bool(conductivity[idx]) and all(
            [min_score <= x for x in score[idx]])
      else:
        is_extracted = bool(conductivity[idx])

      # surface name
      bf.add_comment('Surface name: "{}"'.format(srf.Name))
      if hasattr(srf, 'LongName') and srf.LongName:
        bf.add_comment('Surface long name: "{}"'.format(srf.LongName))

      # surface type
      bf.add_comment('Surface type: "{}"'.format(srf.is_a()))

      # global id
      bf.add_comment('Global id: "{}"'.format(srf.GlobalId))

      # storey
      storey = get_storey(srf)
      if storey is not None:
        bf.add_comment('Storey: "{}"'.format(storey.Name))
        bf.add_empty_line()

      if names[idx]:
        # BIM material named
        bf.add_comment('Material names:')
        for mat_idx, name in enumerate(names[idx]):
          bf.add_comment('Material {}: "{}"'.format(mat_idx + 1, name))
        bf.add_empty_line()

        # matched material name
        if is_extracted:
          bf.add_comment('Matched material names:')
          for mat_idx, name in enumerate(matched_names[idx]):
            bf.add_comment('Material {}: "{}"'.format(mat_idx + 1, name))
          bf.add_empty_line()

      # boundary conditions
      if wall_heat_flux and is_extracted:
        bf.add_value('type', 'externalWallHeatFluxTemperature')
        bf.add_value('mode', 'coefficient')
        bf.add_value('Ta', 'uniform ' + str(temperature))
        bf.add_value('h', 'uniform ' + str(heat_transfer_coefficient))
        bf.add_value('thicknessLayers', to_openfoam_vector(thickness[idx]))
        bf.add_value('kappaLayers', to_openfoam_vector(conductivity[idx]))
        bf.add_value('kappaMethod', 'solidThermo')
      else:
        if wall_heat_flux and not is_extracted:
          bf.add_comment('Material information not specified')

        bf.add_value('type', 'fixedValue')
        bf.add_value('value', temperature)

      bfs[srf_name] = bf
      bfs.add_empty_line()

    return bfs

  def openfoam_rough_wall_nut_dict(self,
                                   solver,
                                   surface,
                                   surface_name=None,
                                   opening_names=None,
                                   min_score=None,
                                   roughness_constant=0.5,
                                   roughness_factor=1.0):
    """오픈폼 0/nut에 입력하기 위한 boundaryField 항목 생성
    난류 해석이 가능한 경우 nutURoughWallFunction 설정
    https://www.openfoam.com/documentation/guides/latest/api/classFoam_1_1nutURoughWallFunctionFvPatchScalarField.html
    
    Parameters
    ----------
    solver : str
        OpenFOAM 솔버 종류
    surface : List[IfcEntity]
        경계면을 구성하는 IfcWall, IfcSlab
    surface_name : List[str], optional
        surface 이름. 미지정 시 순서대로 'Surface_n'으로 저장
    opening_names : list[str], optional
        opening surface 이름. 미지정 시 경계조건 설정 안함
    min_score : Union[int, float], optional
        재료 매칭의 최소 점수 (0-100)
    roughness_constant: float
        Roughness height [non-dimensional]
    roughness_factor: float
        Roughness constant [non-dimensional]

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

    layer_info = [self._layers_info(x) for x in surface]
    # 가장 첫번째 재료 이름을 추출함
    first_material = [x[0][0] if x else None for x in layer_info]
    friction_match = [self.materiam_match.friction(x) for x in first_material]

    roughness = [x[0] for x in friction_match]
    material_name = [x[1] for x in friction_match]
    matched_name = [x[2] for x in friction_match]
    score = [x[3] for x in friction_match]

    turbulence = OpenFoamCase.is_turbulence_available(solver)
    bfs = BoundaryFieldDict()
    width = 17

    # opening
    if opening_names:
      opening_bf = OrderedDict([('type'.ljust(width), 'nutkWallFunction'),
                                ('value'.ljust(width), 'uniform 0')])
      for op in opening_names:
        bfs[op] = opening_bf
        bfs.add_empty_line()

    # wall
    for idx, (srf, srf_name) in enumerate(zip(surface, surface_name)):
      bf = BoundaryFieldDict()
      bf.set_width(width)

      # surface name
      bf.add_comment('Surface name: "{}"'.format(srf.Name))
      if hasattr(srf, 'LongName') and srf.LongName:
        bf.add_comment('Surface long name: "{}"'.format(srf.LongName))

      # surface type
      bf.add_comment('Surface type: "{}"'.format(srf.is_a()))
      bf.add_empty_line()

      bf.add_comment('Material names: {}'.format(material_name[idx]))

      is_extracted = not min_score or (min_score <= score[idx])
      if turbulence and is_extracted:
        bf.add_comment('Matched material names: {}'.format(matched_name[idx]))
        bf.add_empty_line()

        bf.add_value('type', 'nutURoughWallFunction')
        bf.add_value('roughnessHeight', roughness[idx])
        bf.add_value('roughnessConstant', roughness_constant)
        bf.add_value('roughnessFactor',
                     roughness_factor)  # fixme: semicolon 추가 안됨
      else:
        if turbulence and not is_extracted:
          bf.add_comment('Material not matched')
          bf.add_empty_line()

        bf.add_value('type', 'nutkWallFunction')
        bf.add_value('value', '$internalField')

      bfs[srf_name] = bf
      bfs.add_empty_line()

    return bfs

  @staticmethod
  def openfoam_zero_boundary_field(case: OpenFoamCase,
                                   wall_names,
                                   opening_names,
                                   target_fields: list = None,
                                   drop_fields: list = None):
    ffs = [x for x in case.foam_files if x.location == '"0"']

    if target_fields:
      bf_names = [x.name for x in ffs]
      invalid_fields = [x for x in target_fields if x not in bf_names]

      if invalid_fields:
        raise ValueError('Invalid target boundary fields: {}'.format(
            str(invalid_fields)))

      ffs = [x for x in ffs if x.name in target_fields]

    if drop_fields:
      ffs = [x for x in ffs if x.name not in drop_fields]

    wall_names = [
        (x if x.startswith('Surface_') else 'Surface_' + x) for x in wall_names
    ]

    for ff in ffs:
      bf_template = ff.values['boundaryField']
      opening_field = bf_template['opening']
      wall_field = bf_template['wall']

      bf = BoundaryFieldDict()

      for opening in opening_names:
        bf[opening] = opening_field
        bf.add_empty_line()

      for wall in wall_names:
        bf[wall] = wall_field
        bf.add_empty_line()

      case.change_boundary_field(variable=ff.name, boundary_field=bf)

  def openfoam_temperature_info(self,
                                walls,
                                surface_names=None,
                                external_temperature=None,
                                heat_transfer_coefficient=None):
    from textwrap import indent

    def _wall_name(wall):
      if hasattr(wall, 'LongName') and wall.LongName:
        n = '{} ({})'.format(wall.Name, wall.LongName)
      else:
        n = str(wall.Name)
      return n

    if external_temperature is None:
      external_temperature = 300
    if heat_transfer_coefficient is None:
      heat_transfer_coefficient = '1e8'

    name, thickness, matched_name, conductivity, _ = self.match_thermal_props(
        walls)
    with open(PATH_TEMPERATURE, 'r') as f:
      temperature_template = f.readlines()
    with open(PATH_MATERIAL_LAYER, 'r') as f:
      field_template = f.readlines()

    temperature_template = ''.join(temperature_template)
    field_template = ''.join(field_template)

    names = [_wall_name(x) for x in walls]
    types = [x.is_a() for x in walls]

    print('Extracting surfaces...')
    mf = '  // material {}: "{}"\n'
    cf = '  // Surface name: "{}"\n' \
         '  // Surface type: "{}"\n\n' \
         '  // Material names:\n' \
         '{}\n' \
         '  // Matched material names:\n' \
         '{}'
    fields = []
    for idx in range(len(name)):
      if surface_names is None:
        surface_name = 'Surface_' + str(idx)
      else:
        surface_name = 'Surface_' + surface_names[idx]

      name_list = [mf.format(x, y) for x, y in enumerate(name[idx])]
      matched_list = [mf.format(x, y) for x, y in enumerate(matched_name[idx])]

      comment = cf.format(names[idx], types[idx], ''.join(name_list),
                          ''.join(matched_list))

      thickness_layers = ' '.join(map(str, thickness[idx]))
      kappa_layers = ' '.join(map(str, conductivity[idx]))

      field = field_template.format(surface_name, comment, external_temperature,
                                    heat_transfer_coefficient, thickness_layers,
                                    kappa_layers)
      fields.append(field)

    fields = indent('\n\n'.join(fields), '  ')
    temperature = temperature_template.format(fields)

    return temperature

  @staticmethod
  def openfoam_cf_mesh_dict(max_cell_size=1.0,
                            min_cell_size=None,
                            boundary_cell_size=None,
                            boundary_layers_args: dict = None):
    mesh_dict = BoundaryFieldDict()
    mesh_dict.set_width(27)

    mesh_dict.add_comment('Path to the surface mesh')
    mesh_dict.add_value('surfaceFile', '"geometry.fms"')
    mesh_dict.add_empty_line()

    mesh_dict.add_comment('Maximum cell size in the mesh (mandatory)')
    mesh_dict.add_value('maxCellSize', max_cell_size)
    mesh_dict.add_empty_line()

    mesh_dict.add_comment('Minimum cell size in the mesh (optional)')
    if min_cell_size is None:
      mesh_dict.add_comment('minCellSize  null;')
    else:
      mesh_dict.add_value('minCellSize', min_cell_size)
    mesh_dict.add_empty_line()

    mesh_dict.add_comment('Maximum cell size at the boundary (optional)')
    if boundary_cell_size is None:
      mesh_dict.add_comment('boundaryCellSize  null;')
    else:
      mesh_dict.add_value('boundaryCellSize', boundary_cell_size)
    mesh_dict.add_empty_line()

    mesh_dict.add_comment('Stops the meshing process when it is not possible')
    mesh_dict.add_comment('to capture all geometric features (optional)')
    mesh_dict.add_value('enforceGeometryConstraints', 1)
    mesh_dict.add_empty_line()

    mesh_dict.add_comment('Keep cells in the mesh template which')
    mesh_dict.add_comment('intersect selected patches/subsets (optional)')
    keep_cells = {'keepCells': '1'}
    patches = OrderedDict([('"Opening.*"', keep_cells),
                           ('"Surface.*"', keep_cells),
                           ('"Interior.*"', keep_cells)])
    mesh_dict['keepCellsIntersectingPatches'] = patches
    mesh_dict.add_empty_line()

    if boundary_layers_args is not None:
      boundary_layers = {
          'patchBoundaryLayers': {
              '"(Surface|Opening).*"': {
                  'nLayers       ':
                      str(boundary_layers_args.get('nLayers', 5)),
                  'thicknessRatio':
                      str(boundary_layers_args.get('thicknessRatio', 1.1))
              }
          }
      }
      mesh_dict['boundaryLayers'] = boundary_layers

    return mesh_dict

  def convert_space(
      self,
      targets: Union[List, int, IfcEntity],
      tol=0.01
  ) -> Tuple[TopoDS_Compound, TopoDS_Compound, List[IfcEntity],
             List[TopoDS_Shape]]:
    if not isinstance(targets, Iterable):
      targets = [targets]

    for idx in range(len(targets)):
      if isinstance(targets[idx], int):
        targets[idx] = self.ifc.by_id(int(targets[idx]))

    boundaries = set(chain.from_iterable([get_bounded_by(x) for x in targets]))
    walls = [x for x in boundaries if x is not None and x.is_a('IfcWall')]
    openings = list(
        set(chain.from_iterable([self.get_openings(w) for w in walls])))

    if len(targets) == 1 and not openings:
      shape = ifcopenshell.geom.create_shape(self.settings, targets[0]).geometry
      space = shape
    else:
      shape = TopoDS_Compound()
      builder_shape = TopoDS_Builder()
      builder_shape.MakeCompound(shape)

      space = TopoDS_Compound()
      builder_space = TopoDS_Builder()
      builder_space.MakeCompound(space)

      for target in targets:
        try:
          solid = self.create_geometry(target)
          builder_shape.Add(shape, solid)
          builder_space.Add(space, solid)
        except RuntimeError as e:
          raise ValueError('Shape 변환 실패:\n{}\n{}\n{}'.format(
              self.file_path, target, e))

      distance = [shapes_distance(shape, op, tol) for op in openings]
      openings = [
          opening for opening, dist in zip(openings, distance)
          if dist is not None and dist < tol
      ]

      assert all([dist is not None for dist in distance])

      for opening in openings:
        builder_shape.Add(shape, opening)

    return shape, space, walls, openings

  def simplify_space(self,
                     spaces: Union[List[int], List[IfcEntity]],
                     save_dir=None,
                     case_name=None,
                     threshold_volume=None,
                     threshold_dist=None,
                     threshold_angle=None,
                     fuzzy=0.0,
                     relative_threshold=True,
                     preserve_opening=True,
                     opening_volume=True):
    if save_dir:
      if not os.path.exists(save_dir):
        os.mkdir(save_dir)
      save_path = os.path.join(save_dir, case_name)
    else:
      save_path = None

    shape, space, walls, openings = self.convert_space(spaces)
    wall_names = self.component_code(walls, 'W')

    slabs = self.ifc.by_type('IfcSlab')
    slab_names = self.component_code(slabs, 'S')

    walls.extend(slabs)
    wall_names.extend(slab_names)

    def _geom(entity):
      try:
        shape = self.create_geometry(entity)
      except RuntimeError:
        shape = None

      return shape

    wall_shapes = [_geom(x) for x in walls]
    if any(x is None for x in wall_shapes):
      na_index = [i for i, x in enumerate(wall_shapes) if x is None]
      valid_index = [x for x in range(len(walls)) if x not in na_index]
      walls = [walls[x] for x in valid_index]
      wall_names = [wall_names[x] for x in valid_index]
      wall_shapes = [wall_shapes[x] for x in valid_index]

    if not preserve_opening:
      openings = None

    if threshold_volume is None:
      threshold_volume = self.threshold_internal_volume
    if threshold_dist is None:
      threshold_dist = self.threshold_surface_dist
    if threshold_angle is None:
      threshold_angle = self.threshold_surface_angle

    if not any([threshold_volume, threshold_dist, threshold_angle]):
      msg = '단순화 조건이 설정되지 않았습니다. 원본 형상을 저장합니다.'
      self._logger.info(msg)
      simplified = shape
      fused = shape
      is_simplified = False
    else:
      try:
        simplified = simplify_space(shape=shape,
                                    openings=openings,
                                    brep_deflection=self.brep_deflection,
                                    threshold_internal_volume=threshold_volume,
                                    threshold_internal_length=0.0,
                                    threshold_surface_dist=threshold_dist,
                                    threshold_surface_angle=threshold_angle,
                                    relative_threshold=relative_threshold,
                                    fuzzy=fuzzy,
                                    tol_bbox=self.tol_bbox,
                                    tol_cut=self.tol_cut,
                                    buffer_size=5.0)
        fused = fuse_compound(simplified)
      except RuntimeError as e:
        self._logger.error('단순화 실패:\n{}'.format(e))
        simplified = None
        fused = None

      is_simplified = (simplified is not None)

    valid_names, valid_walls = None, None
    if save_path and simplified is not None:
      try:
        DataExchange.write_stl_file(a_shape=simplified,
                                    filename=save_path + '.stl',
                                    linear_deflection=self.brep_deflection[0],
                                    angular_deflection=self.brep_deflection[1])
        compare = compare_shapes(shape, simplified)
        DataExchange.write_stl_file(a_shape=compare,
                                    filename=save_path + '_compare.stl',
                                    linear_deflection=self.brep_deflection[0],
                                    angular_deflection=self.brep_deflection[1])
      except (IOError, RuntimeError) as e:
        self._logger.error('stl 저장 실패:\n{}'.format(e))

      try:
        DataExchange.write_step_file(a_shape=simplified,
                                     filename=save_path + '.stp')
      except (IOError, RuntimeError) as e:
        self._logger.error('stp 저장 실패:\n{}'.format(e))

      try:
        write_obj(compound=simplified,
                  space=space,
                  openings=openings,
                  walls=wall_shapes,
                  obj_path=(save_path + '_interior.obj'),
                  deflection=self.brep_deflection,
                  wall_names=wall_names,
                  extract_interior=True,
                  extract_opening_volume=opening_volume)
        valid_names = write_obj(compound=simplified,
                                space=space,
                                openings=openings,
                                walls=wall_shapes,
                                obj_path=(save_path + '.obj'),
                                deflection=self.brep_deflection,
                                wall_names=wall_names,
                                extract_interior=False,
                                extract_opening_volume=opening_volume)
        valid_walls = [walls[wall_names.index(x)] for x in valid_names]
      except RuntimeError as e:
        self._logger.error('obj 저장 실패:\n{}'.format(e))

    info_original_geom = geometric_features(shape)
    if simplified is None:
      info_simplified_geom = None
    else:
      info_simplified_geom = geometric_features(simplified)

    if fused is None:
      info_fused_geom = None
    else:
      info_fused_geom = geometric_features(fused)

    info_simplification = {
        'threshold_volume': threshold_volume,
        'threshold_distance': threshold_dist,
        'threshold_angle': threshold_angle,
        'fuzzy': fuzzy,
        'relative_threshold': relative_threshold,
        'preserve_opening': preserve_opening,
        'extract_opening_volume': opening_volume,
        'is_simplified': is_simplified
    }
    info = {
        'simplification': info_simplification,
        'original_geometry': info_original_geom,
        'simplified_geometry': info_simplified_geom,
        'fused_geometry': info_fused_geom
    }

    if save_path:
      info_json = json.dumps(info, indent=4)
      with open(save_path + '_info.json', 'w') as f:
        f.write(info_json)

    results = {
        'original': shape,
        'simplified': simplified,
        'fused': fused,
        'space': space,
        'walls': walls if valid_walls is None else valid_walls,
        'wall_shapes': wall_shapes,
        'wall_names': wall_names if valid_names is None else valid_names,
        'openings': openings,
        'opening_names': ['Opening_{}'.format(x) for x in range(len(openings))],
        'info': info
    }

    return results

  def save_simplified_space(self, simplified: dict, path: str):
    is_simplified = simplified['info']['simplification']['is_simplified']

    if is_simplified:
      shape = simplified['simplified']
    else:
      shape = simplified['original']

    try:
      DataExchange.write_stl_file(a_shape=shape,
                                  filename=os.path.join(path, 'geometry.stl'),
                                  linear_deflection=self.brep_deflection[0],
                                  angular_deflection=self.brep_deflection[1])
      if is_simplified:
        compare = compare_shapes(simplified['original'], shape)
        DataExchange.write_stl_file(a_shape=compare,
                                    filename=os.path.join(
                                        path, 'geometry_compare.stl'),
                                    linear_deflection=self.brep_deflection[0],
                                    angular_deflection=self.brep_deflection[1])
    except (IOError, RuntimeError) as e:
      self._logger.error('stl 저장 실패:\n{}'.format(e))

    try:
      DataExchange.write_step_file(a_shape=shape,
                                   filename=os.path.join(path, 'geometry.stp'))
    except (IOError, RuntimeError) as e:
      self._logger.error('stp 저장 실패:\n{}'.format(e))

    try:
      extract_opening_volume = simplified['info']['simplification'][
          'extract_opening_volume']
      write_obj(compound=shape,
                space=simplified['space'],
                openings=simplified['openings'],
                walls=simplified['wall_shapes'],
                obj_path=os.path.join(path, 'geometry_interior.obj'),
                deflection=self.brep_deflection,
                wall_names=simplified['wall_names'],
                extract_interior=True,
                extract_opening_volume=extract_opening_volume)
      write_obj(compound=shape,
                space=simplified['space'],
                openings=simplified['openings'],
                walls=simplified['wall_shapes'],
                obj_path=os.path.join(path, 'geometry.obj'),
                deflection=self.brep_deflection,
                wall_names=simplified['wall_names'],
                extract_interior=False,
                extract_opening_volume=extract_opening_volume)
    except RuntimeError as e:
      self._logger.error('obj 저장 실패:\n{}'.format(e))

    return

  def openfoam_case(self,
                    simplified,
                    save_dir,
                    case_name,
                    openfoam_options: dict = None):
    """OpenFOAM 케이스 생성 및 저장

    Parameters
    ----------
    simplified : dict
        simplification 결과
    save_dir : PathLike
        저장 경로
    case_name : str
        저장할 케이스 이름 (save_dir/case_name에 결과 저장)
    openfoam_options : dict, optional
        options, by default None

    Raises
    ------
    ValueError
        OpenFOAM 설정 입력 오류
    """
    if openfoam_options is None:
      opt = self.openfoam_options.copy()
    else:
      opt = openfoam_options

    missing = [x for x in self.default_openfoam_options if x not in opt]
    if missing:
      msg = '다음 옵션에 기본값 적용: {}'.format(missing)
      self._logger.info(msg)
    for key in missing:
      opt[key] = self.default_openfoam_options[key]

    unused = [x for x in opt if x not in self.default_openfoam_options]
    if unused:
      self._logger.warning('적용되지 않는 OpenFOAM 옵션들: {}'.format(unused))

    solver = opt['solver']
    if not OpenFoamCase.is_supported_solver(solver):
      raise ValueError('지원하지 않는 solver입니다: {}'.format(solver))

    working_dir = os.path.normpath(os.path.join(save_dir, case_name))
    if not os.path.exists(working_dir):
      os.mkdir(working_dir)

    is_simplified = simplified['info']['simplification']['is_simplified']

    flag_external_zone = (opt['flag_external_zone'] and
                          opt['external_zone_size'] > 1)
    if flag_external_zone:
      if is_simplified:
        shape = simplified['simplified']
      else:
        shape = simplified['original']

      zone, zone_faces = make_external_zone(
          shape, buffer_size=opt['external_zone_size'])

      write_obj(compound=shape,
                space=simplified['space'],
                openings=simplified['openings'],
                walls=[self.create_geometry(x) for x in simplified['walls']],
                wall_names=simplified['wall_names'],
                obj_path=os.path.join(working_dir, 'geometry',
                                      'geometry_external.obj'),
                deflection=self.brep_deflection,
                additional_faces=zone_faces,
                extract_interior=False,
                extract_opening_volume=self.extract_opening_volume)

    # geometry 폴더에 저장된 obj 파일을 OpenFOAM 케이스 폴더 가장 바깥에 복사
    if flag_external_zone:
      geom_file = 'geometry_external.obj'
    elif opt['flag_interior_faces']:
      geom_file = 'geometry_interior.obj'
    else:
      geom_file = 'geometry.obj'

    geom_path = os.path.join(working_dir, 'geometry', geom_file)

    if os.path.exists(geom_path):
      copy2(src=geom_path, dst=os.path.join(working_dir, 'geometry.obj'))
    else:
      self._logger.warning('obj 추출에 실패했습니다.')

    # OpenFOAM 케이스 파일 생성
    open_foam_case = OpenFoamCase.from_template(solver=solver,
                                                save_dir=save_dir,
                                                name=case_name)

    # temperature = self.openfoam_temperature_info(
    #     walls=simplify_result['walls'],
    #     surface_names=simplify_result['wall_names'],
    #     external_temperature=condition['external_temperature'],
    #     heat_transfer_coefficient=condition['heat_transfer_coefficient'])

    # temperature_path = os.path.join(save_dir, case_name, '0', 'T')
    # assert os.path.exists(temperature_path)
    # with open(temperature_path, 'w') as f:
    #   f.writelines(temperature)

    wall_names = ['Surface_' + x for x in simplified['wall_names']]
    opening_names = simplified['opening_names']

    if OpenFoamCase.is_energy_available(solver) and opt['flag_energy']:
      bf_t = self.openfoam_temperature_dict(
          solver=solver,
          surface=simplified['walls'],
          surface_name=wall_names,
          opening_names=opening_names,
          min_score=self.minimum_match_score,
          temperature=opt.get('external_temperature', 300),
          heat_transfer_coefficient=opt.get('heat_transfer_coefficient', '1e8'))
      open_foam_case.change_boundary_field(variable='T', boundary_field=bf_t)

    if OpenFoamCase.is_turbulence_available(solver) and opt['flag_friction']:
      bf_nut = self.openfoam_rough_wall_nut_dict(
          solver=solver,
          surface=simplified['walls'],
          surface_name=wall_names,
          opening_names=opening_names,
          min_score=self.minimum_match_score)
      open_foam_case.change_boundary_field(variable='nut',
                                           boundary_field=bf_nut)

    self.openfoam_zero_boundary_field(case=open_foam_case,
                                      wall_names=wall_names,
                                      opening_names=opening_names,
                                      drop_fields=['T', 'nut'])

    if opt.get('max_cell_size', None):
      max_cell_size = opt['max_cell_size']
    else:
      geom_info = simplified['info'][('fused_geometry' if is_simplified else
                                      'original_geometry')]
      max_cell_size = geom_info['characteristic_length'] / opt['grid_resolution']

    mesh_dict = self.openfoam_cf_mesh_dict(
        max_cell_size=max_cell_size,
        min_cell_size=opt.get('min_cell_size', None),
        boundary_cell_size=opt.get('boundary_cell_size', None),
        boundary_layers_args=opt.get('boundary_layers', None))
    open_foam_case.change_foam_file_value('meshDict', mesh_dict)

    open_foam_case.save(overwrite=False, minimum=False)
    open_foam_case.save_shell()
    # TODO: decomposeParDict 설정
    return

  def component_code(self, entities, prefix, storey_prefix='F'):
    if self._storeys is None:
      self._storeys = self.ifc.by_type('IfcBuildingStorey')
      self._storeys.sort(key=lambda x: x.Elevation)
      self._storeys.insert(0, None)

    storeys = [get_storey(x) for x in entities]
    storeys_index = [self._storeys.index(x) for x in storeys]
    storeys_len = len(str(len(self._storeys)))

    def storey_id(x):
      if x is None:
        return 0
      else:
        return x.GlobalId

    entity_index = {storey_id(x): 1 for x in storeys}
    entity_len = {
        storey_id(x): len(str(storeys.count(x))) for x in set(storeys)
    }

    codes = [None for _ in range(len(entities))]

    for idx in range(len(entities)):
      storey = storeys[idx]
      code = '{}{:0{}d}{}{:0{}d}'.format(
          storey_prefix,
          storeys_index[idx],
          storeys_len,
          prefix,
          entity_index[storey_id(storey)],
          entity_len[storey_id(storey)],
      )
      entity_index[storey_id(storey)] += 1
      codes[idx] = code

    return codes


def get_storey(entity):
  contained_in_structure = entity.ContainedInStructure
  if len(contained_in_structure) == 0:
    return None

  assert len(contained_in_structure) == 1
  storey = contained_in_structure[0].RelatingStructure
  assert storey.is_a('IfcBuildingStorey')

  return storey


def get_bounded_by(space: IfcEntity):
  if not space.is_a('IfcSpace'):
    raise ValueError('Need IfcSpace, not {}'.format(space.is_a()))

  try:
    boundaries = [x.RelatedBuildingElement for x in space.BoundedBy]
  except AttributeError:
    boundaries = None

  return boundaries


def write_each_shapes(shape: TopoDS_Compound, save_dir, mkdir=False):
  """compound를 구성하는 각 shape을 저장

  Parameters
  ----------
  shape : TopoDS_Compound
      대상 compound
  save_dir : PathLike
      저장 위치
  """
  exp = TopologyExplorer(shape)
  if exp.number_of_solids() <= 1:
    raise ValueError('대상 shape에 solid가 없음')

  if not os.path.exists(save_dir) and mkdir:
    os.mkdir(save_dir)

  solids = exp.solids()
  for idx, solid in enumerate(solids):
    path = os.path.join(save_dir, '{}.stl'.format(idx))
    DataExchange.write_stl_file(a_shape=solid, filename=path)


def entity_name(entity: IfcEntity) -> str:
  if hasattr(entity, 'LongName') and entity.LongName is not None:
    res = '{} ({})'.format(entity.Name, entity.LongName)
  else:
    res = entity.Name
    assert isinstance(res, str)

  return res
