import os
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain, combinations
from pathlib import Path
from shutil import copy2
from typing import Any, Generator, Iterable, List, Optional, Tuple, Union

import ifcopenshell
import ifcopenshell.geom
from ifcopenshell import entity_instance as IfcEntity
from loguru import logger
from OCC.Core.TopoDS import TopoDS_Builder, TopoDS_Compound, TopoDS_Shape
from OCC.Extend import DataExchange

from converter import geom_utils
from converter.ifc_utils import get_bounded_by, get_storey
from converter.obj_convert import ObjConverter, write_obj, write_obj_from_dict
from converter.openfoam_converter import OpenFoamConverter
from converter.simplify import simplify_space

from .converter import Converter


@dataclass
class BuildingShapes:
  shape: TopoDS_Compound
  space: TopoDS_Compound

  wall_entities: List[IfcEntity]
  wall_shapes: List[TopoDS_Shape]
  wall_names: List[str]

  opening_shapes: Optional[List[TopoDS_Shape]]
  opening_names: Optional[List[str]] = None

  def update_opening_names(self):
    count = len(self.opening_shapes) if self.opening_shapes else 0
    self.opening_names = [f'Opening_{x}' for x in range(count)]

  def set_valid_walls(self, names):
    indices = [i for i, x in enumerate(self.wall_names) if x in names]
    if not indices:
      raise ValueError('No valid walls')

    self.wall_entities = [self.wall_entities[x] for x in indices]
    self.wall_shapes = [self.wall_shapes[x] for x in indices]
    self.wall_names = [self.wall_names[x] for x in indices]


class IfcConverter(Converter):

  def __init__(self,
               path: str,
               wall_types=('IfcWall',),
               slab_types=('IfcSlab',)):
    self._file_path = os.path.normpath(path)
    self._ifc = ifcopenshell.open(self._file_path)
    self.__geom_settings = ifcopenshell.geom.settings()
    self.__geom_settings.set(self.__geom_settings.USE_PYTHON_OPENCASCADE, True)

    self._ifc_types_wall = wall_types
    self._ifc_types_slab = slab_types
    self._storeys = None

    self._brep_deflection = (0.9, 0.5)

  @property
  def ifc(self) -> ifcopenshell.file:
    return self._ifc

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
      self._brep_deflection = tuple(value)
    else:
      raise TypeError

  def create_geometry(self, entity: IfcEntity) -> TopoDS_Shape:
    return ifcopenshell.geom.create_shape(self.__geom_settings, entity).geometry

  def _get_entities(self, types: Iterable[str]):
    return chain.from_iterable((self.ifc.by_type(x) for x in types))

  def get_relating_space(
      self,
      target: Union[int, IfcEntity],
  ) -> Generator[IfcEntity, None, None]:
    if isinstance(target, int):
      target = self._ifc.by_id(target)

    for pb in target.ProvidesBoundaries:
      yield pb.RelatingSpace

  def get_space_wall_dict(self) -> defaultdict:
    """
    Returns
    -------
    defaultdict
        {space: List[wall]}
    """
    spaces_walls = defaultdict(set)

    walls = self._get_entities(self._ifc_types_wall)
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
    assert any(wall.is_a(x) for x in self._ifc_types_wall)

    for opening in wall.HasOpenings:
      try:
        yield self.create_geometry(opening.RelatedOpeningElement)
      except RuntimeError:
        logger.exception('Opening 추출 실패: {}, {}', self.file_path,
                         opening.RelatedOpeningElement)

  def convert_space(
      self,
      targets: Union[List, int, IfcEntity],
      tol=0.01
  ) -> Tuple[TopoDS_Compound, TopoDS_Compound, List[IfcEntity],
             List[TopoDS_Shape]]:
    if not isinstance(targets, Iterable):
      targets = [targets]
    targets = [self.ifc.by_id(x) if isinstance(x, int) else x for x in targets]

    boundaries = set(chain.from_iterable([get_bounded_by(x) for x in targets]))
    walls = [x for x in boundaries if x is not None and x.is_a('IfcWall')]
    openings = list(
        set(chain.from_iterable([self.get_openings(w) for w in walls])))

    if len(targets) == 1 and not openings:
      shape: Any = self.create_geometry(targets[0])
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
          raise ValueError('Shape 변환 실패:\n{}\n{}'.format(
              self.file_path, target)) from e

      distance = [geom_utils.shapes_distance(shape, op, tol) for op in openings]
      openings = [
          opening for opening, dist in zip(openings, distance)
          if dist is not None and dist < tol
      ]

      assert all(dist is not None for dist in distance)

      for opening in openings:
        builder_shape.Add(shape, opening)

    return shape, space, walls, openings

  def _building_shapes(self, spaces: Union[List[int], List[IfcEntity]]):
    openings: Optional[List[TopoDS_Shape]]
    shape, space, walls, openings = self.convert_space(spaces)
    wall_names = self.component_code(walls, prefix='W')

    slabs = list(self._get_entities(self._ifc_types_slab))
    slab_names = self.component_code(slabs, prefix='S')

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

    return BuildingShapes(shape=shape,
                          space=space,
                          wall_entities=walls,
                          wall_shapes=wall_shapes,
                          wall_names=wall_names,
                          opening_shapes=openings)

  def simplify_space(self,
                     spaces: Union[List[int], List[IfcEntity]],
                     threshold_volume=0.0,
                     threshold_dist=0.0,
                     threshold_angle=0.0,
                     relative_threshold=True,
                     preserve_opening=True,
                     opening_volume=True):
    shapes = self._building_shapes(spaces=spaces)

    if not preserve_opening:
      shapes.opening_shapes = None

    if not any([threshold_volume, threshold_dist, threshold_angle]):
      logger.info('단순화 조건이 설정되지 않았습니다. 원본 형상을 저장합니다.')
      simplified: Optional[TopoDS_Compound] = shapes.shape
      fused: Optional[TopoDS_Compound] = shapes.shape
      is_simplified = False
    else:
      try:
        simplified = simplify_space(shape=shapes.shape,
                                    openings=shapes.opening_shapes,
                                    brep_deflection=self.brep_deflection,
                                    threshold_internal_volume=threshold_volume,
                                    threshold_internal_length=0.0,
                                    threshold_surface_dist=threshold_dist,
                                    threshold_surface_angle=threshold_angle,
                                    relative_threshold=relative_threshold,
                                    buffer_size=5.0)
        fused = geom_utils.fuse_compound(simplified)
      except RuntimeError as e:
        logger.error('단순화 실패: {}', e)
        simplified = None
        fused = None

      is_simplified = (simplified is not None)

    info = {
        'simplification': {
            'threshold_volume': threshold_volume,
            'threshold_distance': threshold_dist,
            'threshold_angle': threshold_angle,
            'relative_threshold': relative_threshold,
            'preserve_opening': preserve_opening,
            'extract_opening_volume': opening_volume,
            'is_simplified': is_simplified
        },
        'original_geometry': geom_utils.geometric_features(shapes.shape)
    }
    if is_simplified:
      info['simplified_geometry'] = geom_utils.geometric_features(simplified)
      info['fused_geometry'] = geom_utils.geometric_features(fused)
    else:
      info['simplified_geometry'] = info['original_geometry']
      info['fused_geometry'] = info['original_geometry']

    objcnv = ObjConverter(
        compound=(simplified if is_simplified else shapes.shape),
        space=shapes.space,
        openings=shapes.opening_shapes,
        walls=shapes.wall_shapes,
        wall_names=shapes.wall_names)
    objcnv.classify_compound()
    objcnv.classify_opening(opening_volume=True)
    objcnv.classify_walls()
    valid_wall_names = objcnv.valid_surfaces()

    if valid_wall_names:
      shapes.set_valid_walls(valid_wall_names)

    shapes.update_opening_names()
    results = {
        'original': shapes.shape,
        'simplified': simplified,
        'fused': fused,
        'space': shapes.space,
        'walls': shapes.wall_entities,
        'wall_shapes': shapes.wall_shapes,
        'wall_names': shapes.wall_names,
        'openings': shapes.opening_shapes,
        'opening_names': shapes.opening_names,
        'info': info
    }

    return results

  def save_simplified_space(self, simplified: dict, path: str):
    is_simplified = simplified['info']['simplification']['is_simplified']

    if is_simplified:
      shape = simplified['simplified']

      original = simplified['original']
      DataExchange.write_stl_file(a_shape=original,
                                  filename=os.path.join(
                                      path, 'original_geometry.stl'),
                                  linear_deflection=self.brep_deflection[0],
                                  angular_deflection=self.brep_deflection[1])
    else:
      shape = simplified['original']

    try:
      DataExchange.write_stl_file(a_shape=shape,
                                  filename=os.path.join(path, 'geometry.stl'),
                                  linear_deflection=self.brep_deflection[0],
                                  angular_deflection=self.brep_deflection[1])
      if is_simplified:
        compare = geom_utils.compare_shapes(simplified['original'], shape)
        DataExchange.write_stl_file(a_shape=compare,
                                    filename=os.path.join(
                                        path, 'geometry_compare.stl'),
                                    linear_deflection=self.brep_deflection[0],
                                    angular_deflection=self.brep_deflection[1])
    except (IOError, RuntimeError) as e:
      logger.error('stl 저장 실패: {}', e)

    # try:
    #   DataExchange.write_step_file(a_shape=shape,
    #                                filename=os.path.join(path, 'geometry.stp'))
    # except (IOError, RuntimeError) as e:
    #   logger.error('stp 저장 실패: {}', e)

    try:
      extract_opening_volume = simplified['info']['simplification'][
          'extract_opening_volume']
      # write_obj(compound=shape,
      #           space=simplified['space'],
      #           openings=simplified['openings'],
      #           walls=simplified['wall_shapes'],
      #           obj_path=os.path.join(path, 'geometry_interior.obj'),
      #           deflection=self.brep_deflection,
      #           wall_names=simplified['wall_names'],
      #           extract_interior=True,
      #           extract_opening_volume=extract_opening_volume)
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
      logger.error('obj 저장 실패: {}', e)

  def _write_openfoam_object(self, options: dict, simplified: dict,
                             working_dir: Path):
    flag_external_zone = (options['flag_external_zone'] and
                          options['external_zone_size'] > 1)
    if flag_external_zone:
      shape = (simplified.get('space', None) or simplified['simplified'] or
               simplified['original'])

      _, external_faces = geom_utils.make_external_zone(
          shape, buffer=options['external_zone_size'])
      write_obj_from_dict(
          faces=external_faces,
          obj_path=working_dir.joinpath('geometry/geometry_external.obj'),
          deflection=self.brep_deflection)

    # geometry 폴더에 저장된 obj 파일을 OpenFOAM 케이스 폴더 가장 바깥에 복사
    if flag_external_zone:
      geom_file = 'geometry_external.obj'
    elif options['flag_interior_faces']:
      geom_file = 'geometry_interior.obj'
    else:
      geom_file = 'geometry.obj'

    geom_path = os.path.join(working_dir, 'geometry', geom_file)

    if os.path.exists(geom_path):
      copy2(src=geom_path, dst=os.path.join(working_dir, 'geometry.obj'))
    else:
      logger.warning('obj 추출에 실패했습니다.')

  def component_code(self, entities, prefix, storey_prefix='F'):
    if self._storeys is None:
      self._storeys = self.ifc.by_type('IfcBuildingStorey')
      self._storeys.sort(key=lambda x: x.Elevation)
      self._storeys.insert(0, None)

    storeys = [get_storey(x) for x in entities]
    storeys_index = [self._storeys.index(x) for x in storeys]
    storeys_len = len(str(len(self._storeys)))

    def storey_id(x):
      return 0 if x is None else x.GlobalId

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
