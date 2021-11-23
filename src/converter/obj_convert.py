import os
import re
import subprocess
from collections import defaultdict
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Collection, Iterable, Union

import utils

import numpy as np
from loguru import logger
from OCC.Core.BRepAlgo import BRepAlgo_Common
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape
from OCC.Extend.DataExchange import write_stl_file
from OCC.Extend.TopologyUtils import TopologyExplorer

from converter.geom_utils import get_boundingbox, shapes_distance
from converter.simplify import face_info
from OCCUtils.Common import GpropsFromShape
from OCCUtils.Construct import compound

BLENDER_FOUNDATION_PATH = Path(r'C:\Program Files\Blender Foundation')

EMPTY_BLEND_PATH = utils.DIR.RESOURCE.joinpath('empty.blend')
BLENDER_SCRIPT_PATH = utils.DIR.RESOURCE.joinpath('blender.py')
EMPTY_BLEND_PATH.stat()
BLENDER_SCRIPT_PATH.stat()


def find_blender_path(path=None):
  path = Path(path) if path else BLENDER_FOUNDATION_PATH
  if not path.exists():
    res = None
  else:
    if path.is_file() and path.name == 'blender.exe':
      res = path
    else:
      if path.is_file():
        path = path.parent

      ls = list(path.rglob('blender.exe'))
      if not ls:
        res = None
      else:
        res = sorted(ls)[-1]

  return res


def stl_to_obj(obj_path, blender_path: Union[None, str, Path], *args):
  obj_path = os.path.abspath(obj_path)
  stl_path = [os.path.abspath(x) for x in args]

  blender_path = find_blender_path(blender_path)
  logger.info('blender path: "{}"', blender_path)
  if blender_path is None or not blender_path.exists():
    raise FileNotFoundError('blender의 경로를 찾을 수 없습니다.')

  run_args = [
      blender_path.as_posix(),
      EMPTY_BLEND_PATH.as_posix(),
      '--background',
      '--python',
      BLENDER_SCRIPT_PATH.as_posix(),
      obj_path,
  ]
  run_args.extend(stl_path)
  subprocess.run(run_args, stdout=subprocess.PIPE)


def get_face_id(shape: TopoDS_Shape, hash_upper=1e8):
  faces = tuple(TopologyExplorer(shape).faces())
  ids = tuple(face.HashCode(int(hash_upper)) for face in faces)
  return faces, ids


def _bbox_ndarray(shape):
  return np.array(get_boundingbox(shape)).reshape([2, 3])


def is_same_face(op_faces, op_norm, op_center, srf_norm, srf_center, tol):
  vec = (op_norm.reshape([-1, 1, 3]) *
         (op_center.reshape([-1, 1, 3]) - srf_center))
  dot = np.abs(np.sum(vec, axis=2))
  is_coplanar = np.isclose(dot, 0.0, atol=tol)

  op_norm = op_norm / np.linalg.norm(op_norm, axis=1).reshape([-1, 1])
  srf_norm = srf_norm / np.linalg.norm(srf_norm, axis=1).reshape([-1, 1])
  is_parallel = np.isclose(
      np.abs(np.sum(op_norm.reshape([-1, 1, 3]) * srf_norm, axis=2)), 1.0)

  bbox = np.array([_bbox_ndarray(face) for face in op_faces])
  is_in = (bbox.reshape([-1, 2, 1, 3]) - srf_center) >= 0
  is_in = np.logical_xor(is_in[:, 0], is_in[:, 1])
  is_in = np.sum(is_in, axis=2) == 3

  return is_coplanar & is_parallel & is_in


class ObjConverter:

  def __init__(self,
               compound: TopoDS_Compound,
               space: TopoDS_Shape,
               openings: Union[None, Collection[TopoDS_Shape]] = None,
               walls: Union[None, Collection[TopoDS_Shape]] = None,
               wall_names: Union[None, Collection[str]] = None,
               additional_faces: dict = None,
               hash_upper=1e8):
    self._compound = compound
    self._space = space
    self._hash_upper = hash_upper
    self._additional = additional_faces

    # 최종적으로 추출할 face
    self._obj_interior = None
    self._obj_surface = None
    self._obj_opening = None

    self._comp_faces, self._comp_ids = get_face_id(self._compound,
                                                   self._hash_upper)

    if openings:
      self._op_faces = tuple(
          tuple(TopologyExplorer(opening).faces()) for opening in openings)
      self._opening_count = len(openings)
    else:
      self._op_faces = None
      self._opening_count = 0

    if walls:
      self._walls = tuple(walls)
    else:
      self._walls = None

    if wall_names:
      self._wall_names = tuple(wall_names)
    else:
      self._wall_names = None

  @property
  def surface(self):
    return self._obj_surface

  @property
  def opening(self):
    return self._obj_opening

  @property
  def interior(self):
    return self._obj_interior

  @property
  def ifc_walls(self):
    return self._walls

  def classify_compound(self):
    # 접하는 solid가 2개인 면은 interior, 1개인 면은 surface로 분리
    face_solid = defaultdict(list)

    for solid in TopologyExplorer(self._compound).solids():
      for face in TopologyExplorer(solid).faces():
        face_solid[face.HashCode(int(self._hash_upper))].append(solid)

    self._obj_interior = [
        f for f, i in zip(self._comp_faces, self._comp_ids)
        if len(face_solid[i]) > 1
    ]

    self._obj_surface = [
        f for f, i in zip(self._comp_faces, self._comp_ids)
        if len(face_solid[i]) == 1
    ]

  def classify_opening(self, tol=1e-4, opening_volume=False):
    if self._obj_interior is None:
      self.classify_compound()

    self._obj_opening = list()
    if self._op_faces:
      for faces in self._op_faces:
        if opening_volume:
          # opening의 부피를 추출
          if len(faces) != 6:
            logger.warning('Opening이 직육면체 모양이 아닙니다. 추출에 오류가 발생할 수 있습니다.')

          # 공간과 거리가 떨어져 있는 face (= inlet / outlet) 판단
          dist = [shapes_distance(face, self._space, tol) for face in faces]
          opening_face = [f for f, d in zip(faces, dist) if d > tol]

        else:
          # space와 겹치는 face만을 opening으로 추출
          common = [
              BRepAlgo_Common(face, self._space).Shape() for face in faces
          ]
          area = [GpropsFromShape(x).surface().Mass() for x in common]

          opening_face = [f for f, a in zip(faces, area) if a >= tol]

          if len(opening_face) > 1:
            opening_face = faces

        self._obj_opening.extend(opening_face)

      if self._obj_opening:
        # surface 중 opening face 제거
        _, op_norm, _, op_center, _ = face_info(self._obj_opening)
        _, sur_norm, _, sur_center, _ = face_info(self._obj_surface)

        # 표면(벽), 중 opening에 해당하는 면 추출
        surface_opening_mask = is_same_face(op_faces=self._obj_opening,
                                            op_norm=op_norm,
                                            op_center=op_center,
                                            srf_norm=sur_norm,
                                            srf_center=sur_center,
                                            tol=tol)

        distsq = np.square(op_center.reshape([-1, 1, 3]) - sur_center)
        surface_closest_opening = np.argmin(np.sum(distsq, axis=2), axis=0)
        surface_opening_idx = np.argwhere(np.any(surface_opening_mask, axis=0))

        if opening_volume:
          self._obj_opening = [(surface_closest_opening[i], f)
                               for i, f in enumerate(self._obj_surface)
                               if i in surface_opening_idx]
          self._obj_surface = [
              f for i, f in enumerate(self._obj_surface)
              if i not in surface_opening_idx
          ]
        else:
          # interior 중 opening에 해당하는 면 추출
          _, intr_norm, _, intr_center, _ = face_info(self._obj_interior)

          intr_opening_mask = is_same_face(op_faces=self._obj_opening,
                                           op_norm=op_norm,
                                           op_center=op_center,
                                           srf_norm=intr_norm,
                                           srf_center=intr_center,
                                           tol=tol)

          distsq = np.square(op_center.reshape([-1, 1, 3]) - intr_center)
          intr_closest_opening = np.argmin(np.sum(distsq, axis=2), axis=0)
          intr_opening_idx = np.argwhere(np.any(intr_opening_mask, axis=0))

          opening_from_surface = [(surface_closest_opening[i], f)
                                  for i, f in enumerate(self._obj_surface)
                                  if i in surface_opening_idx]
          opening_from_interior = [(-intr_closest_opening[i], f)
                                   for i, f in enumerate(self._obj_interior)
                                   if i in intr_opening_idx]
          self._obj_interior = [
              f for i, f in enumerate(self._obj_interior)
              if i not in intr_opening_idx
          ]
          self._obj_opening = opening_from_surface + opening_from_interior

          # 표면(벽) 중 opening 부피와 일치하는 표면 제거
          op_vol_faces = list(chain.from_iterable(self._op_faces))
          _, op_vol_norm, _, op_vol_center, _ = face_info(op_vol_faces)
          surface_opening_vol_mask = is_same_face(op_faces=op_vol_faces,
                                                  op_norm=op_vol_norm,
                                                  op_center=op_vol_center,
                                                  srf_norm=sur_norm,
                                                  srf_center=sur_center,
                                                  tol=tol)

          surface_opening_vol_idx = np.argwhere(
              np.any(surface_opening_vol_mask, axis=0))

          self._obj_surface = [
              f for i, f in enumerate(self._obj_surface)
              if i not in surface_opening_vol_idx
          ]

  def classify_walls(self, tol=1e-4):
    if self._walls is None:
      return

    # 각 surface에 대해 가장 가까운 wall의 번호 계산
    center_gp = [
        GpropsFromShape(x).surface().CentreOfMass() for x in self._obj_surface
    ]
    center_vertex = [BRepBuilderAPI_MakeVertex(x).Vertex() for x in center_gp]
    dist = np.array([
        [shapes_distance(c, w, tol) for w in self._walls] for c in center_vertex
    ])
    closest = np.argmin(dist, axis=1)

    if self._wall_names is None:
      length = len(str(int(np.max(closest))))
      names = ['{:0{}d}'.format(x, length) for x in closest]
    else:
      names = [self._wall_names[x] for x in closest]

    self._obj_surface = tuple((x, y) for x, y in zip(names, self._obj_surface))

    return

  def extract_faces(self,
                    obj_path,
                    blender_path,
                    tol,
                    extract_interior=True,
                    deflection=(0.9, 1.0)):
    if self._obj_surface is None:
      self.classify_opening()
      self.classify_walls(tol=tol)

    if len(self._obj_opening) < self._opening_count:
      self.classify_compound()
      self.classify_opening(opening_volume=True)
      self.classify_walls(tol=tol)

    with TemporaryDirectory() as temp_dir:
      path_list = list()

      if self._walls is None:
        surface_path = os.path.join(temp_dir, 'surface.stl')
        write_stl_file(a_shape=compound(self._obj_surface),
                       filename=surface_path,
                       linear_deflection=deflection[0],
                       angular_deflection=deflection[1])
        path_list.append(surface_path)
      else:
        surfaces = defaultdict(list)
        for name, surface in self._obj_surface:
          surfaces[name].append(surface)

        for name, shapes in surfaces.items():
          shape = shapes[0] if (len(shapes) == 1) else compound(shapes)
          path = os.path.join(temp_dir, 'Surface_{}.stl'.format(name))
          write_stl_file(a_shape=shape,
                         filename=path,
                         linear_deflection=deflection[0],
                         angular_deflection=deflection[1])
          path_list.append(path)

      if extract_interior and self._obj_interior:
        interior_path = os.path.join(temp_dir, 'interior.stl')
        write_stl_file(a_shape=compound(self._obj_interior),
                       filename=interior_path,
                       linear_deflection=deflection[0],
                       angular_deflection=deflection[1])
        path_list.append(interior_path)

      if self._obj_opening:
        openings = defaultdict(list)
        for idx, opening in self._obj_opening:
          openings[idx].append(opening)

        length = int(
            np.ceil(np.log10(np.max([x[0] for x in self._obj_opening] + [1]))))

        op_shapes = [compound(x) for x in openings.values()]

        # Opening을 좌표에 따라 번호 수정
        op_center = [
            GpropsFromShape(x).volume().CentreOfMass() for x in op_shapes
        ]
        op_center = np.array([[p.X(), p.Y(), p.Z()] for p in op_center])
        op_index = np.lexsort(op_center.T[::-1])

        for idx in range(len(openings)):
          path = os.path.join(temp_dir,
                              'opening_{:0{}d}.stl'.format(idx, length))
          write_stl_file(a_shape=op_shapes[op_index[idx]],
                         filename=path,
                         linear_deflection=deflection[0],
                         angular_deflection=deflection[1])
          path_list.append(path)

      if self._additional is not None:
        for face_name, face in self._additional.items():
          path = os.path.join(temp_dir, face_name + '.stl')
          write_stl_file(a_shape=face,
                         filename=path,
                         linear_deflection=deflection[0],
                         angular_deflection=deflection[1])
          path_list.append(path)

      stl_to_obj(obj_path, blender_path, *path_list)
      fix_surface_name(obj_path)

  def valid_surfaces(self):
    if (self._obj_surface is not None) and (self._wall_names is not None):
      names = list(set(x[0] for x in self._obj_surface))
      names.sort()
    else:
      names = None

    return names


def fix_surface_name(obj_path):
  """
  blender에서 저장한 표면 이름 (e.g. Surface_0_Surface_0_None)을 수정

  :param obj_path:
  :return: None
  """
  try:
    with open(obj_path) as f:
      geom = f.readlines()
  except FileNotFoundError:
    logger.error('Obj 표면 이름 변환 오류')
    return

  p = re.compile(r'((\w+_[a-z0-9]+)_){2}None',
                 re.IGNORECASE | re.MULTILINE | re.DOTALL)

  new_geom = [None for _ in range(len(geom))]

  for idx, line in enumerate(geom):
    new_geom[idx] = p.sub('\\2', line)

  assert all([x is not None for x in new_geom])

  if new_geom != geom:
    with open(obj_path, 'w') as f:
      f.writelines(new_geom)

  return


def write_obj(compound: TopoDS_Compound,
              space: TopoDS_Shape,
              openings: Union[None, Iterable[TopoDS_Shape]],
              walls: Union[None, Iterable[TopoDS_Shape]],
              obj_path: str,
              deflection,
              wall_names=None,
              additional_faces: dict = None,
              blender_path=None,
              extract_interior=True,
              extract_opening_volume=False,
              hash_upper=1e8,
              tol=1e-4):
  """

  :param compound: 단순화를 마친 compound
  :param space: opening을 포함하지 않은 공간의 shape
  :param openings: 대상 공간과 함께 추출하는 opening의 목록
  :param walls: 벽 목록 (슬라브 포함)
  :param obj_path: obj 파일 저장 경로
  :param deflection: deflection
  :param wall_names: 추출할 벽 표면의 이름 리스트
  :param additional_faces: 추가로 추출할 face dict {name: TopoDS_Face}
  :param blender_path: blender가 설치된 경로
  :param extract_interior: 내부 경계 면을 추출할지 여부
  :param extract_opening_volume: opening을 부피 혹은 면으로 추출하는 옵션
  :param hash_upper: face 구분에 사용되는 hash의 최댓값
  :param tol: opening face 구분에 사용되는 거리 허용 오차
  :return:
  """
  converter = ObjConverter(compound=compound,
                           space=space,
                           openings=openings,
                           walls=walls,
                           wall_names=wall_names,
                           additional_faces=additional_faces,
                           hash_upper=hash_upper)
  converter.classify_compound()
  converter.classify_opening(tol=tol, opening_volume=extract_opening_volume)
  converter.classify_walls(tol=tol)
  converter.extract_faces(obj_path=obj_path,
                          blender_path=blender_path,
                          tol=tol,
                          extract_interior=extract_interior,
                          deflection=deflection)

  return converter.valid_surfaces()
