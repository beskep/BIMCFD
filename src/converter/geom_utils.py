import os
from collections import OrderedDict
from collections.abc import Collection
from itertools import chain
from typing import List, Tuple, Union

import numpy as np
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BOPAlgo import BOPAlgo_MakerVolume, BOPAlgo_Splitter
from OCC.Core.BRep import BRep_Tool, BRep_Tool_Surface
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Fuse
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeSolid,
                                     BRepBuilderAPI_Sewing,
                                     BRepBuilderAPI_Transform)
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.Geom import Geom_Plane
from OCC.Core.gp import gp_Pnt, gp_Quaternion, gp_Trsf, gp_Vec
from OCC.Core.ShapeFix import ShapeFix_Shape
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import (TopoDS_Compound, TopoDS_Face, TopoDS_Shape,
                             TopoDS_Solid)
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Extend import DataExchange
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCCUtils.Common import GpropsFromShape
from OCCUtils.Construct import compound, face_normal, make_plane
from OCCUtils.face import Face
from tqdm import tqdm

# todo: tqdm 지우기


def make_box(*args):
  box = BRepPrimAPI_MakeBox(*args)
  box.Build()

  return box.Shape()


def get_boundingbox(shape, tol=1e-4):
  bbox = Bnd_Box()
  bbox.SetGap(tol)
  brepbndlib_Add(shape, bbox)

  return bbox.Get()


def common_volume(shape1, shape2):
  common = BRepAlgoAPI_Common(shape1, shape2).Shape()

  if common is None:
    res = 0.0
  else:
    res = GpropsFromShape(common).volume().Mass()

  return res


def fix_shape(shape: TopoDS_Shape, precision=None):
  fix = ShapeFix_Shape()
  fix.Init(shape)

  if precision:
    fix.Precision(precision)

  fix.Perform()

  return fix.Shape()


def _maker_volume_sequential_helper(shape, plane, fuzzy):
  mv = BOPAlgo_MakerVolume()
  mv.SetRunParallel(True)

  if fuzzy:
    mv.SetFuzzyValue(fuzzy)

  mv.AddArgument(shape)
  mv.AddArgument(plane)
  mv.Perform()

  return mv.Shape()


def maker_volume_sequential(shape,
                            planes: Collection,
                            fuzzy=0.0,
                            verbose=False):
  it = tqdm(planes) if verbose else planes

  for plane in it:
    shape = _maker_volume_sequential_helper(shape, plane, fuzzy)

  return shape


def maker_volume(shapes: Collection, boundary: TopoDS_Shape = None, fuzzy=0.0):
  """
  BOPAlgo_MakerVolume
  https://github.com/tpaviot/pythonocc-core/issues/554

  :param shapes: Volume을 만드는데 사용되는 shape의 목록
  :param boundary: Optional, 지정 시 결과 중
  boundary 내부에 존재하는 solid만 모아 compound를 만들어 반환
  :param fuzzy: BOPAlgo_MakerVolume의 fuzzy 옵션
  :return:
  """
  if len(shapes) > 1:
    shapes = np.array(shapes)
    _, ind = np.unique([shape.HashCode(int(1e8)) for shape in shapes],
                       return_index=True)
    shapes = shapes[ind]

  mv = BOPAlgo_MakerVolume()
  mv.SetRunParallel(True)

  if fuzzy:
    mv.SetFuzzyValue(fuzzy)

  ls = TopTools_ListOfShape()

  for shape in shapes:
    ls.Append(shape)

  mv.SetArguments(ls)
  mv.Perform()
  result = fix_shape(mv.Shape())

  if boundary is not None:
    solids = list(TopologyExplorer(result).solids())
    is_in = [
        _is_in(boundary, GpropsFromShape(x).volume().CentreOfMass(), on=False)
        for x in solids
    ]
    result = compound([s for s, i in zip(solids, is_in) if i])

  return result


def _is_in(solid, pnt: gp_Pnt, tol=0.001, on=True) -> bool:
  """
  pnt가 solid 내부에 있는지 여부를 반환

  :param solid: TopoDS_Shape
  :param pnt: gp_Pnt
  :return: bool
  """
  classifier = BRepClass3d_SolidClassifier(solid)
  classifier.Perform(pnt, tol)
  result = classifier.State()
  result = result in [TopAbs_IN, TopAbs_ON] if on else result == TopAbs_IN
  classifier.Destroy()

  return result


def calc_split_vol_ratio(split_original: TopoDS_Compound,
                         split_buffer: TopoDS_Compound) -> np.ndarray:
  """
  major plane으로 나눈 원본 shape과 buffer의 부피 비율 계산

  :param split_original: major plane으로 나눈 원본 shape
  :param split_buffer: major plane으로 나눈 buffer
  :return: 표면 단순화 결과에 포함 여부를 결정하는 부피 비율
  """
  # todo: 오류 발생 시 solids_original에서 평면 (mass가 음수)을 제거하게 수정
  solids_original = list(TopologyExplorer(split_original).solids())
  solids_buffer = list(TopologyExplorer(split_buffer).solids())

  assert len(solids_original) > 0
  assert len(solids_buffer) > 0

  props_original = [GpropsFromShape(s).volume() for s in solids_original]
  props_buffer = [GpropsFromShape(s).volume() for s in solids_buffer]

  vol_original = np.array([p.Mass() for p in props_original])
  vol_buffer = np.array([p.Mass() for p in props_buffer])
  vol_buffer[vol_buffer == 0.0] = np.nan

  center_original_gppnt = np.array([p.CentreOfMass() for p in props_original])
  center_original = np.array(
      [[p.X(), p.Y(), p.Z()] for p in center_original_gppnt])

  center_buffer = [p.CentreOfMass() for p in props_buffer]
  center_buffer = np.array([[p.X(), p.Y(), p.Z()] for p in center_buffer])

  center_buffer_ = np.repeat(center_buffer, center_original.shape[0], axis=0)
  center_buffer_ = center_buffer_.reshape(
      [center_buffer.shape[0], center_original.shape[0], 3])

  center_original_ = np.tile(center_original, (center_buffer.shape[0], 1))
  center_original_ = center_original_.reshape(
      [center_buffer.shape[0], center_original.shape[0], 3])

  # 각 buffer의 solid에 대한 original의 solid의 거리
  # todo: 거리 말고 다른 판단방법으로 바꾸기
  dist = np.sum(np.square(center_buffer_ - center_original_), axis=2)

  dist_argsort = np.argsort(dist, axis=1)
  matching_idx = [
      _find_matching_solid(argsort, solid, center_original_gppnt)
      for argsort, solid in zip(dist_argsort, solids_buffer)
  ]

  vol_ratio = np.array(
      [np.nan if np.isnan(x) else vol_original[x] for x in matching_idx])
  assert vol_ratio.shape == vol_buffer.shape
  vol_ratio /= vol_buffer
  vol_ratio[np.isnan(vol_ratio)] = 0.0

  # closest = np.argmin(dist, axis=1)
  # is_in = [_is_in(solid, p) for solid, p
  #          in zip(solids_buffer, center_original_gppnt[closest])]
  #
  # vol_ratio = vol_original[closest] / vol_buffer
  # vol_ratio[np.logical_not(is_in)] = 0.0

  assert np.all(vol_ratio >= -1e-4)
  assert np.all(vol_ratio <= 1.001)

  return vol_ratio


def _find_matching_solid(argsort, buffer_solid, original_center):
  for idx in argsort:
    if _is_in(buffer_solid, original_center[idx], on=False):
      return idx

  return np.nan


def get_face_vertices(face: TopoDS_Face) -> List[List[gp_Pnt]]:
  """
  대상 face의 mesh face별 vertic 좌표를 반환

  :param face: 대상 face
  :return: 세 개의 gp_pnts로 표현된 각 mesh face의 vertex 좌표
  """

  trsf: gp_Trsf = face.Location().Transformation()

  loc = TopLoc_Location()
  trsf_tri: gp_Trsf = loc.Transformation()
  triangulation = BRep_Tool().Triangulation(face, loc)

  if triangulation is None:
    raise ValueError('Brep 메쉬가 형성되지 않았습니다.')

  tri_count = triangulation.NbTriangles()
  nodes = triangulation.Nodes()
  triangles = triangulation.Triangles()

  index = [triangles.Value(i).Get() for i in range(1, tri_count + 1)]
  pnts = [[nodes.Value(i[0]),
           nodes.Value(i[1]),
           nodes.Value(i[2])] for i in index]
  pnts = [[p.Transformed(trsf).Transformed(trsf_tri)
           for p in node_pnts]
          for node_pnts in pnts]

  return pnts


def get_faces_vertices(faces: list) -> List[List[gp_Pnt]]:
  """
  곡면 face 목록을 받아서 triangulation 실행 후 각 점의 좌표 반환

  :param faces:
  :return: list[[gp_pnt, gp_pnt, gp_pnt]
  """
  vertices = [get_face_vertices(face) for face in faces]
  vertices = list(chain.from_iterable(vertices))

  return vertices


def planes_from_faces(faces: List[TopoDS_Shape]) -> List[Geom_Plane]:
  """
  각 face를 포함하는 plane 반환

  :param faces: 대상 face 리스트
  :return: List[Geom_Plane]
  """
  surfaces = [BRep_Tool_Surface(face) for face in faces]
  planes = [Geom_Plane.DownCast(surface) for surface in surfaces]

  return planes


def split_by_own_faces(shape: TopoDS_Shape, face_range=1000.0):
  # splitter = GEOMAlgo_Splitter()
  splitter = BOPAlgo_Splitter()
  splitter.AddArgument(shape)

  faces = TopologyExplorer(shape).faces()

  for face in faces:
    face_util = Face(face)
    _, center = face_util.mid_point()
    norm = face_normal(face_util)
    face_split = make_plane(center=center,
                            vec_normal=gp_Vec(norm.XYZ()),
                            extent_x_min=-face_range,
                            extent_x_max=face_range,
                            extent_y_min=-face_range,
                            extent_y_max=face_range)
    splitter.AddTool(face_split)

  splitter.Perform()

  return splitter.Shape()


def split_by_faces(shape: TopoDS_Shape,
                   faces: List[TopoDS_Face],
                   parallel=False,
                   step=False,
                   verbose=False):
  """
  shape을 각 faces로 나눈 결과를 반환

  :param shape: 분할 대상 shape
  :param faces: shape을 분할할 face의 List
  :param parallel: GEOMAlgo_Splitter의 SetParallelMode 여부
  :param step: True일 경우 각 face를 통해 순차적으로 slit 시행.
  False일 경우 한번에 split.
  :param verbose:

  :return: TopoDS_Compound
  """
  # splitter = GEOMAlgo_Splitter()
  splitter = BOPAlgo_Splitter()

  if parallel:
    splitter.SetParallelMode(True)

  if not step:
    splitter.AddArgument(shape)

    for face in faces:
      splitter.AddTool(face)

    splitter.Perform()
    result = splitter.Shape()
  else:
    it = tqdm(faces) if verbose else faces

    for face in it:
      splitter.AddArgument(shape)
      splitter.AddTool(face)
      splitter.Perform()
      shape = splitter.Shape()
      splitter.Clear()

    result = shape

  return result


def sew_faces(shape: TopoDS_Shape, tol=1e-3) -> TopoDS_Solid:
  """
  shape의 모든 face로 형성되는 solid 반환

  :param shape: 대상 shape
  :param tol: BRepBuilderAPI_Sewing의 허용 오차
  :return: TopoDS_Solid
  """
  sew = BRepBuilderAPI_Sewing(tol)

  for face in TopologyExplorer(shape).faces():
    sew.Add(face)

  sew.Perform()
  solid = BRepBuilderAPI_MakeSolid(sew.SewedShape()).Solid()

  return solid


def fuse_compound(target: Union[List, TopoDS_Shape]):
  if isinstance(target, TopoDS_Shape):
    target = list(TopologyExplorer(target).solids())

  if not target:
    result = None
  elif len(target) == 1:
    result = target[0]
  else:
    fuse = BRepAlgoAPI_Fuse()
    arguments = TopTools_ListOfShape()
    tools = TopTools_ListOfShape()

    arguments.Append(target[0])

    for solid in target[1:]:
      tools.Append(solid)

    fuse.SetArguments(arguments)
    fuse.SetTools(tools)
    fuse.SetRunParallel(True)
    fuse.Build()

    result = fuse.Shape() if fuse.IsDone() else None

  return result


def compare_shapes(original: TopoDS_Shape, simplified: TopoDS_Shape):
  bbox_original = get_boundingbox(original)
  bbox_simplified = get_boundingbox(simplified)

  trsf_x = np.average([
      np.abs(bbox_original[3] - bbox_original[0]),
      np.abs(bbox_simplified[3] - bbox_simplified[0])
  ])

  trsf = gp_Trsf()
  trsf.SetTranslation(gp_Vec(1.5 * trsf_x, 0.0, 0.0))

  transform = BRepBuilderAPI_Transform(trsf)
  transform.Perform(simplified, False)
  simplified = transform.ModifiedShape(simplified)

  compare = compound([original, simplified])

  return compare


def geometric_features(shape: TopoDS_Shape):
  gprops = GpropsFromShape(shape)
  exp = TopologyExplorer(shape)

  volume = gprops.volume().Mass()
  area = gprops.surface().Mass()

  features = OrderedDict([
      ('volume', volume),
      ('area', area),
      ('characteristic_length', volume / area),
      ('solid_count', exp.number_of_solids()),
      ('face_count', exp.number_of_faces()),
      ('edge_count', exp.number_of_edges()),
      ('vertex_count', exp.number_of_vertices()),
  ])

  return features


def shapes_distance(shape1: TopoDS_Shape, shape2: TopoDS_Shape,
                    deflection: float):
  dist = BRepExtrema_DistShapeShape(shape1, shape2, deflection)

  return dist.Value() if dist.IsDone() else None


def make_external_zone(shape: TopoDS_Shape,
                       buffer_size=5,
                       vertical_dim=2) -> Tuple[TopoDS_Shape, dict]:
  """주어진 shape을 둘러싸는 external zone 생성

  Parameters
  ----------
  shape
      대상 shape
  buffer_size
      external zone의 여유 공간 크기. 대상 shape의 높이 (H)의 배수.
  vertical_dim
      연직 방향 dimension (x: 0, y: 1, z: 2),
      ifcopenshell의 기본 설정은 z 방향 (2)

  Returns
  -------
  tuple
      TopoDS_Shape, {face_name: TopoDS_Face}
  """
  bbox = np.array(get_boundingbox(shape)).reshape([2, 3])
  height = np.abs(bbox[0, vertical_dim] - bbox[1, vertical_dim])

  zone_pnts = [
      np.min(bbox, axis=0) - buffer_size * height,
      np.max(bbox, axis=0) + buffer_size * height
  ]
  zone_pnts[0][vertical_dim] = np.min(bbox[:, vertical_dim])  # 바닥

  zone_gp_pnts = [gp_Pnt(*xyz) for xyz in zone_pnts]
  zone = make_box(*zone_gp_pnts)

  faces = list(TopologyExplorer(zone).faces())
  gp_center = [GpropsFromShape(x).surface().CentreOfMass() for x in faces]
  center = np.array([[p.X(), p.Y(), p.Z()] for p in gp_center])

  arg_sort = np.argsort(center[:, vertical_dim])
  ground = faces[arg_sort[0]]
  ceiling = faces[arg_sort[-1]]
  vertical = [faces[x] for x in arg_sort[1:-1]]

  zone_dict = {'External_' + str(i): f for i, f in enumerate(vertical)}
  zone_dict['Ground'] = ground
  zone_dict['Ceiling'] = ceiling

  return zone, zone_dict


def align_model(shape: TopoDS_Shape) -> TopoDS_Shape:
  """건물 모델의 주요 표면이 xyz 평면과 수평하도록 회전,
  원점 근처에 위치하도록 평행이동

  Parameters
  ----------
  shape : TopoDS_Shape
      target shape

  Returns
  -------
  TopoDS_Shape
      aligned shape
  """
  if not isinstance(shape, TopoDS_Shape):
    raise TypeError('Need TopoDS_Shape, got {}'.format(type(shape)))

  faces = list(TopologyExplorer(shape).faces())
  faces_area = [GpropsFromShape(x).surface().Mass() for x in faces]
  area_argsort = np.argsort(faces_area)
  faces_sorted = [faces[x] for x in area_argsort[::-1]]

  for face in faces_sorted:
    plane = planes_from_faces([face])[0]
    if plane is None:
      continue

    norm_gp_dir = plane.Axis().Direction()
    norm = np.array([norm_gp_dir.X(), norm_gp_dir.Y(), norm_gp_dir.Z()])

    zero_axis_count = np.sum(np.isclose(norm, 0.0))

    if zero_axis_count < 2:
      arg_sort = np.argsort(norm)
      align_to = np.array([0.0, 0.0, 0.0])
      align_to[arg_sort[-1]] = 1.0

      quaternion = gp_Quaternion(gp_Vec(*norm), gp_Vec(*align_to))
      trsf = gp_Trsf()
      trsf.SetRotation(quaternion)
      brep_trnf = BRepBuilderAPI_Transform(shape, trsf, False)
      rotated = brep_trnf.Shape()

      return rotated

  return None


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
