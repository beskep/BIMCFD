from collections import OrderedDict
from collections.abc import Collection, Iterable
from itertools import chain
from typing import List, Tuple, Union

import numpy as np
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BOPAlgo import BOPAlgo_MakerVolume, BOPAlgo_Splitter
from OCC.Core.BRep import BRep_Tool, BRep_Tool_Surface
from OCC.Core.BRepAlgoAPI import (BRepAlgoAPI_Common, BRepAlgoAPI_Cut,
                                  BRepAlgoAPI_Fuse)
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeSolid,
                                     BRepBuilderAPI_Sewing,
                                     BRepBuilderAPI_Transform)
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.Geom import Geom_Plane
from OCC.Core.gp import gp_Pnt, gp_Trsf, gp_Vec
from OCC.Core.ShapeFix import ShapeFix_Shape
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import (TopoDS_Compound, TopoDS_Face, TopoDS_Shape,
                             TopoDS_Solid)
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Extend.TopologyUtils import TopologyExplorer
from scipy.spatial import ConvexHull
from tqdm import tqdm

from OCCUtils.Common import GpropsFromShape
from OCCUtils.Construct import compound, face_normal, make_plane
from OCCUtils.face import Face


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


def _calc_split_vol_ratio(split_original: TopoDS_Compound,
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


def _get_face_vertices(face: TopoDS_Face) -> List[List[gp_Pnt]]:
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


def _get_faces_vertices(faces: list) -> List[List[gp_Pnt]]:
  """
  곡면 face 목록을 받아서 triangulation 실행 후 각 점의 좌표 반환

  :param faces:
  :return: list[[gp_pnt, gp_pnt, gp_pnt]
  """
  vertices = [_get_face_vertices(face) for face in faces]
  vertices = list(chain.from_iterable(vertices))

  return vertices


def _planes_from_faces(faces: List[TopoDS_Shape]) -> List[Geom_Plane]:
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


def flat_face_info(faces: List[TopoDS_Shape]):
  """
  평면 (face)의 목록을 받아서 표면 단순화에 필요한 face의 정보 반환

  :param faces: face 리스트
  :return: (면적, 법선 벡터, 법선 벡터 (gp_Vec), 중앙점, 중앙점 (gp_Pnt), 평면)
  """

  planes = _planes_from_faces(faces)
  norm_gp_dir = [pln.Axis().Direction() for pln in planes]
  norm_gp_vec = [gp_Vec(d) for d in norm_gp_dir]
  norms = np.array([[d.X(), d.Y(), d.Z()] for d in norm_gp_dir])

  # 각 face의 면적 추출
  areas = np.array([GpropsFromShape(face).surface().Mass() for face in faces])

  center_gp_pnts = [Face(face).mid_point()[1] for face in faces]
  center = np.array([[p.X(), p.Y(), p.Z()] for p in center_gp_pnts])

  if not areas.size:
    areas = np.empty((0, 3))
    norms = np.empty((0, 3))
    center = np.empty((0, 3))

  return areas, norms, norm_gp_vec, center, center_gp_pnts


def curved_face_info(faces: List[TopoDS_Shape]):
  """
  곡면 (face)의 목록을 받아서 표면 단순화에 필요한 face의 정보 반환

  :param faces: face 리스트
  :return: (면적, 법선 벡터, 법선 벡터 (gp_Vec), 중앙점, 중앙점 (gp_Pnt), 평면)
  """
  vertices_gp = _get_faces_vertices(faces)
  vectors = [[gp_Vec(x, y), gp_Vec(x, z)] for x, y, z in vertices_gp]
  norm_gp_vec = [v.Crossed(w) for v, w in vectors]
  norms = np.array([[v.X(), v.Y(), v.Z()] for v in norm_gp_vec])

  vertices = np.array(vertices_gp).flatten()
  vertices = np.array([[p.X(), p.Y(), p.Z()] for p in vertices])
  vertices = vertices.reshape([-1, 3, 3])

  center = np.average(vertices, axis=1)
  center_gp_pnts = [gp_Pnt(*p) for p in center]
  areas = np.abs(np.linalg.det(vertices)) / 2.0

  if not areas.size:
    areas = np.empty((0, 3))
    norms = np.empty((0, 3))
    center = np.empty((0, 3))

  return areas, norms, norm_gp_vec, center, center_gp_pnts


def face_info(faces: List[TopoDS_Face]):
  planes = _planes_from_faces(faces)

  if any([plane is None for plane in planes]):
    # 곡면이 존재하는 경우
    curved_idx = [i for i, x in enumerate(planes) if x is None]
    curved_faces = [faces[i] for i in curved_idx]
    flat_faces = [faces[i] for i, x in enumerate(faces) if i not in curved_idx]

    # 평면 정보 추출
    areas, norms, norm_gp, center, center_gp = flat_face_info(flat_faces)

    # 곡면 정보 추출
    ca, cn, cngp, cc, ccgp = curved_face_info(curved_faces)

    # 평면 정보에 곡면 정보를 덧붙임
    areas = np.append(areas, ca)
    norms = np.vstack([norms, cn])
    norm_gp.extend(cngp)
    center = np.vstack([center, cc])
    center_gp.extend(ccgp)
    faces_count = areas.shape[0]

    assert faces_count == norms.shape[0]
    assert faces_count == center.shape[0]
  else:
    # 평면만 존재하는 경우
    areas, norms, norm_gp, center, center_gp = flat_face_info(faces)

  return areas, norms, norm_gp, center, center_gp


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


def classify_minor_faces(area: np.ndarray,
                         center: np.ndarray,
                         norm: np.ndarray,
                         threshold_dist: float,
                         threshold_cos: float,
                         openings=None) -> set:
  """
  merge_minor_faces()를 위해 단순화 대상 face (minor face)와 단순화 하지 않는
  주요 face (major face)를 구분
  
  :param area: face의 면적
  :param center: face의 중심 좌표
  :param norm: face의 법선 벡터
  :param threshold_dist: minor face 판단 거리 기준
  :param threshold_cos: minor face 판단 각도 기준 (cos theta)
  :param openings: opening의 face 리스트. 지정할 경우 minor face로 분류되지 않음
  :return: minor face의 index set
  """
  faces_idx = np.arange(area.shape[0])
  opening_face_idx = None

  # 법선 벡터 정규화
  norm = norm / np.linalg.norm(norm, axis=1).reshape([-1, 1])

  # 각 face의 다른 face들과의 cos(theta) (다른 face의 면적으로 가중 평균)
  avg_cos = np.repeat(norm, norm.shape[0], axis=0)
  avg_cos = avg_cos.reshape([norm.shape[0], norm.shape[0], norm.shape[1]])
  avg_cos = np.sum(norm * avg_cos, axis=2)
  avg_cos = np.average(np.abs(avg_cos), axis=1, weights=area)
  # avg_cos = np.average(np.square(avg_cos), axis=1, weights=area)
  # avg_cos = np.max(np.abs(avg_cos) * area, axis=1)

  if openings:
    opening_faces = chain.from_iterable(
        [TopologyExplorer(opening).faces() for opening in openings])
    opening_faces = list(set(opening_faces))
    op_area, op_norm, _, op_center, _ = face_info(opening_faces)

    op_norm = op_norm / np.linalg.norm(op_norm, axis=1).reshape([-1, 1])

    # 두 face가 평행한지 (법선 내적이 1인지) 확인
    norm_ = np.repeat(norm, op_norm.shape[0], axis=0)
    norm_ = norm_.reshape([norm.shape[0], op_norm.shape[0], 3])
    op_norm_ = np.tile(op_norm, (norm.shape[0], 1))
    op_norm_ = op_norm_.reshape([norm.shape[0], op_norm.shape[0], 3])
    is_parallel = np.isclose(np.abs(np.sum(norm_ * op_norm_, axis=2)), 1.0)

    # 두 face가 동일 평면에 존재하는지 (중앙점 간 벡터와 법선의 내적 0) 확인
    center_ = np.repeat(center, op_center.shape[0], axis=0)
    center_ = center_.reshape([center.shape[0], op_center.shape[0], 3])
    op_center_ = np.tile(op_center, (center.shape[0], 1))
    op_center_ = op_center_.reshape([center.shape[0], op_center.shape[0], 3])
    is_coplanar = np.isclose(np.sum(norm_ * (center_ - op_center_), axis=2),
                             0.0)

    # 두 face의 면적이 동일한지 확인
    area_ = np.repeat(area, op_area.shape[0], axis=0)
    area_ = area_.reshape([area.shape[0], op_area.shape[0]])
    op_area_ = np.tile(op_area, area.shape[0])
    op_area_ = op_area_.reshape([area.shape[0], op_area.shape[0]])
    is_same_area = np.isclose(area_, op_area_)

    # 세 기준 모두 일치하는 face를 opening face로 판단
    is_opening_face = np.all([is_parallel, is_coplanar, is_same_area], axis=0)
    is_opening_face = np.any(is_opening_face, axis=1)
    opening_face_idx = np.argwhere(is_opening_face).flatten()

  # buffer를 split하지 않는 (=생략되는) face
  minor_faces = set()

  # 주요 face/생략 대상 face 판단
  # 판단 순서는 1. 면적이 큰 순서대로,
  # 면적이 동일할 경우 2. 다른 면들과 각도가 비슷한 순서대로
  # TODO: opening face는 아예 기준에서 빼버리기?
  for idx in np.lexsort([-avg_cos, -area]):
    if idx in minor_faces or (openings and idx in opening_face_idx):
      continue

    # 거리 기준 판단
    vec = center - center[idx]
    dist = 2 * np.abs(np.sum(norm[idx] * vec, axis=1))
    minor_dist = faces_idx[dist <= threshold_dist]

    # 각도 기준 판단
    cos = np.sum(norm[idx] * norm, axis=1)
    minor_cos = faces_idx[np.abs(cos) >= threshold_cos]

    # 면적 기준 판단
    small = np.argwhere(area <= area[idx]).flatten()

    minor = np.intersect1d(minor_dist, minor_cos, assume_unique=True)
    minor = np.intersect1d(minor, small, assume_unique=True)
    minor_faces.update(minor[minor != idx])

  if openings:
    minor_faces -= set(opening_face_idx)

  return minor_faces


def merge_inner_volume(
    shape: TopoDS_Shape,
    threshold_internal_volume=0.0,
    threshold_internal_length: Union[float, list, np.ndarray] = 0.0,
    brep_deflection: Union[Tuple[float], float] = (1.0, 0.5),
    tol_bbox=1e-8,
    tol_cut=0.0,
):
  """
  shape의 내부 빈 공간 (e.g. 기둥) 중 기준을 충족하는 공간을 단순화
  단순화 대상 공간이 없으면 None을 반환
  내부 공간이 복잡하여 (직육면체 형상이 아니어서) 여러 shape으로 분할될 경우
  원하지 않는 빈 공간이 생략될 수 있음
  B-rep 형성한 shape 넣지 말것

  :param shape: 단순화 대상 shape
  :param threshold_internal_volume:
  기준 부피. 해당 수치 이하의 내부 빈 공간은 삭제함.
  :param threshold_internal_length:
  기준 길이. 내부 빈 공간의 edge 중 하나 이상
  해당 수치보다 작은 게 있으면 해당 공간을 삭제함.
  :param brep_deflection: BRepMesh_IncrementalMesh의 옵션.
  (linear_deflection, angular_deflection) 혹은 linear_deflection.
  :param tol_bbox: Bounding box 허용 오차
  :param tol_cut: BRepAlgoAPI_Cut의 허용 오차
  :return: 단순화의 필요가 없을 경우 None,
  단순화를 시행한 경우 단순화 한 TopoDS_Shape 반환
  """
  assert threshold_internal_volume >= 0.0
  if isinstance(threshold_internal_length, Iterable):
    assert all([x >= 0.0 for x in threshold_internal_length])
  else:
    assert threshold_internal_length >= 0.0

  shape = fuse_compound(shape)

  if TopologyExplorer(shape).number_of_solids() < 1:
    shape = sew_faces(shape)

    if TopologyExplorer(shape).number_of_solids() < 1:
      raise ValueError('단순화 대상 shape에 solid가 없습니다.')

  flag_simplify = True
  simplified_shape = None
  solids_cut = None
  inner_solids = None

  # space를 둘러싸는 bounding box 생성
  bbox_pnts = get_boundingbox(shape, tol=tol_bbox)
  bbox_gp_pnts = [gp_Pnt(*xyz) for xyz in [bbox_pnts[:3], bbox_pnts[3:]]]
  bbox = make_box(*bbox_gp_pnts)

  # bbox에 space를 제거하여 나머지 공간 cut_shape 생성
  algo_cut = BRepAlgoAPI_Cut(bbox, shape)
  if tol_cut:
    algo_cut.SetFuzzyValue(tol_cut)

  cut_shape = algo_cut.Shape()

  if cut_shape is None:
    flag_simplify = False
  else:
    # 나머지 공간의 solid 목록
    solids_cut = list(TopologyExplorer(cut_shape).solids())
    if not solids_cut:
      flag_simplify = False

  if flag_simplify:
    for solid_ in [shape] + solids_cut:
      BRepMesh_IncrementalMesh(solid_, brep_deflection[0], False,
                               brep_deflection[1], True)

    # 나머지 공간의 solid별 face 목록
    faces_cut = [list(TopologyExplorer(solid).faces()) for solid in solids_cut]

    # face의 brep mesh vertex를 추출하기 위한 함수
    def get_vertices(face):
      return np.array([[p.X(), p.Y(), p.Z()]
                       for p in chain.from_iterable(_get_face_vertices(face))])

    # 나머지 공간의 face vertices
    pnts_cut_list = [
        [get_vertices(f) for f in solid_face] for solid_face in faces_cut
    ]
    pnts_cut = np.vstack(list(chain.from_iterable(pnts_cut_list)))

    # ConvexHull 판단 후 나머지 solid와 매칭을 위한 vertex별 인덱스
    solid_idx = np.repeat(
        range(len(solids_cut)),
        [np.sum([p.shape[0] for p in vertices]) for vertices in pnts_cut_list])

    # 원본 공간의 face vertices
    pnts_original = [get_vertices(f) for f in TopologyExplorer(shape).faces()]
    pnts_original = np.vstack(list(chain.from_iterable(pnts_original)))

    # 내외부 점 / 부피 판단 (Convex Hull)
    hull = ConvexHull(np.vstack([pnts_cut, pnts_original]))
    # TODO: coplanar 점 고려

    outer_pnts = hull.vertices[hull.vertices < pnts_cut.shape[0]]
    outer_solid_idx = np.unique(solid_idx[outer_pnts])
    inner_solid_idx = [
        x for x in range(len(solids_cut)) if x not in outer_solid_idx
    ]
    inner_solids = [solids_cut[x] for x in inner_solid_idx]
    if not inner_solids:
      flag_simplify = False

  if flag_simplify:
    # 내부 shape의 부피 추출
    inner_vols = np.array(
        [GpropsFromShape(solid).volume().Mass() for solid in inner_solids])
    # 내부 shape의 최소 edge 길이 추출
    inner_exp = [TopologyExplorer(solid) for solid in inner_solids]
    inner_edges = [list(exp.edges()) for exp in inner_exp]

    inner_min_length = np.array([
        np.min([GpropsFromShape(edge).linear().Mass()
                for edge in edges])
        for edges in inner_edges
    ])

    # 제거해야하는 내부 shape 판단 (부피 기준 혹은 길이 기준 충족)
    merge_mask = np.logical_or(inner_vols <= threshold_internal_volume,
                               inner_min_length <= threshold_internal_length)

    if np.any(merge_mask):
      # 단순화 대상 내부 shape을 merge 시킴
      solids = [
          inner_solids[x] for x in range(len(inner_solids)) if merge_mask[x]
      ]
      solids.append(shape)
      simplified_shape = fuse_compound(solids)
      BRepMesh_IncrementalMesh(simplified_shape, brep_deflection[0], False,
                               brep_deflection[1], True)

  return simplified_shape


def merge_minor_faces(shape: TopoDS_Shape,
                      threshold_dist: float,
                      threshold_angle: float,
                      threshold_vol_ratio=0.5,
                      openings: List[TopoDS_Shape] = None,
                      brep_deflection: Union[Tuple[float], float] = (1.0, 0.5),
                      tol_bbox=1e-8,
                      buffer_size=2,
                      split_limit=None):
  """
  비슷한 face를 통합하여 형상을 단순화
  Ref: Kada, M. (2006). 3D building generalization based on half-space modeling.
  International Archives of Photogrammetry, Remote Sensing and Spatial
  Information Sciences, 36(2), 58-64.

  :param shape: 단순화를 시행할 대상 shape
  :param threshold_dist: 통합할 face의 최대 거리
  :param threshold_angle: 통합할 face의 최대 각도 차 [rad]
  :param threshold_vol_ratio: 분할한 공간 중 추출 여부를 판단하기 위한 부피 비율
  :param openings: 표면 단순화를 하지 않을 Opening의 목록
  :param brep_deflection: BRepMesh_IncrementalMesh의 옵션.
  (linear_deflection, angular_deflection) 혹은 linear_deflection.
  :param tol_bbox: Bounding box의 허용 오차
  :param buffer_size: Bounding box에 대한 buffer의 상대 크기
  :param split_limit: 분할하는 face 수가 해당 수 이상일 경우 순차적으로 분할 시행
  :return: 단순화의 필요가 없을 경우 None을,
  단순화를 시행한 경우 단순화 한 TopoDS_Shape 반환
  """
  assert threshold_dist >= 0.0
  assert 0.0 <= threshold_angle <= np.pi / 2.0
  assert 0.0 <= threshold_vol_ratio <= 1.0

  # shape = maker_volume(list(TopologyExplorer(shape).faces()))
  solids_count = TopologyExplorer(shape).number_of_solids()
  if solids_count <= (0 if openings is None else len(openings)):
    shape = maker_volume(list(TopologyExplorer(shape).faces()))

  shape = fuse_compound(shape)

  if brep_deflection:
    if isinstance(brep_deflection, float):
      brep_deflection = (brep_deflection, 0.5)

    for shape_ in [shape] + (list(openings) if openings else []):
      BRepMesh_IncrementalMesh(shape_, brep_deflection[0], False,
                               brep_deflection[1], True)

  # face 간 각도 기준
  threshold_cos = np.cos(threshold_angle)

  # shape의 모든 face 추출
  faces = list(TopologyExplorer(shape).faces())
  area, norm, norm_gp, center, center_gp = face_info(faces)

  # buffer를 split하지 않는 (=생략되는) face
  minor_faces = classify_minor_faces(area=area,
                                     center=center,
                                     norm=norm,
                                     threshold_dist=threshold_dist,
                                     threshold_cos=threshold_cos,
                                     openings=openings)

  # 단순화 대상 face (minor_faces)가 있을 경우 단순화 시행
  if minor_faces:
    # shape을 둘러싸는 bounding box의 좌표
    bbox_pnts = get_boundingbox(shape, tol=tol_bbox)
    bbox_pnts = np.array(bbox_pnts).reshape([2, 3])

    # bbox를 buffer_size만큼 키운 buffer를 만듬
    bbox_center = np.average(bbox_pnts, axis=0)
    buffer_pnts = buffer_size * (bbox_pnts - bbox_center) + bbox_center
    buffer_pnts = [gp_Pnt(*xyz) for xyz in buffer_pnts]
    buffer = make_box(*buffer_pnts)

    # split을 하기 위한 plane의 길이를 정하기 위한 변수
    extent = buffer_size * np.max(np.abs(bbox_pnts[0] - bbox_pnts[1]))

    # buffer를 split하는데 사용될 face들
    major_faces = [x for x in range(area.shape[0]) if x not in minor_faces]
    major_planes = [(center_gp[i], norm_gp[i]) for i in major_faces]
    major_planes = [
        make_plane(c, v, -extent, extent, -extent, extent)
        for c, v in major_planes
    ]

    # 원본 형상과 buffer를 모든 major face로 나눔
    # if fuzzy:
    #   split_original = maker_volume_sequential(shape, major_planes, fuzzy)
    #   split_buffer = maker_volume_sequential(buffer, major_planes, fuzzy)
    # else:
    #   step = (split_limit is not None and len(major_planes) >= split_limit)
    #   split_original = split_by_faces(
    #     shape, major_planes, parallel=True, step=step)
    #   split_buffer = split_by_faces(
    #     buffer, major_planes, parallel=True, step=step)

    step = (split_limit is not None and len(major_planes) >= split_limit)
    # split_original = split_by_faces(
    #   fix_shape(shape), major_planes, parallel=True, step=step)
    split_buffer = split_by_faces(shape=buffer,
                                  faces=major_planes,
                                  parallel=True,
                                  step=step)
    split_buffer = fix_shape(split_buffer)

    # split_buffer의 solid 추출
    buffer_solids = list(TopologyExplorer(split_buffer).solids())
    buffer_solids = [
        x for x in buffer_solids if GpropsFromShape(x).volume().Mass() > 0.0
    ]
    vol_ratio = [
        common_volume(x, shape) / GpropsFromShape(x).volume().Mass()
        for x in buffer_solids
    ]

    # # 원본 shape과 buffer의 교차되는 부분 부피비 비교
    # try:
    #   vol_ratio = _calc_split_vol_ratio(split_original, split_buffer)
    #   assert len(buffer_solids) == vol_ratio.shape[0]
    # except AssertionError:
    #   buffer_solids = [x for x in buffer_solids
    #                    if GpropsFromShape(x).volume().Mass() > 0.0]
    #   vol_ratio = [common_volume(x, shape) / GpropsFromShape(x).volume().Mass()
    #                for x in buffer_solids]

    # 분할된 buffer의 solid 중 원본 shape의 비중이 기준 이상인 경우만 추출
    simplified_shape = [
        s for s, r in zip(buffer_solids, vol_ratio) if r >= threshold_vol_ratio
    ]
    simplified_shape = compound(simplified_shape)

    # if fuzzy:
    #   simplified_shape = fix_shape(simplified_shape)
    #
    #   # split 시 생긴 오류 (틈새)를 수정하기 위해
    #   # fuzzy를 설정하고 BOPAlgo_MakerVolume 시행
    #   simplified_shape = maker_volume(
    #     list(TopologyExplorer(simplified_shape).faces()), fuzzy=fuzzy)

    if brep_deflection:
      BRepMesh_IncrementalMesh(simplified_shape, brep_deflection[0], False,
                               brep_deflection[1], True)

  else:
    # 단순화의 필요가 없는 경우
    simplified_shape = None

  return simplified_shape


def simplify_space(
    shape: TopoDS_Shape,
    openings: List[TopoDS_Shape] = None,
    brep_deflection=(1.0, 0.5),
    threshold_internal_volume=0.0,
    threshold_internal_length: Union[float, list, np.ndarray] = 0.0,
    threshold_surface_dist=0.0,
    threshold_surface_angle=0.0,
    threshold_surface_vol_ratio=0.5,
    relative_threshold=False,
    fuzzy=0.0,
    tol_bbox=1e-8,
    tol_cut=0.0,
    buffer_size=2,
    split_limit=None,
) -> TopoDS_Shape:
  """
  형상 단순화 함수. 내부 빈 공간 단순화와 외부 표면 단순화를 순차적으로 실행

  표면 단순화 Ref:
  Kada, M. (2006). 3D building generalization based on half-space modeling.
  International Archives of Photogrammetry, Remote Sensing and Spatial
  Information Sciences, 36(2), 58-64.

  :param shape: 단순화 대상 shape
  :param openings: 표면 단순화를 하지 않을 Opening의 목록
  :param brep_deflection: BRep mesh의 옵션. None일 경우 mesh를 다시 하지 않음.
  linear_deflection 또는 (linear_deflection, angular_deflection)
  :param threshold_internal_volume: 내부 공간의 단순화 기준 부피
  :param threshold_internal_length: 내부 공간의 단순화 기준 길이.
  내부 공간의 edge 중 하나 이상 기준 길이보다 짧은 edge가 있으면 해당 공간을 생략.
  :param threshold_surface_dist: 표면 단순화의 기준 길이.
  :param threshold_surface_angle: 표면 단순화의 기준 각도 [rad]
  :param threshold_surface_vol_ratio: 표면 단순화의 기준 부피비
  :param relative_threshold: True일 경우,
      threshold_internal_volume는 shape의 부피 대비 비율,
      threshold_internal_length는 shape의 특성길이 (부피 / 표면적) 대비 비율로 계산함.
  :param fuzzy: fuzzy operation 옵션
  :param tol_bbox: bounding box 허용 오차
  :param tol_cut: BRepAlgoAPI_Cut의 허용 오차
  :param buffer_size: bounding box 대비 버퍼 크기
  :param split_limit: 분할하는 face 수가 해당 수 이상일 경우 순차적으로 분할 시행
  :return: 형상 단순화를 시행했다면 단순화된 TopoDS_Shape,
  단순화가 필요 없다면 None 반환
  """
  flag_simplified = False

  assert BRepCheck_Analyzer(shape, True).IsValid()

  if brep_deflection:
    if isinstance(brep_deflection, float):
      brep_deflection = (brep_deflection, 0.5)

  if relative_threshold:
    props = GpropsFromShape(shape)
    vol = props.volume().Mass()
    area = props.surface().Mass()
    threshold_internal_volume *= vol
    threshold_surface_dist *= (vol / area)

  # 내부 빈 공간 단순화
  if threshold_internal_volume or np.all(threshold_internal_length):
    simplified_shape = merge_inner_volume(
        shape=shape,
        threshold_internal_length=threshold_internal_length,
        threshold_internal_volume=threshold_internal_volume,
        brep_deflection=brep_deflection,
        tol_bbox=tol_bbox,
        tol_cut=tol_cut)

    if simplified_shape:
      flag_simplified = True
      shape = fix_shape(simplified_shape)

  # 외부 표면 단순화
  if any([threshold_surface_dist, threshold_surface_angle]):
    simplified_shape = merge_minor_faces(
        shape=shape,
        openings=openings,
        threshold_dist=threshold_surface_dist,
        threshold_angle=threshold_surface_angle,
        threshold_vol_ratio=threshold_surface_vol_ratio,
        brep_deflection=brep_deflection,
        tol_bbox=tol_bbox,
        buffer_size=buffer_size,
        split_limit=split_limit)

    if simplified_shape:
      flag_simplified = True
      shape = fix_shape(simplified_shape)

  if flag_simplified:
    # result = maker_volume(list(TopologyExplorer(shape).faces()), fuzzy=fuzzy)
    result = shape
  else:
    result = None

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

  features = OrderedDict([('volume', volume), ('area', area),
                          ('characteristic_length', volume / area),
                          ('solid_count', exp.number_of_solids()),
                          ('face_count', exp.number_of_faces()),
                          ('edge_count', exp.number_of_edges()),
                          ('vertex_count', exp.number_of_vertices())])

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
  from OCC.Core.gp import gp_Quaternion

  if not isinstance(shape, TopoDS_Shape):
    raise TypeError('Need TopoDS_Shape, got {}'.fotmat(type(shape)))

  faces = list(TopologyExplorer(shape).faces())
  faces_area = [GpropsFromShape(x).surface().Mass() for x in faces]
  area_argsort = np.argsort(faces_area)
  faces_sorted = [faces[x] for x in area_argsort[::-1]]

  for face in faces_sorted:
    plane = _planes_from_faces([face])[0]
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
