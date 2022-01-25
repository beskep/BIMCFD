from collections.abc import Iterable
from itertools import chain
from typing import List, Optional, Tuple, Union

import utils

import numpy as np
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Face, TopoDS_Shape
from OCC.Extend.TopologyUtils import TopologyExplorer
from scipy.spatial import ConvexHull  # pylint: disable=no-name-in-module

from OCCUtils.Common import GpropsFromShape
from OCCUtils.Construct import compound, make_plane
from OCCUtils.face import Face

from converter import geom_utils


def flat_face_info(faces: List[TopoDS_Face]):
  """
  평면 (face)의 목록을 받아서 표면 단순화에 필요한 face의 정보 반환

  Parameters
  ----------
  faces : List[TopoDS_Shape]

  Returns
  -------
  tuple
      (면적, 법선 벡터, 법선 벡터 (gp_Vec), 중앙점, 중앙점 (gp_Pnt), 평면)
  """

  planes = geom_utils.planes_from_faces(faces)
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


def curved_face_info(faces: List[TopoDS_Face]):
  """
  곡면 (face)의 목록을 받아서 표면 단순화에 필요한 face의 정보 반환

  Parameters
  ----------
  faces : List[TopoDS_Shape]

  Returns
  -------
  tuple
      (면적, 법선 벡터, 법선 벡터 (gp_Vec), 중앙점, 중앙점 (gp_Pnt), 평면)
  """
  vertices_gp = geom_utils.get_faces_vertices(faces)
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
  planes = geom_utils.planes_from_faces(faces)

  if any(plane is None for plane in planes):
    # 곡면이 존재하는 경우
    curved_idx = [i for i, x in enumerate(planes) if x is None]
    curved_faces = [faces[i] for i in curved_idx]
    flat_faces = [faces[i] for i in range(len(faces)) if i not in curved_idx]

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


def classify_minor_faces(area: np.ndarray,
                         center: np.ndarray,
                         norm: np.ndarray,
                         threshold_dist: float,
                         threshold_cos: float,
                         openings=None) -> set:
  """
  merge_minor_faces()를 위해 단순화 대상 face (minor face)와 단순화 하지 않는
  주요 face (major face)를 구분

  Parameters
  ----------
  area : np.ndarray
      face의 면적
  center : np.ndarray
      face의 중심 좌표
  norm : np.ndarray
      face의 법선 벡터
  threshold_dist : float
      minor face 판단 거리 기준
  threshold_cos : float
      minor face 판단 각도 기준 (cos theta)
  openings : list, optional
      opening의 face 리스트. 지정할 경우 minor face로 분류되지 않음.

  Returns
  -------
  set
      minor face의 index set
  """
  faces_idx = np.arange(area.shape[0])
  opening_face_idx = set()

  # 법선 벡터 정규화
  norm = norm / np.linalg.norm(norm, axis=1).reshape([-1, 1])

  # 각 face의 다른 face들과의 cos(theta) (다른 face의 면적으로 가중 평균)
  avg_cos = np.repeat(norm, norm.shape[0], axis=0)
  avg_cos = avg_cos.reshape([norm.shape[0], norm.shape[0], norm.shape[1]])
  avg_cos = np.sum(norm * avg_cos, axis=2)
  avg_cos = np.average(np.abs(avg_cos), axis=1, weights=area)

  if openings:
    opening_faces = list(
        set(
            chain.from_iterable(
                [TopologyExplorer(opening).faces() for opening in openings])))
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
    opening_face_idx = np.argwhere(is_opening_face).flatten()  # type: ignore

  # buffer를 split하지 않는 (=생략되는) face
  minor_faces = set()

  # 주요 face/생략 대상 face 판단
  # 판단 순서는 1. 면적이 큰 순서대로,
  # 면적이 동일할 경우 2. 다른 면들과 각도가 비슷한 순서대로
  # XXX opening face는 아예 기준에서 빼버리기?
  for idx in np.lexsort([-avg_cos, -area]):  # pylint: disable=invalid-unary-operand-type
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
    brep_deflection=(1.0, 0.5),
    tol_bbox=1e-8,
    tol_cut=0.0,
) -> Optional[TopoDS_Shape]:
  """
  shape의 내부 빈 공간 (e.g. 기둥) 중 기준을 충족하는 공간을 단순화.
  단순화 대상 공간이 없으면 None을 반환.
  내부 공간이 복잡하여 (직육면체 형상이 아니어서) 여러 shape으로 분할될 경우
  원하지 않는 빈 공간이 생략될 수 있음.
  B-rep 형성한 shape 넣지 말것.

  Parameters
  ----------
  shape : TopoDS_Shape
      단순화 대상 shape
  threshold_internal_volume : float, optional
      기준 부피. 해당 수치 이하의 내부 빈 공간은 삭제함.
  threshold_internal_length : Union[float, list, np.ndarray], optional
      기준 길이. 내부 빈 공간의 edge 중 하나 이상 해당 수치보다 작은 게 있으면
      해당 공간을 삭제함.
  brep_deflection : tuple, optional
      BRepMesh_IncrementalMesh의 옵션. (linear_deflection, angular_deflection)
      혹은 linear_deflection.
  tol_bbox : float, optional
      Bounding box 허용 오차
  tol_cut : float, optional
      BRepAlgoAPI_Cut의 허용 오차

  Returns
  -------
  Optional[TopoDS_Shape]
  """
  assert threshold_internal_volume >= 0.0
  if isinstance(threshold_internal_length, Iterable):
    assert all(x >= 0.0 for x in threshold_internal_length)
  else:
    assert threshold_internal_length >= 0.0

  shape = geom_utils.fuse_compound(shape)

  if TopologyExplorer(shape).number_of_solids() < 1:
    shape = geom_utils.sew_faces(shape)

    if TopologyExplorer(shape).number_of_solids() < 1:
      raise ValueError('단순화 대상 shape에 solid가 없습니다.')

  flag_simplify = True
  simplified_shape = None
  solids_cut = []
  inner_solids = []

  # space를 둘러싸는 bounding box 생성
  bbox_pnts = geom_utils.get_boundingbox(shape, tol=tol_bbox)
  bbox_gp_pnts = [gp_Pnt(*xyz) for xyz in [bbox_pnts[:3], bbox_pnts[3:]]]
  bbox = geom_utils.make_box(*bbox_gp_pnts)

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
    for solid_ in [shape] + solids_cut:  # type: ignore
      BRepMesh_IncrementalMesh(solid_, brep_deflection[0], False,
                               brep_deflection[1], True)

    # 나머지 공간의 solid별 face 목록
    faces_cut = [list(TopologyExplorer(solid).faces()) for solid in solids_cut]

    # face의 brep mesh vertex를 추출하기 위한 함수
    def get_vertices(face):
      return np.array(
          [[p.X(), p.Y(), p.Z()]
           for p in chain.from_iterable(geom_utils.get_face_vertices(face))])

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
    pnts_original = np.vstack(
        list(
            chain.from_iterable(
                (get_vertices(f) for f in TopologyExplorer(shape).faces()))))

    # 내외부 점 / 부피 판단 (Convex Hull)
    hull = ConvexHull(np.vstack([pnts_cut, pnts_original]))
    # TODO coplanar 점 고려

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
      solids.append(shape)  # type: ignore
      simplified_shape = geom_utils.fuse_compound(solids)
      BRepMesh_IncrementalMesh(simplified_shape, brep_deflection[0], False,
                               brep_deflection[1], True)

  return simplified_shape


def merge_minor_faces(
    shape: TopoDS_Shape,
    threshold_dist: float,
    threshold_angle: float,
    threshold_vol_ratio=0.5,
    openings: List[TopoDS_Shape] = None,
    brep_deflection: Union[Tuple[float, float], float] = (1.0, 0.5),
    tol_bbox=1e-8,
    buffer_size=2,
    split_limit=None,
) -> Optional[TopoDS_Compound]:
  """
  비슷한 face를 통합하여 형상을 단순화

  References
  ----------
  Kada, M. (2006). 3D building generalization based on half-space modeling.
  International Archives of Photogrammetry, Remote Sensing and Spatial
  Information Sciences, 36(2), 58-64.

  Parameters
  ----------
  shape : TopoDS_Shape
      단순화를 시행할 대상 shape
  threshold_dist : float
      통합할 face의 최대 거리
  threshold_angle : float
      통합할 face의 최대 각도 차 [rad]
  threshold_vol_ratio : float, optional
      분할한 공간 중 추출 여부를 판단하기 위한 부피 비율
  openings : List[TopoDS_Shape], optional
      표면 단순화를 하지 않을 Opening의 목록
  brep_deflection : Union[Tuple[float, float], float], optional
      BRepMesh_IncrementalMesh의 옵션. (linear_deflection, angular_deflection)
      혹은 linear_deflection.
  tol_bbox : float, optional
      Bounding box의 허용 오차
  buffer_size : int, optional
      Bounding box에 대한 buffer의 상대 크기
  split_limit : int, optional
      분할하는 face 수가 해당 수 이상일 경우 순차적으로 분할 시행

  Returns
  -------
  Optional[TopoDS_Compound]
  """
  assert threshold_dist >= 0.0
  assert 0.0 <= threshold_angle <= np.pi / 2.0
  assert 0.0 <= threshold_vol_ratio <= 1.0

  # shape = maker_volume(list(TopologyExplorer(shape).faces()))
  solids_count = TopologyExplorer(shape).number_of_solids()
  if solids_count <= (0 if openings is None else len(openings)):
    shape = geom_utils.maker_volume(list(TopologyExplorer(shape).faces()))

  shape = geom_utils.fuse_compound(shape)

  if not brep_deflection:
    bd = None
  else:
    if isinstance(brep_deflection, float):
      bd = (brep_deflection, 0.5)
    else:
      bd = brep_deflection

    for shape_ in [shape] + (list(openings) if openings else []):
      BRepMesh_IncrementalMesh(shape_, bd[0], False, bd[1], True)

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
    bbox_pnts = geom_utils.get_boundingbox(shape, tol=tol_bbox)
    bbox_pnts = np.array(bbox_pnts).reshape([2, 3])

    # bbox를 buffer_size만큼 키운 buffer를 만듬
    bbox_center = np.average(bbox_pnts, axis=0)
    buffer_pnts = buffer_size * (bbox_pnts - bbox_center) + bbox_center
    buffer_pnts = [gp_Pnt(*xyz) for xyz in buffer_pnts]
    buffer = geom_utils.make_box(*buffer_pnts)

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
    split_buffer = geom_utils.split_by_faces(shape=buffer,
                                             faces=major_planes,
                                             parallel=True,
                                             step=step)
    split_buffer = geom_utils.fix_shape(split_buffer)

    # split_buffer의 solid 추출
    buffer_solids = list(TopologyExplorer(split_buffer).solids())
    buffer_solids = [
        x for x in buffer_solids if GpropsFromShape(x).volume().Mass() > 0.0
    ]
    vol_ratio = [
        geom_utils.common_volume(x, shape) / GpropsFromShape(x).volume().Mass()
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
    simplified_shape = compound([
        s for s, r in zip(buffer_solids, vol_ratio) if r >= threshold_vol_ratio
    ])

    # if fuzzy:
    #   simplified_shape = fix_shape(simplified_shape)
    #
    #   # split 시 생긴 오류 (틈새)를 수정하기 위해
    #   # fuzzy를 설정하고 BOPAlgo_MakerVolume 시행
    #   simplified_shape = maker_volume(
    #     list(TopologyExplorer(simplified_shape).faces()), fuzzy=fuzzy)

    if bd:
      BRepMesh_IncrementalMesh(simplified_shape, bd[0], False, bd[1], True)

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
    tol_bbox=1e-8,
    tol_cut=0.0,
    buffer_size=2,
    split_limit=None,
) -> Optional[TopoDS_Shape]:
  """
  형상 단순화 함수. 내부 빈 공간 단순화와 외부 표면 단순화를 순차적으로 실행

  References
  ----------
  Kada, M. (2006). 3D building generalization based on half-space modeling.
  International Archives of Photogrammetry, Remote Sensing and Spatial
  Information Sciences, 36(2), 58-64.

  Parameters
  ----------
  shape : TopoDS_Shape
      단순화 대상 shape
  openings : List[TopoDS_Shape], optional
      표면 단순화를 하지 않을 Opening의 목록
  brep_deflection : tuple, optional
      BRep mesh의 옵션. None일 경우 mesh를 다시 하지 않음.
      (linear_deflection, angular_deflection)
  threshold_internal_volume : float, optional
      내부 공간의 단순화 기준 부피.
  threshold_internal_length : Union[float, list, np.ndarray], optional
      내부 공간의 단순화 기준 길이. 내부 공간의 edge 중 하나 이상 기준 길이보다
      짧은 edge가 있으면 해당 공간을 생략.
  threshold_surface_dist : float, optional
      표면 단순화의 기준 길이.
  threshold_surface_angle : float, optional
      표면 단순화의 기준 각도 [rad]
  threshold_surface_vol_ratio : float, optional
      표면 단순화의 기준 부피비
  relative_threshold : bool, optional
      `True`일 경우, threshold_internal_volume는 shape의 부피 대비 비율,
      threshold_internal_length는 shape의 특성길이 (부피 / 표면적)
      대비 비율로 계산함.
  tol_bbox : float, optional
      bounding box 허용 오차
  tol_cut : float, optional
      BRepAlgoAPI_Cut의 허용 오차
  buffer_size : int, optional
      bounding box 대비 버퍼 크기
  split_limit : int, optional
      분할하는 face 수가 해당 수 이상일 경우 순차적으로 분할 시행

  Returns
  -------
  Optional[TopoDS_Shape]
      형상 단순화를 시행했다면 단순화된 TopoDS_Shape, 단순화가 필요 없다면 None 반환
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
      shape = geom_utils.fix_shape(simplified_shape)

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
      shape = geom_utils.fix_shape(simplified_shape)

  if not flag_simplified:
    result = None
  else:
    # result = maker_volume(list(TopologyExplorer(shape).faces()), fuzzy=fuzzy)
    result = shape

  return result
