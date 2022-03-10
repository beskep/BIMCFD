from collections import OrderedDict, defaultdict
from collections.abc import Collection
from itertools import chain
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BOPAlgo import BOPAlgo_MakerVolume, BOPAlgo_Splitter
from OCC.Core.BRep import BRep_Tool, BRep_Tool_Surface
from OCC.Core.BRepAlgoAPI import (BRepAlgoAPI_Common, BRepAlgoAPI_Cut,
                                  BRepAlgoAPI_Fuse)
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeSolid,
                                     BRepBuilderAPI_MakeVertex,
                                     BRepBuilderAPI_Sewing,
                                     BRepBuilderAPI_Transform)
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
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
from tqdm import tqdm

from OCCUtils.Common import GpropsFromShape
from OCCUtils.Construct import compound, face_normal, make_plane
from OCCUtils.face import Face


def make_box(*args):
  box = BRepPrimAPI_MakeBox(*args)
  box.Build()

  return box.Shape()


def get_boundingbox(shape, tol=1e-8, use_mesh=True):
  bbox = Bnd_Box()
  bbox.SetGap(tol)

  if use_mesh:
    mesh = BRepMesh_IncrementalMesh()
    mesh.SetParallelDefault(True)
    mesh.SetShape(shape)
    mesh.Perform()
    if not mesh.IsDone():
      raise AssertionError("Mesh not done.")

  brepbndlib_Add(shape, bbox, use_mesh)

  return bbox.Get()


def common_volume(shape1, shape2):
  common = BRepAlgoAPI_Common(shape1, shape2).Shape()

  if common is None:
    vol = 0.0
  else:
    vol = GpropsFromShape(common).volume().Mass()

  return vol


def fix_shape(shape: TopoDS_Shape, precision=None):
  fix = ShapeFix_Shape()
  fix.Init(shape)

  if precision:
    fix.SetPrecision(precision)

  fix.Perform()

  return fix.Shape()


def _maker_volume_sequential_helper(shape, plane, fuzzy=0.0):
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


def maker_volume(shapes: Collection,
                 boundary: TopoDS_Shape = None,
                 fuzzy=0.0) -> Union[TopoDS_Shape, TopoDS_Compound]:
  """
  BOPAlgo_MakerVolume
  https://github.com/tpaviot/pythonocc-core/issues/554

  Parameters
  ----------
  shapes : Collection
      Volume을 만드는데 사용되는 shape 목록
  boundary : TopoDS_Shape, optional
      지정 시 결과 중 boundary 내부에 존재하는 solid만 모아 compound를 만들어 반환,
      by default None
  fuzzy : float, optional
      BOPAlgo_MakerVolume의 fuzzy 옵션, by default 0.0

  Returns
  -------
  Union[TopoDS_Shape, TopoDS_Compound]
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
  volume = fix_shape(mv.Shape())

  if boundary is not None:
    solids = TopologyExplorer(volume).solids()
    is_in = (_is_in(boundary,
                    GpropsFromShape(x).volume().CentreOfMass(),
                    on=False) for x in solids)
    volume = compound([s for s, i in zip(solids, is_in) if i])

  return volume


def _is_in(solid, pnt: gp_Pnt, tol=0.001, on=True) -> bool:
  """
  pnt가 solid 내부에 있는지 여부를 반환

  Parameters
  ----------
  solid : TopoDS_Shape
  pnt : gp_Pnt
  tol : float, optional
      tolerance
  on : bool, optional
      표면에 존재하는 경우도 `True`로 판단할지 여부

  Returns
  -------
  bool
  """
  classifier = BRepClass3d_SolidClassifier(solid)
  classifier.Perform(pnt, tol)
  state = classifier.State()
  flag = state in [TopAbs_IN, TopAbs_ON] if on else state == TopAbs_IN
  classifier.Destroy()

  return flag


def _solid_center_distance(xyz1: np.ndarray, xyz2: np.ndarray):
  shape = [xyz1.shape[0], xyz2.shape[0], 3]
  arr1 = np.repeat(xyz1, xyz2.shape[0], axis=0).reshape(shape)
  arr2 = np.tile(xyz2, (xyz1.shape[0], 1)).reshape(shape)
  dist = np.sum(np.square(arr1 - arr2), axis=2)

  return dist


def calc_split_vol_ratio(split_original: TopoDS_Compound,
                         split_buffer: TopoDS_Compound) -> np.ndarray:
  """
  major plane으로 나눈 원본 shape과 buffer의 부피 비율 계산

  Parameters
  ----------
  split_original : TopoDS_Compound
      major plane으로 나눈 원본 shape
  split_buffer : TopoDS_Compound
      major plane으로 나눈 buffer

  Returns
  -------
  np.ndarray
      표면 단순화 결과에 포함 여부를 결정하는 부피 비율
  """
  # 오류 발생 시 solids_original에서 평면 (mass가 음수)을 제거하게 수정
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

  center_buffer = np.array(
      [[p.X(), p.Y(), p.Z()] for p in [p.CentreOfMass() for p in props_buffer]])

  # 각 buffer의 solid에 대한 original의 solid의 거리
  # XXX 거리 말고 다른 판단 방법으로 바꾸기?
  dist = _solid_center_distance(center_buffer, center_original)

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

  Parameters
  ----------
  face : TopoDS_Face

  Returns
  -------
  List[List[gp_Pnt]]
      세 개의 gp_pnts로 표현된 각 mesh face의 vertex 좌표
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


def get_faces_vertices(faces: List[TopoDS_Face]) -> List[List[gp_Pnt]]:
  """
  곡면 face 목록을 받아서 triangulation 실행 후 각 점의 좌표 반환

  Parameters
  ----------
  faces : List[TopoDS_Face]

  Returns
  -------
  List[List[gp_Pnt]]
  """
  vertices = list(
      chain.from_iterable([get_face_vertices(face) for face in faces]))

  return vertices


def plane_from_face(face: TopoDS_Shape) -> Geom_Plane:
  return Geom_Plane.DownCast(BRep_Tool_Surface(face))  # type: ignore


def planes_from_faces(faces: List[TopoDS_Shape]):
  """
  각 face를 포함하는 plane 반환

  Parameters
  ----------
  faces : List[TopoDS_Shape]
      faces

  Returns
  -------
  Generator[Geom_Plane, None, None]
  """
  return (plane_from_face(x) for x in faces)


def face2plane(face: TopoDS_Face, extent=1e8):
  face_util = Face(face)
  _, center = face_util.mid_point()
  norm = face_normal(face)

  return make_plane(center=center,
                    vec_normal=gp_Vec(norm.XYZ()),
                    extent_x_min=-extent,
                    extent_x_max=extent,
                    extent_y_min=-extent,
                    extent_y_max=extent)


def split_by_own_faces(shape: TopoDS_Shape, face_range=1000.0):
  # splitter = GEOMAlgo_Splitter()
  splitter = BOPAlgo_Splitter()
  splitter.AddArgument(shape)

  faces = TopologyExplorer(shape).faces()

  for face in faces:
    face_split = face2plane(face=face, extent=face_range)
    splitter.AddTool(face_split)

  splitter.Perform()

  return splitter.Shape()


def split_by_faces(shape: TopoDS_Shape,
                   faces: Iterable[TopoDS_Face],
                   parallel=False,
                   step=False,
                   verbose=False) -> TopoDS_Compound:
  """
  shape을 각 face로 나눔

  Parameters
  ----------
  shape : TopoDS_Shape
      shape
  faces : Iterable[TopoDS_Face]
      faces
  parallel : bool, optional
      GEOMAlgo_Splitter의 SetParallelMode 여부
  step : bool, optional
      True일 경우 각 face를 통해 순차적으로 slit 시행
  verbose : bool, optional
      verbose

  Returns
  -------
  TopoDS_Compound
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
    split = splitter.Shape()
  else:
    it = tqdm(faces) if verbose else faces

    for face in it:
      splitter.AddArgument(shape)
      splitter.AddTool(face)
      splitter.Perform()
      shape = splitter.Shape()
      splitter.Clear()

    split = shape

  return split  # type: ignore


def sew_faces(shape: TopoDS_Shape, tol=1e-3) -> TopoDS_Solid:
  """
  shape의 모든 face로 형성되는 solid 반환

  Parameters
  ----------
  shape : TopoDS_Shape
  tol : [type], optional
      BRepBuilderAPI_Sewing의 허용 오차

  Returns
  -------
  TopoDS_Solid
  """
  sew = BRepBuilderAPI_Sewing(tol)

  for face in TopologyExplorer(shape).faces():
    sew.Add(face)

  sew.Perform()
  solid = BRepBuilderAPI_MakeSolid(sew.SewedShape()).Solid()  # type: ignore

  return solid


def fuse_compound(target: Union[List, TopoDS_Shape]):
  if isinstance(target, TopoDS_Shape):
    target = list(TopologyExplorer(target).solids())

  if not target:
    compound = None
  elif len(target) == 1:
    compound = target[0]
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

    compound = fuse.Shape() if fuse.IsDone() else None

  return compound


def geometric_features(shape: TopoDS_Shape):
  gprops = GpropsFromShape(shape)
  exp = TopologyExplorer(shape)

  volume = gprops.volume().Mass()
  area = gprops.surface().Mass()

  fs = defaultdict(list)
  for solid in exp.solids():
    for face in TopologyExplorer(solid).faces():
      fs[face.HashCode(100000000)].append(solid)
  surface_count = sum(1 for x in fs.values() if len(x) == 1)

  features = OrderedDict([
      ('volume', volume),
      ('area', area),
      ('characteristic_length', volume / area),
      ('solid_count', exp.number_of_solids()),
      ('face_count', exp.number_of_faces()),
      ('surface_count', surface_count),
      ('edge_count', exp.number_of_edges()),
      ('vertex_count', exp.number_of_vertices()),
  ])

  return features


def shapes_distance(shape1: TopoDS_Shape, shape2: TopoDS_Shape,
                    deflection: float):
  dist = BRepExtrema_DistShapeShape(shape1, shape2, deflection)

  return dist.Value() if dist.IsDone() else None


def fuzzy_cut(shape1, shape2, tol=5e-5, parallel=False):
  cut = BRepAlgoAPI_Cut()
  l1 = TopTools_ListOfShape()
  l1.Append(shape1)
  l2 = TopTools_ListOfShape()
  l2.Append(shape2)

  cut.SetArguments(l1)
  cut.SetTools(l2)
  cut.SetFuzzyValue(tol)
  cut.SetRunParallel(parallel)
  cut.Build()

  return cut.Shape()


def _make_external_zone(shape: TopoDS_Shape, buffer_size=5, vertical_dim=2):
  bbox = np.array(get_boundingbox(shape)).reshape([2, 3])
  height = np.abs(bbox[0, vertical_dim] - bbox[1, vertical_dim])

  zone_pnts = [
      np.min(bbox, axis=0) - buffer_size * height,
      np.max(bbox, axis=0) + buffer_size * height
  ]
  zone_pnts[0][vertical_dim] = np.min(bbox[:, vertical_dim])  # 바닥

  zone_gp_pnts = [gp_Pnt(*xyz) for xyz in zone_pnts]
  zone = make_box(*zone_gp_pnts)

  return zone, bbox


def _classify_zone_face(zone: TopoDS_Shape, vertical_dim=2):
  faces = list(TopologyExplorer(zone).faces())
  gp_center = [GpropsFromShape(x).surface().CentreOfMass() for x in faces]
  center = np.array([[p.X(), p.Y(), p.Z()] for p in gp_center])

  arg_sort = np.argsort(center[:, vertical_dim])
  ground = faces[arg_sort[0]]
  ceiling = faces[arg_sort[-1]]
  vertical = [faces[x] for x in arg_sort[1:-1]]

  face_dict = {f'External_{i}': f for i, f in enumerate(vertical)}
  face_dict['Ground'] = ground
  face_dict['Ceiling'] = ceiling

  return face_dict


def _external_zone_interior(zone: TopoDS_Shape,
                            ground: TopoDS_Face,
                            bbox: np.ndarray,
                            buffer=0.2):
  if buffer < 0.0:
    raise ValueError('buffer < 1.0')

  if buffer == 0.0:
    points: Any = bbox
  else:
    min_pnts, max_pnts = np.min(bbox, axis=0), np.max(bbox, axis=0)
    length = max_pnts - min_pnts
    points = (min_pnts - length * buffer, max_pnts + length * buffer)

  # bounding box보다 (1+buffer)배 큰 박스의 face 추출
  box = make_box(gp_Pnt(*points[0]), gp_Pnt(*points[1]))
  faces = (x for x in TopologyExplorer(box).faces()
           if not _is_in(ground,
                         GpropsFromShape(x).surface().CentreOfMass()))

  # zone을 plane으로 나누고 interior 추출
  split = split_by_faces(shape=zone, faces=(face2plane(x) for x in faces))
  face_solid = defaultdict(list)
  for solid in TopologyExplorer(split).solids():
    for face in TopologyExplorer(solid).faces():
      face_solid[face.HashCode(100000000)].append(solid)

  interiors = (x for x in TopologyExplorer(split).faces()
               if len(face_solid[x.HashCode(100000000)]) > 1)

  return interiors


def make_external_zone(shape: TopoDS_Shape,
                       buffer=5,
                       inner_buffer=0.2,
                       vertical_dim=2) -> Tuple[TopoDS_Shape, dict]:
  """주어진 shape을 둘러싸는 external zone 생성

  Parameters
  ----------
  shape
      대상 shape
  buffer
      external zone의 여유 공간 크기. 대상 shape의 높이 (H)의 배수.
  inner_buffer
      mesh 설계를 위한 interior 경계면의 크기.
      대상 shape의 bounding box의 (1.0 + inner_buff)배 크기인 내부 공간 생성.
  vertical_dim
      연직 방향 dimension (x: 0, y: 1, z: 2),
      ifcopenshell의 기본 설정은 z 방향 (2)

  Returns
  -------
  tuple
      TopoDS_Shape, {face_name: TopoDS_Face}
  """
  zone, bbox = _make_external_zone(shape=fuse_compound(shape) or shape,
                                   buffer_size=buffer,
                                   vertical_dim=vertical_dim)
  shape2 = fuzzy_cut(zone, shape, tol=1e-2)
  if not shape2:
    raise RuntimeError('BRepAlgoAPI_Cut (외부 영역 - 건물 형상) 오류')

  zone_faces = _classify_zone_face(zone=zone, vertical_dim=vertical_dim)

  def _face_name(face):
    center = GpropsFromShape(face).surface().CentreOfMass()
    vertex = BRepBuilderAPI_MakeVertex(center).Vertex()
    name = 'Surface_0'
    for zn, zf in zone_faces.items():
      # if _is_in(zf, center, tol=0.1, on=True):
      #   name = zn
      #   break

      # XXX _is_in 오류 때문에 임의의 거리 기준으로 변경
      dist = shapes_distance(zf, vertex, deflection=1e-2)
      if dist <= 1.0:
        name = zn
        break

    return name

  faces = defaultdict(list)
  for face in TopologyExplorer(shape2).faces():
    faces[_face_name(face)].append(face)

  interiors = _external_zone_interior(zone=zone,
                                      ground=zone_faces['Ground'],
                                      bbox=bbox,
                                      buffer=inner_buffer)
  faces['Interior'] = list(interiors)

  return shape2, faces


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


def align_model(shape: TopoDS_Shape) -> Optional[TopoDS_Shape]:
  """건물 모델의 주요 표면이 xyz 평면과 수평하도록 회전,
  원점 근처에 위치하도록 평행이동

  Parameters
  ----------
  shape : TopoDS_Shape
      target shape

  Returns
  -------
  Optional[TopoDS_Shape]
      aligned shape
  """
  if not isinstance(shape, TopoDS_Shape):
    raise TypeError('Need TopoDS_Shape, got {}'.format(type(shape)))

  faces = list(TopologyExplorer(shape).faces())
  area_argsort = np.argsort(
      [GpropsFromShape(x).surface().Mass() for x in faces])

  for face in (faces[x] for x in area_argsort[::-1]):
    plane = plane_from_face(face)
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


def write_each_shapes(shape: TopoDS_Compound, save_dir: Path, mkdir=False):
  """compound를 구성하는 각 shape을 저장

  Parameters
  ----------
  shape : TopoDS_Compound
      대상 compound
  save_dir : PathLike
      저장 위치
  mkdir : bool
      make dir if `save_dir` not exist
  """
  exp = TopologyExplorer(shape)
  if exp.number_of_solids() <= 1:
    raise ValueError('대상 shape에 solid가 없음')

  if not save_dir.exists():
    if mkdir:
      save_dir.mkdir()
    else:
      raise FileNotFoundError(save_dir)

  solids = exp.solids()
  for idx, solid in enumerate(solids):
    path = save_dir.joinpath(f'{idx}.stl').as_posix()
    DataExchange.write_stl_file(a_shape=solid, filename=path)


def align_obj_to_origin(path):
  """obj 파일 bbox 최소값을 (0, 0, 0으로 변경)

  Parameters
  ----------
  path : pathlike
      대상 obj 파일 경로
  """
  with open(path, 'r') as f:
    lines = f.readlines()

  coord = []
  line_indices = []

  for idx, line in enumerate(lines):
    if line.startswith('v '):
      split_line = line.strip().split(' ')
      vertex = [float(x) for x in split_line[1:]]
      coord.append(vertex)
      line_indices.append(idx)

  coord_array = np.array(coord)
  coord_min = np.min(coord_array, axis=0)

  new_coord = coord_array - coord_min

  for cidx, lidx in enumerate(line_indices):
    lines[lidx] = 'v {:.6f} {:.6f} {:.6f}\n'.format(*new_coord[cidx])

  with open(path, 'w') as f:
    f.writelines(lines)


def outer_components(shape, target='face', hash_upper=1e8):
  if target not in ('face', 'edge'):
    raise ValueError(target)

  hash_upper = int(hash_upper)
  comp_solid = defaultdict(int)

  exp = TopologyExplorer(shape)
  for solid in exp.solids():
    if target == 'face':
      it = TopologyExplorer(solid).faces()
    else:
      it = TopologyExplorer(solid).edges()

    for comp in it:
      comp_solid[comp.HashCode(hash_upper)] += 1

  if target == 'face':
    it = exp.faces()
  else:
    it = exp.edges()

  components = [x for x in it if comp_solid[x.HashCode(hash_upper)] == 1]

  return components
