"""
D2 shape similarity
"""

import os
import re
from itertools import chain

import numpy as np
import pandas as pd
from OCC.Core import BRep
from OCC.Core.BRep import BRep_Tool_Surface, BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Extend import DataExchange
from OCC.Extend import TopologyUtils
from scipy import stats
from tqdm import tqdm

from OCCUtils.Common import GpropsFromShape
from OCCUtils import Construct
from OCCUtils.face import Face
from .visualize import Vis
from .simplify import (maker_volume, fuse_compound, fix_shape, shapes_distance)

pattern = re.compile('.*vol([.0-9]+)_dist([.0-9]+)_fuzzy([.0-9]+).*')

ANGLE_ADJUSTED = False


class SamplingFace(Face):

  def __init__(self, face):
    super(SamplingFace, self).__init__(face)

    self.u_min, self.u_max, self.v_min, self.v_max = self.domain()
    self.u_range = self.u_max - self.u_min
    self.v_range = self.v_max - self.v_min

  def denormalize(self, u, v):
    return (u * self.u_range + self.u_min), (v * self.v_range + self.v_min)

  def norm_param_to_point(self, u, v):
    if not all([0 <= u <= 1, 0 <= v <= 1]):
      raise ValueError

    return self.parameter_to_point(*self.denormalize(u, v))

  @property
  def surface(self):
    if self._srf is None or self.is_dirty:
      self._h_srf = BRep_Tool_Surface(self)
      self._srf = self._h_srf
    return self._srf


def d2_distribution(dist1: np.ndarray, dist2: np.ndarray, bins=None):
  if dist1.size != dist2.size:
    raise ValueError

  if bins is None:
    bins = 'auto'

  dist_range = (np.minimum(dist1.min(), dist2.min()),
                np.maximum(dist1.max(), dist2.max()))  # yapf: disable

  hist1, bin_edges = np.histogram(dist1,
                                  bins=bins,
                                  range=dist_range,
                                  density=False)
  hist2, _ = np.histogram(dist2,
                          bins=hist1.size,
                          range=dist_range,
                          density=False)

  return hist1, hist2, bin_edges


def complexity(shape):
  exp = TopologyUtils.TopologyExplorer(shape)

  faces = list(TopologyUtils.TopologyExplorer(shape).faces())
  try:
    shape = maker_volume(faces, fuzzy=0.001)
  except RuntimeError:
    pass

  gprops = GpropsFromShape(shape)

  try:
    ce = cartesian_edges(shape)
  except (ValueError, ZeroDivisionError):
    ce = None

  comp = {
      'solid_count': exp.number_of_solids(),
      'face_count': exp.number_of_faces(),
      'edge_count': exp.number_of_edges(),
      'vertex_count': exp.number_of_vertices(),
      'volume': gprops.volume().Mass(),
      'area': gprops.surface().Mass(),
      'cartesian_edges': ce
  }

  return comp


def surface_sample(face, uv):
  sf = SamplingFace(face)

  def np_point(uv):
    gp_pnt = sf.norm_param_to_point(*uv)
    return [gp_pnt.X(), gp_pnt.Y(), gp_pnt.Z()]

  points = np.array([np_point(x) for x in uv])
  return points


def surfaces_sample(faces, uv):
  area = np.array([GpropsFromShape(x).surface().Mass() for x in faces])
  choice = np.random.choice(area.shape[0],
                            uv.shape[0],
                            replace=True,
                            p=area / np.sum(area))
  points_list = [
      surface_sample(f, uv[choice == i])
      for i, f in enumerate(faces)
      if i in choice
  ]
  points = np.concatenate(points_list, axis=0)
  return points


def dist_sample(shape, size: int, rng: np.random.RandomState = None):
  rand = np.random.rand if rng is None else rng.rand

  faces = list(TopologyUtils.TopologyExplorer(shape).faces())
  uv = rand(size * 4).reshape([-1, 2])

  points = surfaces_sample(faces, uv)
  dist = np.linalg.norm((points[:size] - points[size:]), axis=1)

  return dist


def get_vertices(shape, tool: BRep.BRep_Tool = None):
  if tool is None:
    tool = BRep.BRep_Tool()
  BRepMesh_IncrementalMesh(shape, 0.5, False, 0.5, True)

  exp = TopologyUtils.TopologyExplorer(shape)
  gp_pnts = [tool.Pnt(x) for x in exp.vertices()]
  pnts = np.array([[p.X(), p.Y(), p.Z()] for p in gp_pnts])

  return pnts


def bhattacharyya_coefficient(distribution1: np.ndarray,
                              distribution2: np.ndarray):
  if distribution1.size != distribution2.size:
    raise ValueError

  ss = np.sqrt(np.sum(distribution1) * np.sum(distribution2))
  bc = np.sum(np.sqrt(distribution1 * distribution2) / ss)
  assert 0 <= bc <= 1

  return bc


def stl_list(path, fuzzy=None):
  list_dir = os.listdir(path)
  list_dir = [
      x for x in list_dir if x.endswith('bpy.stl') and 'interior' not in x
  ]
  list_match = [pattern.match(x) for x in list_dir]
  if not list_dir or any([x is None for x in list_match]):
    raise FileNotFoundError

  vol_dist = np.array(
      [[float(x.group(1)),
        float(x.group(2)),
        float(x.group(3))] for x in list_match])

  if fuzzy is not None:
    index = vol_dist[:, 2] == fuzzy
    list_dir = [x for x, y in zip(list_dir, index) if y]
    vol_dist = vol_dist[index]

  original_idx = np.lexsort(vol_dist.T[::-1])[0]
  original = list_dir[original_idx]
  list_dir.pop(original_idx)

  return original, list_dir


def read_from_stl(path, interior=False):
  """
  interior가 포함된 stl 읽고 interior를 제거 시도...

  :param path:
  :param interior:
  :return: 
  """
  shape = DataExchange.read_stl_file(path)

  if not interior:
    exp = TopologyUtils.TopologyExplorer(shape)

    if exp.number_of_solids() != 1:
      res = shape
    else:
      faces = list(exp.faces())
      mv = maker_volume(faces)
      res = fuse_compound(mv)

      # if TopologyUtils.TopologyExplorer(res).number_of_solids() != 1:
      #   raise ValueError

      mv_dir, mv_file = os.path.split(path)
      mv_file = mv_file.replace('simplified', 'mv')
      DataExchange.write_stl_file(res, os.path.join(mv_dir, mv_file))

    # Vis.visualize(res, os.path.join(mv_dir, mv_file.replace('.stl', '.png')))
  else:
    res = None

  return res


def calculate_d2_similarity_seq(path, size, fuzzy=None, verbose=True):
  try:
    original_file, files = stl_list(path, fuzzy)
  except FileNotFoundError:
    return None

  orig_shape = read_from_stl(os.path.join(path, original_file))
  orig_dist = dist_sample(orig_shape, size)

  df = {
      'case': [original_file],
      'Bhattacharyya': [None],
      'corr': [None],
      'p-value': [None],
      'sd': [np.std(orig_dist)]
  }
  df.update({key: [value] for key, value in complexity(orig_shape).items()})

  it = tqdm(files) if verbose else files
  for file in it:
    shape = read_from_stl(os.path.join(path, file))
    dist = dist_sample(shape, size)

    hist1, hist2, _ = d2_distribution(orig_dist, dist)

    bc = bhattacharyya_coefficient(hist1, hist2)
    corr, p_value = stats.pearsonr(hist1, hist2)

    df['case'].append(file)
    df['Bhattacharyya'].append(bc)
    df['corr'].append(corr)
    df['p-value'].append(p_value)
    df['sd'].append(np.std(dist))

    comp = complexity(shape)
    for key, value in comp.items():
      df[key].append(value)

  df = pd.DataFrame(df)
  return df


def calculate_d2_similarity(shape1, shape2, size):
  dist1 = dist_sample(shape1, size)
  dist2 = dist_sample(shape2, size)
  hist1, hist2, _ = d2_distribution(dist1, dist2)

  bc = bhattacharyya_coefficient(hist1, hist2)
  corr, p_value = stats.pearsonr(hist1, hist2)

  return {'Bhattacharyya': bc, 'corr': corr, 'p-value': p_value}


def visualize_seq(path):
  global ANGLE_ADJUSTED

  try:
    original_file, files = stl_list(path, fuzzy=None)
  except FileNotFoundError:
    return None

  files += [original_file]
  print(original_file)

  for file in tqdm(files):
    shape = read_from_stl(os.path.join(path, file))
    vis_path = os.path.normpath(
        os.path.join('../result/mesh_quality/visualize',
                     file.replace('.stl', '.png')))

    Vis.visualize(shape, color=[0.2] * 3, init_kwargs={'size': (1920, 1080)})

    if not ANGLE_ADJUSTED:
      Vis.start_display()
      # os.system('pause')
      ANGLE_ADJUSTED = True

    Vis.save_image(vis_path)
    # Vis.start_display()


def edge_to_vec(edge):
  exp = TopologyUtils.TopologyExplorer(edge)
  vertices = list(exp.vertices())
  if len(vertices) == 2:
    gp_pnts = [BRep_Tool.Pnt(x) for x in vertices]
    pnts = np.array([[v.X(), v.Y(), v.Z()] for v in gp_pnts])
    vec = pnts[0] - pnts[1]
    vec /= np.sqrt(np.sum(np.square(vec)))
  else:
    vec = np.full((3,), np.nan)
  return vec


def _connected_vertex_count(face, edge):
  vertices = list(TopologyUtils.TopologyExplorer(edge).vertices())
  assert len(vertices) <= 2
  dist = [shapes_distance(face, v, deflection=1e-3) for v in vertices]
  res = np.sum(np.isclose(0.0, dist, atol=0.0))
  return res


def get_face_norm(face, exp=None):
  if exp is None:
    exp = TopologyUtils.TopologyExplorer(face)

  face_vertices = list(exp.vertices_from_face(face))
  assert len(face_vertices) >= 3
  face_gp_pnts = [BRep_Tool.Pnt(x) for x in face_vertices]
  face_pnts = np.array([[v.X(), v.Y(), v.Z()] for v in face_gp_pnts])
  face_norm = np.cross(face_pnts[1] - face_pnts[0], face_pnts[2] - face_pnts[0])
  face_norm /= np.sqrt(np.sum(np.square(face_norm)))
  return face_norm


def _is_parallel(faces, exp):
  assert len(faces) == 2
  norms = [get_face_norm(face, exp) for face in faces]
  cross = np.cross(norms[0], norms[1])
  is_parallel = np.all(np.isclose(0.0, cross, atol=0.0))
  return is_parallel


def cartesian_edges(shape):
  exp = TopologyUtils.TopologyExplorer(shape)
  faces = list(exp.faces())
  # edges = list(exp.edges())

  # edge_faces = {
  #     edge:
  #     [face for face in faces if _connected_vertex_count(face, edge) == 2]
  #     for edge in edges
  # }
  # # assert all([len(x) == 2 for x in edge_faces.values()])
  # outer_edges = [
  #     edge for edge in edges
  #     if (len(edge_faces) > 2) or (not _is_parallel(edge_faces[edge], exp))
  # ]

  edges_count = 0
  cartesian_count = 0

  for face in faces:
    # connected_edges = [
    #     x for x in edges if _connected_vertex_count(face, x) == 1
    # ]
    #
    # if not connected_edges:
    #   # triangles = _get_face_vertices(face)
    #   # edges_count += len(triangles)
    #   return None
    #   # continue
    #
    # vec_edges = np.array([edge_to_vec(x) for x in connected_edges])
    # face_norm = get_face_norm(face, exp)
    # cos = np.abs(np.sum(face_norm * vec_edges, axis=1))
    # is_cartesian = np.logical_or(np.isclose(0.0, cos, atol=0.0),
    #                              np.isclose(1.0, cos, atol=0.0))

    face_vertices = list(exp.vertices_from_face(face))
    all_edges = set(
        chain.from_iterable([exp.edges_from_vertex(x) for x in face_vertices]))
    face_edges = set(exp.edges_from_face(face))
    assert face_edges.issubset(all_edges)

    edges = all_edges.difference(face_edges)
    if not edges:
      continue

    vec_edges = np.array([edge_to_vec(x) for x in edges])
    vec_edges = vec_edges[np.logical_not(np.any(np.isnan(vec_edges), axis=1))]

    norm_gp_dir = Construct.face_normal(face)
    norm = np.array([norm_gp_dir.X(), norm_gp_dir.Y(), norm_gp_dir.Z()])

    cos = np.abs(np.sum(norm * vec_edges, axis=1))
    assert cos.ndim == 1
    is_cartesian = np.logical_or(np.isclose(0.0, cos, atol=0.0),
                                 np.isclose(1.0, cos, atol=0.0))

    edges_count += len(is_cartesian)
    cartesian_count += np.sum(is_cartesian)

  return cartesian_count / edges_count


def characteristic_length(path):
  list_dir = os.listdir(path)
  files = [x for x in list_dir if x.endswith('simplified.stl')]

  def _cl(file):
    shape = DataExchange.read_stl_file(file)
    shape = fix_shape(shape)
    exp = TopologyUtils.TopologyExplorer(shape)
    if exp.number_of_solids() > 1:
      shape = fuse_compound(list(exp.solids()))
    gp = GpropsFromShape(shape)
    vol = gp.volume().Mass()
    area = gp.surface().Mass()
    return vol, area

  va = [_cl(os.path.join(path, x)) for x in files]
  df = pd.DataFrame({
      'case': files,
      'volume': [x[0] for x in va],
      'area': [x[1] for x in va]
  })
  df['cl'] = df['volume'] / df['area']

  return df


if __name__ == '__main__':
  base_dir = r'D:\Python\IFCConverter\IFCConverter' \
             r'\result\mesh_quality'
  size = 1000_000
  # size = 100

  # Vis._init_display()
  # Vis.start_display()
  # os.system('pause')

  p_case = re.compile(r'^\[[0-9]+\].*')
  cl = []

  for root, dirs, files in os.walk(base_dir):
    # print(root)
    prj = os.path.split(root)[1]

    for directory in dirs:
      if not p_case.match(directory):
        continue

      print(os.path.join(root, directory))
      df_path = os.path.join(r'D:\Python\IFCConverter\IFCConverter\result',
                             'd2_{}_{}.csv'.format(prj, directory))

      # if ('Hospital_Parking Garage' in prj) or ('Conference Center' in prj):
      #   continue

      # cl_path = '../result/mesh_quality/cl_{}_{}.csv'.format(prj, directory)
      # cl_case = characteristic_length(os.path.join(root, directory))
      # cl_case.to_csv(cl_path, index=False)
      # cl.append(cl_case)

      # visualize_seq(os.path.join(root, directory))

      if not os.path.isfile(df_path):
        df = calculate_d2_similarity_seq(os.path.join(root, directory),
                                         size=size,
                                         fuzzy=None,
                                         verbose=True)
        if df is not None:
          df.to_csv(df_path)

  # cl_df = pd.concat(cl, ignore_index=True)
  # cl_df.to_csv('../result/mesh_quality/cl.csv', index=False)
