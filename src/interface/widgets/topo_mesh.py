from itertools import chain
from typing import List

import numpy as np
from OCC.Core import TopoDS, gp
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Extend.TopologyUtils import TopologyExplorer


def _get_face_vertices(face: TopoDS.TopoDS_Face) -> List[List[gp.gp_Pnt]]:
  """
  대상 face의 mesh face별 vertic 좌표를 반환

  :param face: 대상 face
  :return: 세 개의 gp_pnts로 표현된 각 mesh face의 vertex 좌표
  """
  trsf: gp.gp_Trsf = face.Location().Transformation()

  loc = TopLoc_Location()
  trsf_tri: gp.gp_Trsf = loc.Transformation()
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


def _get_faces_vertices(faces: list) -> List[List[gp.gp_Pnt]]:
  """
  곡면 face 목록을 받아서 triangulation 실행 후 각 점의 좌표 반환

  :param faces:
  :return: list[[gp_pnt, gp_pnt, gp_pnt]
  """
  vertices = [_get_face_vertices(face) for face in faces]
  vertices = list(chain.from_iterable(vertices))

  return vertices


def face_info(faces: List[TopoDS.TopoDS_Shape]):
  vertices_gp = _get_faces_vertices(faces)
  vectors = [[gp.gp_Vec(x, y), gp.gp_Vec(x, z)] for x, y, z in vertices_gp]
  norm_gp_vec = [v.Crossed(w) for v, w in vectors]
  norms = np.array([[v.X(), v.Y(), v.Z()] for v in norm_gp_vec])

  vertices = np.array(vertices_gp).flatten()
  vertices = np.array([[p.X(), p.Y(), p.Z()] for p in vertices])
  vertices = vertices.reshape([-1, 3, 3])

  return vertices, norms


class TopoDsMesh:

  def __init__(self,
               shapes: List[TopoDS.TopoDS_Shape],
               linear_deflection: float,
               angular_deflection=0.5,
               color=(1.0, 1.0, 1.0, 1.0)) -> None:
    self.shapes = shapes
    self.linear_deflection = linear_deflection
    self.angular_deflection = angular_deflection

    if color is None:
      self.color = (1.0, 1.0, 1.0, 1.0)
    else:
      self.color = color

    self._vertices = None
    self._norms = None

    self._mesh_vertices = None
    self._mesh_index = None
    self._mesh_format = None

    self.generate_mesh()

  @property
  def mesh_vertices(self):
    return list(self._mesh_vertices.flatten())

  @property
  def mesh_vertices_array(self):
    return self._mesh_vertices

  @property
  def mesh_index(self):
    return self._mesh_index

  @property
  def mesh_format(self):
    return self._format

  def brep_mesh_incremental(self, shape):
    BRepMesh_IncrementalMesh(shape, self.linear_deflection, False,
                             self.angular_deflection, True)

  def generate_mesh(self):
    faces = []
    for shape in self.shapes:
      self.brep_mesh_incremental(shape)
      exp = TopologyExplorer(shape)
      faces.extend(list(exp.faces()))

    self._vertices, self._norms = face_info(faces)

    vertices_ = self._vertices.reshape([-1, 3])
    norms_ = np.repeat(self._norms, 3, axis=0)

    self._format = [(b'v_pos', 3, 'float'), (b'v_normal', 3, 'float')]
    self._mesh_vertices = np.hstack([vertices_, norms_])
    self._mesh_index = list(range(self._vertices.shape[0] * 3))

  def bbox(self):
    v = self._vertices.reshape([-1, 3])
    minimum = np.min(v, axis=0)
    maximum = np.max(v, axis=0)
    return np.vstack([minimum, maximum])


if __name__ == "__main__":
  from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

  make_box = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0)
  box = make_box.Shape()

  topo_mesh = TopoDsMesh(shapes=[box],
                         linear_deflection=0.1,
                         angular_deflection=0.5)
