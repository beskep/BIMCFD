import sys
from pathlib import Path
from typing import List

import numpy as np
# from kivy.core.window import Window
from kivy.graphics import (Callback, Color, Mesh, PopMatrix, PushMatrix,
                           RenderContext, Rotate, Scale, Translate,
                           UpdateNormalMatrix)
from kivy.graphics.opengl import GL_DEPTH_TEST, glDisable, glEnable
from kivy.graphics.transformation import Matrix
from kivy.resources import resource_find
from kivy.uix.widget import Widget

_SRC_DIR = Path(__file__).parents[1]
if str(_SRC_DIR) not in sys.path:
  sys.path.append(str(_SRC_DIR))

from utils import RESOURCE_DIR
from interface.topo_mesh import TopoDsMesh
from interface.utfapp import UtfApp

_GLSL_PATH = RESOURCE_DIR.joinpath('color.glsl')
assert _GLSL_PATH.exists()


class BaseRenderer(Widget):

  ROTATE_SPEED = 1.0
  SCALE_RANGE = (0.01, 100.0)
  SCALE_FACTOR = 2.0

  def __init__(self, **kwargs):
    super(BaseRenderer, self).__init__(**kwargs)

    self.canvas = RenderContext(compute_normal_mat=True)
    self.canvas.shader.source = resource_find(str(_GLSL_PATH))

    self.rotx = None
    self.roty = None
    self.scale = None

    with self.canvas:
      self.cb = Callback(self.setup_gl_context)
      PushMatrix()
      self.setup_scene()
      PopMatrix()
      self.cb = Callback(self.reset_gl_context)
    self.update_glsl()
    self.canvas['diffuse_light'] = (1.0, 1.0, 0.8)
    self.canvas['ambient_light'] = (0.1, 0.1, 0.1)

  def setup_gl_context(self, *args):
    glEnable(GL_DEPTH_TEST)

  def reset_gl_context(self, *args):
    glDisable(GL_DEPTH_TEST)

  def update_glsl(self):
    asp = self.width / float(self.height)
    proj = Matrix().view_clip(left=-asp,
                              right=asp,
                              bottom=-1,
                              top=1,
                              near=1,
                              far=100,
                              perspective=1)
    self.canvas['projection_mat'] = proj
    # self.canvas['diffuse_light'] = (1.0, 1.0, 0.8)
    # self.canvas['ambient_light'] = (0.1, 0.1, 0.1)

  def setup_scene(self):
    Color(1, 1, 1, 1)

    PushMatrix()
    Translate(0, 0, -3)

    self.rotx = Rotate(0, 1, 0, 0)
    self.roty = Rotate(0, 0, 1, 0)
    self.scale = Scale(1.0)

    # UpdateNormalMatrix()
    # self.mesh = Mesh(
    #     vertices=m.vertices,
    #     indices=m.indices,
    #     fmt=m.vertex_format,
    #     mode='triangles',
    # )
    PopMatrix()

  def rotate_angle(self, touch):
    x_angle = (touch.dx / self.width) * 360.0 * self.ROTATE_SPEED
    y_angle = (-touch.dy / self.height) * 360.0 * self.ROTATE_SPEED
    return x_angle, y_angle

  def scale_screen(self, touch):
    if touch.button == 'scrolldown':
      new_scale = self.scale.xyz[0] / self.SCALE_FACTOR
    elif touch.button == 'scrollup':
      new_scale = self.scale.xyz[0] * self.SCALE_FACTOR
    else:
      raise AssertionError(str(touch.button))

    if self.SCALE_RANGE[0] <= new_scale <= self.SCALE_RANGE[1]:
      self.scale.xyz = (new_scale,) * 3

  def ignore_under_touch(fn):

    def wrap(self, touch):
      gl = touch.grab_list
      if not len(gl) or (self is gl[0]()):
        return fn(self, touch)

    return wrap

  @ignore_under_touch
  def on_touch_down(self, touch):
    touch.grab(self)
    if touch.is_mouse_scrolling:
      self.scale_screen(touch)

  @ignore_under_touch
  def on_touch_up(self, touch):
    touch.ungrab(self)

  @ignore_under_touch
  def on_touch_move(self, touch):
    ax, ay = self.rotate_angle(touch)
    self.roty.angle += ax
    self.rotx.angle += ay
    self.update_glsl()


class _TopoDsMesh:

  def __init__(self, shape: TopoDsMesh) -> None:
    self.mesh_vertices_array = shape.mesh_vertices_array
    self.mesh_index = shape.mesh_index
    self.mesh_format = shape.mesh_format
    self.bbox = shape.bbox()
    self.color = shape.color

  @property
  def mesh_vertices(self):
    return list(self.mesh_vertices_array.flatten())

  def to_center(self, center):
    self.mesh_vertices_array[:, :3] -= center


class TopoRenderer(BaseRenderer):

  DEFAULT_SCALE = 0.25

  def __init__(self, shapes: List[TopoDsMesh] = None, **kwargs):
    # if shapes:
    #   self.shapes = [_TopoDsMesh(x) for x in shapes]

    #   bboxs = np.array([x.bbox for x in self.shapes])
    #   bbox = np.vstack([np.min(bboxs, axis=0), np.max(bboxs, axis=0)])
    #   center = np.average(bbox, axis=0)
    #   for shape in self.shapes:
    #     shape.to_center(center)
    # else:
    #   self.shapes = []
    self.save_shapes(shapes)

    super(TopoRenderer, self).__init__(**kwargs)

  def save_shapes(self, shapes: List[TopoDsMesh] = None):
    if shapes:
      self.shapes = [_TopoDsMesh(x) for x in shapes]

      bboxs = np.array([x.bbox for x in self.shapes])
      bbox = np.vstack([np.min(bboxs, axis=0), np.max(bboxs, axis=0)])
      center = np.average(bbox, axis=0)
      for shape in self.shapes:
        shape.to_center(center)
    else:
      self.shapes = []

  def update_glsl(self):
    asp = self.width / float(self.height)
    proj = Matrix().view_clip(left=-asp,
                              right=asp,
                              bottom=-1,
                              top=1,
                              near=1,
                              far=100,
                              perspective=1)
    self.canvas['projection_mat'] = proj

  def setup_scene(self):
    self.rotx = []
    self.roty = []
    self.scale = []
    self.mesh = []

    for shape in self.shapes:
      Color(*shape.color)

      PushMatrix()
      Translate(0, 0, -3)

      self.rotx.append(Rotate(0, 1, 0, 0))
      self.roty.append(Rotate(0, 0, 1, 0))
      self.scale.append(Scale(self.DEFAULT_SCALE))

      UpdateNormalMatrix()
      mesh = Mesh(vertices=shape.mesh_vertices,
                  indices=shape.mesh_index,
                  fmt=shape.mesh_format,
                  mode='triangles')
      self.mesh.append(mesh)
      PopMatrix()

  def on_touch_move(self, touch):
    ax, ay = self.rotate_angle(touch)
    for rx, ry in zip(self.rotx, self.roty):
      rx.angle += ay
      ry.angle += ax
    self.update_glsl()

  def scale_screen(self, touch):
    if touch.button == 'scrolldown':
      new_scale = self.scale[0].xyz[0] / self.SCALE_FACTOR
    elif touch.button == 'scrollup':
      new_scale = self.scale[0].xyz[0] * self.SCALE_FACTOR
    else:
      raise AssertionError(str(touch.button))

    if self.SCALE_RANGE[0] <= new_scale <= self.SCALE_RANGE[1]:
      for scale in self.scale:
        scale.xyz = (new_scale,) * 3


if __name__ == "__main__":
  from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCone

  box = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()
  cone = BRepPrimAPI_MakeCone(1.2, 0.8, 0.5).Shape()
  box_mesh = TopoDsMesh(shapes=[box],
                        linear_deflection=0.1,
                        color=(0.5, 1.0, 1.0, 0.5))
  cone_mesh = TopoDsMesh(shapes=[cone],
                         linear_deflection=0.1,
                         color=(1.0, 0.5, 1.0, 0.5))

  class _RendererApp(UtfApp):

    def build(self):
      return TopoRenderer(shapes=[box_mesh, cone_mesh])

  _RendererApp().run()
