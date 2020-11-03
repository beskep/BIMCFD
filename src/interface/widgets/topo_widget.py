from typing import List

import numpy as np
from kivy import graphics
from kivy.graphics.transformation import Matrix
from kivy.resources import resource_find
from kivy.uix.widget import Widget

from interface.utf_app import UtfApp
from interface.widgets.topo_mesh import TopoDsMesh
from utils import RESOURCE_DIR

_GLSL_PATH = RESOURCE_DIR.joinpath('color.glsl')
assert _GLSL_PATH.exists()


class BaseRenderer(Widget):

  ROTATE_SPEED = 1.0
  SCALE_RANGE = (0.01, 100.0)
  SCALE_FACTOR = 2.0

  def __init__(self, **kwargs):
    super(BaseRenderer, self).__init__(**kwargs)

    self.canvas = graphics.RenderContext(compute_normal_mat=True)
    self.canvas.shader.source = resource_find(str(_GLSL_PATH))

    self.rotx = None
    self.roty = None
    self.scale = None

    with self.canvas:
      self.cb = graphics.Callback(self.setup_gl_context)
      graphics.PushMatrix()
      self.setup_scene()
      graphics.PopMatrix()
      self.cb = graphics.Callback(self.reset_gl_context)
    self.update_glsl()
    self.canvas['diffuse_light'] = (1.0, 1.0, 0.8)
    self.canvas['ambient_light'] = (0.1, 0.1, 0.1)

  def setup_gl_context(self, *args):
    graphics.opengl.glEnable(graphics.opengl.GL_DEPTH_TEST)

  def reset_gl_context(self, *args):
    graphics.opengl.glDisable(graphics.opengl.GL_DEPTH_TEST)

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
    graphics.Color(1, 1, 1, 1)

    graphics.PushMatrix()
    graphics.Translate(0, 0, -3)

    self.rotx = graphics.Rotate(0, 1, 0, 0)
    self.roty = graphics.Rotate(0, 0, 1, 0)
    self.scale = graphics.Scale(1.0)

    # UpdateNormalMatrix()
    # self.mesh = Mesh(
    #     vertices=m.vertices,
    #     indices=m.indices,
    #     fmt=m.vertex_format,
    #     mode='triangles',
    # )
    graphics.PopMatrix()

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
  # TODO: 바깥 창 기준으로 mesh가 위치하게 수정

  DEFAULT_SCALE = 0.25

  def __init__(self, shapes: List[TopoDsMesh] = None, **kwargs):
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
      graphics.Color(*shape.color)

      graphics.PushMatrix()
      graphics.Translate(0, 0, -3)

      self.rotx.append(graphics.Rotate(0, 1, 0, 0))
      self.roty.append(graphics.Rotate(0, 0, 1, 0))
      self.scale.append(graphics.Scale(self.DEFAULT_SCALE))

      graphics.UpdateNormalMatrix()
      mesh = graphics.Mesh(vertices=shape.mesh_vertices,
                           indices=shape.mesh_index,
                           fmt=shape.mesh_format,
                           mode='triangles')
      self.mesh.append(mesh)
      graphics.PopMatrix()

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
