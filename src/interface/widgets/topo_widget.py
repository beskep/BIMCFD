from typing import List

from utils import RESOURCE_DIR

import numpy as np
from kivy import graphics
from kivy.app import App
from kivy.graphics.transformation import Matrix
from kivy.resources import resource_find
from kivy.uix.widget import Widget

from interface.widgets.topo_mesh import TopoDsMesh

_GLSL_PATH = RESOURCE_DIR.joinpath('color.glsl')
assert _GLSL_PATH.exists()


class BaseRenderer(Widget):

  ROTATE_SPEED = 1.0
  SCALE_RANGE = (0.01, 100.0)
  SCALE_FACTOR = 2.0

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

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
    self.mesh_vertices_array: np.ndarray = shape.mesh_vertices_array
    self.mesh_index = shape.mesh_index
    self.mesh_format = shape.mesh_format
    self.bbox: np.ndarray = shape.bbox()
    self.color = shape.color

  @property
  def mesh_vertices(self):
    return list(self.mesh_vertices_array.flatten())

  def to_center(self, center):
    self.mesh_vertices_array[:, :3] -= center
    self.bbox -= center


class TopoRenderer(BaseRenderer):
  # TODO: 생성 직후 업데이트

  DEFAULT_SCALE = 0.25

  def __init__(self, shapes: List[TopoDsMesh] = None, **kwargs):
    self.save_shapes(shapes)

    super().__init__(**kwargs)

  def save_shapes(self, shapes: List[TopoDsMesh] = None):
    if shapes:
      self.shapes = [_TopoDsMesh(x) for x in shapes]

      bboxs = np.vstack([x.bbox for x in self.shapes])
      bbox = np.vstack([np.min(bboxs, axis=0), np.max(bboxs, axis=0)])
      center = np.average(bbox, axis=0)
      for shape in self.shapes:
        shape.to_center(center)
      self.bbox = bbox
      self.topo_center = center
    else:
      self.shapes = []
      self.bbox = [[-1, -1, -1], [1, 1, 1]]
      self.topo_center = [0, 0, 0]

  def update_glsl(self):
    """
    Widget의 중앙에 mesh가 표시되도록 예제를 변경
    https://stackoverflow.com/questions/59664392/kivy-how-to-render-3d-model-in-a-given-layout

    TODO: topo 위치, 스케일 조정
    """
    app = App.get_running_app()

    # Calculate new left edge for the clip matrix
    ratio = app.root.width / self.width
    scene_width0 = np.abs(self.bbox[0, 0] - self.bbox[1, 0])
    scene_width1 = ratio * scene_width0
    left = np.min(self.bbox[:, 0]) - (scene_width1 - scene_width0)
    right = np.max(self.bbox[:, 0])

    # calculate new top and bottom of clip frustum
    # that maintains topo aspect ratio
    scene_vertical_center = np.mean(self.bbox[:, 1])
    scene_height0 = np.abs(self.bbox[0, 1] - self.bbox[1, 1])
    scene_height1 = ratio * scene_height0
    bottom = scene_vertical_center - scene_height1 / 2.0
    top = scene_vertical_center + scene_height1 / 2.0

    # create new clip matrix
    far = np.abs(self.bbox[0, 2] - self.bbox[1, 2]) - self.translate.z
    proj = Matrix().view_clip(left=left,
                              right=right,
                              bottom=bottom,
                              top=top,
                              near=1,
                              far=far,
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

      self.translate = graphics.Translate(0, 0, -3)

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
  from kivymd.app import MDApp

  box = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()
  cone = BRepPrimAPI_MakeCone(1.2, 0.8, 0.5).Shape()
  box_mesh = TopoDsMesh(shapes=[box],
                        linear_deflection=0.1,
                        color=(0.5, 1.0, 1.0, 0.5))
  cone_mesh = TopoDsMesh(shapes=[cone],
                         linear_deflection=0.1,
                         color=(1.0, 0.5, 1.0, 0.5))

  class _RendererApp(MDApp):

    def build(self):
      return TopoRenderer(shapes=[box_mesh, cone_mesh])

  _RendererApp().run()
