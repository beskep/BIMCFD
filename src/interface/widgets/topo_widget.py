from typing import List

import utils

import numpy as np
from kivy import graphics
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.transformation import Matrix
from kivy.resources import resource_find
from kivy.uix.widget import Widget

from interface.widgets.topo_mesh import TopoDsMesh

_GLSL_PATH = utils.DIR.RESOURCE.joinpath('color.glsl')
_GLSL_PATH.stat()


def ignore_under_touch(fn):

  def wrap(self, touch):
    gl: list = touch.grab_list

    if not gl or (self is gl[0]()):
      fn_ = fn(self, touch)
    else:
      fn_ = fn

    return fn_

  return wrap


class BaseRenderer(Widget):

  ROTATE_SPEED = 1.0
  SCALE_RANGE = (0.01, 100.0)
  SCALE_FACTOR = 1.5

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.canvas = graphics.RenderContext(compute_normal_mat=True)
    self.canvas.shader.source = resource_find(str(_GLSL_PATH))

    self.rotx = None
    self.roty = None
    self.scale = None

    with self.canvas:
      self.cb = graphics.Callback(self.setup_gl_context)
      self.setup_scene()
      self.cb = graphics.Callback(self.reset_gl_context)

    self.canvas['diffuse_light'] = (1.0, 1.0, 0.8)
    self.canvas['ambient_light'] = (0.1, 0.1, 0.1)

    Clock.schedule_interval(self.update_glsl, 1 / 24.0)

  def setup_gl_context(self, *args):
    graphics.opengl.glEnable(graphics.opengl.GL_DEPTH_TEST)

  def reset_gl_context(self, *args):
    graphics.opengl.glDisable(graphics.opengl.GL_DEPTH_TEST)

  def update_glsl(self, *args):
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

  def translate(self, xyz):
    self.mesh_vertices_array[:, :3] += xyz
    self.bbox += xyz

  def to_center(self, center):
    self.mesh_vertices_array[:, :3] -= center
    self.bbox -= center


class TopoRenderer(BaseRenderer):

  def __init__(self,
               shapes: List[TopoDsMesh] = None,
               default_scale=1.0,
               near=1.0,
               far=None,
               perspective=1.0,
               **kwargs):
    """표면 Mesh 시각화 Widget

    Parameters
    ----------
    shapes : List[TopoDsMesh], optional
        Shapes to visualize, by default None
    default_scale : float, optional
        default scale, by default 1.0
    near : float, optional
        (probably) coordinate of near clipping plane, by default 1.0
    far : [None, float], optional
        (probably) coordinate of far clipping plane, by default None
    perspective : float, optional
        dunno, by default 1.0
    """
    self._shapes: List = None
    self._bbox: np.ndarray = None
    self._topo_center: np.ndarray = None
    self._depth = 1.0

    self._save_shapes(shapes)

    self._default_scale = default_scale
    self._near = near
    self._far = far
    self._perspective = perspective

    super().__init__(**kwargs)

  def _save_shapes(self, shapes: List[TopoDsMesh] = None):
    if shapes:
      self._shapes = [_TopoDsMesh(x) for x in shapes]

      bboxs = np.vstack([x.bbox for x in self._shapes])
      bbox = np.vstack([np.min(bboxs, axis=0), np.max(bboxs, axis=0)])
      center = np.average(bbox, axis=0)

      if not np.all(np.isclose(center, 0.0, rtol=0.001)):
        # 중앙 정렬
        bbox -= center
        for shape in self._shapes:
          shape.to_center(center)

      self._bbox = bbox
      self._topo_center = center
    else:
      self._shapes = []
      self._bbox = np.array([[-1, -1, -1], [1, 1, 1]])
      self._topo_center = np.array([0, 0, 0])

    self._depth = np.max(self._bbox) - np.min(self._bbox)

  def _boundary(self):
    app = App.get_running_app()

    # Calculate new left edge for the clip matrix
    if app.root:
      ratio = app.root.width / self.width
    else:
      ratio = 1.0

    scene_width0 = np.abs(self._bbox[0, 0] - self._bbox[1, 0])
    scene_width1 = ratio * scene_width0
    left = np.min(self._bbox[:, 0]) - (scene_width1 - scene_width0)
    right = np.max(self._bbox[:, 0])

    # calculate new top and bottom of clip frustum that maintains topo aspect ratio
    scene_vertical_center = np.mean(self._bbox[:, 1])
    scene_height0 = np.abs(self._bbox[0, 1] - self._bbox[1, 1])
    scene_height1 = scene_height0 * ratio
    # scene_height1 = scene_height0 * (self.height / self.width)
    bottom = scene_vertical_center - scene_height1 / 2.0
    top = scene_vertical_center + scene_height1 / 2.0

    return left, right, top, bottom

  def update_glsl(self, *args):
    """
    Widget의 중앙에 mesh가 표시되도록 예제를 변경
    - https://stackoverflow.com/questions/59664392/kivy-how-to-render-3d-model-in-a-given-layout
    - http://www.opengl-tutorial.org/beginners-tutorials/tutorial-3-matrices/#fn:projection
    """
    left, right, top, bottom = self._boundary()

    # create new clip matrix
    if self._far is None:
      far = 2.0 * self._depth
    else:
      far = self._far

    proj = Matrix().view_clip(
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        near=self._near,  # near clipping plane
        far=far,  # far clipping plane
        perspective=self._perspective)

    self.canvas['projection_mat'] = proj

  def setup_scene(self):
    self.rotx = []
    self.roty = []
    self.scale = []
    self.mesh = []
    z_translate = -1.5 * self._depth

    for shape in self._shapes:
      graphics.Color(*shape.color)
      graphics.PushMatrix()
      graphics.Translate(0, 0, z_translate)

      self.rotx.append(graphics.Rotate(0, 1, 0, 0))
      self.roty.append(graphics.Rotate(0, 0, 1, 0))
      self.scale.append(graphics.Scale(self._default_scale))

      graphics.UpdateNormalMatrix()
      mesh = graphics.Mesh(vertices=shape.mesh_vertices,
                           indices=shape.mesh_index,
                           fmt=shape.mesh_format,
                           mode='triangles')
      self.mesh.append(mesh)
      graphics.PopMatrix()

    for rx, ry in zip(self.rotx, self.roty):
      rx.angle += 30.0
      ry.angle += 30.0

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
  from kivymd.uix.boxlayout import MDBoxLayout
  from kivymd.uix.label import MDLabel

  box = BRepPrimAPI_MakeBox(2.0, 1.0, 1.0).Shape()
  cone = BRepPrimAPI_MakeCone(1.2, 0.8, 0.5).Shape()
  box_mesh = TopoDsMesh(shapes=[box],
                        linear_deflection=0.1,
                        color=(0.5, 1.0, 1.0, 0.5))
  cone_mesh = TopoDsMesh(shapes=[cone],
                         linear_deflection=0.1,
                         color=(1.0, 0.5, 1.0, 0.5))

  class _RendererApp(MDApp):

    def __init__(self, **kwargs):
      super().__init__(**kwargs)
      self._topo_renderer = None

    def build(self):
      self._topo_renderer = TopoRenderer(shapes=[box_mesh, cone_mesh],
                                         default_scale=1.0,
                                         near=0.01,
                                         perspective=0.01)

      box_layout = MDBoxLayout()
      box_layout.add_widget(MDLabel())
      box_layout.add_widget(self._topo_renderer)

      return box_layout

  _RendererApp().run()
