from typing import Callable, Optional

from OCC.Display import OCCViewer, SimpleGui


class Vis:
  _display: Optional[OCCViewer.Viewer3d] = None
  _start: Optional[Callable] = None

  @classmethod
  def init_display(cls, **kwargs):
    if cls._display is not None:
      raise AttributeError('Display already initialized')

    for color in ['background_gradient_color1', 'background_gradient_color2']:
      if color not in kwargs:
        kwargs[color] = [255, 255, 255]

    if 'display_triedron' not in kwargs:
      kwargs['display_triedron'] = False

    cls._display, cls._start, _, _ = SimpleGui.init_display(**kwargs)
    cls._display.EnableAntiAliasing()

  @classmethod
  def visualize(cls, shape, erase=True, **kwargs):
    init_kwargs = kwargs.pop('init_kwargs', None)

    if cls._display is None:
      if init_kwargs is None:
        init_kwargs = dict()
      cls.init_display(**init_kwargs)
    elif erase:
      cls._display.EraseAll()

    if 'color' in kwargs and not isinstance(kwargs['color'], str):
      kwargs['color'] = OCCViewer.rgb_color(*kwargs['color'])
    if 'transparency' not in kwargs:
      kwargs['transparency'] = 0.0

    cls._display.DisplayShape(shape, **kwargs)
    cls._display.FitAll()

  @classmethod
  def save_image(cls, path):
    cls._display.FitAll()
    cls._display.ExportToImage(path)

  @property
  def display(self) -> OCCViewer.Viewer3d:
    return self.display

  @classmethod
  def start_display(cls):
    # pylint: disable=not-callable
    cls._start()
