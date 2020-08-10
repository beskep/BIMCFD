import sys
from pathlib import Path

from kivy.core.text import LabelBase
from kivy.properties import ObjectProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button

_SRC_DIR = Path(__file__).parents[1]
assert _SRC_DIR.exists(), str(_SRC_DIR)
if str(_SRC_DIR) not in sys.path:
  sys.path.append(str(_SRC_DIR))

from interface.utfapp import UtfApp
from interface.topo_widget import TopoRenderer, TopoDsMesh
from converter.ifc_converter import IfcConverter

_SRC_DIR = Path(__file__).parents[1]


def register_font():
  font_path = _SRC_DIR / './etc/NotoSansCJKkr-DemiLight.otf'
  assert font_path.exists()
  LabelBase.register(name='NotoSansKR', fn_regular=str(font_path))


class FileChooseDialog(FloatLayout):
  # file_choose = ObjectProperty(None)
  # cancel = ObjectProperty(None)
  dismiss_popup = ObjectProperty(None)


class TopoLayout(FloatLayout):
  ifc_path = ObjectProperty(None)

  def dismiss_popup(self):
    self._popup.dismiss()

  def show_file_choose_dialog(self):
    content = FileChooseDialog(
        dismiss_popup=self.dismiss_popup,
        # cancel=self.dismiss_popup,
        # file_choose=self.file_choose,
    )
    self._popup = Popup(title='File choose',
                        content=content,
                        size_hint=(0.9, 0.9))
    self._popup.open()

  def file_choose(self, path, file_path):
    return file_path[0]


class TopoApp(UtfApp):
  linear_deflection = 0.1
  angular_deflection = 0.5

  def __init__(self, **kwargs):
    super(TopoApp, self).__init__(**kwargs)
    self._ifc_path = None
    self._converter = None
    self._space = None
    self._vis_layout: BoxLayout = None

  def file_choose(self, path):
    if len(path) != 1:
      # message
      return

    path = Path(path[0])
    if path.suffix.lower() == '.ifc':
      self._ifc_path = path
      try:
        self._converter = IfcConverter(path)
        self.send_message('IFC 파일 로드 성공')
      except:
        self.send_message('IFC 로드 실패')

    else:
      self.send_message('IFC 파일을 골라주세요')

  def send_message(self, text):
    self.root.ids.message.text = text

  def show_space_list(self, button: Button):
    if self._converter is None:
      self.send_message('IFC 파일이 로드되지 않았습니다')
      return

    spaces = self._converter.ifc.by_type('IfcSpace')

    drop_down = DropDown()
    drop_down.bind(on_select=self.select_space)
    for space in spaces:
      id_ = space.id()
      name = space.Name

      item = Button(text='{} (id: {})'.format(name, id_),
                    size_hint_y=None,
                    height=50)
      item.space = space

      item.bind(on_release=lambda btn: drop_down.select(btn))
      drop_down.add_widget(item)
    drop_down.open(button)

  def select_space(self, drop_down, button: Button, *args):
    self._space = button.space
    self.send_message('Space 선택됨')

  def visualize(self):
    if self._converter is None:
      self.send_message('IFC 파일이 로드되지 않았습니다')
      return

    if self._space is None:
      self.send_message('공간이 선택되지 않았습니다')
      return

    try:
      _, space, _, openings = self._converter.convert_space(self._space)
      space_mesh = TopoDsMesh([space],
                              linear_deflection=self.linear_deflection,
                              angular_deflection=self.angular_deflection,
                              color=(1.0, 1.0, 1.0, 0.5))
      mesh = [space_mesh]
      if openings:
        opening_mesh = TopoDsMesh(openings,
                                  linear_deflection=self.linear_deflection,
                                  angular_deflection=self.angular_deflection,
                                  color=(0.5, 0.5, 1.0, 0.5))
        mesh.append(opening_mesh)

      if self._vis_layout is None:
        self._vis_layout = self.root.ids.vis_layout

      self._vis_layout.clear_widgets()
      self._vis_layout.add_widget(TopoRenderer(shapes=mesh))

      self.send_message('')
    except:
      self.send_message('FAILED')


def run_topo_app():
  register_font()

  file_dir = Path(__file__).parent
  kv_path = file_dir / 'topo_app.kv'
  assert kv_path.exists()

  TopoApp().run(file_path=kv_path)


if __name__ == "__main__":
  run_topo_app()
