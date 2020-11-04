import itertools

from kivy.lang.builder import Builder
from kivy.properties import NumericProperty, StringProperty
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.dropdownitem import MDDropDownItem
from kivymd.uix.menu import MDDropdownMenu, RightContent
from kivymd.uix.textfield import MDTextField


class RightContentCls(RightContent):
  flag_kv_loaded = False
  kv = '''
<RightContentCls>
  disabled: True

  MDIconButton:
    icon: root.icon
    user_font_size: "16sp"
    pos_hint: {"center_y": .5}

  MDLabel:
    text: root.text
    font_style: "Caption"
    size_hint_x: None
    width: self.texture_size[0]
    text_size: None, None
  '''

  def __init__(self, *args, **kwargs):
    self.load_kv()
    super(RightContentCls, self).__init__(*args, **kwargs)

  @classmethod
  def load_kv(cls):
    if not cls.flag_kv_loaded:
      Builder.load_string(cls.kv)
      cls.flag_kv_loaded = True


class DropDownMenu(MDBoxLayout):
  text = StringProperty('')
  text_width = NumericProperty(0)
  font_name = StringProperty('')

  # TODO: 초기 theme_text_color, 설정 후 색 변경

  def __init__(self, *args, **kwargs):
    super(DropDownMenu, self).__init__(*args, **kwargs)

    self._button = MDDropDownItem()
    self._button.size_hint_x = 1
    self._menu: MDDropdownMenu = None
    self._text_width = None
    self._selected_item = None

    self.add_widget(self._button)

  @property
  def button(self):
    return self._button

  @property
  def menu(self):
    if self._menu is None:
      self._menu = MDDropdownMenu()
      self._menu.width_mult = 8
      self._items = None
      # TODO: items 없애고 menu 아이템으로 연결

      self._menu.caller = self._button
      self._menu.callback = self.select_item
      self._button.on_release = self._menu.open

    return self._menu

  @property
  def selected_item(self):
    return self._selected_item

  def selected_item_text(self) -> str:
    return self.selected_item.text

  @property
  def items(self):
    return self._items

  @items.setter
  def items(self, value: list):
    self._items = value
    self.menu.items = value

  def set_items(self, text: list, right_text=None, icon=None, right_icon=None):
    # TODO: 첫 번째 right_text가 None으로 설정되는 문제 해결
    # TODO: 'icon' 설정 안되는 문제 해결
    if right_text is None or isinstance(right_text, str):
      right_text = itertools.repeat(right_text, len(text))

    if icon is None or isinstance(icon, str):
      icon = itertools.repeat(icon, len(text))

    if right_icon is None or isinstance(right_icon, str):
      right_icon = itertools.repeat(right_icon, len(text))

    # test = [list(x) for x in zip(text, right_text, icon, right_icon)]

    items = [
        make_drop_down_item(*x) for x in zip(text, right_text, icon, right_icon)
    ]
    self.items = items

  def set_button_text(self, value: str):

    if self._text_width is not None:
      value = value.ljust(self._text_width)

    self._button.text = value

  def select_item(self, item):
    self._selected_item = item
    self.set_button_text(item.text)
    self.menu.dismiss()

  def on_text(self, instance, value):
    assert isinstance(value, str)
    self.set_button_text(value)

  def on_font_name(self, instance, value):
    self._button.ids.label_item.font_name = value

  def on_text_width(self, instance, value):
    assert isinstance(value, (int, float))
    self._text_width = int(value)
    self.set_button_text(self._button.text)


def make_drop_down_item(text, right_text=None, icon=None, right_icon=None):
  item = {'text': text}

  if right_text is not None:
    rc = (RightContentCls(text=right_text) if right_icon is None else
          RightContentCls(text=right_text, icon=right_icon))
    item['right_content_cls'] = rc

  if icon is not None:
    item['icon'] = icon

  return item


if __name__ == "__main__":
  from kivy.lang import Builder
  from kivymd.app import MDApp

  kv = '''
MDBoxLayout:
    MDRaisedButton:
        text: "1"
        size_hint: (1, 1)
    DropDownMenu:
        id: test
        text: "PRESS ME"
        pos_hint: {"center_x": .5, "center_y": .5}
        size_hint: (1, 1)
    MDRaisedButton:
        text: "2"
        size_hint: (1, 1)
'''

  class Tester(MDApp):

    def build(self):
      # widget = DropDownMenu()
      # widget.items = [{"text": f"Item {i}"} for i in range(5)]

      widget = Builder.load_string(kv)
      ddm: DropDownMenu = widget.ids.test
      # widget.ids.test.items = [{"text": f"Item {i}"} for i in range(5)]
      ddm.set_items(text=['test {}'.format(x) for x in range(7)],
                    right_text=['rtest {}'.format(x) for x in range(7)],
                    icon='git',
                    right_icon='apple-keyboard-command')
      ddm._menu.open()

      return widget

  Tester().run()
