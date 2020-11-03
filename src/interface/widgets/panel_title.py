from kivy.metrics import dp
from kivy.properties import StringProperty
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDIcon, MDLabel


class PanelTitle(MDBoxLayout):
  icon = StringProperty('')
  text = StringProperty('')

  def __init__(self, **kwagrgs):
    super(PanelTitle, self).__init__(**kwagrgs)

    self._icon = MDIcon()
    self._icon.size_hint = None, None
    self._icon.width = self._icon.font_size

    self._title = MDLabel()
    self._title.theme_text_color = 'Secondary'
    self._title.size_hint_y = None
    self._title.height = self._title.font_size

    self.add_widget(self._icon)
    self.add_widget(self._title)

    self.spacing = dp(5)
    self.size_hint_y = None
    self.height = max(self._icon.font_size, self._title.font_size)

  def on_icon(self, instance, value):
    self._icon.icon = value

  def on_text(self, instance, value):
    self._title.text = value
