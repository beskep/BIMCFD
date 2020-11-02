import sys
from pathlib import Path

import pytest

_SRC_DIR = Path(__file__).parents[1] / 'src'
assert _SRC_DIR.exists()
sys.path.append(str(_SRC_DIR))

from converter.ifc_converter import IfcConverter
from interface import topo_widget as wg


@pytest.mark.skip
def test_space_visualization():
  ifc_path = Path(
      r'D:\repo\IFC\DURAARK Datasets\Academic_Autodesk\Academic_Autodesk-AdvancedSampleProject_Arch.ifc'
  )
  cnv = IfcConverter(ifc_path)

  space = cnv.ifc.by_id(3744)
  shape, space, walls, openings = cnv.convert_space(space)
  # walls = [cnv.create_geometry(x) for x in walls]

  space_mesh = wg.TopoDsMesh([space],
                             linear_deflection=0.1,
                             color=(1.0, 1.0, 1.0, 0.5))
  opening_mesh = wg.TopoDsMesh(openings,
                               linear_deflection=0.1,
                               color=(0.5, 0.5, 1.0, 0.5))

  class RendererApp(wg.UtfApp):

    def build(self):
      return wg.TopoRenderer(shapes=[space_mesh, opening_mesh])

  RendererApp().run()


if __name__ == "__main__":
  pytest.main(['-v', '-s', '-k', 'test_gui'])