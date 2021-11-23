import sys

import ifcopenshell
import ifcopenshell.geom
from fuzzywuzzy import fuzz, process
from scipy.spatial.transform import _rotation_groups

from src import utils
from src.converter import ifc_converter, simplify
from src.interface import bim_cfd_app, bim_cfd_base
from src.OCCUtils import Common, Construct, edge, face

if __name__ == '__main__':
  level = 10 if any('debug' in x.lower() for x in sys.argv) else 20
  utils.set_logger(level=level)

  bim_cfd_app.main()
