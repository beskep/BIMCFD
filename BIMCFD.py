import fuzzywuzzy
import ifcopenshell
import ifcopenshell.geom
from fuzzywuzzy import fuzz, process

from src import utils
from src.converter import ifc_converter, simplify
from src.interface import bim_cfd_app, bim_cfd_base
from src.OCCUtils import Common, Construct, edge, face

bim_cfd_app.main()
