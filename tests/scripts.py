import os
import sys

SRC_DIR = os.path.abspath(os.path.join(__file__, '../../src'))
assert os.path.exists(SRC_DIR)
if SRC_DIR not in sys.path:
  sys.path.append(SRC_DIR)

from converter import ifc_converter


def align_shape():
  from pathlib import Path

  from converter.simplify import align_model
  from OCC.Extend import DataExchange

  ifc_path = r'D:\repo\IFC\DURAARK Datasets\Academic\Academic_DDB-Soeholm_Arch.ifc'
  space_name = 'K04'
  space_id = 694
  res_dir = Path(r'D:\CFD\test')

  converter = ifc_converter.IfcConverter(ifc_path)
  space = converter.ifc.by_id(space_id)
  shape = converter.convert_space(space)[0]

  aligned = align_model(shape)
  aligned_path = res_dir / '{}_aligned_original.stl'.format(space_name)
  DataExchange.write_stl_file(aligned, str(aligned_path))

  for dist in [0.1, 0.5, 1.0]:
    res = converter.simplify_space(spaces=space_id,
                                   threshold_dist=dist,
                                   relative_threshold=True)
    simplfied_shape = res['simplified']
    aligned = align_model(simplfied_shape)
    res_path = res_dir / '{}_aligned_dist{}.stl'.format(space_name, dist)
    DataExchange.write_stl_file(aligned, str(res_path))


def write_each():
  from pathlib import Path

  ifc_path = r'D:\Python\IFCConverter\IFCConverter\src\NIBS\Clinic_A_20110906_optimized.ifc'
  space_id = 230

  converter = ifc_converter.IfcConverter(ifc_path)
  # spaces = converter.ifc.by_type('IfcSpace')
  space = converter.ifc.by_id(space_id)

  for dist in [0.0, 0.1, 0.5, 1.0]:
    res = converter.simplify_space(spaces=space_id,
                                   threshold_dist=dist,
                                   relative_threshold=True)
    simplified_shape = res['simplified']

    res_dir = Path(r'D:\CFD\test\{}_each_dist{}'.format(space.Name, dist))
    ifc_converter.write_each_shapes(shape=simplified_shape, save_dir=res_dir)
