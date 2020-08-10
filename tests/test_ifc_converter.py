import os
import sys

import ifcopenshell
import ifcopenshell.geom
import pytest

_SRC_DIR = os.path.abspath(os.path.join(__file__, '../../src'))
assert os.path.exists(_SRC_DIR)
if _SRC_DIR not in sys.path:
  sys.path.append(_SRC_DIR)

from converter import openfoam
from converter import ifc_converter


@pytest.fixture
def converter():
  ifc_path = r'D:\repo\IFC\DURAARK Datasets\Academic_Autodesk\Academic_Autodesk-AdvancedSampleProject_Arch.ifc'
  assert os.path.exists(ifc_path)

  return ifc_converter.IfcConverter(ifc_path)


def test_openfoam_temperature(converter):
  target_space = converter.ifc.by_id(3744)

  shape, space, walls, openings = converter.convert_space(target_space)

  valid_walls = [x for x in walls if converter._layers_info(x) is not None]

  temperature = converter.openfoam_temperature_info(valid_walls)

  assert isinstance(temperature, str)
  assert temperature != ''

  with open(r'D:\CFD\test\T', 'w') as f:
    f.write(temperature)


def test_get_storey_of_walls(converter: ifc_converter.IfcConverter):
  walls = converter.ifc.by_type('IfcWall')
  for wall in walls[:10]:
    storey = ifc_converter.get_storey(wall)
    if storey is not None:
      assert isinstance(storey, ifcopenshell.entity_instance)


def test_get_storey_of_slabs(converter: ifc_converter.IfcConverter):
  slabs = converter.ifc.by_type('IfcSlab')
  for slab in slabs[:5]:
    storey = ifc_converter.get_storey(slab)
    if storey is not None:
      assert isinstance(storey, ifcopenshell.entity_instance)


def test_simpl_and_temperature_info(converter: ifc_converter.IfcConverter):
  result_dir = r'D:\CFD\test'
  space = converter.ifc.by_id(3744)

  converter.simplify_space(spaces=space,
                           threshold_volume=0.0,
                           threshold_dist=0.0,
                           threshold_angle=0.0,
                           save_dir=result_dir,
                           case_name='00_original')
  result = converter.simplify_space(spaces=space,
                                    threshold_volume=0.0,
                                    threshold_dist=5.0,
                                    threshold_angle=(3.141592 / 2.0),
                                    save_dir=result_dir,
                                    case_name='02_simplified')
  temperature = converter.openfoam_temperature_info(
      walls=result['walls'],
      surface_names=result['wall_names'],
      external_temperature=300,
      heat_transfer_coefficient=None)
  with open(os.path.join(result_dir, 'T'), 'w') as f:
    f.writelines(temperature)


def test_simplify(converter: ifc_converter.IfcConverter):
  from converter import simplify
  from pprint import pprint

  result_dir = r'D:\CFD\test'
  case_name = 'IfcConvert'

  converter.threshold_surface_dist = 5.0
  converter.threshold_surface_angle = (3.141592 / 2)

  space = converter.ifc.by_id(3744)

  result = converter.simplify_space(spaces=space,
                                    save_dir=result_dir,
                                    case_name=case_name,
                                    relative_threshold=True,
                                    preserve_opening=True,
                                    opening_volume=True)

  pprint(result['info']['fused_geometry'])

  simplified_shape = result['fused']
  assert simplified_shape is not None
  exp = simplify.TopologyExplorer(simplified_shape)
  assert exp.number_of_solids() == 1


def test_single_openfoam_case(converter: ifc_converter.IfcConverter):
  result_dir = r'D:\CFD\test'
  case_name = 'IfcConvert'

  converter.threshold_surface_dist = 5.0
  converter.threshold_surface_angle = (3.141592 / 2)
  converter.minimum_match_score = 20

  converter.set_openfoam_options(solver='buoyantSimpleFoam',
                                 min_cell_size=0.0005,
                                 boundary_layers={
                                     'nLayers': 3,
                                     'thicknessRatio': 1.2
                                 })

  space = converter.ifc.by_id(3744)

  converter.openfoam_case(spaces=space,
                          save_dir=result_dir,
                          case_name=case_name,
                          openfoam_options=None)

  for d in ('geometry', '0', 'constant', 'system'):
    assert os.path.exists(os.path.join(result_dir, case_name, d))

  assert os.path.exists(os.path.join(result_dir, case_name, 'geometry.obj'))


def test_openfoam_case_all_solvers(converter: ifc_converter.IfcConverter):
  result_dir = r'D:\CFD\test'

  converter.threshold_surface_dist = 5.0
  converter.threshold_surface_angle = (3.141592 / 2)
  converter.minimum_match_score = 20

  space = converter.ifc.by_id(3744)

  for solver in openfoam.supported_solvers():
    case_name = 'IfcConvert' + solver[0].upper() + solver[1:]
    converter.set_openfoam_options(solver=solver)
    converter.openfoam_case(spaces=space,
                            save_dir=result_dir,
                            case_name=case_name)

    for d in ('geometry', '0', 'constant', 'system'):
      assert os.path.exists(os.path.join(result_dir, case_name, d))

    assert os.path.exists(os.path.join(result_dir, case_name, 'geometry.obj'))


def test_openfoam_wall_temperature_dict(converter: ifc_converter.IfcConverter):
  from collections import OrderedDict

  space = converter.ifc.by_id(3744)
  converter.threshold_surface_dist = 5.0
  converter.threshold_surface_angle = (3.141592 / 2)

  simplify_result = converter.simplify_space(spaces=space,
                                             save_dir=None,
                                             case_name=None)
  bf = converter.openfoam_temperature_dict(
      solver='buoyantSimpleFoam',
      surface=simplify_result['walls'],
      surface_name=simplify_result['wall_names'],
      min_score=70,
      temperature=300,
      heat_transfer_coefficient=1e10)

  assert isinstance(bf, OrderedDict)


def test_openfoam_rough_wall_nut_dict(converter: ifc_converter.IfcConverter):
  from collections import OrderedDict

  space = converter.ifc.by_id(3744)
  converter.threshold_surface_dist = 5.0
  converter.threshold_surface_angle = (3.141592 / 2)

  simplify_result = converter.simplify_space(spaces=space,
                                             save_dir=None,
                                             case_name=None)

  bf = converter.openfoam_rough_wall_nut_dict(
      solver='buoyantSimpleFoam',
      surface=simplify_result['walls'],
      surface_name=simplify_result['wall_names'])

  assert isinstance(bf, OrderedDict)


def test_openfoam_zero_boundary_field(converter: ifc_converter.IfcConverter):
  from converter.openfoam import OpenFoamCase
  from copy import deepcopy

  space = converter.ifc.by_id(3744)
  converter.threshold_surface_dist = 5.0
  converter.threshold_surface_angle = (3.141592 / 2)

  simplify_result = converter.simplify_space(spaces=space,
                                             save_dir=None,
                                             case_name=None)

  case = OpenFoamCase.from_template(solver='buoyantSimpleFoam',
                                    save_dir='',
                                    name=None)

  with pytest.raises(ValueError):
    converter.openfoam_zero_boundary_field(
        case=case,
        wall_names=simplify_result['wall_names'],
        opening_names=simplify_result['opening_names'],
        target_fields=['spam', 'egg', 'bacon'])

  ff_template = deepcopy(case.foam_files)
  wall_names = ['Surface_' + x for x in simplify_result['wall_names']]
  opening_names = simplify_result['opening_names']
  drop_fields = ['T', 'nut']
  converter.openfoam_zero_boundary_field(case=case,
                                         wall_names=wall_names,
                                         opening_names=opening_names,
                                         drop_fields=drop_fields)
  ff = case.foam_files
  assert ff_template != ff

  ff_target = [
      x for x in ff if x.location == '"0"' and x.name not in drop_fields
  ]
  wall_names_set = set(wall_names)
  opening_names_set = set(opening_names)
  for ff in ff_target:
    bf = ff.values['boundaryField']
    bf_keys = set(bf.keys())
    assert wall_names_set <= bf_keys
    assert opening_names_set <= bf_keys


def test_external_zone(converter: ifc_converter.IfcConverter):
  from converter.simplify import make_external_zone
  from converter.obj_convert import write_obj
  from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face

  result_dir = r'D:\CFD\test'

  space = converter.ifc.by_id(3744)
  converter.threshold_surface_dist = 5.0
  converter.threshold_surface_angle = (3.141592 / 2)

  simplified = converter.simplify_space(spaces=space,
                                        save_dir=None,
                                        case_name=None)

  zone, zone_faces = make_external_zone(simplified['simplified'],
                                        buffer_size=5,
                                        vertical_dim=2)
  assert isinstance(zone, TopoDS_Shape)
  assert isinstance(zone_faces, dict)

  for key, value in zone_faces:
    assert isinstance(key, str)
    assert isinstance(value, TopoDS_Face)

  obj_path = os.path.join(result_dir, 'zone_test.obj')
  write_obj(compound=simplified['simplified'],
            space=simplified['space'],
            openings=simplified['openings'],
            walls=[converter.create_geometry(x) for x in simplified['walls']],
            obj_path=obj_path,
            deflection=converter.brep_deflection,
            wall_names=simplified['wall_names'],
            additional_faces=zone_faces)
  assert os.path.exists(obj_path)


def test_boundary_field_dict():
  bfd = openfoam.BoundaryFieldDict()

  bfd['key1'] = 'value1'
  bfd.add_comment('test comment')
  bfd.add_empty_line()
  bfd['key2'] = 'value2'
  bfd.add_empty_line()

  from pprint import pprint

  pprint(bfd)


def test_openfoam_option(converter: ifc_converter.IfcConverter):
  from pprint import pprint

  default_options = converter._default_openfoam_options.copy()
  print('default options:')
  pprint(default_options)

  converter.set_openfoam_options(solver='buoyantSimpleFoam',
                                 grid_resolution=1.0,
                                 temperature=300)
  options = converter.openfoam_options
  assert all(x in options for x in default_options.keys())
  assert options['solver'] == 'buoyantSimpleFoam'
  assert options['grid_resolution'] == 1.0
  assert options['temperature'] == 300

  print('options:')
  pprint(options)


if __name__ == '__main__':
  pytest.main(['-v', '-s', '-k', 'test_single_openfoam_case'])
