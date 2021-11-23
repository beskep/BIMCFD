import os

import utils

import ifcopenshell
import ifcopenshell.geom
import pytest

from converter import ifc_converter, openfoam
from converter.openfoam import OpenFoamCase

ifc_path = r'test\data\Academic_Autodesk-AdvancedSampleProject_Arch.ifc'
converter = ifc_converter.IfcConverter(ifc_path)
result_dir = r'D:\CFD\test'


def test_openfoam_temperature():
  target_space = converter.ifc.by_id(3744)

  shape, space, walls, openings = converter.convert_space(target_space)

  valid_walls = [x for x in walls if converter._layers_info(x) is not None]

  temperature = converter.openfoam_temperature_info(valid_walls)

  assert isinstance(temperature, str)
  assert temperature != ''

  with open(r'D:\CFD\test\T', 'w') as f:
    f.write(temperature)


def test_get_storey_of_walls():
  walls = converter.ifc.by_type('IfcWall')
  for wall in walls[:10]:
    storey = ifc_converter.get_storey(wall)
    if storey is not None:
      assert isinstance(storey, ifcopenshell.entity_instance)


def test_get_storey_of_slabs():
  slabs = converter.ifc.by_type('IfcSlab')
  for slab in slabs[:5]:
    storey = ifc_converter.get_storey(slab)
    if storey is not None:
      assert isinstance(storey, ifcopenshell.entity_instance)


def test_simpl_and_temperature_info():
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


def test_simplify():
  from pprint import pprint

  from converter import simplify

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


def test_single_openfoam_case():
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


def test_openfoam_case_all_solvers():
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


def test_openfoam_wall_temperature_dict():
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


def test_openfoam_rough_wall_nut_dict():
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


def test_openfoam_zero_boundary_field():
  from copy import deepcopy

  from converter.openfoam import OpenFoamCase

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


def test_external_zone():
  from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Shape

  from converter.obj_convert import write_obj
  from converter.simplify import make_external_zone

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

  for key, value in zone_faces.items():
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


def test_openfoam_option():
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


def test_openfoam_case_from_template():
  solver = 'simpleFoam'
  save_dir = r'D:\test'
  name = 'test_case'

  OpenFoamCase.from_template(solver=solver, save_dir=save_dir, name=name)
