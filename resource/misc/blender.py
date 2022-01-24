import sys


def _stl_to_obj(obj_path, *args):
  import bpy

  for path in args:
    bpy.ops.import_mesh.stl(filepath=path, axis_forward='-Z', axis_up='Y')

  bpy.ops.export_scene.obj(filepath=obj_path,
                           axis_forward='-Z',
                           axis_up='Y',
                           use_materials=True,
                           group_by_object=False,
                           group_by_material=True)


if __name__ == '__main__':
  assert len(sys.argv) > 4
  _stl_to_obj(sys.argv[5], *sys.argv[6:])
