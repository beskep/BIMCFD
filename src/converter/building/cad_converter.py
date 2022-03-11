from pathlib import Path

from loguru import logger
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh

from Extend.DataExchange import read_step_file, read_stl_file

from converter.geom_utils import (align_obj_to_origin, fix_shape,
                                  geometric_features, make_external_zone)
from converter.obj_convert import write_obj_from_dict

from .converter import Converter


def _read_cad(path):
  path = Path(path)
  suffix = path.suffix.lower()

  if suffix == '.stl':
    shape = read_stl_file(path.as_posix())
  elif suffix in ('.step', '.stp'):
    shape = read_step_file(path.as_posix())
  else:
    raise ValueError(f'File extension not in (.stl, .step, .stp): {path}')

  return shape


class CADConverter(Converter):

  def __init__(self, path) -> None:
    self._shape = _read_cad(path=path)

  @property
  def shape(self):
    return self._shape

  def simplify_space(self, *args, **kwargs):
    logger.debug('CADConverter -> simplify하지 않음')

    fix_shape(self._shape)
    BRepMesh_IncrementalMesh(self._shape, self._brep_deflection[0], False,
                             self._brep_deflection[1], True)

    simplified = {
        'original': self.shape,
        'simplified': self.shape,
        'wall_names': ('0',),
        'info': {
            'simplification': {
                'is_simplified': False
            },
            'original_geometry': geometric_features(self.shape)
        }
    }
    return simplified

  def save_simplified_space(self, *args, **kwargs):
    logger.debug('CADConverter -> skip save_simplified_space')

  def _write_openfoam_object(self, options: dict, simplified: dict,
                             working_dir: Path):
    _, external_faces = make_external_zone(
        shape=self.shape,
        buffer=options['external_zone_size'],
        inner_buffer=options.get('inner_buffer', 0.2),
        vertical_dim=int(options.get('vertical_dimension', 2)),
        each_wall=options.get('flag_each_wall', False))

    path = working_dir.joinpath('geometry.obj')
    write_obj_from_dict(faces=external_faces,
                        obj_path=path,
                        deflection=self._brep_deflection)
    align_obj_to_origin(path)
