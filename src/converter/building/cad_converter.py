from pathlib import Path

from loguru import logger
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh

from Extend.DataExchange import read_step_file, read_stl_file

from converter.geom_utils import (align_obj_to_origin, fix_shape,
                                  make_external_zone)
from converter.obj_convert import write_obj, write_obj_from_dict

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

  def __init__(self, path, vertical_dim=2) -> None:
    self._shape = _read_cad(path=path)
    self._vertical_dim = vertical_dim
    self._brep_deflection = (0.9, 0.5)

  def simplify_space(self, *args, **kwargs):
    logger.debug('CADConverter -> simplify하지 않음')

    fix_shape(self._shape)
    BRepMesh_IncrementalMesh(self._shape, self._brep_deflection[0], False,
                             self._brep_deflection[1], True)

    simplified = {
        'original': self._shape,
        'simplified': self._shape,
        'wall_names': ('0',),
        'info': {
            'simplification': {
                'is_simplified': False
            }
        }
    }
    return simplified

  def save_simplified_space(self, *args, **kwargs):
    logger.debug('CADConverter -> skip save_simplified_space')

  def _write_openfoam_object(self, options: dict, simplified: dict,
                             working_dir: Path):
    shape = self._shape
    _, external_faces = make_external_zone(shape=shape,
                                           buffer=options['external_zone_size'],
                                           vertical_dim=self._vertical_dim)

    path = working_dir.joinpath('geometry.obj')
    write_obj_from_dict(faces=external_faces,
                        obj_path=path,
                        deflection=self._brep_deflection)
    align_obj_to_origin(path)
