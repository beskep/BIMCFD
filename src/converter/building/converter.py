from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import Iterable

from converter.openfoam_converter import OpenFoamConverter


class Converter(ABC):
  _brep_deflection = (0.9, 0.5)

  @property
  def brep_deflection(self):
    return self._brep_deflection

  @brep_deflection.setter
  def brep_deflection(self, value):
    if isinstance(value, (int, float)):
      self._brep_deflection = (value, 0.5)
    elif isinstance(value, Iterable):
      self._brep_deflection = tuple(value)
    else:
      raise TypeError

  @abstractmethod
  def simplify_space(self,
                     spaces,
                     threshold_volume=0.0,
                     threshold_dist=0.0,
                     threshold_angle=0.0,
                     relative_threshold=True,
                     preserve_opening=True,
                     opening_volume=True):
    pass

  @abstractmethod
  def save_simplified_space(self, simplified: dict, path: str):
    pass

  @abstractmethod
  def _write_openfoam_object(self, options: dict, simplified: dict,
                             working_dir: Path):
    pass

  def openfoam_case(self, simplified: dict, save_dir: PathLike, case_name: str,
                    options: dict):
    """
    OpenFOAM 케이스 생성 및 저장

    Parameters
    ----------
    simplified : dict
        simplification 결과
    save_dir : PathLike
        저장 경로
    case_name : str
        저장할 케이스 이름 ("save_dir/case_name"에 결과 저장)
    options : dict
        OpenFOAM 옵션
    """
    save_dir = Path(save_dir)
    save_dir.stat()
    working_dir = save_dir.joinpath(case_name)
    working_dir.mkdir(exist_ok=True)

    if (options['flag_external_zone'] and options['external_zone_size'] > 1):
      simplified['wall_names'] = ['0']

    self._write_openfoam_object(options=options,
                                simplified=simplified,
                                working_dir=working_dir)

    oc = OpenFoamConverter(options=options)
    oc.write_openfoam_case(simplified=simplified,
                           save_dir=save_dir,
                           name=case_name)
