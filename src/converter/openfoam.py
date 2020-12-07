import logging
import os
from collections import OrderedDict, namedtuple
from itertools import chain
from pathlib import Path
from shutil import copy2, rmtree
from typing import List, Tuple

from utils import TEMPLATE_DIR

from butterfly.case import Case as ButterflyCase
from butterfly.foamfile import FoamFile
from butterfly.version import Header

_DEFAULT_HEADER = \
    r'''/*--------------------------------*- C++ -*----------------------------------*\
    | =========                 |                                                 |
    | \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
    |  \\    /   O peration     | Version:  v1812                                 |
    |   \\  /    A nd           | Web:      www.OpenFOAM.com                      |
    |    \\/     M anipulation  |                                                 |
    \*---------------------------------------------------------------------------*/
    '''
Header.set_header(_DEFAULT_HEADER)

_SOLVERS = (
    'simpleFoam',
    'buoyantSimpleFoam',
    'buoyantBousinessqSimpleFoam',
)
_SOLVERS_ENERGY = (
    'buoyantSimpleFoam',
    'buoyantBousinessqSimpleFoam',
)
_SOLVERS_CONDUCTIVITY = ('buoyantSimpleFoam',)
_SOLVERS_TURBULENCE = (
    'simpleFoam',
    'buoyantSimpleFoam',
    'buoyantBousinessqSimpleFoam',
)
_SOLVER_PATH = {x.lower(): TEMPLATE_DIR.joinpath(x) for x in _SOLVERS}
for x in _SOLVER_PATH.values():
  assert x.exists(), x

_MESH_SH = (b'surfaceFeatureEdges -angle 0 geometry.obj geometry.fms\n'
            b'cartesianMesh\n'
            b'checkMesh | tee log.mesh')


def supported_solvers(energy=False,
                      conductivity=False,
                      turbulence=False) -> List[str]:
  solvers = set(_SOLVERS)

  if energy:
    solvers = solvers.intersection(set(_SOLVERS_ENERGY))
  if conductivity:
    solvers = solvers.intersection(set(_SOLVERS_CONDUCTIVITY))
  if turbulence:
    solvers = solvers.intersection(set(_SOLVERS_TURBULENCE))

  solvers = sorted(list(solvers))

  return solvers


def iter_files(folder, fullpath=False):
  """list files in a folder."""
  if not os.path.isdir(folder):
    yield None

  for f in os.listdir(folder):
    if os.path.isfile(os.path.join(folder, f)):
      if fullpath:
        yield os.path.join(folder, f)
      else:
        yield f
    else:
      yield


def load_case_files(folder, fullpath=False):
  """load openfoam files from a folder."""
  files = []
  for p in ('0', 'constant', 'system'):
    fp = os.path.join(folder, p)
    files.append(tuple(iter_files(fp, fullpath)))

  Files = namedtuple('Files', 'zero constant system')
  return Files(*files)


def read_foam_file_header(path):
  header_lines = []
  with open(path, 'r') as f:
    for line in f:
      if line[0].isalnum():
        break

      header_lines.append(line)

  header = ''.join(header_lines)
  if 'OpenFOAM' not in header:
    header = None

  return header


class BoundaryFieldDict(OrderedDict):
  """OpenFOAM의 boundary field에 주석과 빈 줄을 추가하기 위한 클래스
  """

  def __init__(self, width=None, **kwargs):
    super().__init__(**kwargs)

    self.__key = ' '
    self._width = width

  def set_width(self, width: int):
    self._width = width

  def add_value(self, key: str, value):
    if self._width is not None:
      key = key.ljust(self._width)

    self[key] = value

  def add_comment(self, comment: str):
    if not comment.startswith('//'):
      comment = '// ' + comment

    self[comment] = ' '

  def add_empty_line(self):
    if self.__key in self:
      while True:
        self.__key += ' '

        if self.__key not in self:
          break

    self[self.__key] = ' '
    self.__key += ' '


class OpenFoamCase(ButterflyCase):
  logger = logging.getLogger(Path(__file__).stem)

  SUBFOLDERS = ('0', 'constant', 'constant\\polyMesh', 'system')

  # minimum list of files to be able to run blockMesh and snappyHexMesh
  MINFOAMFIles = ('fvSchemes', 'fvSolution', 'controlDict')

  def __init__(self,
               name: str,
               working_dir: str,
               foamfiles: list = None,
               geometries: list = None):
    if foamfiles is None:
      foamfiles = []

    if geometries is None:
      geometries = []

    super().__init__(name=name, foamfiles=foamfiles, geometries=geometries)

    self.working_dir = os.path.normpath(working_dir)
    self._original_dir = None
    self._solver = None

  @property
  def foam_files(self) -> Tuple[FoamFile]:
    """Get all the foam_files."""
    return super().foam_files

  @property
  def original_dir(self):
    return self._original_dir

  @original_dir.setter
  def original_dir(self, value):
    path = os.path.normpath(value)
    if os.path.exists(path):
      self._original_dir = path

  def load_mesh(self):
    pass

  def load_points(self):
    pass

  def update_bc_in_zero_folder(self):
    pass

  @staticmethod
  def is_energy_available(solver: str):
    return solver.lower() in [x.lower() for x in _SOLVERS_ENERGY]

  @staticmethod
  def is_conductivity_available(solver: str):
    return solver.lower() in [x.lower() for x in _SOLVERS_CONDUCTIVITY]

  @staticmethod
  def is_turbulence_available(solver: str):
    return solver.lower() in [x.lower() for x in _SOLVERS_TURBULENCE]

  @classmethod
  def create_foamfile_from_file(cls, p, convert_to_meters):
    return cls._Case__create_foamfile_from_file(
        p=p, convertToMeters=convert_to_meters)

  @classmethod
  def from_folder(cls, path, working_dir, name=None, convert_from_meters=1):
    """Create a Butterfly case from a case folder.

    Args:
        path: Full path to case folder.
        name: An optional new name for this case.
        convert_from_meters: A number to be multiplied to stl file vertices
            to be converted to the new units if not meters. This value will
            be the inverse of convertToMeters.
    """
    # collect foam files
    __originalName = os.path.split(path)[-1]
    if not name:
      name = __originalName

    _files = load_case_files(path, fullpath=True)

    # convert files to butterfly objects
    ff = []
    flag_header = True
    for p in chain.from_iterable([_files.zero, _files.constant, _files.system]):
      if not p:
        continue

      if flag_header:
        header = read_foam_file_header(p)
        if header is not None:
          Header.set_header(header)
          flag_header = False

      try:
        foam_file = cls.create_foamfile_from_file(
            p=p, convert_to_meters=(1.0 / convert_from_meters))

        if foam_file:
          ff.append(foam_file)

        cls.logger.debug('Imported %s from case', os.path.normpath(p))

      except Exception as e:
        cls.logger.error('Failed to import %s:\n\t%s', os.path.normpath(p), e)
        raise e

    bf_geometries = []

    _case = cls(name=name,
                working_dir=working_dir,
                foamfiles=ff,
                geometries=bf_geometries)
    _case.original_dir = path

    # original name is a variable to address the current limitation to change
    # the name of stl file in snappyHexMeshDict. It will be removed once the
    # limitation is addressed.
    _case.__originalName = __originalName

    return _case

  @classmethod
  def is_supported_solver(cls, solver: str):
    return solver.lower() in _SOLVER_PATH.keys()

  @classmethod
  def from_template(cls, solver, save_dir, name):
    if not cls.is_supported_solver(solver):
      raise ValueError('지원하지 않는 solver입니다: {}'.format(solver))

    _case = cls.from_folder(path=_SOLVER_PATH[solver.lower()].as_posix(),
                            working_dir=save_dir,
                            name=name)
    _case._solver = solver.lower()
    return _case

  def save(self, overwrite=False, minimum=False):
    """Save case to folder.

    Args:
        overwrite: If True all the current content will be overwritten
            (default: False).
        minimum: Write minimum necessary files for case. These files will
            be enough for meshing the case but not running any commands.
            Files are ('fvSchemes', 'fvSolution', 'controlDict',
            'blockMeshDict','snappyHexMeshDict'). Rest of the files will be
            created from a Solution.
    """
    # create folder and subfolders if they are not already created
    if overwrite and os.path.exists(self.project_dir):
      rmtree(self.project_dir, ignore_errors=True)

    for f in self.SUBFOLDERS:
      p = os.path.join(self.project_dir, f)

      if not os.path.exists(p):
        try:
          os.makedirs(p)
        except Exception as e:
          msg = 'Failed to create foam file {}\n\t{}'.format(p, e)

          if str(e).startswith('[Error 183]'):
            self.logger.warning(msg)
          else:
            raise IOError(msg) from e

    # save foamfiles
    if minimum:
      foam_files = (
          ff for ff in self.foam_files if ff.name in self.MINFOAMFIles)
    else:
      foam_files = list(self.foam_files)

    for fname in self.MINFOAMFIles:
      if fname not in [x.name for x in foam_files]:
        ff = FoamFile(name=fname, cls='dictionary', location='system')
        foam_files.append(ff)

    for f in foam_files:
      if self.original_dir and (f.location != '"0"') and (f.name != 'meshDict'):
        sub_folder = f.location.replace('"', '')
        file_name = f.name

        path_from = os.path.join(self.original_dir, sub_folder, file_name)
        assert os.path.exists(path_from)

        path_to = os.path.join(self.project_dir, sub_folder, file_name)
        copy2(path_from, path_to)
      else:
        f.save(self.project_dir)

    # add .foam file
    foam_path = os.path.join(self.project_dir, self.project_name + '.foam')
    with open(foam_path, 'w') as f:
      f.write('')

    self.logger.info('%s is saved to: %s', self.project_name, self.project_dir)

  def save_shell(self):
    # todo: decomposePar, run 지원
    mesh = os.path.join(self.project_dir, 'mesh.sh')

    with open(mesh, 'wb') as f:
      f.write(_MESH_SH)

  def change_boundary_field(self, variable, boundary_field: BoundaryFieldDict):
    foam_file = None
    for ff in self.foam_files:
      if ff.name == variable:
        foam_file = ff
        break

    if foam_file is None:
      raise ValueError('Variable not found: {}'.format(variable))

    interior_key = '"(Interior).*"'
    if interior_key not in boundary_field:
      boundary_field[interior_key] = OrderedDict([('type', 'empty')])

    foam_file.values['boundaryField'] = boundary_field

  def change_foam_file_value(self, foam_file_name: str, value: dict):
    foam_file = None
    for ff in self.foam_files:
      if ff.name == foam_file_name:
        foam_file = ff
        break

    if foam_file is None:
      raise ValueError('Variable not found: {}'.format(foam_file_name))

    foam_file.values = value
