import os
from collections import namedtuple, OrderedDict
from shutil import rmtree, copy2
from typing import Tuple
from warnings import warn

from butterfly.case import Case as ButterflyCase
from butterfly.fields import Field
from butterfly.foamfile import FoamFile
from butterfly.geometry import bf_geometry_from_stl_file
from butterfly.refinementRegion import refinementRegions_from_stl_file
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

_FILE_DIR = os.path.dirname(__file__)
_SOLVERS = ('simpleFoam', 'buoyantSimpleFoam', 'buoyantBousinessqSimpleFoam')
_SOLVERS_ENERGY = ('buoyantSimpleFoam', 'buoyantBousinessqSimpleFoam')
_SOLVERS_CONDUCTIVITY = ('buoyantSimpleFoam',)
_SOLVERS_TURBULENCE = ('simpleFoam', 'buoyantSimpleFoam',
                       'buoyantBousinessqSimpleFoam')
_SOLVER_PATH = {
    x.lower(): os.path.normpath(os.path.join(_FILE_DIR, '../template', x))
    for x in _SOLVERS
}
assert all([os.path.exists(x) for x in _SOLVER_PATH.values()])


def supported_solvers() -> Tuple[str]:
  return _SOLVERS[:]


def list_files(folder, fullpath=False):
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
    files.append(tuple(list_files(fp, fullpath)))

  Files = namedtuple('Files', 'zero constant system')
  return Files(*files)


def read_foam_file_header(path):
  header_lines = []
  with open(path, 'r') as f:
    for line in f:
      if line[0].isalnum():
        break
      else:
        header_lines.append(line)

  header = ''.join(header_lines)
  if 'OpenFOAM' not in header:
    header = None

  return header


class BoundaryFieldDict(OrderedDict):
  """OpenFOAM의 boundary field에 주석과 빈 줄을 추가하기 위한 클래스
  """

  def __init__(self, *args, **kwargs):
    super(OrderedDict, self).__init__(*args, **kwargs)
    self.__key = ' '
    self._width = None

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

    super(OpenFoamCase, self).__init__(name=name,
                                       foamfiles=foamfiles,
                                       geometries=geometries)

    self.working_dir = os.path.normpath(working_dir)
    self._original_dir = None
    self._solver = None

  @property
  def foam_files(self) -> Tuple[FoamFile]:
    """Get all the foam_files."""
    return tuple(f for f in self._Case__foamfiles)

  @property
  def original_dir(self):
    return self._original_dir

  @original_dir.setter
  def original_dir(self, value):
    path = os.path.normpath(value)
    if os.path.exists(path):
      self._original_dir = path

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
    for f in (_files.zero, _files.constant, _files.system):
      for p in f:
        if not p:
          continue

        if flag_header:
          header = read_foam_file_header(p)
          if header is not None:
            Header.set_header(header)
            flag_header = False

        try:
          foam_file = cls._Case__create_foamfile_from_file(
              p, 1.0 / convert_from_meters)
          if foam_file:
            ff.append(foam_file)
          print('Imported {} from case.'.format(p))
        except Exception as e:
          print('Failed to import {}:\n\t{}'.format(p, e))

    s_hmd = cls._Case__get_foam_file_by_name('snappyHexMeshDict', ff)

    if s_hmd:
      s_hmd.project_name = name

      stlfiles = tuple(f for f in _files.stl if f.lower().endswith('.stl'))
      bf_geometries = tuple(
          geo for f in stlfiles
          for geo in bf_geometry_from_stl_file(f, convert_from_meters)
          if os.path.split(f)[-1][:-4] in s_hmd.stl_file_names)

    else:
      bf_geometries = []

    _case = cls(name=name,
                working_dir=working_dir,
                foamfiles=ff,
                geometries=bf_geometries)
    _case.original_dir = path

    # update each field of boundary condition for geometries
    if s_hmd:
      for ff in _case.get_foam_files_from_location('0'):
        for geo in _case.geometries:
          try:
            f = ff.get_boundary_field(geo.name)
          except AttributeError as e:
            if not geo.name.endswith('Conditions'):
              print(str(e))
          else:
            # set boundary condition for the field
            if not f:
              setattr(geo.boundary_condition, ff.name, None)
            else:
              setattr(geo.boundary_condition, ff.name, Field.from_dict(f))

    if s_hmd:
      refinementRegions = tuple(
          ref for f in _files.stl
          if os.path.split(f)[-1][:-4] in s_hmd.refinementRegion_names
          for ref in refinementRegions_from_stl_file(
              f, s_hmd.refinementRegion_mode(os.path.split(f)[-1][:-4])))

      _case.add_refinementRegions(refinementRegions)

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

    _case = cls.from_folder(path=_SOLVER_PATH[solver.lower()],
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
            warn(msg)
          else:
            raise IOError(msg)

    # save foamfiles
    if minimum:
      foam_files = (
          ff for ff in self.foam_files if ff.name in self.MINFOAMFIles)
    else:
      foam_files = self.foam_files

    for f in foam_files:
      if self.original_dir and f.location != '"0"' and f.name != 'meshDict':
        sub_folder = f.location.replace('"', '')
        file_name = f.name

        path_from = os.path.join(self.original_dir, sub_folder, file_name)
        # if file_name == 'blockMeshDict' and not os.path.exists(path_from):
        #   file_name = 'meshDict'
        #   path_from = os.path.join(self.original_dir, sub_folder, file_name)

        assert os.path.exists(path_from)

        path_to = os.path.join(self.project_dir, sub_folder, file_name)
        copy2(path_from, path_to)
      else:
        # todo: foam file 저장 함수 새로 만들기
        f.save(self.project_dir)

    # add .foam file
    foam_path = os.path.join(self.project_dir, self.project_name + '.foam')
    with open(foam_path, 'w') as f:
      f.write('')

    print('{} is saved to: {}'.format(self.project_name, self.project_dir))

  def save_shell(self):
    mesh = os.path.join(self.project_dir, 'mesh.sh')
    with open(mesh, 'w') as f:
      f.write('''surfaceFeatureEdges -angle 0 geometry.obj geometry.fms
      cartesianMesh
      checkMesh | tee log.mesh''')

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
