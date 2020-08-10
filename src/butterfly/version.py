# coding=utf-8
"""Versions and header."""
from copy import deepcopy
from datetime import datetime


class Version(object):
  """Version class."""

  bf_ver = "0.0.4"
  of_ver = "4.0"
  of_full_ver = "v1706+"
  is_using_docker_machine = True  # useful to run OpenFOAM container
  last_updated = datetime(year=2017, month=8, day=24, hour=13, minute=40)

  def duplicate(self):
    """Return a copy of this object."""
    return deepcopy(self)

  def ToString(self):
    """Overwrite .NET ToString method."""
    return self.__repr__()

  def __repr__(self):
    """Version."""
    return 'Version::Butterfly{}::OpenFOAM{}'.format(self.bf_ver, self.OFVer)


class Header(object):
  """Input files header.

    Usage:
        Header.header()
    """

  _header_format = \
      "/*--------------------------------*- C++ -*----------------------------------*\\\n" + \
      "| =========                 |                                                 |\n" + \
      "| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n" + \
      "|  \\\\    /   O peration     | Version:  {}                                |\n" + \
      "|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n" + \
      "|    \\\\/     M anipulation  |                                                 |\n" + \
      "\\*---------------------------------------------------------------------------*/\n" + \
      "/* Butterfly {}                https://github.com/ladybug-tools/butterfly *\\\n" + \
      "\\*---------------------------------------------------------------------------*/\n"
  _header = _header_format.format(Version.of_full_ver, Version.bf_ver)

  # @staticmethod
  # def header(of_version=Version.of_full_ver, butterfly_version=Version.bf_ver):
  #   """Retuen OpenFOAM file header."""
  #   header = \
  #       "/*--------------------------------*- C++ -*----------------------------------*\\\n" + \
  #       "| =========                 |                                                 |\n" + \
  #       "| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n" + \
  #       "|  \\\\    /   O peration     | Version:  {}                                |\n" + \
  #       "|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n" + \
  #       "|    \\\\/     M anipulation  |                                                 |\n" + \
  #       "\\*---------------------------------------------------------------------------*/\n" + \
  #       "/* Butterfly {}                https://github.com/ladybug-tools/butterfly *\\\n" + \
  #       "\\*---------------------------------------------------------------------------*/\n"
  #
  #   return header.format(of_version, butterfly_version)

  @classmethod
  def header(cls, **kwargs):
    return cls._header

  @classmethod
  def set_header(cls, header):
    cls._header = header

  def duplicate(self):
    """Return a copy of this object."""
    return deepcopy(self)

  def ToString(self):
    """Overwrite .NET ToString method."""
    return self.__repr__()

  def __repr__(self):
    """Header."""
    return self.header
