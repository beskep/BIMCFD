import argparse
import os

import ifcopenshell
import ifcopenshell.geom
from fuzzywuzzy import fuzz, process
from scipy.spatial.transform import _rotation_groups

from src import utils

if __name__ == '__main__':
  os.environ['KIVY_NO_ARGS'] = '1'

  parser = argparse.ArgumentParser()

  parser.add_argument('-l',
                      '--loglevel',
                      help='log level',
                      type=int,
                      default=20)
  parser.add_argument('-d',
                      '--debug',
                      help='debug logging',
                      action='store_true')

  group = parser.add_mutually_exclusive_group()
  group.add_argument('--loguru',
                     action='store_true',
                     dest='handle_kivy_logger',
                     help='loguru handles kivy log')
  group.add_argument('--kivy',
                     action='store_false',
                     dest='handle_kivy_logger',
                     help='kivy handles it\'s log')
  parser.set_defaults(handle_kivy_logger=True)

  args = parser.parse_args()
  level = 10 if args.debug else args.loglevel
  utils.set_logger(level=level, handle_kivy_logger=args.handle_kivy_logger)

  from src.converter import ifc_converter, simplify
  from src.interface import bim_cfd_app, bim_cfd_base
  from src.OCCUtils import Common, Construct, edge, face

  bim_cfd_app.main()
