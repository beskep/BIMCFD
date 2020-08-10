import os
import sys

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT_DIR not in sys.path:
  sys.path.append(_ROOT_DIR)
