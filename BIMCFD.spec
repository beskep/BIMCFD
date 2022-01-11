# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path

from kivy.tools.packaging.pyinstaller_hooks import (get_deps_minimal, hookspath,
                                                    runtime_hooks)
from PyInstaller import compat

sys.setrecursionlimit(5000)

SRC_DIR = Path(__file__).parent.joinpath('src').as_posix()
if SRC_DIR not in sys.path:
  sys.path.insert(0, SRC_DIR)

from kivymd import hooks_path

block_cipher = None

deps = get_deps_minimal(audio=False, camera=False, spelling=False)
deps['hiddenimports'].append('kivymd.stiffscroll')

mkldir = Path(compat.base_prefix).joinpath('Library/bin')
deps['binaries'].extend([(x.as_posix(), '.') for x in mkldir.glob('mkl*.dll')])

libpng = list(
    Path(compat.base_prefix).joinpath('share/sdl2/bin').glob('libpng*'))[0]
deps['binaries'].append((libpng.as_posix(), '.'))

a = Analysis(['BIMCFD.py'],
             pathex=['D:\\Python\\BIMCFD'],
             datas=[('resource\\', 'resource\\')],
             hookspath=[hooks_path, SRC_DIR],
             runtime_hooks=runtime_hooks(),
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False,
             **deps)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz,
          a.scripts, [],
          exclude_binaries=True,
          name='BIMCFD',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True)
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='BIMCFD')
