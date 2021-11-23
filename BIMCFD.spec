# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from pathlib import Path

from PyInstaller import compat

sys.setrecursionlimit(5000)

SRC_DIR = os.path.normpath(os.path.abspath('./src'))
if SRC_DIR not in sys.path:
  sys.path.insert(0, SRC_DIR)

from kivymd import hooks_path

mkldir = Path(compat.base_prefix).joinpath('Library/bin')
mklbins = [(x.as_posix(), '.') for x in mkldir.glob('mkl*.dll')]

block_cipher = None

a = Analysis(['BIMCFD.py'],
             pathex=['D:\\Python\\BIMCFD'],
             binaries=mklbins,
             datas=[('resource\\', 'resource\\')],
             hiddenimports=['kivymd.stiffscroll'],
             hookspath=[hooks_path, SRC_DIR],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
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
