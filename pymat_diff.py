"""Check diff status between python and matlab codebase."""
import os

from pathlib import Path

current_path = Path(os.path.dirname(__file__))
py_dir = Path(current_path, 'emccd_detect')
mat_dir = Path(current_path, 'emccd_detect_matlab')

py_exclude = ['config']
py_list = sorted([f.stem for f in py_dir.rglob('[!__]*.py')
                  if f.stem not in py_exclude])
mat_list = sorted([f.stem for f in mat_dir.rglob('*.m')])