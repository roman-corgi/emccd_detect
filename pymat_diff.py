"""Check diff status between python and matlab codebase."""
import os

import difflib
from pathlib import Path


class PymatDiffException(Exception):
    """Exception class for pymat_diff module."""


def get_filenames(path, ext, exclude=None):
    """Get list of all filenames of a given extension in a directory.

    Parameters
    ----------
    path : str
        Full path of directory.
    ext : str
        Extension of files to be searched.
    exclude : list, optional
        List of names to exclude. Defaults to None.

    Returns
    -------
    out : list
        List of filenames stripped of extensions.

    """
    path = Path(path)
    if not path.is_dir():
        raise PymatDiffException('No such directory: {:}'.format(path))

    if not exclude:
        exclude = []

    return sorted([f for f in path.rglob('*.' + ext) if f.stem not in exclude])


if __name__ == '__main__':
    current_path = Path(os.path.dirname(__file__))
    dir_py = Path(current_path, 'emccd_detect')
    dir_mat = Path(current_path, 'emccd_detect_m')

    exclude_py = ['__init__', 'config', 'imagesc']
    exclude_mat = ['autoArrangeFigures', 'histbn']
    list_py = get_filenames(dir_py, 'py', exclude_py)
    list_mat = get_filenames(dir_mat, 'm', exclude_mat)

    stems_py = [f.stem for f in list_py]
    stems_mat = [f.stem for f in list_mat]
    common_stems = sorted(set(stems_py).intersection(stems_mat))

    common_py = sorted([f for f in set(list_py) if f.stem in common_stems])
    common_mat = sorted([f for f in set(list_mat) if f.stem in common_stems])
    diff_py = [f for f in set(list_py) if f.stem not in set(stems_mat)]
    diff_mat = [f for f in set(list_mat) if f.stem not in set(stems_py)]

    if diff_py:
        print('Python directory contains extra files not in Matlab:\n')
        for name in diff_py:
            print('   + ' + str(name))
    print('')
    if diff_mat:
        print('Matlab directory contains extra files not in Python:\n')
        for name in diff_mat:
            print('   + ' + str(name))
    print('')

    for py, mat in zip(common_py, common_mat):
        with open(py, 'r') as file_py:
            text_py = file_py.read().splitlines()
        with open(mat, 'r') as file_mat:
            text_mat = file_mat.read().splitlines()

        for line in difflib.unified_diff(text_py, text_mat):
            print(line)
