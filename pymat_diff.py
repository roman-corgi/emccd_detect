"""Check diff status between python and matlab codebase."""
import os

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

    return sorted([f.stem for f in path.rglob('*.' + ext)
                   if f.stem not in exclude])


if __name__ == '__main__':
    current_path = Path(os.path.dirname(__file__))
    py_dir = Path(current_path, 'emccd_detect')
    mat_dir = Path(current_path, 'emccd_detect_m')

    py_exclude = ['__init__', 'config', 'imagesc']
    mat_exclude = ['autoArrangeFigures', 'histbn']
    py_list = get_filenames(py_dir, 'py', py_exclude)
    mat_list = get_filenames(mat_dir, 'm', mat_exclude)

    py_diff = [f for f in set(py_list) if f not in set(mat_list)]
    mat_diff = [f for f in set(mat_list) if f not in set(py_list)]

    if py_diff:
        print('Python directory contains extra files not in Matlab:\n')
        for name in py_diff:
            print('   + ' + name)
    print('\n')
    if mat_diff:
        print('Matlab directory contains extra files not in Python:\n')
        for name in mat_diff:
            print('   + ' + name)
