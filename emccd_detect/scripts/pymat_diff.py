"""Check diff status between python and matlab codebase."""
import os
import pickle
from pathlib import Path

import difflib
import inquirer
import webbrowser


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

    return sorted([f for f in path.rglob('*.' + ext) if f.stem not in exclude],
                  key=lambda f: f.stem)


def matlabize(text_py):
    """Make a python file text more matlab.

    Parameters
    ----------
    text_py : list
        Text of python file.

    Returns
    -------
    modified_py : list
        Text of edited python file.

    """
    modified_py = []

    # Line by line modifications
    for line in text_py:
        # Remove import statements
        if line[:6] == 'import':
            continue
        elif line[:4] == 'from':
            continue

        # Remove leading whitespace
        line = line.lstrip()

        # Remove docstrings
        line = line.lstrip('"')
        line = line.rstrip('"')

        # Remove block comments
        line = line.lstrip('#')
        line = line.lstrip()

        # Remove numpy, and scipy
        line = line.replace('np.', '')
        line = line.replace('sc.', '')

        # Change matplotlib specific terminology
        line = line.replace('plt.', '')
        if line == 'figure()':
            line = 'figure'
        if line == 'grid(True)':
            line = 'grid'

        # Compensate for Matlab's lack of +=
        if '+=' in line:
            var_end_i = line.index('+') - 1
            line = line.replace('+=', '= {:} +'.format(line[:var_end_i]))

        modified_py.append(line)

    # File modifications
    # Remove encoding declaration
    if '-*- coding: utf-8 -*-' in modified_py:
        modified_py.remove('-*- coding: utf-8 -*-')

    return modified_py


def pythonize(text_mat):
    """Make a matlab file text more python.

    Parameters
    ----------
    text_mat : list
        Text of matlab file.

    Returns
    -------
    modified_mat : list
        Text of modified matlab file.

    """
    modified_mat = []

    # Line by line modifications
    for line in text_mat:
        # Remove leading whitespace
        line = line.lstrip()

        # Remove end statements
        if line == 'end':
            continue

        # Remove terminating semicolons even if followed by comments
        line = line.rstrip(';')
        line = line.replace(';  %', '  %')

        # Remove block comments
        line = line.lstrip('%')
        line = line.lstrip()

        # Replace equivalent syntax
        line = line.replace('% ', '# ')
        line = line.replace('^', '**')

        # Remove elementwise symbol
        line = line.replace('.*', '*')
        line = line.replace('./', '/')

        # Replace indexing commas with brackets
        line = line.replace('(i)', '[i]')

        # Replace booleans
        line = line.replace(' true', ' True')
        line = line.replace(' false', ' False')

        # Remove line break dots
        if line.endswith('...'):
            line = line.rstrip('.')

        # Modify function calls
        if line[:8] == 'function':
            equals_i = line.index('=')
            line = 'def' + line[equals_i+1:] + ':'

        modified_mat.append(line)

    # File modifications

    return modified_mat


def color_grad(num):
    """Put a number between 0 and 100 on a red to green color scale."""
    if num < 30:
        return 91
    elif num >= 30 and num < 60:
        return 93
    else:
        return 92


if __name__ == '__main__':
    # Clear console
    os.system('cls' if os.name == 'nt' else 'clear')

    current_path = Path(os.path.dirname(__file__))
    dir_py = Path(current_path)
    dir_mat = Path(current_path, 'emccd_detect_m')

    exclude_py = ['__init__', 'config', 'imagesc', 'pymat_diff', 'setup']
    exclude_mat = ['autoArrangeFigures']
    list_py = get_filenames(dir_py, 'py', exclude_py)
    list_mat = get_filenames(dir_mat, 'm', exclude_mat)

    stems_py = [f.stem for f in list_py]
    stems_mat = [f.stem for f in list_mat]
    common_stems = sorted(set(stems_py).intersection(stems_mat))

    common_py = [f for f in list_py if f.stem in common_stems]
    common_mat = [f for f in list_mat if f.stem in common_stems]
    diff_py = [f for f in list_py if f.stem not in stems_mat]
    diff_mat = [f for f in list_mat if f.stem not in stems_py]

    # Disply filenames that exist only in either the matlab or python directory
    if diff_py:
        print('Python directory contains extra files not in Matlab:\n')
        for name in diff_py:
            print('    + {: <20} [{:}]'.format(str(name.stem), str(name)))
    print('')
    if diff_mat:
        print('Matlab directory contains extra files not in Python:\n')
        for name in diff_mat:
            print('    + {: <20} [{:}]'.format(str(name.stem), str(name)))
    print('')

    ratio_list = []

    # Open files side by side for comparison
    for py, mat in zip(common_py, common_mat):
        with open(py, 'r') as file_py:
            text_py = file_py.read().splitlines()
        with open(mat, 'r') as file_mat:
            text_mat = file_mat.read().splitlines()

        # Remove blank lines
        text_py = list(filter(None, text_py))
        text_mat = list(filter(None, text_mat))

        # Modify files to remove irrelevant language-specific syntax
        modified_py = matlabize(text_py)
        modified_mat = pythonize(text_mat)

        # Take diff and create html for each common file
        diff = difflib.HtmlDiff().make_file(modified_py, modified_mat,
                                            '{:}'.format(py.name),
                                            '{:}'.format(mat.name))
        diff_name = '{:}.html'.format(py.stem)
        with open(Path(current_path, 'diff', diff_name), 'w') as file:
            file.write(diff)

        # Get diff ratio
        ratio = difflib.SequenceMatcher(None, modified_py, modified_mat).ratio()
        ratio_list.append(round(ratio*100))

    # Unpickle selected for use in Checkbox defaults
    selected_path = Path(current_path, 'diff', 'selected.txt')
    if selected_path.exists():
        with open(selected_path, 'rb') as f:
            selected_past = pickle.load(f)
    else:
        selected_past = []

    # Display list of both strings and color-coded similarity score
    dir_diff = get_filenames(Path(current_path, 'diff'), 'html')
    dir_diff_stems = [f.stem for f in dir_diff]

    display_list = []
    selected_past_long = []
    for name, ratio in zip(dir_diff_stems, ratio_list):
        color = color_grad(ratio)
        display_str = '{: <20} \033[{:}m{: >4}\033[0m'.format(name, color, ratio)
        display_list.append(display_str)
        if name in selected_past:
            selected_past_long.append(display_str)

    # Display options to opening diffs in browser
    questions = [
        inquirer.Checkbox('diffs',
                          message='Diffs available',
                          choices=display_list,
                          default=selected_past_long,
                          ),
    ]
    answers = inquirer.prompt(questions)
    selected_long = answers['diffs']
    selected = [s.split(' ', 1)[0] for s in selected_long]

    # Pickle selected for use in Checkbox defaults in next run
    with open(selected_path, 'wb') as f:
        pickle.dump(selected, f)

    # Open selected in browser
    for name in dir_diff:
        if name.stem in selected:
            webbrowser.open('file://' + str(name))
