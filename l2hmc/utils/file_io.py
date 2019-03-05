"""
Helper methods for performing file IO.

Author: Sam Foreman (github: @saforem2)
Created: 2/27/2019
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import pickle

# pylint:disable=invalid-name

try:
    import horovod.tensorflow as hvd

    HAS_HOROVOD = True
    hvd.init()

except ImportError:
    HAS_HOROVOD = False

from definitions import ROOT_DIR


def write(s, f, mode='a', nl=True):
    """Write string `s` to file `f` if and only if hvd.rank() == 0."""
    try:
        if HAS_HOROVOD and hvd.rank() != 0:
            return
        with open(f, mode) as ff:
            ff.write(s + '\n' if nl else '')
    except NameError:
        with open(f, mode) as ff:
            ff.write(s + '\n' if nl else '')


def log(s, nl=True):
    """Print string `s` to stdout if and only if hvd.rank() == 0."""
    try:
        if HAS_HOROVOD and hvd.rank() != 0:
            return
        print(s, end='\n' if nl else '')
    except NameError:
        print(s, end='\n' if nl else '')


def save_params_to_pkl_file(params, out_dir):
    """Save `params` dictionary to `parameters.pkl` in `out_dir.`"""
    check_else_make_dir(out_dir)
    params_file = os.path.join(out_dir, 'parameters.pkl')
    #  print(f"Saving params to: {params_file}.")
    log(f"Saving params to: {params_file}.")
    with open(params_file, 'wb') as f:
        pickle.dump(params, f)


def check_else_make_dir(d):
    """If directory `d` doesn't exist, it is created."""
    if not os.path.isdir(d):
        log(f"Creating directory: {d}")
        #  print(f"Creating directory: {d}.")
        os.makedirs(d)
    else:
        log(f"Directory {d} already exists.")


def _create_log_dir(base_name):
    """Create directory for storing information about experiment."""
    root_log_dir = os.path.join(os.path.split(ROOT_DIR)[0], base_name)
    log_dir = make_run_dir(root_log_dir)
    info_dir = os.path.join(log_dir, 'run_info')
    figs_dir = os.path.join(log_dir, 'figures')
    if not os.path.isdir(info_dir):
        os.makedirs(info_dir)
    if not os.path.isdir(figs_dir):
        os.makedirs(figs_dir)
    return log_dir, info_dir, figs_dir


def create_log_dir(base_name):
    """Create directory for storing information about experiment."""
    dirs = _create_log_dir(base_name)

    return dirs


def _check_log_dir(log_dir):
    """Check that log_dir and subdirectories `run_info`, `figures` exist."""
    if not os.path.isdir(log_dir):
        raise ValueError(f'Unable to locate {log_dir}, exiting.')
    else:
        if not log_dir.endswith('/'):
            log_dir += '/'
        info_dir = log_dir + 'run_info/'
        figs_dir = log_dir + 'figures/'
        if not os.path.isdir(info_dir):
            os.makedirs(info_dir)
        if not os.path.isdir(figs_dir):
            os.makedirs(figs_dir)
    return log_dir, info_dir, figs_dir


def check_log_dir(log_dir):
    """Check that log_dir and subdirectories `run_info`, `figures` exist."""
    dirs = _check_log_dir(log_dir)
    return dirs


def _get_run_num(log_dir):
    #  if not os.path.isdir(log_dir):
    #      os.makedirs(log_dir)
    check_else_make_dir(log_dir)

    contents = os.listdir(log_dir)
    if contents in ([], ['.DS_Store']):
        return 1

    run_nums = []
    for item in contents:
        try:
            run_nums.append(int(item.split('_')[-1]))
        except ValueError:
            continue
    if run_nums == []:
        return 1

    return sorted(run_nums)[-1] + 1


def get_run_num(log_dir):
    """Determine the next sequential number to use for new run directory."""
    run_num = _get_run_num(log_dir)

    return run_num


def _make_run_dir(log_dir):
    """Create directory for new run called `run_num` where `num` is unique."""
    if log_dir.endswith('/'):
        _dir = log_dir
    else:
        _dir = log_dir + '/'
    run_num = get_run_num(_dir)
    run_dir = _dir + f'run_{run_num}/'
    if os.path.isdir(run_dir):
        raise f'Directory: {run_dir} already exists, exiting!'
    else:
        log(f'Creating directory for new run: {run_dir}')
        os.makedirs(run_dir)
    return run_dir


def make_run_dir(log_dir):
    """Create directory for new run called `run_num` where `num` is unique."""
    #  try:
    #      if HAS_HOROVOD and hvd.rank() != 0:
    #          return
    #      run_dir = _make_run_dir(log_dir)
    #  except NameError:
    run_dir = _make_run_dir(log_dir)
    return run_dir
