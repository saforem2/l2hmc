import tensorflow as tf
import os

def get_run_num(log_dir):
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    contents = os.listdir(log_dir)
    if contents == []:
        return 1
    else:
        run_nums = []
        for item in contents:
            try:
                run_nums.append(int(item.split('_')[-1]))
                #  run_nums.append(int(''.join(x for x in item if x.isdigit())))
            except ValueError:
                continue
        return sorted(run_nums)[-1] + 1
    #  if contents == ['.DS_Store']:
    #      return 1
    #  else:
    #      for item in contents:
    #          if os.path.isdir(log_dir + item):
    #              run_dirs.append(item)
    #      run_nums = [int(str(i)[3:]) for i in run_dirs]
    #      prev_run_num = max(run_nums)
    #      return prev_run_num + 1

def make_run_dir(log_dir):
    if log_dir.endswith('/'):
        _dir = log_dir
    else:
        _dir = log_dir + '/'
    run_num = get_run_num(_dir)
    run_dir = _dir + f'run_{run_num}/'
    if os.path.isdir(run_dir):
        raise f'Directory: {run_dir} already exists, exiting!'
    else:
        print(f'Creating directory for new run: {run_dir}')
        os.makedirs(run_dir)
    return run_dir


def check_log_dir(log_dir):
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


def create_log_dir():
    """Create directory for storing information about experiment."""
    #  root_log_dir = '../../log_mog_tf/'
    #  root_log_dir = os.path.join(ROOT_DIR, log_mog_tf)
    root_log_dir = os.path.join(os.path.split(ROOT_DIR)[0], 'log_mog_tf')
    log_dir = make_run_dir(root_log_dir)
    info_dir = log_dir + 'run_info/'
    figs_dir = log_dir + 'figures/'
    if not os.path.isdir(info_dir):
        os.makedirs(info_dir)
    if not os.path.isdir(figs_dir):
        os.makedirs(figs_dir)
    return log_dir, info_dir, figs_dir


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)"""
    #  with tf.name_scope('summaries'):
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
