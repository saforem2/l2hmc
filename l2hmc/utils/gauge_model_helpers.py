"""
Helper functions for handling run data for U(1) gauge model.

Specifically, provides methods for dealing with directory structures;
saving/printing relevant information during training, and creating plots from
the results of a run.
"""
# pylint: disable=wildcard-import, no-member, too-many-arguments, invalid-name
import os
import pickle
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from definitions import ROOT_DIR

from lattice.gauge_lattice import u1_plaq_exact
#  from utils.tf_logging import make_run_dir


def get_run_num(log_dir):
    """Determine the next sequential number to use for new run directory."""
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    contents = os.listdir(log_dir)
    if contents == [] or contents == ['.DS_Store']:
        return 1
    else:
        run_nums = []
        for item in contents:
            try:
                run_nums.append(int(item.split('_')[-1]))
            except ValueError:
                continue
        if run_nums == []:
            return 1
        else:
            return sorted(run_nums)[-1] + 1


def make_run_dir(log_dir):
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
        print(f'Creating directory for new run: {run_dir}')
        os.makedirs(run_dir)
    return run_dir


def check_log_dir(log_dir):
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


def create_log_dir(base_name):
    """Create directory for storing information about experiment."""
    root_log_dir = os.path.join(os.path.split(ROOT_DIR)[0], base_name)
    log_dir = make_run_dir(root_log_dir)
    info_dir = log_dir + 'run_info/'
    figs_dir = log_dir + 'figures/'
    if not os.path.isdir(info_dir):
        os.makedirs(info_dir)
    if not os.path.isdir(figs_dir):
        os.makedirs(figs_dir)
    return log_dir, info_dir, figs_dir


def print_run_data(data, header=False):
    """Print information about current run to std out."""
    data_str = format_run_data(data)
    if header:
        header = data_header(test_flag=True)
        print(header)
    print(data_str)


def write_run_data(file_path, data, header=False):
    """Write run `data` to human-readable file at `file_path`."""
    data_str = format_run_data(data)

    if header:
        header_str = data_header()
        with open(file_path, 'a') as f:
            f.write(header_str)
            f.write('\n')

    step = data['step']
    #  if step == 1 or step % 100 == 0:
    #      header = data_header(test_flag=True)
    with open(file_path, 'a') as f:
        f.write(data_str)
        f.write('\n')
        #  f.write('avg_plaquettes: {}\n'.format(data['avg_plaquettes']))
        #  f.write('topological_charges: {}\n'.format(data['top_charges']))
        #  f.write('total_actions: {}\n'.format(data['total_actions']))
        #  f.write(separator)


def write_run_parameters(file_path, parameters):
    """Write `parameters` to human-readable file at `file_path`.

    Args:
        file_path: Path to file to save `parameters` to.
        parameters (dict)
    """
    with open(file_path, 'w') as f:
        f.write('Parameters:\n')
        f.write(80 * '-' + '\n')
        for key, val in parameters.items():
            if isinstance(val, (int, float, str)):
                print(f'{key}: {val}\n')
        #  for key, val in parameters.items():
        #      f.write(f'{key}: {val}\n')
        f.write(80*'=')
        f.write('\n')


def data_header():
    """Create formatted (header) string containing labels for printing data."""
    h_str = ("{:^15s}{:^14s}{:^14s}{:^14s}{:^14s}{:^14s}{:^14s}")
    h_strf = h_str.format("STEP", "LOSS", "NORM. TIME", "ACCEPT %",
                          "EPS", "BETA", "LR")#, "ACTION", "TOP Q", "PLAQ")
    dash0 = (len(h_strf) + 1) * '-'
    dash1 = (len(h_strf) + 1) * '-'
    header_str = dash0 + '\n' + h_strf + '\n' + dash1

    return header_str


def format_run_data(data):
    """Create formatted string containing relevant information from `data`."""
    data_str = (
        f"{data['step']:>6g}/{data['train_steps']:<7g} "
        f"{data['loss']:^13.4g} "
        f"{data['step_time']:^13.4g} "
        f"{np.mean(data['accept_prob']):^13.4g} "
        f"{data['eps']:^13.4g} "
        f"{data['beta']:^13.4g} "
        f"{data['learning_rate']:^13.4g}"
    )

    return data_str


def _plot_data(x_data, y_data, x_label, y_label, skip_steps=1):
    """Create plot consisting of `x_data`, `y_data` with x and y labels. 

    NOTE: 
        `skip_steps` allows for plotting every `skip_steps` data, since we
        are interested in time-series data, we can 'spread-out' the data by
        plotting only every third data point, for example.
    """
    fig, ax = plt.subplots()
    num_samples = y_data.shape[1]
    for sample in range(num_samples):
        label_str = f"sample {sample} (avg: {np.mean(y_data[:, sample]):^5.4g})"
        ax.plot(x_data[::skip_steps], y_data[:, sample][::skip_steps],
                marker='', ls='-', label=label_str)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.legend(loc='best', fontsize=10)

    return fig, ax


def plot_run_data(data, params, steps_arr, out_dir, skip_steps=1):
    """Create plots of relevant lattice observables using `_plot_data` above."""
    avg_plaqs_arr = data['average_plaquettes_arr']
    avg_top_charge_arr = data['topological_charges_arr']
    total_actions_arr = data['total_actions_arr']
    num_samples = params['num_samples']
    beta = params['beta']

    fig, ax = _plot_data(steps_arr,
                         avg_plaqs_arr,
                         'Step',
                         'Average Plaquette',
                         skip_steps=skip_steps)

    ax.axhline(u1_plaq_exact(beta), xmin=0, xmax=max(steps_arr),
               color='k', ls='-', label=f"Exact ({u1_plaq_exact(beta):^5.4g})")

    ax.legend(loc='best')

    file_name = os.path.join(out_dir, 'average_plaquettes_vs_step.pdf')
    print(f"Saving plot to {file_name}...")
    fig.savefig(file_name, dpi=400, bbox_inches='tight')

    fig, ax = _plot_data(steps_arr, avg_top_charge_arr,
                         'Step', 'Average Topological Charge',
                         skip_steps=skip_steps)

    file_name = os.path.join(out_dir, 'average_topological_charge_vs_step.pdf')
    print(f"Saving plot to {file_name}...")
    fig.savefig(file_name, dpi=400, bbox_inches='tight')

    fig, ax = _plot_data(steps_arr, total_actions_arr,
                         'Step', 'Average Total Action',
                         skip_steps=skip_steps)

    file_name = os.path.join(out_dir, 'average_total_action_vs_step.pdf')
    print(f"Saving plot to {file_name}...")
    fig.savefig(file_name, dpi=400, bbox_inches='tight')
    print('done.')


def save_run_data(checkpointer, log_dir, files, data, samples, params):
    """Save run `data` to `files` in `log_dir` using `checkpointer`"""
    saved_path = checkpointer.save(file_prefix=os.path.join(log_dir, "ckpt"))
    print(f"Saved checkpoint to: {saved_path}")
    np.save(files['average_plaquettes_file'], data['average_plaquettes_arr'])
    np.save(files['total_actions_file'], data['total_actions_arr'])
    np.save(files['topological_charges_file'], data['topological_charges_arr'])
    np.save(files['samples_file'], data['samples'])
    #  print('avg_plaquettes: {}\n'.format(data['avg_plaquettes']))
    with open(files['data_pkl_file'], 'wb') as f:
        pickle.dump(data, f)
    with open(files['parameters_pkl_file'], 'wb') as f:
        pickle.dump(params, f)
    with open(file['samples_pkl_file'], 'wb') as f:
        pickle.dump(samples, f)
