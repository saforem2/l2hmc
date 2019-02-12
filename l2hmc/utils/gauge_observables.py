"""
Methods for calculating and plotting relevant physical observables using
sample configurations generated from the (L2)HMC sampler.

Author: Sam Foreman (twitter/github @saforem2)
Date: 12 / 9 / 2018
"""
#pylint: disable=invalid-name, too-many-locals
import os
import sys
import time
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

try:
    import horovod.tensorflow as hvd
    HAS_HOROVOD = True
except ImportError:
    HAS_HOROVOD = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


from scipy.optimize import curve_fit
from pandas.plotting import autocorrelation_plot
from collections import Counter

from astropy.stats import jackknife_resampling, jackknife_stats

from lattice.lattice import GaugeLattice, u1_plaq_exact
#  from lattice.gauge_lattice import GaugeLattice, u1_plaq_exact
#  from l2hmc_eager import gauge_dynamics_eager as gde

from .plot_helper import plot_broken_xaxis, plot_multiple_lines
from .gauge_model_helpers import log, write

from .autocorr import (
    integrated_time, autocorr_func_1d, AutocorrError, calc_ESS,
    calc_iat, autocorr_fast, autocorr, autocovariance, acl_spectrum
)

COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
MARKERS = ['o', 's', 'x', 'v', 'h', '^', 'p', '<', 'd', '>', 'o']
LINESTYLES = ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--']


##############################################################################
# File I/O helpers.
##############################################################################
def check_else_make_dir(d):
    """Checks if directory exists, otherwise creates directory."""
    if not os.path.isdir(d):
        log(f"Making directory: {d}.")
        os.makedirs(d)

def _load_params(log_dir):
    """Load in model parameters from `log_dir`.

    Returns:
        params: Dictionary containing relevant parameters necessary to recreate
            lattice and calculate observables.
    """
    info_dir = os.path.join(log_dir, 'run_info')
    #  assert os.path.isdir(info_dir)
    params_file = os.path.join(info_dir, 'parameters.pkl')
    try:
        with open(params_file, 'rb') as f:
            params = pickle.load(f)
        return params

    except FileNotFoundError:
        log(f"Unable to find {params_file} in {info_dir}. Returning 0.")
        return 0


##############################################################################
# File I/O helpers.
##############################################################################
def jackknife(x, fn):
    """Jackknife estimate of the estimator fn."""
    n = len(x)
    idx = np.arange(n)
    return np.sum(fn(x[idx!=i]) for i in range(n)) / float(n)

def jackknife_var(x, fn):
    """Jackknife estimate of the variance of the estimator fn."""
    n = len(x)
    idx = np.arange(n)
    j_est = jackknife(x, fn)
    j_var = (n - 1) / (n + 0.) * np.sum(
        (fn(x[idx!=i]) - j_est)**2 for i in range(n)
    )
    return j_var


##############################################################################
# Create lattice and calculate relevant observables.
##############################################################################

def _make_lattice(params):
    """Return GaugeLattice object with attributes defined by `params` dict."""
    return GaugeLattice(time_size=params['time_size'],
                        space_size=params['space_size'],
                        dim=params['dim'],
                        link_type=params['link_type'],
                        num_samples=params['num_samples'],
                        rand=params['rand'])

def calc_susceptibility_stats(charges):
    """Calculate statistics for the topological susceptibility from charges.

    Args:
        charges: Array (shape: (num_steps, num_samples) containing topological
            charge history of L2HMC chain, comprised of multiple samples. 

    Returns:
        estimate_arr: List containing the estimated sample statistic for each
            sample in charges.
        bias_arr: List containing the estimated bias statistic for each sample
            in charges.
        stderr_arr: List containing the stderr of the sample statistic for each
            sample in charges.
        conf_interval: List containing the confidence interval for the sample
            statistic for each sample in charges. 
            NOTE: intervals aligned column-wise.
    """
    #  mean_squared = lambda x: np.mean(x) ** 2
    squared_mean = lambda x: np.mean(x ** 2)
    test_statistic = lambda x: squared_mean(x)
    #  test_statistic = lambda x: mean_squared(x)
    estimate_arr = []
    bias_arr = []
    stderr_arr = []
    conf_interval_arr = []
    #  stats_strings_arr = []
    for idx, q in enumerate(charges.T):  # q is the top. charge of sample idx
        #  q_squared = q ** 2
        q_rs = jackknife_resampling(q)
        # want to calculate the susceptibility as 
        # susceptibility = mean(q ** 2) - mean(q) ** 2
        estimate, bias, stderr, conf_interval = jackknife_stats(q,
                                                                test_statistic,
                                                                0.95)

        str0 = (f"Topological susceptibility statistics for sample {idx}, "
                f"consisting of {q_rs.shape[0]} L2HMC steps.")
        sep_str = len(str0) * '-' + '\n'

        log(sep_str)
        log(str0)
        log(f'estimate: {estimate}')
        log(f'bias: {bias}')
        log(f'stderr: {stderr}')
        log(f'conf_interval: {conf_interval}\n')
        log(sep_str)

        estimate_arr.append(estimate)
        bias_arr.append(bias)
        stderr_arr.append(stderr)
        conf_interval_arr.extend(conf_interval)

    # Compute susceptibility statistics over all (tot) samples 
    #  q_rs_all = jackknife_resampling(np.array(charges).flatten())
    #  stats_all = jackknife_stats(q_rs_all,
    #                                test_statistic,
    #                                0.95)
    #  estimate_all, bias_all, stderr_all, conf_interval_all = stats_all

    #  str0 = (f"Topological susceptibility statistics for combined samples"
    #          f"consisting of {q_rs.shape[0]} L2HMC steps.")
    #  sep_str1 = len(str0) * '=' + '\n'
    #  sep_str0 = len(str0) * '-' + '\n'
    #
    #  log(sep_str1)
    #  log(str0)
    #  log(f'estimate: {estimate_all}')
    #  log(f'bias: {bias_all}')
    #  log(f'stderr: {stderr_all}')
    #  log(f'conf_interval: {conf_interval_all}\n')
    #  log(sep_str)

    stats = (estimate_arr, bias_arr, stderr_arr, conf_interval_arr)
    #  stats_all = (estimate_all, bias_all, stderr_all, conf_interval_all)

    return stats #, stats_all

# pylint: disable=invalid-name, too-many-locals
def _calc_observables(samples, params):
    """Calculate lattice observables for each sample in `samples`.
    
    Args:
        samples: numpy.ndarray containing GaugeLattice link configurations. 
        params: Relevant parameters defining properties of GaugeLattice.

    Returns:
        total_actions: numpy.ndarray containing the total actions of each
            sample contained in `samples_arr`.
        avg_plaquettes: numpy.ndarray containing the average plaquette of each
            sample contained in `samples_arr`.
        top_charges: numpy.ndarray containing the topological charge for each
            samples contained in `samples_arr`.
    """
    if not isinstance(samples, np.ndarray):
        samples = np.array(samples)

    observables_fn = None
    lattice = _make_lattice(params)

    total_actions = []
    avg_plaquettes = []
    top_charges = []
    #  beta = params['beta_final']
    for idx, sample in enumerate(samples):
        t0 = time.time()
        try:
            observables = np.array(lattice.calc_plaq_observables(sample))
        except:
            observables = np.array(lattice._calc_plaq_observables(sample))

        actions, plaqs, charges = observables

        total_actions.append(actions)
        avg_plaquettes.append(plaqs)
        top_charges.append(charges)

        log(f"step: {idx} "
            f"time/step: {time.time() - t0:^6.4g} "
            f"avg action: {np.mean(actions):^6.4g} "
            f"avg plaquette: {np.mean(plaqs):^6.4g} ")
        #  "top charges: ")
        log('\n')
        log('top_charges: ', nl=False)
        log(charges)
        log('\n')

    return (np.array(total_actions),
            np.array(avg_plaquettes),
            np.array(top_charges))

##############################################################################
# Calculate observables directly from `log_dir` using the methods above to help
# load samples.
##############################################################################

def calc_observables(log_dir, observables_dicts=None, training=False):
    """Calculate and save observables from samples contained in `log_dir`.

    Args:
        log_dir: String representing the path containing the relevant samples.
            The samples files themselves should be in a subdirectory of
            `log_dir` called `samples_history`.
        observables_dicts: Tuple of dictionaries (actions_dict, 
                                                  avg_plaquettes_dict, 
                                                  charges_dict, 
                                                  susceptibility_stats_dict).
        training: Boolean value specifying if samples should be loaded from
        separate subdirectory containing samples generated during the training
            process.

    Returns:
        observables_dicts: Tuple of dictionaries (actions_dict, 
                                                  avg_plaquettes_dict, 
                                                  charges_dict, 
                                                  susceptibility_stats_dict).

     NOTE: 
         * If observables_dicts is None, the observables will all be
         recalculated. If observables_dicts is not None, observables will only
         be calculated if they're not already contained in observables_dicts,
         so each of the above dictionaries only gets updated (instead of
         entirely recalculated).  
    """
    #  if training:
    #      output = find_training_samples(log_dir)
    #  else:
    #      output = find_samples(log_dir)

    output = find_samples(log_dir, training)
    params, step_keys, beta_keys, samples_files, figs_dir_dict = output

    if observables_dicts is None:
        actions_dict, plaqs_dict, charges_dict = {}, {}, {}
        susceptibility_stats_dict = {}
        susceptibility_stats_all_dict = {}
    else:
        actions_dict = observables_dicts[0]
        plaqs_dict = observables_dicts[1]
        charges_dict = observables_dicts[2]
        susceptibility_stats_dict = observables_dicts[3]
        #  susceptibility_stats_all_dict = observables_dicts[4]

    for idx, sample_file in enumerate(samples_files):
        step_key = step_keys[idx]
        beta_key = beta_keys[idx]
        if (step_key, beta_key) not in charges_dict.keys():
            log(f"Calculating observables for {step_key}...")
            with open(sample_file, 'rb') as f:
                samples = pickle.load(f)

            actions, plaqs, charges = _calc_observables(samples, params)
            actions_dict[step_key, beta_key] = actions
            plaqs_dict[step_key, beta_key] = plaqs
            charges_dict[step_key, beta_key] = charges

            suscept_stats = calc_susceptibility_stats(charges)

            #  suscept_stats, suscept_stats_all = output

            susceptibility_stats_dict[step_key, beta_key] = {
                'estimate': suscept_stats[0],
                'bias': suscept_stats[1],
                'stderr': suscept_stats[2],
                'conf_interval': suscept_stats[3]
            }

            #  susceptibility_stats_all_dict[step_key] = {
            #      'estimate': suscept_stats_all[0],
            #      'bias': suscept_stats_all[1],
            #      'stderr': suscept_stats_all[2],
            #      'conf_interval': suscept_stats_all[3]
            #  }

            del samples
        else:
            log(f"Observables alredy calculated for {step_key} eval steps.")

    observables_dir = os.path.join(log_dir, 'observables/')
    if training:
        observables_dir = os.path.join(observables_dir, 'training')

    for idx, step_key in enumerate(step_keys):
        beta = beta_keys[idx]
        obs_dir = os.path.join(observables_dir,
                               f'{step_key}_steps_beta_{beta}')
        check_else_make_dir(obs_dir)

        actions_file = os.path.join(obs_dir,
                                    f'actions_{step_key}_beta_{beta}.pkl')
        plaqs_file = os.path.join(obs_dir,
                                  f'plaqs_{step_key}_beta_{beta}.pkl')
        charges_file = os.path.join(obs_dir,
                                    f'charges_{step_key}_beta_{beta}.pkl')

        susceptibility_stats_pkl_file = os.path.join(
            obs_dir, f'susceptibility_stats_{step_key}_beta_{beta}.pkl'
        )
        #  susceptibility_stats_all_pkl_file = os.path.join(
        #      obs_dir, f'susceptibility_stats_all_{step_key}.pkl'
        #  )
        susceptibility_stats_txt_file = os.path.join(
            obs_dir, f'susceptibility_stats_{step_key}_beta_{beta}.txt'
        )

        log(80 * '-' + '\n')
        log(f"Saving actions to: {actions_file}.")
        with open(actions_file, 'wb') as f:
            pickle.dump(actions_dict[step_key, beta], f)

        log(f"Saving plaquettes to: {plaqs_file}.")
        with open(plaqs_file, 'wb') as f:
            pickle.dump(plaqs_dict[step_key, beta], f)

        log(f"Saving topological charges to: {charges_file}.")
        with open(charges_file, 'wb') as f:
            pickle.dump(charges_dict[step_key, beta], f)

        log(f"Saving suscept. stats to: {susceptibility_stats_pkl_file}")
        with open(susceptibility_stats_pkl_file, 'wb') as f:
            pickle.dump(susceptibility_stats_dict[step_key, beta], f)

        #  log(f"Saving ALL suscept. stats to: {susceptibility_stats_pkl_file}")
        #  with open(susceptibility_stats_all_pkl_file, 'wb') as f:
        #      pickle.dump(susceptibility_stats_all_dict[step_key], f)

        log(f"Writing suscept. stats to: {susceptibility_stats_txt_file}")
        strings = []
        for k, v in susceptibility_stats_dict[step_key, beta].items():
            strings.append(f'{k}:\n  {v}')

        #  strings_all = []
        #  for k, v in susceptibility_stats_all_dict[step_key].items():
        #      strings_all.append(f'{k}:\n {v}')

        str0 = f'Topological suscept. stats for {step_key} steps.\n'
        sep_str0 = len(str0) * '-'
        sep_str1 = len(str0) * '-' + '\n'

        write(str0, susceptibility_stats_txt_file, 'w')
        write(sep_str0, susceptibility_stats_txt_file, 'a')
        _ = [write(s, susceptibility_stats_txt_file, 'a') for s in strings]
        write(sep_str0, susceptibility_stats_txt_file, 'a')
        log('\n' + 80 * '-' + '\n')
        write(sep_str0, susceptibility_stats_txt_file, 'a')
        #  _ = [write(s, susceptibility_stats_txt_file, 'a') for s in strings_all]
        #  write(sep_str1, susceptibility_stats_txt_file, 'a')

    observables_dicts = (actions_dict, plaqs_dict, charges_dict,
                         susceptibility_stats_dict)
                         #  susceptibility_stats_all_dict)

    return observables_dicts

def _calc_thermalization_time(samples):
    """Calculate the time required for `samples` to sufficiently thermalize.

    Note:
        There doesn't seem to be a general consesus on how best to approach
        this problem. Further thought required.
    """
    def exp_fn(x, A, B, C, D):
        return A * np.exp(B * (x - C)) + D

    xdata = np.arange(samples.shape[0])

    popt, pcov = curve_fit(f, xdata=xdata, ydata=samples)

    return popt, pcov

##############################################################################
# Load samples and create necessary directory structure
##############################################################################
class Loader(object):
    def __init__(self, log_dir, training=False):
        self.log_dir = log_dir
        self.training = training
        self.info_dir = os.path.join(self.log_dir, 'run_info')
        self.eval_dir = os.path.join(self.log_dir, 'eval_info')
        self.figs_dir = os.path.join(self.log_dir, 'figures')
        check_else_make_dir(self.figs_dir)

        if training:
            self.eval_dir = os.path.join(self.eval_dir, 'training')
            self.samples_dir = os.path.join(self.eval_dir, 'samples')
            self.figs_dir = os.path.join(self.figs_dir, 'training')
            check_else_make_dir(self.figs_dir)
        else:
            self.samples_dir = os.path.join(self.eval_dir, 'samples')

        self.params = self.load_params()
        output = self.find_samples()
        self.step_keys, self.samples_files, self.figs_dir_dict = output

    def load_params(self):
        """Load in model parameters.

        Returns:
            params: Dictionary containing relevant parameters necessary to
                recreate lattice and calculate observables.
        """
        params_file = os.path.join(self.info_dir, 'parameters.pkl')
        try:
            with open(params_file, 'rb') as f:
                params = pickle.load(f)
            return params

        except FileNotFoundError:
            log(f"Unable to find {params_file} in {self.info_dir}. Exiting.")
            return None

    def find_samples(self):
        """Find sample files from root directory `self.log_dir`. """
        if self.training:
            train_step_idx = 2

        samples_files = [os.path.join(self.samples_dir, f)
                         for f in os.listdir(self.samples_dir)
                         if 'samples' in f and f.endswith('.pkl')]

        run_steps_idx = -3
        step_keys = [int(i.split('/')[-1].split('_')[run_steps_idx])
                     for i in samples_files]

        figs_dir_dict = {}
        for key in step_keys:
            d = os.path.join(self.figs_dir, f'{key}_train_steps')
            check_else_make_dir(d)
            figs_dir_dict[key] = d

        return step_keys, samples_files, figs_dir_dict


def find_samples(log_dir, training=False):
    """Calculate observables from collection of samples stored in log_dir.

    Args:
        log_dir: Root directory containing information about run.
    Returns:
        params: Dictionary of parameters. 
            keys: name of parameter
            value: parameter value
        step_keys: List containing the number of steps the sampler was
            evaluated for.
        samples_files: List of files containing samples generated from sampler. 
        figs_dir_dict: Dictionary of directories containing directories to hold
            figures of observables plots.
    """
    params = _load_params(log_dir)
    info_dir = os.path.join(log_dir, 'run_info')
    eval_dir = os.path.join(log_dir, 'eval_info')
    figs_dir = os.path.join(log_dir, 'figures')
    if training:
        figs_dir = os.path.join(figs_dir, 'training')
        samples_dir = os.path.join(eval_dir, 'training', 'samples')
        step_key_idx = 2
    else:
        samples_dir = os.path.join(eval_dir, 'samples')
        step_key_idx = 3

    beta_key_idx = -1

    samples_files = [
        os.path.join(samples_dir, i) for i in os.listdir(samples_dir)
        if 'samples_history' in i and i.endswith('.pkl')
    ]

    step_keys = [
        int(i.split('/')[-1].split('_')[step_key_idx].rstrip('.pkl'))
        for i in samples_files
    ]

    beta_keys = [
        float(i.split('/')[-1].split('_')[beta_key_idx].rstrip('.pkl'))
        for i in samples_files
    ]

    figs_dir_dict = {}
    check_else_make_dir(figs_dir)

    #  for idx, key in enumerate(step_keys):
    for step, beta in zip(step_keys, beta_keys):
        key = (step, beta)
        if training:
            dir_name = f'{step}_train_steps_beta_{beta}'
        else:
            dir_name = f'{step}_steps_beta_{beta}'
        _figs_dir = os.path.join(figs_dir, dir_name)
        #  _figs_dir = os.path.join(figs_dir, f'{key}_steps')
        check_else_make_dir(_figs_dir)
        figs_dir_dict[key] = _figs_dir

    return params, step_keys, beta_keys, samples_files, figs_dir_dict

def find_training_samples(log_dir):
    """Find files containing samples generated during training."""
    info_dir = os.path.join(log_dir, 'run_info/')
    eval_dir = os.path.join(log_dir, 'eval_info/')
    samples_dir = os.path.join(eval_dir, 'training', 'samples')
    params = _load_params(log_dir)

    figs_dir = os.path.join(log_dir, 'figures')

    samples_files = [
        samples_dir + i for i in os.listdir(samples_dir)
        if 'samples_history' in i and i.endswith('.pkl')
    ]

    step_keys = [int(i.split('/')[-1].split('_')[2]) for i in samples_files]

    training_figs_dir = os.path.join(figs_dir, 'training/')
    check_else_make_dir(training_figs_dir)
    training_figs_dir_dict = {}

    for key in step_keys:
        _dir = os.path.join(training_figs_dir, f'{key}_train_steps/')
        check_else_make_dir(_dir)
        training_figs_dir_dict[key] = _dir

    return params, step_keys, samples_files, training_figs_dir_dict

##############################################################################
# High-level plotting functions.
##############################################################################

def plot_observables(log_dir, observables_dicts, training=False):
    """Plot observables stored in `observables_dicts`."""
    if training:
        #  params, _, _, figs_dir_dict = find_training_samples(log_dir)
        params, _, beta_keys, _, figs_dir_dict = find_samples(log_dir,
                                                              training=True)
        title_str_key = 'training'
    else:
        params, _, beta_keys, _, figs_dir_dict = find_samples(log_dir)
        title_str_key = 'evaluation'

    actions_dict, plaqs_dict, charges_dict, _ = observables_dicts

    plt.close('all')
    figs_axes = []
    for key in charges_dict.keys():
        step, beta = key
        observables = (actions_dict[key], plaqs_dict[key], charges_dict[key])
        title_str = (r"$\beta = $"
                     + f"{beta}, {step} {title_str_key} steps")

        kwargs = {
            'figs_dir': figs_dir_dict[key],
            'title': title_str
        }

        fig_ax = make_multiple_lines_plots(
            beta,
            #  params['beta_final'],
            observables,
            **kwargs
        )

        figs_axes.append(fig_ax)
        log(80 * '-')

    return figs_axes

def plot_top_charges(log_dir, charges_dict, training=False):
    """Plot top. charge history using samples generated during training."""
    if training:
        #  params, _, _, figs_dir_dict = find_training_samples(log_dir)
        params, _, beta_keys, _, figs_dir_dict = find_samples(log_dir,
                                                              training=True)
        title_str_key = 'training'
    else:
        params, _, _, _, figs_dir_dict = find_samples(log_dir)
        title_str_key = 'evaluation'
    #  params, _, _, figs_dir_dict = find_samples(log_dir)

    plt.close('all')


    j = 0
    for key, val in charges_dict.items():
        step, beta = key
        for idx in range(val.shape[1]):
            fig, ax = plt.subplots()
            _ = ax.plot(val[:, idx],
                        marker=MARKERS[idx],
                        color=COLORS[idx],
                        ls='',
                        fillstyle='none',
                        label=f'sample {idx}')
            _ = ax.legend(loc='best')
            _ = ax.set_xlabel('Step', fontsize=14)
            _ = ax.set_ylabel('Topological charge', fontsize=14)
            if training:
                title_str = (r"$\beta = $"
                             + f"{beta}, {step} training steps")
            else:
                title_str = (r"$\beta = $"
                             + f"{beta}, {step} eval steps")
            _ = ax.set_title(title_str, fontsize=16)
            out_dir = os.path.join(
                figs_dir_dict[key], 'topological_charges_history'
            )
            check_else_make_dir(out_dir)
            out_file = os.path.join(
                out_dir,
                f'topological_charge_history_sample_{idx}.pdf'
            )
            log(f"Saving figure to {out_file}.")
            if not os.path.isfile(out_file):
                _ = fig.savefig(out_file, dpi=400, bbox_inches='tight')
        log(80 * '-' + '\n')

def plot_top_charges_counts(log_dir, charges_dict, training=False):
    """Create scatter plot for the count of of unique values in charges_dict."""
    if training:
        #  params, _, _, figs_dir_dict = find_training_samples(log_dir)
        params, _, beta_keys, _, figs_dir_dict = find_samples(log_dir,
                                                              training=True)
        title_str_key = 'training'
    else:
        params, _, _, _, figs_dir_dict = find_samples(log_dir)
        title_str_key = 'evaluation'

    for key, val in charges_dict.items():
        step, beta = key
        for idx in range(val.shape[1]):
            counts = Counter(val[:, idx])
            fig, ax = plt.subplots()
            ax.plot(list(counts.keys()),
                    list(counts.values()),
                    marker=MARKERS[idx],
                    color=COLORS[idx],
                    ls='',
                    #  fillstyle='none',
                    label=f'sample {idx}')
            _ = ax.legend(loc='best')
            _ = ax.set_xlabel('Topological charge', fontsize=14)
            _ = ax.set_ylabel('Number of occurences', fontsize=14)
            title_str = (r"$\beta = $"
                         + f"{beta}, "
                         + f"{step} {title_str_key} steps")
            #  title_str = (r"$\beta = $"
            #               + f"{params['beta_final']}, {key} training steps")
            _ = ax.set_title(title_str, fontsize=16)
            out_dir = os.path.join(
                figs_dir_dict[key], 'topological_charges_counts'
            )
            check_else_make_dir(out_dir)
            out_file = os.path.join(
                out_dir,
                f'topological_charge_counts_sample_{idx}.pdf'
            )
            log(f"Saving figure to {out_file}.")
            if not os.path.isfile(out_file):
                _ = fig.savefig(out_file, dpi=400, bbox_inches='tight')
        log(80 * '-' + '\n')


##############################################################################
# Calculate various autocorrelation spectra.
##############################################################################
def _calc_samples_acl_spectrum(samples):
    """Calculate autocorrelation spectrum of `samples`."""
    samples = np.array(samples)
    _shape = samples.shape
    samples = samples.reshape(_shape[0], _shape[1], -1)
    samples_acl_spectrum = acl_spectrum(samples, scale=1)

    return samples_acl_spectrum

def calc_top_charges_autocorr(top_charges_history):
    """Compute the autocorrelation function from the topological charges data.

    Args:
        top_charges_history: Array containing the topological charge history
            for each L2HMC step. If multiple samples were used, this should
            have shape: (num_steps, num_samples)

    Returns:
        autocorr_arr: Array containing the autocorrelation of the topological
            charge. If multiple samples were used, this should have shape:
                (num_samples, num_steps).
        autocorr_avg: Array containing the autocorrelation of the topological
            charge, averaged over the samples (assuming multiple samples were
            used). This should have shape:
                (num_steps,)
    """
    autocorr_arr = []
    num_samples = top_charges_history.shape[1]
    autocorr_arr = np.array([
        autocorr(top_charges_history[:, i]) for i in range(num_samples)
    ])
    #  for i in range(num_samples):
    #      autocorr_arr.append(autocorr(top_charges_history[:, i]))
    #  autocorr_arr = np.array(autocorr_arr)
    autocorr_avg = np.mean(autocorr_arr, axis=0)

    return autocorr_arr, autocorr_avg

def calc_samples_autocorr(samples_history):
    """Compute the autocorrelation function of individual links.

    Args:
        samples_history: Array containing the histories of sample lattice
            (link) configurations for each L2HMC step. If multiple samples were
            used in the batch, this should have shape: 
                (num_steps, num_samples, lattice_time_size, 
                lattice_space_size, lattice_dimension)

    Returns:
        samples_autocorr_arr: Array containing the autocorrelation of the
            individual links from the lattice configurations. If multiple
            samples were used, this should have shape:
                (num_samples, 
                lattice_time_size * lattice_space_size * lattice_dimension,
                num_steps)
        samples_autocorr_arr_avg: Array containing the autocorrelation of the
            individual links from the lattice configurations, averaged over all
            individual links. This should have shape:
                (num_samples, num_steps)
    """
    if not isinstance(samples_history, np.ndarray):
        samples_history = np.array(samples_history)

    _shape = samples_history.shape
    samples_history = samples_history.reshape(_shape[0], _shape[1], -1)
    num_samples = samples_history.shape[1]
    num_links = samples_history.shape[-1]
    samples_autocorr_arr = []
    for n in range(num_samples):
        links_autocorr_arr = []
        for l in range(num_links):
            links_autocorr_arr.append(autocorr(samples_history[:, n, l]))
        samples_autocorr_arr.append(links_autocorr_arr)
    samples_autocorr_arr = np.array(samples_autocorr_arr)
    samples_autocorr_arr_avg = samples_autocorr_arr.mean(axis=1)

    return samples_autocorr_arr, samples_autocorr_arr_avg

def calc_integrated_autocorr_time(data):
    """Calculate the integrated autocorr. time from time-series `data`."""
    acf_arr = [] # autocorrelation function array
    iat_arr = [] # integrated autocorrelation time array
    for idx in range(data.shape[1]):
        try:
            iat, _flag = integrated_time(data[:, idx], quiet=True)
            if _flag:
                log(f'\n Failed on idx: {idx}\n')
            iat_arr.append(iat)
            # `function_1d` computes the autocorr. fn of 1-D time-series
            acf_arr.append(autocorr_func_1d(data[:, idx]))
        except AutocorrError as err:
            log(f'Failed on idx: {idx}\n')
            log(err)
            log('\n')
            continue

    return np.array(acf_arr), np.array(iat_arr)


##############################################################################
# Plot observables calculated above.
##############################################################################

def make_multiple_lines_plots(beta, observables, **kwargs):
    """Create all relevant plots."""
    figs_dir = kwargs.get('figs_dir', None)
    #  legend = kwargs.get('legend', False)
    title = kwargs.get('title', None)

    actions, avg_plaquettes, top_charges = observables

    steps = np.arange(len(actions))
    multiple_lines_figs_axes = []

    if figs_dir is None:
        charges_file, plaquettes_file, actions_file = None, None, None

    else:
        charges_file = os.path.join(figs_dir,
                                    'topological_charge_vs_step.pdf')
        plaquettes_file = os.path.join(figs_dir,
                                       'average_plaquette_vs_step.pdf')
        actions_file = os.path.join(figs_dir,
                                    'average_action_vs_step.pdf')

    ###########################################################################
    # Topological charge
    ###########################################################################
    kwargs = {
        'out_file': charges_file,
        'markers': True,
        'lines': False,
        'legend': True,
        'title': title,
    }
    fig, ax = plot_multiple_lines(steps, top_charges.T,
                                  x_label='Step',
                                  y_label='Topological charge',
                                  **kwargs)

    multiple_lines_figs_axes.append((fig, ax))

    ###########################################################################
    # Average plaquette
    ###########################################################################
    kwargs['out_file'] = None
    kwargs['lines'] = True
    kwargs['markers'] = False
    fig, ax = plot_multiple_lines(steps, avg_plaquettes.T,
                                  x_label='Step',
                                  y_label='Average plaquette',
                                  **kwargs)

    _ = ax.axhline(y=u1_plaq_exact(beta),
                   color='r', ls='--', lw=2.5, label='exact')

    multiple_lines_figs_axes.append((fig, ax))
    if plaquettes_file is not None:
        log(f"Saving figure to: {plaquettes_file}.")
        plt.savefig(plaquettes_file, dpi=400, bbox_inches='tight')

    ###########################################################################
    # Average action
    ###########################################################################
    kwargs['out_file'] = actions_file
    fig, ax = plot_multiple_lines(steps, actions.T,
                                  x_label='Step',
                                  y_label='Total action',
                                  **kwargs)
    multiple_lines_figs_axes.append((fig, ax))

    return multiple_lines_figs_axes


##############################################################################
# Plot autocorrelations for observables calculated above.
##############################################################################

def make_pandas_autocorrelation_plot(data, x_label, y_label, out_file=None):
    """Make autocorrelation plot using `pandas.plotting.autocorrelation_plot`.

    Args:
        top_charges: Array containing time series of topological charges.
        out_file: String specifying where to save the resultant plot.

    Returns:
        fig, ax: Figure and Axes objects (from matplotlib.pyplot.subplots())
    """
    top_charges_series = pd.Series(data)
    fig, ax = plt.subplots()
    autocorrelation_plot(top_charges_series, ax=ax)
    if x_label:
        ax.set_xlabel(x_label, fontsize=14)
    if y_label:
        ax.set_ylabel(y_label, fontsize=14)

    if out_file:
        log(f"Saving figure to: {out_file}.")
        plt.savefig(out_file, dpi=400, bbox_inches='tight')

    return fig, ax

def make_matplotlib_autocorrelation_plot(data, **kwargs):
    """Make autocorrelation plot using `matplotlib.pyplot.acorr`.

    Args:
        top_charges: Array containing time series of topological charges.
        out_file: String specifying where to save the resultant plot.

    Returns:
        lags: Array (length 2 * maxlags + 1) containing the `lag vector`.
        autocorr_vec: Array (length 2 * maxlags + 1) `autocorrelation vector`.
        fig, ax: Figure and Axes objects (from matplotlib.pyplot.subplots())
    """
    maxlags = kwargs.get('maxlags', None)
    x_label = kwargs.get('x_label', None)
    y_label = kwargs.get('y_label', None)
    label = kwargs.get('label', None)
    color = kwargs.get('color', 'k')
    out_file = kwargs.get('out_file', None)

    fig, ax = plt.subplots()
    lags, autocorr_vec, _, _ = ax.acorr(data,
                                        maxlags=maxlags,
                                        usevlines=True,
                                        normed=True,
                                        color=color,
                                        label=label)
    ax.axhline(0, color='r', lw=2)
    ax.grid(True)
    ax.legend(loc='best')

    if x_label:
        ax.set_xlabel(x_label, fontsize=14)
    if y_label:
        ax.set_ylabel(y_label, fontsize=14)

    if out_file:
        log(f"Saving figure to: {out_file}.")
        plt.savefig(out_file, dpi=400, bbox_inches='tight')

    return lags, autocorr_vec, fig, ax

# pylint: disable=invalid-name
def plot_autocorr_with_iat(acf_arr, iat_arr, ESS_arr, **kwargs):
    """Plot the autocorr fns. labeled by their integrated autocorr. time.

    Args:
        acf_arr: Array containing the (normalized) autocorrelation functions.
        iat_arr: Array containing the autocorrelation functions associated
            integrated autocorrelation times.
        **kwargs: Additional keyword arguments passed along to plotting fns.

    Returns:
        fig: matplotlib.pyplot.Figure object
        ax: matplotlib.pyplot.Axis object
    """
    if not isinstance(ESS_arr, np.ndarray):
        ESS_arr = np.array(ESS_arr)

    steps = np.arange(len(acf_arr[0]))

    x_label = kwargs.get('x_label', None)
    y_label = kwargs.get('y_label', None)
    legend = kwargs.get('legend', True)
    out_file = kwargs.get('out_file', None)

    fig, ax = plt.subplots()
    for idx in range(iat_arr.shape[0]):
        ax.plot(steps, acf_arr[idx], ls='-', alpha=0.9,
                label=(r'$\tau_{{\mathrm{int}}} = $'
                       f'{iat_arr[idx][0]:4.3g}, '
                       f'ESS: {ESS_arr[idx]:4.3g}'))
    ax.plot(steps, acf_arr.mean(axis=0), color='k', ls='-', alpha=0.75,
            label=(r'$\tau_{{\mathrm{int}}}^{{(\mathrm{avg})}} = $'
                   f'{iat_arr.mean(axis=0)[0]:4.3g}, '
                   f'ESS: {ESS_arr.mean(axis=0):4.3g}'))

    if legend:
        ax.legend(loc='best')

    if x_label:
        ax.set_xlabel(x_label, fontsize=14)
    if y_label:
        ax.set_ylabel(y_label, fontsize=14)
    if out_file:
        log(f'Saving figure to: {out_file}.')
        plt.savefig(out_file, dpi=400, bbox_inches='tight')

    return fig, ax

def make_samples_acl_spectrum_plot(samples, out_file=None):
    """Create plot of samples acl spectrum."""
    if not isinstance(samples, np.ndarray):
        samples = np.array(samples)
    _shape = samples.shape
    samples = samples.reshape(_shape[0], _shape[1], -1)
    samples_acl_spectrum = acl_spectrum(samples, scale=1)
    acl_steps = np.arange(len(samples_acl_spectrum))


    fig, ax = plt.subplots()
    ax.plot(acl_steps, samples_acl_spectrum/samples_acl_spectrum[0])
    ax.set_xlabel('Step', fontsize=14)
    ax.set_ylabel('Autocorrelation (avg. over links)', fontsize=14)

    if out_file:
        #  out_file = os.path.join(figs_dir, 'links_autocorrelation_vs_step.pdf')
        log(f"Saving figure to: {out_file}.")
        plt.savefig(out_file, dpi=400, bbox_inches='tight')

    return fig, ax
