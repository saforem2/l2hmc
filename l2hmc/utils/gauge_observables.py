"""
Methods for calculating and plotting relevant physical observables using
sample configurations generated from the (L2)HMC sampler.

Author: Sam Foreman (twitter/github @saforem2)
Date: 12 / 9 / 2018
"""
import os
import sys
import time
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pandas.plotting import autocorrelation_plot

from lattice.lattice import GaugeLattice, u1_plaq_exact
#  from lattice.gauge_lattice import GaugeLattice, u1_plaq_exact
#  from l2hmc_eager import gauge_dynamics_eager as gde

from .plot_helper import plot_broken_xaxis, plot_multiple_lines
from .autocorr import (
    integrated_time, autocorr_func_1d, AutocorrError, calc_ESS,
    calc_iat, autocorr_fast, autocorr, autocovariance, acl_spectrum
)

COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

##############################################################################
# File I/O helpers.
##############################################################################
def check_else_make_dir(d):
    """Checks if directory exists, otherwise creates directory."""
    if not os.path.isdir(d):
        print(f"Making directory: {d}.")
        os.makedirs(d)

def _read_from_file(_file):
    """Helper function to load from `.pkl` file."""
    with open(_file, 'rb') as f:
        data = pickle.load(f)
    return data

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
        print(f"Unable to find {params_file} in {info_dir}. Returning 0.")
        return 0

def _load_samples_from_file(samples_file):
    """Load samples from file `f`."""
    with open(samples_file, 'rb') as f:
        print(f"Reading samples from {samples_file}.")
        samples = pickle.load(f)
    return samples

def _load_multiple_samples(samples_history_dir):
    """Load all of the `samples_history` data from `samples_history_dir`"""
    _files = os.listdir(samples_history_dir)

    samples_files = [
        samples_history_dir + '/' + i for i in _files if i.endswith('.pkl')
    ]

    samples_dict = {}
    for _file in samples_files:
        print(f'Loading samples history from: {_file}.')
        num_steps = int(_file.split('/')[-1].split('_')[-1].rstrip('.pkl'))
        samples_dict[num_steps] = np.array(_load_samples_from_file(_file))
        #  with open(_file, 'rb') as f:
        #      samples_dict[num_steps] = np.array(pickle.load(f))

    return samples_dict

def _load_samples(log_dir):
    """Load sample link configurations in from `log_dir`.

    Returns:
        samples: numpy.ndarray containing samples loaded from `log_dir`.
    """
    info_dir = os.path.join(log_dir, 'run_info')

    samples_history_dir = os.path.join(log_dir, 'samples_history')
    if os.path.isdir(samples_history_dir):
        if os.listdir(samples_history_dir) is not None:
            return _load_multiple_samples(samples_history_dir)

    #  assert os.path.isdir(info_dir)
    samples_file = os.path.join(info_dir, 'samples_history.pkl')
    if os.path.isfile(samples_file):
        print(f"Loading samples from: {samples_file}.")
        with open(samples_file, 'rb') as f: # pylint: disable=invalid-name
            samples_arr = pickle.load(f)
        return samples_arr
    else:
        print(f"Unable to find {samples_file} in {info_dir}. Exiting.")
        sys.exit(0)

def _load_multiple_observables(observables_dir):
    """Load multiple observables from `observables_dir`."""
    _files = os.listdir(observables_dir)
    observables_files = [
        observables_dir + '/' + i for i in _files if i.endswith('.pkl')
    ]
    observables_dict = {}
    for _file in observables_files:
        print(f"Loading observables from {_file}.")
        num_steps = int(_file.split('/')[-1].split('_')[-1].rstrip('.pkl'))
        with open(_file, 'rb') as f:
            observables_dict[num_steps] = np.array(pickle.load(f))

    return observables_dict

def _load_observables(log_dir):
    """Load observables from: `log_dir/run_info/observables.pkl`."""
    info_dir = os.path.join(log_dir, 'run_info')

    observables_dir = os.path.join(log_dir, 'observables')
    if os.path.isdir(observables_dir):
        if os.listdir(observables_dir) is not None:
            return _load_multiple_observables(observables_dir)

    observables_file = os.path.join(info_dir, 'observables.pkl')
    if os.path.isfile(observables_file):
        print(f"Loading observables from: {observables_file}.")
        with open(observables_file, 'rb') as f: # pylint: disable=invalid-name
            observables = pickle.load(f)
        return observables
    else:
        print(f"Unable to find {observables_file} in {info_dir}. Exiting.")
        return False

def _save_observables_to_file(observables, log_dir, out_file=None):
    """Save calculated observables to pickle file."""
    observables_dir = os.path.join(log_dir, 'observables')
    check_else_make_dir(observables_dir)
    if out_file is None:
        observables_file = os.path.join(observables_dir, 'observables.pkl')
    else:
        observables_file = os.path.join(observables_dir, out_file)
    print(f'Saving calculated observables to: {observables_file}.')
    with open(observables_file, 'wb') as f: # pylint:disable=invalid-name
        pickle.dump(observables, f)
    print('done.')


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
    # if len(samples[0].shape) == 4, then each item in `samples` represents a
    # single batch of link configurations, so we use the
    # `lattice.calc_plaq_observables` method for calculating observables.
    # similarly, if len(samples.shape) >= 3, then `samples` either has shape
    #     (num_steps, num_samples, time_size, space_size, dim) == 4
    # or
    #     (num_steps, num_samples, time_size * space_size * dim) == 3
    cond1 = len(samples[0].shape) == 4
    cond2 = len(samples.shape) >= 3
    if cond1 or cond2:
        observables_fn = lattice.calc_plaq_observables
    # if len(samples[0].shape) == 3, then each item in `samples` contains only
    # a single link configuration.
    # similarly, if samples.shape[-1] == lattice.num_links, samples has shape:
    #     (num_steps, lattice.num_links) (i.e. a single MCMC chain)
    # in either of these cases we then use the 
    # `lattice._calc_plaq_observables` method for calculating observables.
    cond3 = len(samples[0].shape) == 3
    cond4 = samples[0].shape[0] == lattice.num_links
    # pylint:disable=protected-access
    if cond3 or cond4:
        observables_fn = lattice._calc_plaq_observables

    if observables_fn is None:
        raise ValueError(f"Incorrect shape for `samples` {samples.shape}")

    total_actions = []
    avg_plaquettes = []
    top_charges = []
    beta = params['beta_final']
    for idx, sample in enumerate(samples):
        t0 = time.time()
        observables = np.array(observables_fn(sample, beta))
        actions, plaqs, charges = observables

        total_actions.append(actions)
        avg_plaquettes.append(plaqs)
        top_charges.append(charges)

        print(f"step: {idx} "
              f"time/step: {time.time() - t0:^6.4g} "
              f"avg action: {np.mean(actions):^6.4g} "
              f"avg plaquette: {np.mean(plaqs):^6.4g} "
              "top charges: ")
        print('\n')
        print([int(i) for i in charges])
        print('\n')


              #  f"top charges: {np.mean(charges):^6.4g}")

    return (np.array(total_actions),
            np.array(avg_plaquettes),
            np.array(top_charges))

def calc_observables_from_log_dir(log_dir):
    """Calculate observables from collection of samples stored in log_dir.

    Args:
        log_dir: Root directory containing information about run.
    Returns:
        params, samples, observables: Tuple of parameters used during run,
            samples generated from learned sampler, and the observables
            calculated from those samples.
    """
    params = _load_params(log_dir)
    samples = _load_samples(log_dir)
    observables = _load_observables(log_dir)

    if isinstance(samples, dict):
        if not observables:
            observables = {}
            for key, val in samples.items():
                observables[key] = _calc_observables(val, params)
                _save_observables_to_file(
                    observables[key],
                    log_dir,
                    out_file=f'observables_{key}.pkl'
                )
    else:
        if not observables:
            observables = _calc_observables(samples, params)
            _save_observables_to_file(observables, log_dir)

    return params, samples, observables

def _calc_thermalization_time(samples):
    """Calculate the time required for `samples` to sufficiently thermalize.

    Note:
        There doesn't seem to be a general consesus on how best to approach
        this problem. Further thought required.
    """
    pass


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
                print(f'\n Failed on idx: {idx}\n')
            iat_arr.append(iat)
            # `function_1d` computes the autocorr. fn of 1-D time-series
            acf_arr.append(autocorr_func_1d(data[:, idx]))
        except AutocorrError as err:
            print(f'Failed on idx: {idx}\n')
            print(err)
            print('\n')
            continue

    return np.array(acf_arr), np.array(iat_arr)


##############################################################################
# Plot observables calculated above.
##############################################################################
def make_broken_xaxis_plots(figs_dir, beta, observables,
                            top_charges_autocorr, legend=False):
    """Make plots with broken xaxis."""
    actions, avg_plaquettes, top_charges = observables
    #  top_charges_autocorr_arr, top_charges_autocorr_avg = top_charges_autocorr
    #  samples_autocorr_arr, samples_autocorr_avg = samples_autocorr

    steps = np.arange(len(actions))

    broken_xaxis_figs_axes = []
    ###########################################################################
    # Topological charge autocorrelation function
    ###########################################################################
    out_file = os.path.join(figs_dir,
                            'topological_charge_autocorr_fn_broken_xaxis.pdf')

    fig, ax, ax2 = plot_broken_xaxis(steps, top_charges_autocorr.T,
                                     x_label='step',
                                     y_label='Autocorrelation (top. charge)',
                                     legend=legend,
                                     out_file=out_file)

    broken_xaxis_figs_axes.append((fig, ax, ax2))

    ###########################################################################
    # Topological charge
    ###########################################################################
    out_file = os.path.join(figs_dir,
                            'topological_charge_vs_step_broken_xaxis.pdf')
    fig, ax, ax2 = plot_broken_xaxis(x_data=steps,
                                     y_data=top_charges,
                                     x_label='step',
                                     y_label='Topological charge',
                                     legend=legend,
                                     out_file=None)
    if legend:
        ax2.legend(loc='lower right')

    broken_xaxis_figs_axes.append((fig, ax, ax2))
    plt.savefig(out_file, dpi=400, bbox_inches='tight')

    ###########################################################################
    # Average plaquette 
    ###########################################################################
    out_file = os.path.join(figs_dir,
                            'average_plaquette_vs_step_broken_xaxis.pdf')
    fig, ax, ax2 = plot_broken_xaxis(steps, avg_plaquettes,
                                     x_label='step',
                                     y_label='Average plaquette',
                                     legend=legend)
    _ = ax.axhline(y=u1_plaq_exact(beta), color='r', ls='--', lw=2.5,
                   label='exact')
    _ = ax2.axhline(y=u1_plaq_exact(beta), color='r', ls='--', lw=2.5,
                    label='exact')
    if legend:
        _ = ax2.legend(loc='lower right', fontsize=10)

    broken_xaxis_figs_axes.append((fig, ax, ax2))
    plt.savefig(out_file, dpi=400, bbox_inches='tight')

    ###########################################################################
    # Average action
    ###########################################################################
    out_file = os.path.join(figs_dir,
                            'average_action_vs_step_broken_xaxis.pdf')
    fig, ax, ax2 = plot_broken_xaxis(steps, actions,
                                     x_label='step',
                                     y_label='Total action',
                                     legend=legend,
                                     out_file=out_file)

    broken_xaxis_figs_axes.append((fig, ax, ax2))

    return broken_xaxis_figs_axes

def make_multiple_lines_plots(beta, observables, figs_dir=None, legend=False):
    """Create all relevant plots."""
    actions, avg_plaquettes, top_charges = observables

    steps = np.arange(len(actions))
    multiple_lines_figs_axes = []

    ###########################################################################
    # Topological charge autocorrelation function
    ###########################################################################
    #  out_file = os.path.join(figs_dir, 'topological_charge_autocorr_fn.pdf')
    #  fig, ax = plot_multiple_lines(steps, top_charges_autocorr,
    #                                x_label='step',
    #                                y_label='Autocorrelation (top. charge)',
    #                                legend=legend,
    #                                out_file=out_file)
    #  multiple_lines_figs_axes.append((fig, ax))

    ###########################################################################
    # Topological charge
    ###########################################################################
    if figs_dir is None:
        charges_file, plaquettes_file, actions_file = None, None, None

    else:
        charges_file = os.path.join(figs_dir,
                                    'topological_charge_vs_step.pdf')
        plaquettes_file = os.path.join(figs_dir,
                                       'average_plaquette_vs_step.pdf')
        actions_file = os.path.join(figs_dir,
                                    'average_action_vs_step.pdf')
    #  kwargs = {
    #      marker=''
    #  }
    fig, ax = plot_multiple_lines(steps, top_charges.T,
                                  x_label='step',
                                  y_label='Topological charge',
                                  markers=True,
                                  lines=False,
                                  legend=True,
                                  out_file=charges_file)
    multiple_lines_figs_axes.append((fig, ax))

    ###########################################################################
    # Average plaquette
    ###########################################################################
    fig, ax = plot_multiple_lines(steps, avg_plaquettes.T,
                                  x_label='step',
                                  y_label='Average plaquette',
                                  legend=legend,
                                  out_file=plaquettes_file)
    _ = ax.axhline(y=u1_plaq_exact(beta),
                   color='r', ls='--', lw=2.5, label='exact')

    multiple_lines_figs_axes.append((fig, ax))
    if plaquettes_file is not None:
        plt.savefig(plaquettes_file, dpi=400, bbox_inches='tight')

    ###########################################################################
    # Average action
    ###########################################################################
    fig, ax = plot_multiple_lines(steps, actions.T,
                                  x_label='step',
                                  y_label='Total action',
                                  legend=legend,
                                  out_file=actions_file)
    multiple_lines_figs_axes.append((fig, ax))

    return multiple_lines_figs_axes

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
        print(f"Saving figure to: {out_file}.")
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
        print(f"Saving figure to: {out_file}.")
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
        print(f'Saving figure to: {out_file}.')
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
        print(f"Saving figure to: {out_file}.")
        plt.savefig(out_file, dpi=400, bbox_inches='tight')

    return fig, ax

def make_plots_from_log_dir(log_dir):
    """All-in-one function for calculating observables and creating plots."""
    params, samples, observables = calc_observables_from_log_dir(log_dir)

    _, _, top_charges = observables

    beta = params['beta']
    figs_dir = os.path.join(log_dir, 'figures')


    top_charges_autocorr, _ = calc_top_charges_autocorr(top_charges)

    #  samples_autocorr, _ = calc_samples_autocorr(samples)

    #  broken_xaxis_figs_axes = make_broken_xaxis_plots(
    #      figs_dir,
    #      beta,
    #      observables,
    #      top_charges_autocorr
    #  )

    multiple_lines_figs_axes = make_multiple_lines_plots(
        figs_dir,
        beta,
        observables,
        top_charges_autocorr
    )

    return multiple_lines_figs_axes #, broken_xaxis_figs_axes

def calc_observables_generate_plots(log_dir):
    """Wrapper function for calculating all relevant observables and plots."""
    figs_dir = os.path.join(log_dir, 'figures')
    autocorr_dir = os.path.join(figs_dir, 'autocorrelation_plots')
    pandas_autocorr_dir = os.path.join(autocorr_dir, 'pandas_autocorr_plots')
    mpl_autocorr_dir = os.path.join(autocorr_dir, 'mpl_autocorr_dir')
    check_else_make_dir(autocorr_dir)
    check_else_make_dir(mpl_autocorr_dir)
    check_else_make_dir(pandas_autocorr_dir)

    #########################################################################
    # Calculate observables
    #########################################################################
    params, samples, observables = calc_observables_from_log_dir(log_dir)
    if isinstance(observables, dict):
        actions = {}
        avg_plaquettes = {}
        top_charges = {}
        for key, val in observables.items():
            _actions, _avg_plaquettes, _top_charges = val
            actions[key] = _actions
            avg_plaquettes[key] = _avg_plaquettes
            top_charges[key] = _top_charges

    else:
        actions, avg_plaquettes, top_charges = observables

    #########################################################################
    # Calculate autocorr fns, integrated autocorr times and ESS
    #########################################################################
    output = calc_top_charges_autocorr(top_charges)
    top_charges_autocorr, top_charges_autocorr_avg = output

    acf_arr, iat_arr = calc_integrated_autocorr_time(top_charges)

    ESS_arr = []
    for acf in acf_arr:
        ESS_arr.append(calc_ESS(acf))

    samples_autocorr, samples_autocorr_avg = calc_samples_autocorr(samples)

    #########################################################################
    # Create plots
    #########################################################################
    multiple_lines_figs_axes = make_multiple_lines_plots(
        params['beta'],
        observables,
        figs_dir=figs_dir,
        legend=False
    )

    #  broken_xaxis_figs_axes = make_broken_xaxis_plots(
    #      params['beta'],
    #      observables,
    #      figs_dir,
    #      legend=False,
    #  )

    for idx in range(top_charges.shape[1]):
        out_file = os.path.join(
            pandas_autocorr_dir,
            f'top_charges_autocorr_pandas_{idx}.pdf'
        )
        fig, ax = make_pandas_autocorrelation_plot(
            top_charges[:, idx],
            x_label='Lag',
            y_label='Autocorrelation (top. charge)',
            out_file=out_file
        )

    for idx in range(top_charges.shape[1]):
        out_file = os.path.join(
            mpl_autocorr_dir,
            f'top_charges_autocorr_mpl_{idx}.pdf'
        )
        kwargs = {
            'x_label': 'Lag',
            'y_label': 'Autocorrelation (top. charge)',
            'label': f'sample {idx}',
            'out_file': out_file,
            'color': COLORS[idx]
        }
        output = make_matplotlib_autocorrelation_plot(
            top_charges[:, idx],
            **kwargs
        )

    out_file = os.path.join(figs_dir, 'links_autocorrelation_vs_step.pdf')
    fig, ax = make_samples_acl_spectrum_plot(samples, out_file)

    out_file = os.path.join(
        figs_dir,
        'integrated_autocorrelation_time_plot.pdf'
    )
    kwargs = {
        'x_label': 'Lag',
        'y_label': 'Autocorrelation (top. charge)',
        'legend': True,
        'out_file': out_file
    }
    fig, ax = plot_autocorr_with_iat(acf_arr, iat_arr, ESS_arr, **kwargs)
