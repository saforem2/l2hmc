import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import functools
import argparse
import sys
import os
import signal
import pickle
#  from pathlib import Path
#  os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from mpl_toolkits.mplot3d import Axes3D
from utils.func_utils import accept, jacobian, autocovariance,\
        get_log_likelihood, binarize, normal_kl, acl_spectrum, ESS
from utils.distributions import GMM
from utils.layers import Linear, Sequential, Zip, Parallel, ScaleTanh
from utils.dynamics import Dynamics
from utils.sampler import propose
from utils.notebook_utils import get_hmc_samples
from utils.tf_logging import variable_summaries, get_run_num, make_run_dir
from utils.tunneling import distance, calc_min_distance, calc_tunneling_rate,\
        find_tunneling_events
from utils.jackknife import block_resampling, jackknife_err


# look at scaling with dimensionality, look at implementing simple U2 model
# into distributions and see if any unforseen prooblems arise. 


###############################################################################
#  TODO:
#   * Go back to 2D case and look at different starting temperatures
#   * Make trajectory length go with root T, go with higher temperature
#   * In 2D start with higher initial temp to get around 50% acceptance rate
# -----------------------------------------------------------------------------
#   * For Lattice model:
#       - Define distance as difference in average plaquette.
#       - Look at site by site difference in plaquette (not sum) to prevent
#       integer values that would be the same across different configurations
#       - Try to get network to be compatible with complex numbers and
#       eventually complex matrices.
###############################################################################

def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

#  TODO: Look at using pathlib to deal with paths

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
    root_log_dir = '../log_mog_tf/'
    log_dir = make_run_dir(root_log_dir)
    info_dir = log_dir + 'run_info/'
    figs_dir = log_dir + 'figures/'
    if not os.path.isdir(info_dir):
        os.makedirs(info_dir)
    if not os.path.isdir(figs_dir):
        os.makedirs(figs_dir)
    return log_dir, info_dir, figs_dir

def distribution_arr(x_dim, n_distributions):
    """Create array describing likelihood of drawing from distributions."""
    assert x_dim >= n_distributions, ("n_distributions must be less than or"
                                      " equal to x_dim.")
    if x_dim == n_distributions:
        big_pi = round(1.0 / n_distributions, x_dim)
        arr = n_distributions * [big_pi]
        return np.array(arr, dtype=np.float32)
    else:
        big_pi = (1.0 / n_distributions) - 1E-16
        arr = n_distributions * [big_pi]
        small_pi = (1. - sum(arr)) / (x_dim - n_distributions)
        arr.extend((x_dim - n_distributions) * [small_pi])
        return np.array(arr, dtype=np.float32)

def network(x_dim, scope, factor):
    with tf.variable_scope(scope):
        net = Sequential([
            Zip([
                Linear(x_dim, 10, scope='embed_1', factor=1.0 / 3),
                Linear(x_dim, 10, scope='embed_2', factor=factor * 1.0 / 3),
                Linear(2, 10, scope='embed_3', factor=1.0 / 3),
                lambda _: 0.,
            ]),
            sum,
            tf.nn.relu,
            Linear(10, 10, scope='linear_1'),
            tf.nn.relu,
            Parallel([
                Sequential([
                    Linear(10, x_dim, scope='linear_s', factor=0.001),
                    ScaleTanh(x_dim, scope='scale_s')
                ]),
                Linear(10, x_dim, scope='linear_t', factor=0.001),
                Sequential([
                    Linear(10, x_dim, scope='linear_f', factor=0.001),
                    ScaleTanh(x_dim, scope='scale_f'),
                ])
            ])
        ])
    return net

def plot_trajectory_and_distribution(samples, trajectory, x_dim=None):
    if samples.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2],
           alpha=0.5, marker='o', s=15, color='C0')
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                color='C1', marker='o', markeredgecolor='C1', alpha=0.75,
                ls='-', lw=1., markersize=2)
    if samples.shape[1] == 2:
        fig, ax = plt.subplots()
        ax.scatter(samples[:, 0], samples[:, 1],  color='C0', alpha=0.6)
        ax.plot(trajectory[:, 0], trajectory[:, 1],
                 color='C1', marker='o', alpha=0.8, ls='-')
    return fig, ax



class GaussianMixtureModel(object):
    """Model for training L2HMC using multiple Gaussian distributions."""
    def __init__(self, params, log_dir=None):
        """Initialize parameters and define relevant directories."""
        self._init_params(params)

        if log_dir is not None:
            dirs = check_log_dir(log_dir)
        else:
            dirs = create_log_dir()

        self.log_dir, self.info_dir, self.figs_dir = dirs

        self.tunneling_rates_file = self.info_dir + 'tunneling_rates.pkl'
        self.params_file = self.info_dir + 'params_dict.pkl'

        if os.path.isfile(self.params_file):
            self._load_variables()
            #  import pdb
            #  pdb.set_trace()
            #  print(f'\nload_variables exception: {e}\n')
            #  self.log_dir, self.info_dir, self.figs_dir = create_log_dir()


        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        tf.add_to_collection('global_step', self.global_step)

        self.learning_rate = tf.train.exponential_decay(
            self.params['lr_init'],
            self.global_step,
            self.params['lr_decay_steps'],
            self.params['lr_decay_rate'],
            staircase=True
        )

    def _init_params(self, params):
        """Parse keys in params dictionary to be used for setting instance
        parameters."""
        self.params = {}
        self.params['x_dim'] = params.get('x_dim', 3)
        #  number of Gaussian distributions to use for target distribution
        self.params['num_distributions'] = params.get('num_distributions', 2)
        self.params['lr_init'] = params.get('lr_init', 1e-3)
        self.params['lr_decay_steps'] = params.get('lr_decay_steps', 1000)
        self.params['lr_decay_rate'] = params.get('lr_decay_rate', 0.96)
        self.params['temp_init'] = params.get('temp_init', 10)
        self.params['annealing_steps'] = params.get('annealing_steps', 100)
        self.params['annealing_rate'] = params.get('annealing_rate', 0.98)
        #  Initial step size (learnable)
        self.params['eps'] = params.get('eps', 0.1)
        self.params['scale'] = params.get('scale', 0.1)
        nts = params.get('num_training_steps', 2e4)
        self.params['num_training_steps'] = nts
        #  number of samples (minibatch size) to use for training
        self.params['num_samples'] = params.get('num_samples', 200)
        #  Length of trajectory to use during training 
        ttl0 = params.get('train_trajectory_length', 10)
        self.params['train_trajectory_length'] = ttl0
        #  Length of trajectory to use when calculating tunneling rate info
        ttl1 = params.get('test_trajectory_length', 2e3)
        self.params['test_trajectory_length'] = ttl1
        #  Standard deviation of Gaussian distributions in target distribution
        self.params['sigma'] = params.get('sigma', 0.05)
        #  If num_distributions < x_dim, the likelihood of drawing a sample
        #  outside of the target distributions.
        self.params['small_pi'] = params.get('small_pi', 2e-16)
        self.params['logging_steps'] = params.get('logging_steps', 100)
        #  How often tunneling rate info should be calculated
        trs = params.get('tunneling_rate_steps', 500)
        self.params['tunneling_rate_steps'] = trs
        self.params['save_steps'] = params.get('save_steps', 2500)

        #  Array containing the position of the Gaussian target distributions
        self.means = params.get('means', np.eye(self.params['x_dim']))
        self.covs, self.distribution = self._distribution(self.params['sigma'],
                                                          self.means)
        # Initial samples drawn from Normal distribution
        self.samples = np.random.randn(self.params['num_samples'],
                                       self.params['x_dim'])
        self.tunneling_rates = {}
        self.tunneling_rates_avg = []
        self.tunneling_rates_err = []
        self.acceptance_rates = {}
        self.acceptance_rates_avg = []
        self.acceptance_rates_err = []
        self.losses = []
        #  self.tunneling_info = []
        self.temp_arr = []
        self.steps_arr = []
        self.temp = self.params['temp_init']

    def _distribution(self, sigma, means):
        """Initialize distribution using utils/distributions.py"""
        means = np.array(means).astype(np.float32)
        cov_mtx = sigma * np.eye(self.params['x_dim']).astype(np.float32)
        covs = np.array([cov_mtx] * self.params['x_dim']).astype(np.float32)
        dist_arr = distribution_arr(self.params['x_dim'],
                                    self.params['num_distributions'])
        distribution = GMM(means, covs, dist_arr)
        return covs, distribution

    def _create_dynamics(self, trajectory_length, eps, use_temperature=True):
        """ Create dynamics object using 'utils/dynamics.py'. """
        energy_function = self.distribution.get_energy_function()
        self.dynamics = Dynamics(self.params['x_dim'],
                                 energy_function,
                                 trajectory_length,
                                 eps,
                                 net_factory=network,
                                 use_temperature=use_temperature)

    def _create_loss(self):
        """ Initialize loss and build recipe for calculating it during
        training. """
        with tf.name_scope('loss'):
            self.x = tf.placeholder(tf.float32, shape=(None,
                                                       self.params['x_dim']),
                                    name='x')
            self.z = tf.random_normal(tf.shape(self.x), name='z')
            self.Lx, _, self.px, self.output = propose(self.x, self.dynamics,
                                                       do_mh_step=True)
            self.Lz, _, self.pz, _ = propose(self.z, self.dynamics,
                                             do_mh_step=False)
            self.loss = tf.Variable(0., trainable=False, name='loss')
            v1 = ((tf.reduce_sum(tf.square(self.x - self.Lx), axis=1) * self.px)
                  + 1e-4)
            v2 = ((tf.reduce_sum(tf.square(self.z - self.Lz), axis=1) * self.pz)
                  + 1e-4)
            scale = self.params['scale']

            #  tf.assign_add(self.loss, (scale * (tf.reduce_mean(1.0 / v1)
            #                                     + tf.reduce_mean(1.0 / v2))))
            #  tf.assign_add(self.loss, (- tf.reduce_mean(v1, name='v1')
            #                            - tf.reduce_mean(v1, name='v2')) / scale)
            self.loss = self.loss + scale * (tf.reduce_mean(1.0 / v1) +
                                             tf.reduce_mean(1.0 / v2))
            self.loss = self.loss + ((- tf.reduce_mean(v1, name='v1')
                                      - tf.reduce_mean(v2, name='v2')) / scale)

    def _create_optimizer(self):
        """Initialize optimizer to be used during training."""
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss,
                                               global_step=self.global_step,
                                               name='train_op')

    def _create_summaries(self):
        """Create summary objects for logging in tensorboard."""
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram_loss', self.loss)
            variable_summaries(self.loss)
            self.summary_op = tf.summary.merge_all()

    def _create_params_file(self):
        """Write relevant parameters to .txt file for reference."""
        params_txt_file = self.info_dir + 'parameters.txt'
        with open(params_txt_file, 'w') as f:
            for key, value in self.params.items():
                f.write(f'\n{key}: {value}\n')
            f.write(f"\nmeans:\n\n {str(self.means)}\n"
                    f"\ncovs:\n\n {str(self.covs)}\n")
        print(f'params file written to: {params_txt_file}')

    def _save_variables(self):
        """Save current values of variables."""
        print(f"Saving parameter values to: {self.info_dir}")
        with open(self.tunneling_rates_file, 'wb') as f:
            pickle.dump(self.tunneling_rates, f)
        with open(self.params_file, 'wb') as f:
            pickle.dump(self.params, f)

        np.save(self.info_dir + 'losses_array', np.array(self.losses))
        np.save(self.info_dir + 'steps_array', np.array(self.steps_arr))
        np.save(self.info_dir + 'temp_array', np.array(self.temp_arr))
        #  np.save(self.info_dir + 'tunneling_info', self.tunneling_info)
        np.save(self.info_dir + 'means', self.means)
        np.save(self.info_dir + 'covariances', self.covs)

        np.save(self.info_dir + 'tunneling_rates_avg',
                np.array(self.tunneling_rates_avg))
        np.save(self.info_dir + 'tunneling_rates_err',
                np.array(self.tunneling_rates_err))
        np.save(self.info_dir + 'acceptance_rates_avg',
                np.array(self.acceptance_rates_avg))
        np.save(self.info_dir + 'acceptance_rates_err',
                np.array(self.acceptance_rates_err))


        print("done!\n")

    def _load_variables(self):
        """Load variables from previously ran experiment."""
        print(f'Loading from previous parameters in from: {self.info_dir}')

        self.params = {}
        with open(self.params_file, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        self.steps_arr = list(np.load(self.info_dir + 'steps_array.npy'))
        self.temp_arr = list(np.load(self.info_dir + 'temp_array.npy'))
        self.temp = self.temp_arr[-1]

        self.losses = list(np.load(self.info_dir + 'losses_array.npy'))
        self.means = np.load(self.info_dir + 'means.npy')
        self.covs = np.load(self.info_dir + 'covariances.npy')

        #  self.tunneling_info = list(np.load(self.info_dir
        #                                     + 'tunneling_info.npy'))
        self.tunneling_rates_avg = list(np.load(self.info_dir
                                                + 'tunneling_rates_avg.npy'))
        self.tunneling_rates_err = list(np.load(self.info_dir
                                                + 'tunneling_rates_err.npy'))
        self.acceptance_rates_avg = list(np.load(self.info_dir +
                                                 'acceptance_rates_avg.npy'))
        self.acceptance_rates_err = list(np.load(self.info_dir +
                                                 'acceptance_rates_err.npy'))

    def build_graph(self):
        """Build the graph for our model."""
        #  if self.log_dir is None:
            #  self._create_log_dir()
        #energy_function = self.distribution.get_energy_function()
        self._create_dynamics(self.params['train_trajectory_length'],
                              self.params['eps'],
                              use_temperature=True)
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()
        self._create_params_file()

    def generate_trajectories(self, sess, temperature=1.):
        """ Generate trajectories using current values from L2HMC update
        method.  """
        _samples = self.distribution.get_samples(self.params['num_samples'])
        _trajectories = []
        _loss_arr = []
        _px_arr = []
        for step in range(self.params['test_trajectory_length']):
            _trajectories.append(np.copy(_samples))
            _feed_dict = {self.x: _samples,
                          self.dynamics.temperature: temperature,}
            _loss, _samples, _px = sess.run([
                self.loss,
                self.output[0],
                self.px
            ], feed_dict=_feed_dict)
            _loss_arr.append(np.copy(_loss))
            _px_arr.append(np.copy(_px))
        return np.array(_trajectories), np.array(_loss_arr), np.array(_px_arr)

    def calc_tunneling_rates(self, trajectories):
        """Calculate tunneling rates from trajectories."""
        tunneling_rates = []
        for i in range(trajectories.shape[1]):
            rate = find_tunneling_events(trajectories[:, i, :], self.means,
                                         self.params['num_distributions'])
            tunneling_rates.append(rate)
        #  tunneling_rate_avg = np.mean(tunneling_rate)
        #  tunneling_rate_std = np.std(tunneling_rate)
        return tunneling_rates

    def calc_avg_vals_errors(self, data, num_blocks=20):
        """ Calculate average values and errors of using block jackknife
        resampling method.

        Args:
            data (dict or array-like):
                Data of interest.
            step (int):
                If data is a dictionary, step provides the key corresponding to
                the relevant data array.
            num_blocks (int):
                Number of blocks to use for block jackknife resampling.
        """
        #  if isinstance(data, dict):
        #      try:
        #          arr = np.array(data[step])
        #      except KeyError:
        #          print(f'Key {step} is invalid. Exiting.')
        #  else:
        #      arr = data
        arr = np.array(data)
        avg_val = np.mean(arr)
        avg_val_rs = []
        arr_rs = block_resampling(arr, num_blocks)
        for block in arr_rs:
            avg_val_rs.append(np.mean(block))
        error = jackknife_err(y_i=avg_val_rs,
                              y_full=avg_val,
                              num_blocks=num_blocks) / len(arr)
        return avg_val, error

    def _temp_refresh(self, tunn_avg_err):
        """
        (EXPERIMENTAL):
            If current value of temperature is less than 5, and the newly
            calculated tunneling rate is smaller than the value it had the last
            time it was calculated (plus the previous values error), implement
            a temperature refresh to prevent  the model from getting stuck in a
            local minimum of phase space.

        Args:
            tunn_avg_err (array-like):
                tunn_avg_err[0] contains average tunneling rate
                tunn_avg_err[1] contains average tunneling rate error.
        """
        if 1 < self.temp < 5:
            new_tunneling_rate = tunn_avg_err[0]
            prev_tunneling_rate = 0
            if len(self.tunneling_rates_avg) > 1:
                prev_tunneling_rate = self.tunneling_rates_avg[-2]

            tunneling_rate_diff = (new_tunneling_rate
                                   - prev_tunneling_rate
                                   + 2 * tunn_avg_err[1])

            #  if the tunneling rate decreased since the last
            #  time it was calculated, restart the temperature 
            if tunneling_rate_diff < 0:
                # the following will revert self.temp to a
                # value slightly smaller than the value it had
                # previously the last time the tunneling rate
                # was calculated
                print("\n\tTunneling rate decreased!")
                print("\tNew tunneling rate:"
                      f" {new_tunneling_rate:.3g}, "
                      "Previous tunneling_rate:"
                      f" {prev_tunneling_rate:.3g}, "
                      f"diff: {tunneling_rate_diff:.3g}\n")
                print("\tResetting temperature...")
                if len(self.temp_arr) > 1:
                    prev_temp = self.temp_arr[-2]
                    new_temp = (prev_temp *
                                self.params['annealing_rate'])
                    print(f"\tCurrent temp: {self.temp:.3g}, "
                          f"\t Previous temp: {prev_temp:.3g}, "
                          f"\t New temp: {new_temp:.3g}\n")
                    self.temp = new_temp
                    self.temp_arr[-1] = self.temp

    def _calc_tunneling_info(self, sess, step, temp_refresh=True):
        """
        Calculate average tunneling rate and error by generating
        trajectories at a temperature of 1.

        Args:
            sess (tf.Session):
                Current tensorflow session.
            temp_refresh (boolean):
                Whether or not to implement the temperature
                refresher. (experimental)
        """
        self.temp_arr.append(self.temp)
        self.steps_arr.append(step+1)

        # obtain trajectories, loss, and acceptance rates for all 'num_samples'
        # samples
        trajectory_stats = self.generate_trajectories(sess, self.temp)
        # trajectories are contained in trajectory_stats[0]
        tunneling_rates = self.calc_tunneling_rates(trajectory_stats[0])
        # tunneling rates for all num_samples samples contained in
        # tunneling_stats[0]
        self.tunneling_rates[step] = tunneling_rates
        # not sure if these are needed
        #  tunneling_rate_avg = tunneling_stats[1]
        #  tunneling_rate_std = tunneling_stats[2]
        #  tunneling_info = [step, tunneling_rate_avg,
        #                    tunneling_rate_std]
        #  self.tunneling_info.append(tunneling_info)
        #  Calculate the average value and error of tunneling rates 
        tunn_avg_err = self.calc_avg_vals_errors(self.tunneling_rates[step])
        self.tunneling_rates_avg.append(tunn_avg_err[0])
        self.tunneling_rates_err.append(tunn_avg_err[1])

        # not sure if this is needed
        loss_arr = trajectory_stats[1]

        # acceptance rates are contained in trajectory_stats[2]
        self.acceptance_rates[step] = trajectory_stats[2]
        accept_avg_err = self.calc_avg_vals_errors(self.acceptance_rates[step])
        self.acceptance_rates_avg.append(accept_avg_err[0])
        self.acceptance_rates_err.append(accept_avg_err[1])

        #  tunneling_stats = self.calc_tunneling_rates(trajectories)
        #  avg_tunneling_info = self.calc_tunneling_rates_errors(step)

        print(f"\n\tStep: {step}, "
              f"Tunneling rate avg: {tunn_avg_err[0]:.4g}, "
              f"Tunneling rate err: {tunn_avg_err[1]:.4g}, "
              f"temp: {self.temp:.3g}\n "
              f"\tAverage loss: {np.mean(loss_arr):.4g}, "
              f"Average acceptance: {accept_avg_err[0]:.4g}")

        temp_check = self.temp * self.params['annealing_rate']
        if temp_refresh and temp_check > 1:
            self._temp_refresh(tunn_avg_err)

    def plot_with_errors(self, x_data, y_data, y_errors,
                         x_label, y_label, **kwargs):
        """Method for plotting tunneling rates during training."""
        #  tunneling_info_arr = np.array(self.tunneling_info)
        #  step_nums = tunneling_info_arr[:, 0]
        x = np.array(x_data)
        y = np.array(y_data)
        y_err = np.array(y_errors)
        if not (x.shape == y.shape == y_err.shape):
            err_str = ("x, y, and y_errs all must have the same shape.\n"
                       f" x_data.shape: {x.shape}"
                       f" y_data.shape: {y.shape}"
                       f" y_err.shape:" " {y_err.shape}")
            raise ValueError(err_str)
        #  properties = {}
        if kwargs is not None:
            color = kwargs.get('color', 'C0')
            marker = kwargs.get('marker', '.')
            ls = kwargs.get('ls', '-')
            fillstyle= kwargs.get('fillstyle', 'none')
            #  for key, val in kwargs:
            #      properties[key] = val
        fig, ax = plt.subplots()
        ax.errorbar(x, y, yerr=y_err, capsize=1.5, capthick=1.5,
                    color=color, marker=marker, ls=ls, fillstyle=fillstyle)
        ax.set_ylabel(y_label)#, fontsize=16)
        ax.set_xlabel(x_label)
        str0 = (f"{self.params['num_distributions']}"
                + f" in {self.params['x_dim']} dims; ")
        str1 = (r'$\mathcal{N}_{\hat \mu}(\sqrt{2}\hat \mu;$'
                + r'${{{0}}}),$'.format(self.params['sigma']))
        ax.set_title(str0 + str1, y=1.15)
        fig.tight_layout()
        x_label_ = x_label.replace(' ', '_').lower()
        y_label_ = y_label.replace(' ', '_').lower()
        out_file = (self.figs_dir +
                    f'{y_label_}_vs_{x_label_}_{int(self.steps_arr[-1])}.pdf')
                    #  + f'tunneling_rate_{int(self.steps_arr[-1])}.pdf')
        print(f'Saving figure to: {out_file}\n')
        fig.savefig(out_file, dpi=400, bbox_inches='tight')
        plt.close('all')
        return fig, ax

            #  fig, ax = plt.subplots()
            #  import pdb
            #  pdb.set_trace()
            #  ax.errorbar(self.steps_arr, self.tunneling_rates_avg,
            #              yerr=self.tunneling_rates_err, capsize=1.5,
            #              capthick=1.5, color='C0', marker='.', ls='--',
            #              fillstyle='none')
            #  ax1 = ax.twiny()
            #  ax1.errorbar(self.temp_arr, self.tunneling_rates_avg,
            #              yerr=self.tunneling_rates_err, capsize=1.5,
            #              capthick=1.5, color='C1', marker='.', ls='-', alpha=0.8,
            #              fillstyle='none')
            #  ax1.set_xlabel('Temperature', color='C1')
            #  ax.tick_params('x', colors='C0')
            #  ax1.tick_params('x', colors='C1')
            #  ax.set_xlabel('Training step')#, fontsize=16)
            #ax.legend(loc='best')#, markerscale=1.5), fontsize=12)
            #  str2 = (r' $\mathcal{N}_2(\sqrt{2}\hat y; $'
            #          + r'${{{0}}})$'.format(self.params['sigma']))
            #ax.set_ylim((-0.05, 1.))
        #  except ValueError:
        #      import pdb
        #      pdb.set_trace()

    def _print_time_info(self, t0, t1, step):
        """Print information about time taken to run 100 training steps and
        time taken to calculate tunneling information. (For informational
        purposes only)."""
        tt = time.time()
        tunneling_time = int(tt - t1)
        elapsed_time = int(tt - t0)
        time_per_step100 = 100*int(tt - t0) / step

        t_str2 = time.strftime("%H:%M:%S",
                               time.gmtime(tunneling_time))
        t_str = time.strftime("%H:%M:%S",
                              time.gmtime(elapsed_time))
        t_str3 = time.strftime("%H:%M:%S",
                              time.gmtime(time_per_step100))

        print(f'\n\tTime to calculate tunneling_rate: {t_str2}')
        print(f'\tTime for 100 training steps: {t_str3}')
        print(f'\tTotal time elapsed: {t_str}\n')

    def train(self, num_train_steps, config=None):
        """Train the model."""
        saver = tf.train.Saver(max_to_keep=3)
        initial_step = 0
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(self.log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print('Restoring previous model from: '
                      f'{ckpt.model_checkpoint_path}')
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Model restored.\n')
                self.global_step = tf.train.get_global_step()
                initial_step = sess.run(self.global_step)
                #  self._load_variables()

            writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            t0 = time.time()
            try:
                for step in range(initial_step, initial_step + num_train_steps):
                    feed_dict = {self.x: self.samples,
                                 self.dynamics.temperature: self.temp}

                    _, loss_, self.samples, px_, lr_, = sess.run([
                        self.train_op,
                        self.loss,
                        self.output[0],
                        self.px,
                        self.learning_rate
                    ], feed_dict=feed_dict)

                    self.losses.append(loss_)
                    eps = sess.run(self.dynamics.eps)

                    if step % self.params['logging_steps'] == 0:
                        summary_str = sess.run(self.summary_op,
                                               feed_dict=feed_dict)
                        writer.add_summary(summary_str, global_step=step)
                        writer.flush()

                        print(f"Step: {step}/{initial_step+num_train_steps}, "
                              f"Loss: {loss_:.4g}, "
                              f"accept rate: {np.mean(px_):.2g}, "
                              f"LR: {lr_:.3g}, "
                              f"temp: {self.temp:.5g}, "
                              f"step size: {eps:.3g}\n")

                    if step % self.params['annealing_steps'] == 0:
                        tt = self.temp * self.params['annealing_rate']
                        if tt > 1.:
                            self.temp = tt

                    if (step + 1) % self.params['tunneling_rate_steps'] == 0:
                        t1 = time.time()
                        self._calc_tunneling_info(sess, step, temp_refresh=True)
                        self._print_time_info(t0, t1, step)
                        self.plot_with_errors(x_data=self.steps_arr,
                                              y_data=self.tunneling_rates_avg,
                                              y_errors=self.tunneling_rates_err,
                                              x_label='Training step',
                                              y_label='Tunneling rate')
                        self.plot_with_errors(x_data=self.temp_arr,
                                              y_data=self.tunneling_rates_avg,
                                              y_errors=self.tunneling_rates_err,
                                              x_label='Temperature',
                                              y_label='Tunneling rate',
                                              kwargs={'color': 'C1'})
                        self.plot_with_errors(x_data=self.steps_arr,
                                              y_data=self.acceptance_rates_avg,
                                              y_errors=self.acceptance_rates_err,
                                              x_label='Training step',
                                              y_label='Acceptance rate',
                                              kwargs={'color': 'C2'})
                        self.plot_with_errors(x_data=self.temp_arr,
                                              y_data=self.acceptance_rates_avg,
                                              y_errors=self.acceptance_rates_err,
                                              x_label='Temperature',
                                              y_label='Acceptance rate',
                                              kwargs={'color': 'C2'})
                        #  self.plot_tunneling_rates()

                    if (step + 1) % self.params['save_steps'] == 0:
                        self._save_variables()
                        ckpt_file = os.path.join(self.log_dir, 'model.ckpt')
                        print(f'Saving checkpoint to: {ckpt_file}\n')
                        saver.save(sess, ckpt_file, global_step=step)

                writer.close()

            except (KeyboardInterrupt, SystemExit):
                print("KeyboardInterrupt detected, saving current state and"
                      " exiting. ")
                #  self.plot_tunneling_rates()
                self._save_variables()
                ckpt_file = os.path.join(self.log_dir, 'model.ckpt')
                print(f'Saving checkpoint to: {ckpt_file}\n')
                saver.save(sess, ckpt_file, global_step=step)
                writer.flush()
                writer.close()


def main(args):
    """Main method for running from command-line."""
    X_DIM = args.dimension
    NUM_DISTRIBUTIONS = args.num_distributions

    MEANS = np.zeros((X_DIM, X_DIM), dtype=np.float32)
    CENTERS = np.sqrt(2)  # center of Gaussian
    for i in range(NUM_DISTRIBUTIONS):
        MEANS[i::NUM_DISTRIBUTIONS, i] = CENTERS


    params = {                          # default parameter values
        'x_dim': X_DIM,
        'num_distributions': NUM_DISTRIBUTIONS,
        'lr_init': 1e-3,
        'temp_init': 20,
        'annealing_rate': 0.98,
        'eps': 0.1,
        'scale': 0.1,
        'num_samples': 200,
        'train_trajectory_length': 10,
        'test_trajectory_length': 2000,
        'means': MEANS,
        'sigma': 0.05,
        'small_pi': 2E-16,
        'num_training_steps': 20000,
        'annealing_steps': 200,
        'tunneling_rate_steps': 500,
        'lr_decay_steps': 1000,
        'save_steps': 2500,
        'logging_steps': 100
    }

    params['x_dim'] = args.dimension

    if args.step_size:
        params['eps'] = args.step_size
    if args.temp_init:
        params['temp_init'] = args.temp_init
    if args.num_samples:
        params['num_samples'] = args.num_samples
    if args.train_trajectory_length:
        params['train_trajectory_length'] = args.train_trajectory_length
    if args.test_trajectory_length:
        params['test_trajectory_length'] = args.test_trajectory_length
    if args.num_steps:
        params['num_training_steps'] = args.num_steps
    if args.annealing_steps:
        params['annealing_steps'] = args.annealing_steps
    if args.annealing_rate:
        params['annealing_rate'] = args.annealing_rate
    if args.tunneling_rate_steps:
        params['tunneling_rate_steps'] = args.tunneling_rate_steps

    if args.log_dir:
        model = GaussianMixtureModel(params, log_dir=args.log_dir)
    else:
        model = GaussianMixtureModel(params)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    model.build_graph()
    model.train(params['num_training_steps'], config=config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('L2HMC model using Mixture of Gaussians '
                     'for target distribution')
    )
    parser.add_argument("-d", "--dimension", type=int, required=True,
                        help="Dimensionality of distribution space.")

    parser.add_argument("-N", "--num_distributions", type=int, required=True,
                        help="Number of distributions to include for GMM model.")

    parser.add_argument("-n", "--num_steps", default=10000, type=int,
                        required=True, help="Define the number of training "
                        "steps. (Default: 10000)")

    parser.add_argument("-T", "--temp_init", default=20, type=int,
                        required=False, help="Initial temperature to use for "
                        "annealing. (Default: 20)")

    parser.add_argument("--num_samples", default=200, type=int, required=False,
                        help="Number of samples to use for batched training. "
                        "(Default: 200)")

    parser.add_argument("--step_size", default=0.1, type=float, required=False,
                        help="Initial step size to use in leapfrog update, "
                        "called `eps` in code. (This will be tuned for an "
                        "optimal value during" "training)")

    parser.add_argument("--test_trajectory_length", default=2000, type=int,
                        required=False, help="Trajectory length to be used "
                        "during testing. (Default: 2000)")

    parser.add_argument("--train_trajectory_length", default=10, type=int,
                        required=False, help="Trajectory length to be used "
                        "during training. (Default: 10)")

    parser.add_argument("--annealing_steps", default=100, type=int,
                        required=False, help="Number of annealing steps."
                        "(Default: 100)")

    parser.add_argument("--tunneling_rate_steps", default=500, type=int,
                        required=False, help="Number of steps after which to "
                        "calculate the tunneling rate."
                        "(Default: 500)")

    parser.add_argument("--annealing_rate", default=0.98, type=float,
                        required=False, help="Annealing rate. (Default: 0.98)")

    parser.add_argument("--log_dir", type=str, required=False,
                        help="Define the log dir to use if restoring from"
                        "previous run (Default: None)")

    args = parser.parse_args()

    main(args)
