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
#  frm pathlib import Path
#  os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from mpl_toolkits.mplot3d import Axes3D
#  from tensorflow.python import debug as tf_debug

from utils.func_utils import accept, jacobian, autocovariance,\
        get_log_likelihood, binarize, normal_kl, acl_spectrum, ESS
from utils.distributions import GMM
from utils.layers import Linear, Sequential, Zip, Parallel, ScaleTanh
from utils.dynamics import Dynamics
from utils.sampler import propose
from utils.notebook_utils import get_hmc_samples
from utils.tf_logging import variable_summaries, get_run_num, make_run_dir
from utils.trajectories import calc_tunneling_rate, calc_avg_distances
#  from utils.jackknife import block_resampling, jackknife_err
from utils.data_utils import calc_avg_vals_errors, block_resampling,\
        jackknife_err
from utils.plot_helper import errorbar_plot, annealing_schedule_plot

###############################################################################
#  TODO: 
# -----------------------------------------------------------------------------
#  (!!)  * For Lattice model:
#          - Define distance as difference in average plaquette.
#          - Look at site by site difference in plaquette (not sum) to prevent
#            integer values that would be the same across different
#            configurations
#          - Try to get network to be compatible with complex numbers and
#            eventually complex matrices.
# -----------------------------------------------------------------------------
#   (!)  * Implement model with pair of Gaussians both separated along a single
#          axis, and separated diagonally across all dimensions.
#   (~)  * Look at using pathlib to deal with paths.
#------------------------------------------------------------------------------
#        * COMPLETED: 
#            (x)  * Look at replacing self.params['...'] with setattr for
#                   initalization.
#            (x)  * Go back to 2D case and look at different starting
#                   temperatures
#            (x)  * Make trajectory length go with root T, go with higher
#                   temperature
#            (x)  * In 2D start with higher initial temp to get around 50%
#                   acceptance rate.
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
        big_pi = (1.0 / n_distributions) - x_dim * 1E-16
        arr = n_distributions * [big_pi]
        small_pi = (1. - sum(arr)) / (x_dim - n_distributions)
        arr.extend((x_dim - n_distributions) * [small_pi])
        return np.array(arr, dtype=np.float32)

def network(x_dim, scope, factor):
    with tf.variable_scope(scope):
        net = Sequential([
            Zip([
                Linear(x_dim, 20, scope='embed_1', factor=1.0 / 3),
                Linear(x_dim, 20, scope='embed_2', factor=factor * 1.0 / 3),
                Linear(2, 20, scope='embed_3', factor=1.0 / 3),
                lambda _: 0.,
            ]),
            sum,
            tf.nn.relu,
            Linear(20, 20, scope='linear_1'),
            tf.nn.relu,
            Parallel([
                Sequential([
                    Linear(20, x_dim, scope='linear_s', factor=0.001),
                    ScaleTanh(x_dim, scope='scale_s')
                ]),
                Linear(20, x_dim, scope='linear_t', factor=0.001),
                Sequential([
                    Linear(20, x_dim, scope='linear_f', factor=0.001),
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
    def __init__(self, params, config, log_dir=None):
        """Initialize parameters and define relevant directories."""
        self._init_params(params)
        #  self._params = params

        if log_dir is not None:
            dirs = check_log_dir(log_dir)
        else:
            dirs = create_log_dir()

        self.log_dir, self.info_dir, self.figs_dir = dirs

        self.files={
            'distances': os.path.join(self.info_dir, 'distances.pkl'),
            'distances_highT': os.path.join(self.info_dir,
                                            'distances_highT.pkl'),
            'tunneling_rates': os.path.join(self.info_dir,
                                            'tunneling_rates.pkl'),
            'tunneling_rates_highT': os.path.join(self.info_dir,
                                                  'tunneling_rates_highT.pkl'),
            'acceptance_rates': os.path.join(self.info_dir,
                                             'acceptance_rates.pkl'),
            'acceptance_rates_highT': os.path.join(self.info_dir,
                                                   'acceptance_rates_highT.pkl')
        }
        self._save_init_variables()

        if os.path.isfile(os.path.join(self.info_dir, 'parameters.txt')):
            self._load_variables()

        if not self.steps_arr:
            self.step_init = 0
        else:
            self.step_init = self.steps_arr[-1]

        self.trajectory_length = 3 * np.sqrt(self.sigma * self.temp_init)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        tf.add_to_collection('global_step', self.global_step)

        self.learning_rate = tf.train.exponential_decay(
            self.lr_init,
            self.global_step,
            self.lr_decay_steps,
            self.lr_decay_rate,
            staircase=True
        )

        self.build_graph()
        self.sess = tf.Session(config=config)
        print(80*'#')
        print('Model parameters:')
        for key, val in self.__dict__.items():
            if isinstance(val, (int, float, str)):
                print(f'{key}: {val}\n')
        print(80*'#')
        print('\n')


    def _init_params(self, params):
        """Parse keys in params dictionary to be used for setting instance
        parameters."""
        self.x_dim = 2
        self.num_distributions = 2
        self.eps = 0.2
        self.scale = 0.1
        self.num_samples = 200
        self.means = None
        self.sigma = None
        self.small_pi = None
        self.lr_init = 1E-2
        self.temp_init = 20
        self.annealing_steps = 200
        self.annealing_factor = 0.98
        self.num_training_steps = 20000
        self.tunneling_rate_steps = 1000
        self.lr_decay_steps = 1000
        self.lr_decay_rate = 0.96
        self.logging_steps = 100
        self.save_steps = 1000
        self.tunneling_rates = {}
        self.acceptance_rates = {}
        self.distances = {}
        self.tunneling_rates_highT = {}
        self.acceptance_rates_highT = {}
        self.distances_highT = {}
        self.temp_arr = []
        self.steps_arr = []
        self.losses_arr = []
        #  self.annealing_temps_arr = []
        #  self.annealing_steps_arr = []
        self.arrangement = 'axes'

        for key, val in params.items():
            setattr(self, key, val)

        self.covs, self.distribution = self._distribution(self.sigma,
                                                          self.means)
        # Initial samples drawn from Normal distribution
        self.samples = np.random.randn(self.num_samples,
                                       self.x_dim)

        self.temp = self.temp_init
        self.step_init = 0
        self._annealing_steps_init = self.annealing_steps
        self._tunneling_rate_steps_init = self.tunneling_rate_steps

    def _save_init_variables(self):
        """Since certain parameters (annealing steps, etc.) may change as the
        training proceeds, we create an additional file `_init_params.pkl` that
        stores the initial values of all parameters just for safekeeping."""
        _init_params_dict = {}
        for key, val in self.__dict__.items():
            if isinstance(val, (int, float)) or key=='means':
                _init_params_dict[key] = val

        _init_params_file = self.info_dir + '_init_params.pkl'
        with open(_init_params_file, 'wb') as f:
            pickle.dump(_init_params_dict, f)
        print(f'Initial parameters written to {_init_params_file}.')

    def _save_variables(self):
        """Save current values of variables."""
        #  print(f"Saving parameter values to: {self.info_dir}")
        self._create_params_file()
        for name, file in self.files.items():
            with open(file, 'wb') as f:
                pickle.dump(getattr(self, name), f)

        _params_file = self.info_dir + '_params.pkl'
        _params_dict = {}
        for key, val in self.__dict__.items():
            if isinstance(val, (int, float)) or key=='means':
                _params_dict[key] = val

        with open(_params_file, 'wb') as f:
            pickle.dump(_params_dict, f)

        np.save(self.info_dir + 'temp_arr.npy', np.array(self.temp_arr))
        np.save(self.info_dir + 'steps_arr.npy', np.array(self.steps_arr))
        np.save(self.info_dir + 'losses_arr.npy', np.array(self.losses_arr))
        np.save(self.info_dir + 'covs_arr.npy', np.array(self.covs))
        #  np.save(self.info_dir + 'annealing_steps_arr.npy',
        #          np.array(self.annealing_steps_arr))
        #  np.save(self.info_dir + 'annealing_temps_arr.npy',
        #          np.array(self.annealing_temps_arr))

    def _save_model(self, saver, writer, step):
        """Save tensorflow model with graph and all quantities of interest."""
        self._save_variables()
        ckpt_file = os.path.join(self.log_dir, 'model.ckpt')
        #  print(f'Saving checkpoint to: {ckpt_file}\n')
        saver.save(self.sess, ckpt_file, global_step=step)
        writer.flush()

    def _load_variables(self):
        """Load variables from previously ran experiment."""
        print(f'Loading from previous parameters in from: {self.info_dir}')
        for name, file in self.files.items():
            with open(file, 'rb') as f:
                setattr(self, name, pickle.load(f))

        # Since certain parameters (annealing steps, etc.) may change as the
        # training proceeds, we create an additional file `_init_params.pkl`
        # that stores the initial values of all parameters just for
        # safekeeping.
        _params_file = self.info_dir + '_params.pkl'
        with open(_params_file, 'rb') as f:
            _params_dict = pickle.load(f)

        for key, val in _params_dict.items():
            setattr(self, key, val)

        self.covs = np.load(self.info_dir + 'covs_arr.npy')
        self.temp_arr = list(np.load(self.info_dir + 'temp_arr.npy'))
        self.steps_arr = list(np.load(self.info_dir + 'steps_arr.npy'))
        self.losses_arr = list(np.load(self.info_dir + 'losses_arr.npy'))
        #  self.annealing_steps_arr = np.load((self.info_dir
        #                                      + 'annealing_steps_arr.npy'),
        #                                     np.array(self.annealing_steps_arr))
        #  self.annealing_temps_arr = np.load((self.info_dir
        #                                      + 'annealing_temps_arr.npy'),
        #                                     np.array(self.annealing_steps_arr))
        try:
            self.temp = self.temp_arr[-1]
            self.step_init = self.steps_arr[-1]
        except IndexError:
            raise IndexError(f"len(self.temp_arr): {len(self.temp_arr)}")
            raise IndexError(f"len(self.steps_arr): {len(self.steps_arr)}")

    def _distribution(self, sigma, means):
        """Initialize distribution using utils/distributions.py"""
        means = np.array(means).astype(np.float32)
        cov_mtx = sigma * np.eye(self.x_dim).astype(np.float32)
        covs = np.array([cov_mtx] * self.x_dim).astype(np.float32)
        dist_arr = distribution_arr(self.x_dim,
                                    self.num_distributions)
        distribution = GMM(means, covs, dist_arr)
        return covs, distribution

    def _create_dynamics(self, trajectory_length, eps, use_temperature=True):
        """ Create dynamics object using 'utils/dynamics.py'. """
        energy_function = self.distribution.get_energy_function()
        tl = 3 * np.sqrt(self.sigma * self.temp_init)
        self.dynamics = Dynamics(self.x_dim,
                                 energy_function,
                                 tl,
                                 eps,
                                 net_factory=network,
                                 use_temperature=use_temperature)

    def _create_loss(self):
        """ Initialize loss and build recipe for calculating it during
        training. """
        with tf.name_scope('loss'):
            self.x = tf.placeholder(tf.float32, shape=(None,
                                                       self.x_dim),
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
            scale = self.scale

            self.loss = self.loss + scale * (tf.reduce_mean(1.0 / v1)
                                             + tf.reduce_mean(1.0 / v2))
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
            #  variable_summaries(self.loss)
            self.summary_op = tf.summary.merge_all()

    def _create_params_file(self):
        """Write relevant parameters to .txt file for reference."""
        params_txt_file = self.info_dir + 'parameters.txt'
        with open(params_txt_file, 'w') as f:
            #  for key, value in self.params.items():
            for key, val in self.__dict__.items():
                if isinstance(val, (int, float, str)):
                    f.write(f'\n{key}: {val}\n')
            f.write(f"\nmeans:\n\n {str(self.means)}\n"
                    f"\ncovs:\n\n {str(self.covs)}\n")
        #  print(f'params file written to: {params_txt_file}')

    def _update_trajectory_length(self, temp):
        """Update the trajectory length to be roughly equal to half the period
        of evolution. """
        new_trajectory_length = max([2, int(3 * np.sqrt(self.sigma * temp))])
        #  if new_trajectory_length < 2:
        #      new_trajectory_length = 3
        self.trajectory_length = new_trajectory_length
        self.dynamics.T = new_trajectory_length

    def build_graph(self):
        """Build the graph for our model."""
        self._create_dynamics(self.trajectory_length,
                              self.eps,
                              use_temperature=True)
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()
        self._save_variables()

    def generate_trajectories(self, temp=1., num_samples=500, num_steps=100):
        """ Generate trajectories using current values from L2HMC update
        method.  """
        if num_steps is None:
            #  num_steps = int(5 * self.trajectory_length)
            num_steps = int(self.trajectory_length)
        _samples = self.distribution.get_samples(num_samples)
        _trajectories = []
        _loss_arr = []
        _px_arr = []
        #  for step in range(self.params['trajectory_length']):
        for step in range(int(num_steps)):
            _trajectories.append(np.copy(_samples))
            _feed_dict = {self.x: _samples,
                          #  self.dynamics.trajectory_length: num_steps,
                          self.dynamics.temperature: temp}
            _loss, _samples, _px = self.sess.run([
                self.loss,
                self.output[0],
                self.px
            ], feed_dict=_feed_dict)
            _loss_arr.append(np.copy(_loss))
            _px_arr.append(np.copy(_px))
        return np.array(_trajectories), np.array(_loss_arr), np.array(_px_arr)

    def _calc_tunneling_rates(self, trajectories):
        """Calculate tunneling rates from trajectories."""
        tunneling_rates = []
        for i in range(trajectories.shape[1]):
            rate = calc_tunneling_rate(trajectories[:, i, :], self.means,
                                         self.num_distributions)
            tunneling_rates.append(rate)
        return tunneling_rates

    def _calc_tunneling_info(self, trajectory_data):
        """Calculate average tunneling rate and error by generating
        trajectories at a temperature of 1.

        Args:
            trajectory_data (list): 
                trajectory_data[0] = Array of trajectories, 
                    shape = [trajectory_length,     
                             number of unique trajectories, 
                             x_dim]
                trajectory_data[1] = Loss array, the loss from each trajectory.
                trajectory_data[3] = Acceptancea rray, the acceptance rates
                    from each trajectory.
        Returns:
            tunn_avg_err (np.ndarray):
                Array containing the tunneling rate averages and their
                respective errors.
            accept_avg_err (np.ndarray):
                Array containing the acceptance rate averages and their
                respective errors.
        """
        # trajectories are contained in trajectory_data[0]
        tunneling_rates = self._calc_tunneling_rates(trajectory_data[0])
        #  Calculate the average value and error of tunneling rates 
        tunn_avg_err = calc_avg_vals_errors(tunneling_rates, num_blocks=100)
        # not sure if this is needed
        loss_arr = trajectory_data[1]
        # acceptance rates are contained in trajectory_data[2]
        accept_avg_err = calc_avg_vals_errors(trajectory_data[2],
                                              num_blocks=100)

        return tunn_avg_err, accept_avg_err

    def get_tunneling_rates(self, step, temp, num_samples, num_steps,
                            trajectory_data=None):
        """Main method for generating trajectories, and calculating the
        tunneling rates with errors."""
        # trajectory_data[0] = trajectories, of shape [ttl, ns, x_dim]
        # trajectory_data[1] = loss_arr, the loss from each trajectory
        # trajectory_data[3] = acceptance_arr, the acceptance rate from each
        if trajectory_data is None:
            trajectory_data = self.generate_trajectories(temp=temp,
                                                         num_samples=num_samples,
                                                         num_steps=num_steps)
        # tunn_rates[0] = average tunneling rate from td
        # tunn_rates[1] = average tunneling rate error from td
        # accept_rates[0] = average acceptance rate from ar
        # accept_rates[1] = average acceptance rate error from ar
        tunn_rates, accept_rates = self._calc_tunneling_info(trajectory_data)

        # avg_distances[0] = average distance traveled over all trajectories
        # avg_distances[1] = error in average distance 
        # NOTE: we swap axes 0 and 1 of the trajectories to reshape
        # them as [num_samples, num_steps, x_dim]
        trajectories = trajectory_data[0].transpose([1, 0, 2])
        avg_distances = calc_avg_distances(trajectories)

        return trajectory_data, tunn_rates, accept_rates, avg_distances

    def _update_annealing_schedule(self):
        """Update the annealing schedule dynamically based on the changes in
        the tunneling rate.

        If the tunneling rate decreases during training, we want to slow down
        the annealing schedule. (?? Might delete) However if the tunneling rate
        stays the same or increases we can speed the annealing schedule up.
        """
        if len(list(self.tunneling_rates.values())) > 1:
            previous_step = self.steps_arr[-2]
            previous_temp = self.temp_arr[-2]

            current_step = self.steps_arr[-1]
            current_temp = self.temp_arr[-1]

            def get_val(step, temp):
                if temp == 1.0:
                    return self.tunneling_rates[(step, temp)]
                else:
                    return self.tunneling_rates_highT[(step, temp)]

            tr0_old, tr0_old_err = get_val(previous_step, 1.0)
            tr0_new, tr0_new_err = get_val(current_step, 1.0)
            tr1_old, tr1_old_err = get_val(previous_step, previous_temp)
            tr1_new, tr1_new_err = get_val(current_step, current_temp)

            #  tr0_old = tr0_arr[-2]
            #  tr0_old_err = tr0_err_arr[-2]
            #  tr0_new = tr0_arr[-1]
            #  tr0_new_err = tr0_err_arr[-1]

            #  tr1_old = tr1_arr[-2]
            #  tr1_old_err = tr1_err_arr[-2]
            #  tr1_new = tr1_arr[-1]
            #  tr1_new_err = tr1_err_arr[-1]
            #
            # want the tunneling to either increase or remain
            # constant (within margin of error)
            delta_tr0_dec = ((tr0_old - tr0_old_err)
                             - (tr0_new + tr0_new_err))
            delta_tr1_dec = ((tr1_old - tr1_old_err)
                             - (tr1_new + tr1_new_err))

            delta_tr0_inc = ((tr0_new - tr0_new_err)
                             - (tr0_old + tr0_old_err))
            delta_tr1_inc = ((tr1_new - tr1_new_err)
                             - (tr1_old + tr1_old_err))

            # if either of the tunneling rates decreased we want
            # to slow down the annealing schedule. In order to do
            # this, we can:
            #  1.) Increase the number of annealing steps
            #      old way: (divide by the annealing factor) 
            #      new way: (Increase by 50)
            #  2.) Increase the annealing factor to reduce the
            #      amount by which the temperature decreases
            #      with each annealing step (divide the 
            #      annealing factor itself to bring it closer
            #      to 1.)
            #  3.) Reset the temperature to a higher value?? 
            if (delta_tr0_dec > 0) or (delta_tr1_dec > 0):
                #  if not (tr1_old > 0.8 or tr1_new > 0.8):
                if tr1_old < 0.9:
                    as_old = self.annealing_steps
                    temp_old = self.temp
                    print('\nTunneling rate decreased.')
                    print(f'Change in tunneling rate (temp = 1):'
                          f' {delta_tr0_dec}')
                    print(f'Change in tunneling rate (temp ='
                          f' {self.temp:.3g}): {delta_tr1_dec}')
                    self.annealing_steps += 50
                    if self.annealing_steps > self.tunneling_rate_steps:
                        self.tunneling_rate_steps = self.annealing_steps + 100

                    self.temp = self.temp_arr[-2] * self.annealing_factor

                    print('Slowing down annealing schedule and resetting'
                          ' temperature.')
                    print(f'Annealing steps: {as_old} -->'
                          f' {self.annealing_steps}')
                    print(f'Temperature: {temp_old:.3g} -->'
                          f' {self.temp:.3g}\n')

            #  if (delta_tr0_inc > 0) or (delta_tr1_inc > 0):
            #      if not (delta_tr0_dec > 0 or delta_tr1_dec > 0):
            #          # Only speed up the annealing schedule if NEITHER the high
            #          # or low temperature tunneling rates have decreased.
            #          as_old = self.annealing_steps
            #          temp_old = self.temp
            #          print('\nTunneling rate increased. Speeding up'
            #                ' annealing schedule.')
            #          print(f'Change in tunneling rate (temp = 1):'
            #                f' {delta_tr0_inc}')
            #          print(f'Change in tunneling rate (temp ='
            #                f' {self.temp:.3g}): {delta_tr1_inc}')
            #
            #          as_new = self.annealing_steps - 50
            #          self.anneling_steps = max(50, as_new)
            #          #  new_as = (self.annealing_steps
            #          #            * self.annealing_factor)
            #
            #          #  self.annealing_steps = int(new_as)
            #                                     #  / self.annealing_factor)
            #          #self.temp = (self.temp * self.annealing_factor)
            #                       #  / self.annealing_factor)
            #          #  self.annealing_factor /= self.annealing_factor
            #          print(f'Annealing steps: {as_old} -->'
            #                f' {self.annealing_steps}')
            #          print(f'Temperature: {temp_old:.3g} -->'
            #                f' {self.temp:.3g}\n')

        else:
            print("Nothing to compare to!")

    def _print_header(self, test_flag=False):
        if test_flag:
            h_str = ('{:^8s}{:^6s}{:^6s}{:^10s}{:^8s}{:^10s}'
                     + '{:^8s}{:^10s}{:^8s}{:^11s}{:^6s}')
            h_strf = h_str.format("STEP", "TEMP", "LOSS", "ACCEPT %", "ERR",
                                  "TUNN %", "ERR", "DIST", "ERR", "STEP SIZE",
                                  "LENGTH")
            dash0 = (len(h_strf) + 1) * '='
            dash1 = (len(h_strf) + 1) * '-'
            print(dash0)
            #  print(h_str)
            print(h_strf)
            print(dash0)
            #  print(dash1)
        else:
            h_str = '{:^13s}{:^8s}{:^13s}{:^13s}{:^13s}{:^13s}{:^13s}'
            h_strf = h_str.format("STEP", "TEMP", "LOSS", "ACCEPT RATE",
                                  "LR", "STEP SIZE", "TRAJ LEN")
            dash = (len(h_strf) + 1) * '-'
            print(dash)
            #  print(h_str)
            print(h_strf)
            print(dash)

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

        print(f'\nTime to calculate tunneling_rate: {t_str2}')
        print(f'Time for 100 training steps: {t_str3}')
        print(f'Total time elapsed: {t_str}\n')

    def _print_tunneling_info(self, step, eps, tr_info, ar_info,
                              dist_info, losses, temp):
        """Print information about quantities of interest calculated from
        sample trajectories.

        Args:
            tr_info (array-like):
                Tunneling rate info, avg_val, error = tr_info[0], tr_info[1].
            ar_info (array-like):
                Acceptance rate info, same as above.
            dist_info (array-like):
                Average (Euclidean) distance traversed over all
                trajectories, same as above.
            losses (array-like):
                Loss values calculated from trajectories used to determine the
                tunneling rate info
            temp (float):
                Temperature

        """
        i_str = (f'{self.steps_arr[-1]:^8g}'
                 + f'{temp:^6.3g}'
                 + f'{np.mean(losses):^6.4g}'
                 + f'{ar_info[0]:^10.4g}'
                 + f'{ar_info[1]:^8.4g}'
                 + f'{tr_info[0]:^10.4g}'
                 + f'{tr_info[1]:^8.4g}'
                 + f'{dist_info[0]:^10.4g}'
                 + f'{dist_info[1]:^8.4g}'
                 + f'{eps:^11.4g}'
                 + f'{int(self.trajectory_length):^6.4g}')
        print(i_str)
        dash1 = (len(i_str) + 1) * '-'
        print(dash1)

    def _generate_plots(self, step):
        """ Plot tunneling rate, acceptance_rate vs. training step for both
        sets of trajectories. 

         Variables with the suffix _highT correspond to trajectories calculated
         during training at temperatures > 1.
        
         Variables without the suffix _highT correspond to trajectories
         calculated during training at temperatures = 1.

         Args:
             step (int):
                 Used as suffix for filename.

        Returns:
            list:
                List consisting of (fig, ax) pairs for each of the plots.
         """

        x_steps = [self.steps_arr, self.steps_arr, self.steps_arr]
        x_temps = [self.temp_arr, self.temp_arr, self.temp_arr]

        def get_vals_as_arr(_dict): return np.array(list(_dict.values()))

        tr_arr = get_vals_as_arr(self.tunneling_rates)
        ar_arr = get_vals_as_arr(self.acceptance_rates)
        dist_arr = get_vals_as_arr(self.distances)

        tr_arr_highT = get_vals_as_arr(self.tunneling_rates_highT)
        ar_arr_highT = get_vals_as_arr(self.acceptance_rates_highT)
        dist_arr_highT = get_vals_as_arr(self.distances_highT)

        y_data = [tr_arr[:, 0], ar_arr[:, 0], dist_arr[:, 0]]
        y_err = [tr_arr[:, 1], tr_arr[:, 1], tr_arr[:, 1]]

        y_data_highT = [tr_arr_highT[:, 0],
                        ar_arr_highT[:, 0],
                        dist_arr_highT[:, 0]]

        y_err_highT = [tr_arr_highT[:, 1],
                       ar_arr_highT[:, 1],
                       dist_arr_highT[:, 1]]

        str0 = (f"{self.num_distributions}"
                + f" in {self.x_dim} dims; ")
        str1 = ''
        if self.arrangement == 'axes':
            str1 = (r'$\mathcal{N}_{\hat \mu}(1\hat \mu;$'
                    + r'${{{0}}}),$'.format(self.sigma))
        if self.arrangement == 'single_axis':
            str1 = (r'$\mathcal{N}_{\hat \mu}(\pm 1\hat \mu;$'
                    + r'${{{0}}}),$'.format(self.sigma))
        if self.arrangement == 'diagonal':
            str1 = (r'$\mathcal{N}_{\hat \mu}(\pm R(\pi/4) 1 \hat \mu;$'
                    + r'${{{0}}}),$'.format(self.sigma))
        #  prefix = str0 + str1
        title = str0 + str1 + r'$T_{trajectory} = 1$'
        title_highT = str0 + str1 + r'$T_{trajectory} > 1$'
        kwargs = {
            'x_label': 'Training step',
            'y_label': '',
            'legend_labels': ['Tunneling rate',
                              'Acceptance rate',
                              'Distance / step'],
            'title': title,
            'grid': True,
            'reverse_x': False,
            'plt_stle': '~/.config/matplotlib/stylelib/ggplot_sam.mplstyle'
        }

        #  def out_file(f, s): return self.figs_dir + f'{f}_{s+1}.pdf'
        def out_file(f): return self.figs_dir + f'{f}.pdf'

        out_file0 = out_file('tr_ar_dist_steps_lowT')#, step)
        out_file1 = out_file('tr_ar_dist_steps_highT')#, step)
        out_file2 = out_file('tr_ar_dist_temps_lowT')#, step)
        out_file3 = out_file('tr_ar_dist_temps_highT')#, step)

        def add_vline(axes, x, **kwargs):
            if len(axes) > 1:
                for ax in axes:
                    ax.axvline(x=x, **kwargs)
            else:
                ax.axvline(x=x, **kwargs)
            return ax

        fig0, axes0 = errorbar_plot(x_steps, y_data, y_err,
                                    out_file=out_file0, **kwargs)

        line_args = {'color': 'C3', 'ls': ':', 'lw': 2.}
        #  axes0 = add_vline(axes0, 1, **line_args)
        #  fig0.savefig(out_file0, dpi=400, bbox_inches='tight')
        #  for ax in axes:
        #      ax.axvline(x=1, color='C3', ls=':', lw=2.)

        # for trajectories with temperature > 1 vs. STEP
        kwargs['title'] = title_highT
        fig1, axes1 = errorbar_plot(x_steps, y_data_highT, y_err_highT,
                                    out_file=out_file1, **kwargs)
        #  axes1 = add_vline(axes1, 1, **line_args)
        #  fig1.savefig(out_file1, dpi=400, bbox_inches='tight')
        #  for ax in axes:
        #      ax.axvline(x=1, color='C3', ls=':', lw=2.)

        # for trajectories with temperature = 1. vs TEMP
        kwargs['x_label'] = 'Temperature'
        kwargs['title'] = title
        kwargs['reverse_x'] = True
        fig2, axes2 = errorbar_plot(x_temps, y_data, y_err,
                                    out_file=out_file2, **kwargs)
        axes2 = add_vline(axes2, 1, **line_args)
        fig2.savefig(out_file2, dpi=400, bbox_inches='tight')

        # for trajectories with temperature > 1. vs TEMP
        kwargs['title'] = title_highT
        fig3, axes3 = errorbar_plot(x_temps, y_data_highT, y_err_highT,
                                    out_file=out_file3, **kwargs)
        axes3 = add_vline(axes3, 1, **line_args)
        fig3.savefig(out_file3, dpi=400, bbox_inches='tight')

        fig4, ax4 = annealing_schedule_plot(**self.__dict__)

        plt.close('all')

    def train(self, num_train_steps):
        """Train the model."""
        saver = tf.train.Saver(max_to_keep=3)
        initial_step = 0
        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(self.log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restoring previous model from: '
                  f'{ckpt.model_checkpoint_path}')
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('Model restored.\n')
            self.global_step = tf.train.get_global_step()
            initial_step = self.sess.run(self.global_step)
        writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        _samples = self.distribution.get_samples(self.num_samples)
        t0 = time.time()
        try:
            self._print_header()
            for step in range(initial_step, initial_step + num_train_steps):
                t00 = time.time()

                feed_dict = {self.x: _samples,
                             self.dynamics.temperature: self.temp}

                #  _, loss_, self.samples, px_, lr_, = self.sess.run([
                _, loss_, _samples, px_, lr_, = self.sess.run([
                    self.train_op,
                    self.loss,
                    self.output[0],
                    self.px,
                    self.learning_rate
                ], feed_dict=feed_dict)


                #  ns = self.num_samples
                #  ttl = self.trajectory_length
                #  _tdata = [_samples, loss_, px_]
                #  _ret_vals = self.get_tunneling_rates(step, self.temp, ns, ttl,
                #                                       trajectory_data=_tdata)
                #  _td, _tr, _ar, _ad = _ret_vals

                #  self.losses.append(loss_)
                self.losses_arr.append(loss_)
                eps = self.sess.run(self.dynamics.eps)

                if (step + 1) % self.save_steps == 0:
                    self._print_header()
                    self._save_model(saver, writer, step)

                if step % self.annealing_steps == 0:
                    temp_ = self.temp * self.annealing_factor
                    if temp_ > 1.:
                        self.temp = temp_
                        #  self.annealing_steps_arr.append(step + 1)
                        #  self.annealing_temps_arr.append(self.temp)
                        self._update_trajectory_length(temp_)

                if step % self.logging_steps == 0:
                    summary_str = self.sess.run(self.summary_op,
                                           feed_dict=feed_dict)
                    writer.add_summary(summary_str, global_step=step)
                    writer.flush()
                    last_step = initial_step + num_train_steps

                    col_str = (f'{step:>5g}/{last_step:<7g}'
                               + f'{self.temp:^8.4g}'
                               + f'{loss_:^13.4g}'
                               + f'{np.mean(px_):^13.4g}'
                               + f'{lr_:^13.4g}'
                               + f'{eps:^13.4g}'
                               + f'{self.trajectory_length:^13g}')

                    print(col_str)

                if (step + 1) % self.tunneling_rate_steps == 0:
                    t1 = time.time()
                    self.temp_arr.append(self.temp)
                    self.steps_arr.append(step + 1)

                    ns = self.num_samples
                    ttl = self.trajectory_length

                    td0, tr0, ar0, ad0 = self.get_tunneling_rates(step, 1.,
                                                                  ns, ttl)
                    self.tunneling_rates[(step+1, 1.)] = tr0
                    self.acceptance_rates[(step+1, 1.)] = ar0
                    self.distances[(step+1, 1.)] = ad0

                    td1, tr1, ar1, ad1 = self.get_tunneling_rates(step,
                                                                  self.temp,
                                                                  ns, ttl)

                    self.tunneling_rates_highT[(step+1, self.temp)] = tr1
                    self.acceptance_rates_highT[(step+1, self.temp)] = ar1
                    self.distances_highT[(step+1, self.temp)] = ad1

                    self._print_header(test_flag=True)
                    self._print_tunneling_info(step, eps, tr0, ar0,
                                               ad0, td0[1], temp=1.)
                    self._print_tunneling_info(step, eps, tr1, ar1,
                                               ad1, td1[1], temp=self.temp)

                    self._print_time_info(t0, t1, step)
                    self._generate_plots(step)

                    self._update_annealing_schedule()
                    self._print_header()

            writer.close()
            self.sess.close()

        except (KeyboardInterrupt, SystemExit):
            print("\nKeyboardInterrupt detected! \n"
                  + "Saving current state and exiting.\n")
            #  self.plot_tunneling_rates()
            self._save_variables()
            self._save_model(saver, writer, step)
            writer.close()
            self.sess.close()


def main(args):
    """Main method for running from command-line."""
    x_dim = args.dimension
    num_distributions = args.num_distributions

    # Create array containing the location of the mean for each Gaussian
    means = np.zeros((x_dim, x_dim), dtype=np.float32)

    #---------------------------------------------------------------------------
    # GMM Model with Gaussians centered along each axis:
    #---------------------------------------------------------------------------
    # If x_dim = num_distributions then we put one along each eaxis as shown
    # below, where X indicates the position of the center of the Gaussian.
    #    [X, 0, 0, ..., 0, 0, 0]
    #    [0, X, 0, ..., 0, 0, 0]
    #    [0, 0, X, ..., 0, 0, 0]
    #    [          \          ]    =    diag(x) 
    #    [0, 0, 0, ..., X, 0, 0]
    #    [0, 0, 0, ..., 0, X, 0]
    #    [0, 0, 0, ..., 0, 0, X]
    #---------------------------------------------------------------------------
    # NOTE: If num_distributions < x_dim we append copies of the first
    # num_distributions gaussians to the remaining x_dim rows. 
    # For example, if num_distributions = 2 and x_dim = 4, with two gaussians
    # located at (1, 0, 0, 0) and (0, 1, 0, 0)
    # 
    #    means = [1, 0, 0, 0]
    #            [0, 1, 0, 0]
    #            [1, 0, 0, 0]
    #            [0, 1, 0, 0]
    #---------------------------------------------------------------------------
    centers = 1 # center of Gaussian
    for i in range(num_distributions):
        means[i::num_distributions, i] = centers
    #  params['arrangement'] = 'axes'
    arrangement = 'axes'


    ###########################################################################
    # TODO: 
    #--------------------------------------------------------------------------
    #   1.) Try experiment using pair of Gaussians both separated along a
    #       single axis, and
    #   2.) With Gaussians separated diagonally across all dimensions.
    ###########################################################################


    #--------------------------------------------------------------------------
    # Pair of Gaussians both separated along a single axis:
    #--------------------------------------------------------------------------
    # Separated Gaussians along a single axis chosen randomly.
    # For example, if the randomly chosen axis was 1 (i.e. vertical axis),
    # the `means` array would consist of two Gaussians and would look like (for
    # each of the Gaussians centered at 1):
    # 
    #    [0, +1, 0, ..., 0, 0, 0]
    #    [0, -1, 0, ..., 0, 0, 0]
    #    [0, +1, 0, ..., 0, 0, 0]
    #    [0, -1, 0, ..., 0, 0, 0]
    #    [          ...         ]
    #    [0, +1, 0, ..., 0, 0, 0]
    #    [0, -1, 0, ..., 0, 0, 0]
    #--------------------------------------------------------------------------
    if args.single_axis:
        means = np.zeros((x_dim, x_dim))
        rand_axis = np.random.randint(x_dim)
        centers = 1
        #  centers = np.sqrt(2)

        means[::2, rand_axis] = centers
        means[1::2, rand_axis] = - centers
        #  params['arrangement'] = 'single_axis'
        arrangement = 'single_axis'

    if args.diagonal:
        means = np.zeros((x_dim, x_dim))
        rand_axis = np.random.randint(x_dim)
        centers = 1
        #  centers = np.sqrt(2)

        means[::2, :] = centers
        means[1::2, :] = - centers
        arrangement = 'diagonal'
        #  params['arrangement'] = 'diagonal'

        #  means[even_rows, rand_axis] = np.cos(np.pi/4) * centers
        #  means[even_rows, rand_axis-1] = np.sin(np.pi/4) * centers
        #  means[odd_rows, rand_axis] = -1 * np.cos(np.pi/4) * centers
        #  means[odd_rows, rand_axis-1] = -1 * np.sin(np.pi/4) * centers


    #  rand_axis = np.random.randint(num_distributions)
    #  means[0, rand_axis] = centers1
    #  means[1, rand_axis] = - centers1
    #  for i in range(num_distributions):
    #      means[]

    params = {'x_dim': args.dimension,
              'num_distributions': num_distributions,
              'eps': 0.1,
              'scale': 0.1,
              'num_samples': 200,
              'means': means,
              'sigma': 0.05,
              'small_pi': 2E-16,
              'lr_init': 1e-2,
              'temp_init': 20,
              'annealing_steps': 200,
              'annealing_factor': 0.98,
              'num_training_steps': 20000,
              'tunneling_rate_steps': 1000,
              'save_steps': 1000,
              'lr_decay_steps': 2500,
              'lr_decay_rate': 0.96,
              'logging_steps': 100,
              'arrangement': arrangement}

    if args.step_size:
        params['eps'] = args.step_size
        #  print(f"eps: {params['eps']}")
    if args.temp_init:
        params['temp_init'] = args.temp_init
        #  print(f'temp_init: {args.temp_init}')
    if args.num_samples:
        params['num_samples'] = args.num_samples

    if args.scale:
        params['scale'] = args.scale
        #  print(f'num_samples: {args.num_samples}')
    #  if args.trajectory_length:
    #      params['trajectory_length'] = args.trajectory_length
    #  if args.test_trajectory_length:
    #      params['test_trajectory_length'] = args.test_trajectory_length
    if args.num_steps:
        params['num_training_steps'] = int(args.num_steps)
        #  print(f'num_training_steps: {args.num_training_steps}')
    if args.annealing_steps:
        params['annealing_steps'] = int(args.annealing_steps)
        #  print(f'annealing_steps: {args.annealing_steps}')
    if args.annealing_factor:
        params['annealing_factor'] = args.annealing_factor
        #  print(f'annealing_factor: {args.annealing_factor}')
    if args.tunneling_rate_steps:
        params['tunneling_rate_steps'] = int(args.tunneling_rate_steps)
        params['save_steps'] = int(args.tunneling_rate_steps)

    if args.save_steps:
        params['save_steps'] = int(args.save_steps)
        #  print(f'tunneling_rate_steps: {args.tunneling_rate_steps}')
    #  if args.make_plots:
    #      plot = args.make_plots
    #  for arg in args:
    #      print(f"{arg}")

    config = tf.ConfigProto()
    #  config.gpu_options.allow_growth = True

    t0 = time.time()
    if args.log_dir:
        model = GaussianMixtureModel(params,
                                     config=config,
                                     log_dir=args.log_dir)
    else:
        model = GaussianMixtureModel(params, config=config)

    t1 = time.time()
    print(f'Time to build and populate graph: {t1 - t0:.4g}s\n')

    model.train(params['num_training_steps'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('L2HMC model using Mixture of Gaussians '
                     'for target distribution')
    )
    parser.add_argument("-d", "--dimension",
                        type=int, required=True,
                        help="Dimensionality of distribution space.")

    parser.add_argument("-N", "--num_distributions",
                        type=int, required=True,
                        help="Number of target distributions for GMM model.")

    parser.add_argument("-n", "--num_steps",
                        default=10000, type=int, required=True,
                        help="Define the number of training "
                        "steps. (Default: 10000)")

    parser.add_argument("-T", "--temp_init",
                        default=20, type=int, required=False,
                        help="Initial temperature to use for "
                        "annealing. (Default: 20)")

    parser.add_argument("--num_samples",
                        default=200, type=int, required=False,
                        help="Number of samples to use for batched training. "
                        "(Default: 200)")

    parser.add_argument("--scale",
                        default=0.1, type=float, required=False,
                        help="Multiplicative factor for scaling hyperbolic"
                        " tangent activation function in neural network."
                        " (Default: 0.1)")

    parser.add_argument("--step_size",
                        default=0.1, type=float, required=False,
                        help="Initial step size to use in leapfrog update, "
                        "called `eps` in code. (This will be tuned for an "
                        "optimal value during" "training)")

    #  parser.add_argument("--t_trajectory_length", default=2000, type=int,
    #                      required=False, help="Trajectory length to be used "
    #                      "during testing. (Default: 2000)")

    #  parser.add_argument("--trajectory_length", default=10, type=int,
    #                      required=False, help="Trajectory length to be used "
    #                      "during. (Default: 10)")

    parser.add_argument("--annealing_steps",
                        default=100, type=int, required=False,
                        help="Number of annealing steps."
                        "(Default: 100)")

    parser.add_argument("--annealing_factor",
                        default=0.98, type=float, required=False,
                        help="Annealing factor by which the temperature is"
                        " multiplied each time we hit the annealing step.")

    #  parser.add_argument("--annealing_factor",
    #                      default=0.98, type=float, required=False,
    #                      help="Annealing factor. (Default: 0.98)")

    parser.add_argument("--tunneling_rate_steps",
                        default=1000, type=int, required=False,
                        help="Number of steps after which to "
                        "calculate the tunneling rate."
                        "(Default: 1000)")

    parser.add_argument("--save_steps",
                        default=1000, type=int, required=False,
                        help="Number of steps after which to "
                        "save the model and all parameters."
                        "(Default: 1000)")

    parser.add_argument("--single_axis",
                        default=False, type=bool, required=False,
                        help="Specifies alternate arrangement where the GMM"
                        " distribution consists of a single pair of Gaussians"
                        " equidistant from the origin separated along a" 
                        " randomly chosen axis.")

    parser.add_argument("--diagonal",
                        default=False, type=bool, required=False,
                        help="Specifies alternate arrangement where the GMM"
                        " distribution consists of a single pair of Gaussians"
                        " equidistant from the origin separated along a"
                        " randomly chosen axis, rotated by 45 deg..")

    #  parser.add_argument('--make_plots', default=True, required=False,
    #                      help="Whether or not to create plots during training."
    #                      " (Default: True)")

    parser.add_argument("--log_dir",
                        type=str, required=False,
                        help="Define the log dir to use if restoring from"
                        "previous run (Default: None)")

    args = parser.parse_args()

    main(args)
