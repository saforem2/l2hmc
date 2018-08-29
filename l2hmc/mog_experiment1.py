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
from tensorflow.python import debug as tf_debug

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
from utils.plot_helper import errorbar_plot

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
    def __init__(self, params, config, log_dir=None, verbose=False):
        """Initialize parameters and define relevant directories."""
        self.verbose = verbose
        self._init_params(params)
        self._params = params

        if log_dir is not None:
            dirs = check_log_dir(log_dir)
        else:
            dirs = create_log_dir()

        self.log_dir, self.info_dir, self.figs_dir = dirs

        self.files={
            '_params': os.path.join(self.info_dir, '_params.pkl'),
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
        #  self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess,
        #                                                   'localhost:6064')

    def _init_params(self, params):
        """Parse keys in params dictionary to be used for setting instance
        parameters."""
        self.x_dim = None
        self.num_distributions = None
        self.eps = None
        self.scale = None 
        self.num_samples = None
        self.means = None
        self.sigma = None
        self.small_pi = None
        self.lr_init = None
        self.temp_init = None
        self.annealing_steps = None
        self.annealing_factor = None
        self.num_training_steps = None
        self.tunneling_rate_steps = None
        self.lr_decay_steps = None
        self.lr_decay_rate = None
        self.logging_steps = None
        self.save_steps = None
        self.tunneling_rates = {}
        self.acceptance_rates = {}
        self.distances = {}
        self.tunneling_rates_highT = {}
        self.acceptance_rates_highT = {}
        self.distances_highT = {}
        self.temp_arr = []
        self.steps_arr = []
        self.losses_arr = []

        for key, val in params.items():
            setattr(self, key, val)

        self.covs, self.distribution = self._distribution(self.sigma,
                                                          self.means)
        # Initial samples drawn from Normal distribution
        self.samples = np.random.randn(self.num_samples,
                                       self.x_dim)

        self.temp = self.temp_init
        self.step_init = 0

    def _load_variables(self):
        """Load variables from previously ran experiment."""
        print(f'Loading from previous parameters in from: {self.info_dir}')
        for name, file in self.files.items():
            with open(file, 'rb') as f:
                setattr(self, name, pickle.load(f))

        for key, val in self._params.items():
            setattr(self, key, val)

        self.covs = np.load(self.info_dir + 'covs_arr.npy')
        self.temp_arr = list(np.load(self.info_dir + 'temp_arr.npy'))
        self.steps_arr = list(np.load(self.info_dir + 'steps_arr.npy'))
        self.losses_arr = list(np.load(self.info_dir + 'losses_arr.npy'))
        try:
            self.temp = self.temp_arr[-1]
            self.step_init = self.steps_arr[-1]
        except IndexError:
            raise IndexError(f"self.temp_arr.shape: {self.temp_arr.shape}")
            raise IndexError(f"self.steps_arr.shape: {self.steps_arr.shape}")

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
        bad_keys = ['global_step', 'learning_rate', 'dynamics',
                    'x', 'z', 'Lx', 'px', 'Lz', 'pz', 'loss',
                    'output', 'train_op', 'summary_op', 'sess']
        with open(params_txt_file, 'w') as f:
            #  for key, value in self.params.items():
            for key, val in self.__dict__.items():
                b1 = not isinstance(val, (dict, list, np.ndarray, tf.Variable))
                b2 = key not in bad_keys
                if b1 and b2:
                    f.write(f'\n{key}: {val}\n')
            f.write(f"\nmeans:\n\n {str(self.means)}\n"
                    f"\ncovs:\n\n {str(self.covs)}\n")
        print(f'params file written to: {params_txt_file}')

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
        self._create_params_file()

    def generate_trajectories(self, temp=1., num_samples=None, num_steps=None):
        """ Generate trajectories using current values from L2HMC update
        method.  """
        if num_samples is None:
            num_samples = self.num_samples
        if num_steps is None:
            num_steps = 5 * self.trajectory_length
        _samples = self.distribution.get_samples(num_samples)
        _trajectories = []
        _loss_arr = []
        _px_arr = []
        #  for step in range(self.params['trajectory_length']):
        for step in range(num_steps):
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
                 + f'{5 * int(self.trajectory_length):^6.4g}')
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
        str1 = (r'$\mathcal{N}_{\hat \mu}(\sqrt{2}\hat \mu;$'
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
            'plt_stle': 'ggplot'
        }

        #  def out_file(f, s): return self.figs_dir + f'{f}_{s+1}.pdf'
        def out_file(f, s): return self.figs_dir + f'{f}_{s+1}.pdf'

        out_file0 = out_file('tr_ar_dist_steps_lowT', step)
        out_file1 = out_file('tr_ar_dist_steps_highT', step)
        out_file2 = out_file('tr_ar_dist_temps_lowT', step)
        out_file3 = out_file('tr_ar_dist_temps_highT', step)

        errorbar_plot(x_steps, y_data, y_err,
                      out_file=out_file0, **kwargs)

        # for trajectories with temperature > 1 vs. STEP
        kwargs['title'] = title_highT
        errorbar_plot(x_steps, y_data_highT, y_err_highT,
                      out_file=out_file1, **kwargs)

        # for trajectories with temperature = 1. vs TEMP
        kwargs['x_label'] = 'Temperature'
        kwargs['title'] = title
        kwargs['reverse_x'] = True
        errorbar_plot(x_temps, y_data, y_err,
                      out_file=out_file2, **kwargs)

        # for trajectories with temperature > 1. vs TEMP
        kwargs['title'] = title_highT
        errorbar_plot(x_temps, y_data_highT, y_err_highT,
                      out_file=out_file3, **kwargs)
        plt.close('all')

    def _save_variables(self):
        """Save current values of variables."""
        print(f"Saving parameter values to: {self.info_dir}")
        for name, file in self.files.items():
            with open(file, 'wb') as f:
                pickle.dump(getattr(self, name), f)
        np.save(self.info_dir + 'temp_arr.npy', np.array(self.temp_arr))
        np.save(self.info_dir + 'steps_arr.npy', np.array(self.steps_arr))
        np.save(self.info_dir + 'losses_arr.npy', np.array(self.losses_arr))
        np.save(self.info_dir + 'covs_arr.npy', np.array(self.covs))

    def _save_model(self, saver, writer, step):
        """Save tensorflow model with graph and all quantities of interest."""
        self._save_variables()
        ckpt_file = os.path.join(self.log_dir, 'model.ckpt')
        print(f'Saving checkpoint to: {ckpt_file}\n')
        saver.save(self.sess, ckpt_file, global_step=step)
        writer.flush()

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
        t0 = time.time()
        tr0_arr = []
        tr0_err_arr = []
        tr1_arr = []
        tr1_err_arr = []
        try:
            self._print_header()
            for step in range(initial_step, initial_step + num_train_steps):
                t00 = time.time()
                feed_dict = {self.x: self.samples,
                             self.dynamics.temperature: self.temp}

                _, loss_, self.samples, px_, lr_, = self.sess.run([
                    self.train_op,
                    self.loss,
                    self.output[0],
                    self.px,
                    self.learning_rate
                ], feed_dict=feed_dict)

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
                        self._update_trajectory_length(temp_)

                if step % self.logging_steps == 0:
                    #  dash = '-' * 100
                    #  print(dash)
                    summary_str = self.sess.run(self.summary_op,
                                           feed_dict=feed_dict)
                    writer.add_summary(summary_str, global_step=step)
                    writer.flush()
                    last_step = initial_step + num_train_steps

                    col_str = (f'{step:>5g}/{last_step:<7g}'
                               + f'{self.temp:^8.4g}{loss_:^13.4g}'
                               + f'{np.mean(px_):^13.4g}{lr_:^13.4g}'
                               + f'{eps:^13.4g}{self.trajectory_length:^13g}')
                    print(col_str)


                if (step + 1) % self.tunneling_rate_steps == 0:
                    t1 = time.time()
                    self.temp_arr.append(self.temp)
                    self.steps_arr.append(step + 1)

                    ns = self.num_samples
                    #ttl = self.trajectory_length
                    ttl = self.trajectory_length

                    # td0 = trajectory_data
                    # td0[0] = trajectories, of shape [ttl, ns, x_dim]
                    # td0[1] = loss_arr, the loss from each trajectory
                    # td0[3] = acceptance_arr, the acceptance rate from each
                    #          trajectory
                    td0 = self.generate_trajectories(temp=1.,
                                                     num_samples=ns,
                                                     num_steps=ttl)
                    # tr0 = tunneling_rates0
                    # tr0[0] = average tunneling rate from td0
                    # tr0[1] = average tunneling rate error from td0
                    # ar0 = acceptance_rates0
                    # ar0[0] = average acceptance rate from ar0
                    # ar0[1] = average acceptance rate error from ar0
                    tr0, ar0 = self._calc_tunneling_info(td0)
                    tr0_arr.append(tr0[0])
                    tr0_err_arr.append(tr0[1])

                    # ad0 = average distances0
                    # ad0[0] = average distance traveled over all trajectories
                    # ad0[1] = error in average distance 
                    # NOTE: we swap axes 0 and 1 of the trajectories to reshape
                    # them as [num_samples, num_steps, x_dim]
                    ad0 = calc_avg_distances(td0[0].transpose([1, 0, 2]))
                    # td1 = trajectory_data1, elements are the same as td0
                    td1 = self.generate_trajectories(temp=self.temp,
                                                     num_samples=ns,
                                                     num_steps=ttl)
                    # tr1 = tunneling_rates1, same elements as tr0
                    # ar0 = acceptance_rates0, same elements as ar0
                    tr1, ar1 = self._calc_tunneling_info(td1)
                    tr1_arr.append(tr1[0])
                    tr1_err_arr.append(tr1[1])

                    # ad1 = average distances1, same elements as ad0
                    ad1 = calc_avg_distances(td1[0].transpose([1, 0, 2]))

                    self._print_header(test_flag=True)
                    self._print_tunneling_info(step, eps, tr0, ar0,
                                               ad0, td0[1], temp=1.)
                    self._print_tunneling_info(step, eps, tr1, ar1,
                                               ad1, td1[1], temp=self.temp)

                    self.tunneling_rates[(step, 1.)] = tr0
                    self.acceptance_rates[(step, 1.)] = ar0
                    self.distances[(step, 1.)] = ad0

                    temp_key = round(self.temp, 3)
                    self.tunneling_rates_highT[(step, temp_key)] = tr1
                    self.acceptance_rates_highT[(step, temp_key)] = ar1
                    self.distances_highT[(step, temp_key)] = ad1

                    self._print_time_info(t0, t1, step)
                    self._generate_plots(step)

                    if len(tr0_arr) > 1:
                        # tunneling_rate at temp = 1
                        tr0_old = tr0_arr[-2]
                        tr0_old_err = tr0_err_arr[-2]
                        tr0_new = tr0_arr[-1]
                        tr0_new_err = tr0_err_arr[-1]

                        tr1_old = tr1_arr[-2]
                        tr1_old_err = tr1_err_arr[-2]
                        tr1_new = tr1_arr[-1]
                        tr1_new_err = tr1_err_arr[-1]

                        # want the tunneling to either increase or remain
                        # constant (within margin of error)
                        delta_tr0 = ((tr0_new + tr0_new_err)
                                     - (tr0_old - tr0_old_err))
                        delta_tr1 = ((tr1_new + tr1_new_err)
                                     - (tr1_old - tr1_old_err))

                        # if either of the tunneling rates decreased we want
                        # to slow down the annealing schedule. In order to do
                        # this, we can:
                            #  1.) Increase the number of annealing steps
                            #      (divide by the annealing factor) 
                            #  2.) Increase the annealing factor to reduce the
                            #      amount by which the temperature decreases
                            #      with each annealing step (divide the 
                            #      annealing factor itself to bring it closer
                            #      to 1.)
                            #  3.) Reset the temperature to a higher value?? 
                        if (delta_tr0 > 0) or (delta_tr1 > 0):
                            as_old = self.annealing_steps
                            temp_old = self.temp
                            print('\nTunneling rate decreased. Slowing down'
                                  ' annealing schedule.')
                            print(f'Change in tunneling rate (temp = 1):'
                                  f' {delta_tr0}')
                            print(f'Change in tunneling rate (temp ='
                                  f' {self.temp:.3g}): {delta_tr1}')
                            #  print('Old annealing steps:'
                            #        f' {self.annealing_steps}')
                            #  print(f'Old temperature: {self.temp}')
                                  #  ' Old annealing factor: '
                                  #  f' {self.annealing_factor}.\n')
                            self.annealing_steps = int((self.annealing_steps
                                                        / self.annealing_factor)
                                                       / self.annealing_factor)
                            self.temp = ((self.temp / self.annealing_factor)
                                         / self.annealing_factor)
                            #  self.annealing_factor /= self.annealing_factor
                            print(f'Annealing steps: {as_old} -->'
                                  f' {self.annealing_steps}')
                            print(f'Temperature: {temp_old:.3g} -->'
                                  f' {self.temp:.3g}\n')
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

    means = np.zeros((x_dim, x_dim), dtype=np.float32)
    centers = np.sqrt(2)  # center of Gaussian
    for i in range(num_distributions):
        means[i::num_distributions, i] = centers

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
              'logging_steps': 100}

    if args.step_size:
        params['eps'] = args.step_size
    if args.temp_init:
        params['temp_init'] = args.temp_init
    if args.num_samples:
        params['num_samples'] = args.num_samples
    #  if args.trajectory_length:
    #      params['trajectory_length'] = args.trajectory_length
    #  if args.test_trajectory_length:
    #      params['test_trajectory_length'] = args.test_trajectory_length
    if args.num_steps:
        params['num_training_steps'] = args.num_steps
    if args.annealing_steps:
        params['annealing_steps'] = args.annealing_steps
    if args.annealing_factor:
        params['annealing_factor'] = args.annealing_factor
    if args.tunneling_rate_steps:
        params['tunneling_rate_steps'] = args.tunneling_rate_steps
    #  if args.make_plots:
    #      plot = args.make_plots

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

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
    parser.add_argument("-d", "--dimension", type=int, required=True,
                        help="Dimensionality of distribution space.")

    parser.add_argument("-N", "--num_distributions", type=int, required=True,
                        help="Number of target distributions for GMM model.")

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

    #  parser.add_argument("--t_trajectory_length", default=2000, type=int,
    #                      required=False, help="Trajectory length to be used "
    #                      "during testing. (Default: 2000)")

    #  parser.add_argument("--trajectory_length", default=10, type=int,
    #                      required=False, help="Trajectory length to be used "
    #                      "during. (Default: 10)")

    parser.add_argument("--annealing_steps", default=100, type=int,
                        required=False, help="Number of annealing steps."
                        "(Default: 100)")

    parser.add_argument("--tunneling_rate_steps", default=1000, type=int,
                        required=False, help="Number of steps after which to "
                        "calculate the tunneling rate."
                        "(Default: 1000)")

    parser.add_argument("--annealing_factor", default=0.98, type=float,
                        required=False, help="Annealing factor. (Default: 0.98)")

    #  parser.add_argument('--make_plots', default=True, required=False,
    #                      help="Whether or not to create plots during training."
    #                      " (Default: True)")

    parser.add_argument("--log_dir", type=str, required=False,
                        help="Define the log dir to use if restoring from"
                        "previous run (Default: None)")

    args = parser.parse_args()

    main(args)



#  def plot_with_errors(self, x_data, y_data, y_errors,
#                   x_label, y_label, **kwargs):
#  #legend_labels=None, **kwargs):
#  """Method for plotting tunneling rates during training."""
#  #  tunneling_info_arr = np.array(self.tunneling_info)
#  #  step_nums = tunneling_info_arr[:, 0]
#  x = np.array(x_data)
#  y = np.array(y_data)
#  y_err = np.array(y_errors)
#  if not (x.shape == y.shape == y_err.shape):
#      err_str = ("x, y, and y_errs all must have the same shape.\n"
#                 f" x_data.shape: {x.shape}"
#                 f" y_data.shape: {y.shape}"
#                 f" y_err.shape:" " {y_err.shape}")
#      raise ValueError(err_str)
#  #  if legend_labels is not None:
#  #      if len(legend_labels) != x.shape[0]:
#  #          err_str = ("If 'legend_labels' is passed, a label must be"
#  #                     " supplied for each data set in 'x_data'.\n"
#  #                     f" len_legend_labels: {len(legend_labels)}"
#  #                     f" x_data.shape[0]: {x.shape[0]}.\n")
#  #          raise ValueError(err_str)
#  #  properties = {}
#  if kwargs is not None:
#      color = kwargs.get('color', 'C0')
#      marker = kwargs.get('marker', '.')
#      ls = kwargs.get('ls', '-')
#      fillstyle= kwargs.get('fillstyle', 'none')
#      #  for key, val in kwargs:
#      #      properties[key] = val
#  colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
#  markers = ['o', 's', 'v', 'h', 'P']
#  linestyles = ['-', '--', ':', '.-', '-']
#  #  fig, axes = plt.subplots(len(legend_labels), sharey=True)
#  fig, ax = plt.subplots()
#  ax.errorbar(x, y, yerr=y_err, capsize=1.5, capthick=1.5,
#              color=color, marker=marker, ls=ls, fillstyle=fillstyle)
#  ax.set_ylabel(y_label, fontsize=16)
#  ax.set_xlabel(x_label, fontsize=16)
#              #  label=legend_labels)
#  #  for idx, row in enumerate(x):
#  #  for idx, ax in enumerate(axes):
#      #  axes[idx].errorbar(x[idx], y[idx], yerr=y_err[idx],
#      #              capsize=1.5, capthick=1.5,
#      #              color=colors[idx], marker=markers[idx],
#      #              ls=linestyles[idx], fillstyle=fillstyle,
#      #              label=legend_label[idx])
#      #  axes[idx].set_ylabel(y_label[idx])#, fontsize=16)
#      #  axes[idx].set_xlabel(x_label[idx])
#      #  axes[idx].legend(loc='best')
#
#  str0 = (f"{self.params['num_distributions']}"
#          + f" in {self.params['x_dim']} dims; ")
#  str1 = (r'$\mathcal{N}_{\hat \mu}(\sqrt{2}\hat \mu;$'
#          + r'${{{0}}}),$'.format(self.params['sigma']))
#  #  axes[0].set_title(str0 + str1)
#  ax.set_title(str0 + str1)#, y=1.15)
#  fig.tight_layout()
#  x_label_ = x_label.replace(' ', '_').lower()
#  y_label_ = y_label.replace(' ', '_').lower()
#  out_file = (self.figs_dir +
#              f'{y_label_}_vs_{x_label_}_{int(self.steps_arr[-1])}.pdf')
#              #  + f'tunneling_rate_{int(self.steps_arr[-1])}.pdf')
#  print(f'Saving figure to: {out_file}\n')
#  fig.savefig(out_file, dpi=400, bbox_inches='tight')
#  plt.close('all')
#  return fig, ax



#  self.params = {}
#  self.params['x_dim'] = params.get('x_dim', 3)
#  #  number of Gaussian distributions to use for target distribution
#  self.params['num_distributions'] = params.get('num_distributions', 2)
#  self.params['lr_init'] = params.get('lr_init', 1e-3)
#  self.params['lr_decay_steps'] = params.get('lr_decay_steps', 1000)
#  self.params['lr_decay_rate'] = params.get('lr_decay_rate', 0.96)
#  self.params['temp_init'] = params.get('temp_init', 10)
#  self.params['annealing_steps'] = params.get('annealing_steps', 100)
#  self.params['annealing_factor'] = params.get('annealing_factor', 0.98)
#  #  Initial step size (learnable)
#  self.params['eps'] = params.get('eps', 0.1)
#  self.params['scale'] = params.get('scale', 0.1)
#  nts = params.get('num_training_steps', 2e4)
#  self.params['num_training_steps'] = nts
#  #  number of samples (minibatch size) to use for training
#  self.params['num_samples'] = params.get('num_samples', 200)
#  #  Length of trajectory to use during training
#  ttl0 = params.get('train_trajectory_length', 10)
#  self.params['train_trajectory_length'] = ttl0
#  #  Length of trajectory to use when calculating tunneling rate info
#  ttl1 = params.get('test_trajectory_length', 2e3)
#  self.params['test_trajectory_length'] = ttl1
#  #  Standard deviation of Gaussian distributions in target distribution
#  self.params['sigma'] = params.get('sigma', 0.05)
#  #  If num_distributions < x_dim, the likelihood of drawing a sample
#  #  outside of the target distributions.
#  self.params['small_pi'] = params.get('small_pi', 2e-16)
#  self.params['logging_steps'] = params.get('logging_steps', 100)
#  #  How often tunneling rate info should be calculated
#  trs = params.get('tunneling_rate_steps', 500)
#  self.params['tunneling_rate_steps'] = trs
#  self.params['save_steps'] = params.get('save_steps', 2500)
#
#  #  Array containing the position of the Gaussian target distributions
#  self.means = params.get('means', np.eye(self.params['x_dim']))



## TEMP REFRESH CODE
#  if 1 < self.temp < 2:
#      new_tunneling_rate = tunn_avg_err[0]
#      prev_tunneling_rate = 0
#      if len(self.tunneling_rates_avg) > 1:
#          prev_tunneling_rate = self.tunneling_rates_avg[-2]
#
#      tunneling_rate_diff = (new_tunneling_rate
#                             - prev_tunneling_rate
#                             + 2 * tunn_avg_err[1])
#
#      #  if the tunneling rate decreased since the last
#      #  time it was calculated, restart the temperature
#      if tunneling_rate_diff < 0:
#          # the following will revert self.temp to a
#          # value slightly smaller than the value it had
#          # previously the last time the tunneling rate
#          # was calculated
#          print("\n\tTunneling rate decreased!")
#          print("\tNew tunneling rate:"
#                f" {new_tunneling_rate:.3g}, "
#                "Previous tunneling_rate:"
#                f" {prev_tunneling_rate:.3g}, "
#                f"diff: {tunneling_rate_diff:.3g}\n")
#          print("\tResetting temperature...")
#          if len(self.temp_arr) > 1:
#              prev_temp = self.temp_arr[-2]
#              new_temp = (prev_temp *
#                          self.params['annealing_factor'])
#              print(f"\tCurrent temp: {self.temp:.3g}, "
#                    f"\t Previous temp: {prev_temp:.3g}, "
#                    f"\t New temp: {new_temp:.3g}\n")
#              self.temp = new_temp
#              self.temp_arr[-1] = self.temp
