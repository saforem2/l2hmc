"""
Module for running either generic HMC or L2HMC algorithms for gauge models
defined on the lattice.

Capable of parsing command line options specifying lattice attributes and
hyperparameters for training L2HMC.

Author: Sam Foreman
"""
# pylint: disable=wildcard-import, no-member, too-many-arguments, invalid-name
import os
import sys
import time
import pickle
import numpy as np
import tensorflow as tf
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from tensorflow.python.client import timeline
#  from tensorflow.profiler import Profiler

from scipy.special import i0, i1  # pylint: disable=no-name-in-module

from l2hmc_eager import gauge_dynamics_eager as gauge_dynamics_eager
from l2hmc_eager.gauge_dynamics_eager import GaugeDynamicsEager
from l2hmc_eager.neural_nets import ConvNet, GenericNet

from utils import gauge_dynamics as gauge_dynamics
from utils.gauge_dynamics import GaugeDynamics
from utils.jacobian import _map, jacobian
from utils.tf_logging import variable_summaries, get_run_num, make_run_dir

from definitions import ROOT_DIR

from HMC.hmc import HMC

from lattice.gauge_lattice import GaugeLattice, pbc, mat_adj, u1_plaq_exact

MODULE_PATH = os.path.abspath(os.path.join('..'))
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

tfe = tf.contrib.eager

PARAMS = {
    'time_size': 8,
    'space_size': 8,
    'link_type': 'U1',
    'dim': 2,
    'beta': 3.5,
    'num_samples': 4,
    'n_steps': 5,
    'eps': 0.2,
    'loss_scale': 0.1,
    'loss_eps': 1e-4,
    'learning_rate_init': 1e-4,
    'learning_rate_decay_steps': 100,
    'learning_rate_decay_rate': 0.96,
    'train_iters': 500,
    'record_loss_every': 50,
    'data_steps': 10,
    'save_steps': 50,
    'print_steps': 1,
    'logging_steps': 50,
    'clip_value': 100,
    'rand': False,
    'conv_net': True,
    'metric': 'l2'
}


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


def create_log_dir():
    """Create directory for storing information about experiment."""
    root_log_dir = os.path.join(os.path.split(ROOT_DIR)[0], 'gauge_logs')
    log_dir = make_run_dir(root_log_dir)
    info_dir = log_dir + 'run_info/'
    figs_dir = log_dir + 'figures/'
    if not os.path.isdir(info_dir):
        os.makedirs(info_dir)
    if not os.path.isdir(figs_dir):
        os.makedirs(figs_dir)
    return log_dir, info_dir, figs_dir


def exact_plaquette(beta):
    """Exact value of the (expectation value) of average plaquette."""
    return i1(beta) / i0(beta)


def eager_step(dynamics, optimizer, samples, **kwargs):
    """Single iteration step of training procedure.

    To be defunnable, the function cannot return an Operation, so the
    `eager_step` method is used for defun or eager.
    """
    loss, grads, samples, accept_prob = gauge_dynamics_eager.loss_and_grads(
        dynamics,
        samples,
        loss_fn=gauge_dynamics_eager.compute_loss,
        scale=kwargs.get('scale', 0.1),
        eps=kwargs.get('eps', 1e-4),
        metric=kwargs.get('metric', 'l2')
    )

    gradients, _ = tf.clip_by_global_norm(grads, kwargs.get('clip_value', 10))

    optimizer.apply_gradients(
        zip(grads, dynamics.trainable_variables)
    )
    return loss, samples, accept_prob, gradients


def graph_step(dynamics, optimizer, samples, **kwargs):
    """Single iteration step of training procedure.

    This function is used in the graph to be able to run the gradient updates,
    when not using eager or defun.
    """
    #  x, dynamics, optimizer, scale=0.1, eps=1e-4, metric='l2', clip_value=10,
    #  global_step=None, loss_fn=gauge_dynamics_eager.compute_loss):

    loss, grads, samples, accept_prob = gauge_dynamics_eager.loss_and_grads(
        dynamics,
        samples,
        loss_fn=gauge_dynamics_eager.compute_loss,
        scale=kwargs.get('scale', 0.1),
        eps=kwargs.get('eps', 1e-4),
        metric=kwargs.get('metric', 'l2')
    )

    gradients, _ = tf.clip_by_global_norm(grads, kwargs.get('clip_value', 10))

    train_op = optimizer.apply_gradients(
        zip(grads, dynamics.trainable_variables)
    )
    return train_op, loss, samples, accept_prob, gradients


def _variable_summaries(var, name):
    """Attach various summaries to a Tensor(for Tensorboard visualization.)"""
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# pylint: ignore  too-many-instance-attributes
class GaugeModel(object):
    """Class for implementing (L2)HMC for a pure gauge model on the lattice."""
    def __init__(self, 
                 config, 
                 params=None,
                 log_dir=None, 
                 restore=False, 
                 eager=False):
        """
        Initialize GaugeModel object.

       Args:
           params (dict): Parameters for model.
           config: Tensorflow config object.
           log_dir (str): String containing directory to use for saving run
               information. If log_dir is None, one will be created.
           restore: Flag for restoring model from previous training
               session.
           eager: Flag specifying whether or not to use tensorflow's eager
               execution.
        """
        _t0 = time.time()
        if params is None:
            params = PARAMS

        self.data = {}

        self._init_params(params)

        if log_dir:
            dirs = check_log_dir(log_dir)
        else:
            dirs = create_log_dir()

        self.log_dir, self.info_dir, self.figs_dir = dirs

        self.files = {
            'parameters_file': os.path.join(self.info_dir, 'parameters.txt'),
            'samples_file': os.path.join(self.info_dir, 'samples.npy'),
            'run_info_file': os.path.join(self.info_dir, 'run_info.txt'),
            'average_plaquettes_file': (
                os.path.join(self.info_dir, 'average_plaquettes.npy')
            ),
            'total_actions_file': (
                os.path.join(self.info_dir, 'total_actions.npy')
            ),
            'topological_charges_file': (
                os.path.join(self.info_dir, 'topological_charges.npy')
            ),
        }

        #  self.summary_writer = tf.contrib.summary.create_file_writer(log_dir)

        print('Creating lattice...')
        self.lattice = self._create_lattice(time_size=self.time_size,
                                            space_size=self.space_size,
                                            dim=self.dim,
                                            beta=self.beta,
                                            link_type=self.link_type,
                                            num_samples=self.num_samples,
                                            rand=self.rand)

        self.samples = tf.convert_to_tensor(self.lattice.samples,
                                            dtype=tf.float32)

        self.potential_fn = self.lattice.get_energy_function(self.samples)
        print('done.')

        print("Creating dynamics...")
        self.dynamics = self._create_dynamics(lattice=self.lattice,
                                              n_steps=self.n_steps,
                                              eps=self.eps,
                                              potential_fn=self.potential_fn,
                                              conv_net=self.conv_net,
                                              eager=eager)
        print('done.')

        self.global_step = tf.train.get_or_create_global_step()
        self.global_step.assign(1)

        #  tf.add_to_collection('global_step', self.global_step)

        self.learning_rate = tf.train.exponential_decay(
            self.learning_rate_init,
            self.global_step,
            self.learning_rate_decay_steps,
            self.learning_rate_decay_rate,
            staircase=True
        )

        print("Building graph...")
        self.build_graph()

        self.sess = tf.Session(config=config)

        if restore:
            self._restore_model()

        print(f"total initialization time: {time.time() - _t0}\n")

    def _init_params(self, params=None):
        """
        Parse key value pairs from params and set as as class attributes.

        Args:
            params: Dictionary containing model parameters.

        Returns:
            None
        """
        self.train_times = {}
        self.total_actions_arr = []
        self.average_plaquettes_arr = []
        self.topological_charges_arr = []
        self.losses_arr = []
        self.steps_arr = []
        self.samples_arr = []
        self.accept_prob_arr = []
        self.data = {
            'step': 0,
            'loss': 0.,
            'time': 0.,
            'accept_prob': 0.,
            'eps': params.get('eps', 0.),
            'avg_action': 0.,
            'avg_top_charge': 0.,
            'avg_plaq': 0.,
            'avg_plaquettes': None,
            'top_charges': None,
            'total_actions': None,
            'average_plaquettes_arr': np.array(
                self.average_plaquettes_arr
            ),
            'topological_charges_arr': np.array(
                self.topological_charges_arr
            ),
            'total_actions_arr': np.array(self.total_actions_arr),
            'samples': None
        }

        if params is None:
            params = PARAMS

        self.params = params

        for key, val in params.items():
            setattr(self, key, val)

    def _update_data(self, total_actions, avg_plaquettes, top_charges):
        """Update lattice observables stored in `self.data` object."""
        self.data['avg_action'] = np.mean(total_actions)
        self.data['avg_top_charge'] = np.mean(top_charges)
        self.data['avg_plaq'] = np.mean(avg_plaquettes)

        self.data['total_actions'] = total_actions
        self.data['top_charges'] = top_charges
        self.data['avg_plaquettes'] = avg_plaquettes

        self.total_actions_arr.append(total_actions)
        self.average_plaquettes_arr.append(avg_plaquettes)
        self.topological_charges_arr.append(top_charges)

        self.data['average_plaquettes_arr'] = np.array(
            self.average_plaquettes_arr
        )
        self.data['topological_charges_arr'] = np.array(
            self.topological_charges_arr
        )
        self.data['total_actions_arr'] = np.array(self.total_actions_arr)

    @staticmethod
    def _create_lattice(time_size, space_size, dim, 
                        beta, link_type, num_samples, rand):
        """Create lattice object with gauge fields of `link_type`.
            Args:
                time_size: Temporal extent of lattice.
                space_size: Spatial extent of lattice.
                dim: Dimensionality of lattice.
                beta: Inverse coupling strength.
                link_type: String defining type of link variables, 
                    one of (U1, SU2, SU3)
                num_samples: Number of lattices to use for training (batch size).
            Returns:
                lattice object
        """
        return GaugeLattice(time_size, space_size, dim, beta, link_type,
                            num_samples, rand)

    @staticmethod
    def _create_dynamics(lattice, n_steps, eps, potential_fn, conv_net, eager):
        """Create dynamics object that implements theL2HMC algorithm.
            Args:
                lattice: Lattice object.
                num_lf_steps: Number of leapfrog steps to be used.
                eps: Leapfrog step size.
                potential_fn: Minus log-likelihood function (potential energy /
                    action) function describing the target distribution.
            Returns:
                dynamics object
        """
        if eager:
            print("Using GaugeDynamicsEager...")
            return GaugeDynamicsEager(
                lattice,
                n_steps=n_steps,
                minus_loglikelihood_fn=potential_fn,
                eps=eps,
                conv_net=conv_net,
                test_HMC=False
            )
        print("Using GaugeDynamics...")
        return GaugeDynamics(
            lattice,
            n_steps=n_steps,
            eps=eps,
            minus_loglikelihood_fn=potential_fn,
            hmc=False,
            conv_net=conv_net,
            eps_trainable=True
        )

    @staticmethod
    def _create_checkpointer(optimizer, dynamics, global_step):
        """Create checkpointer object."""
        return tf.train.Checkpoint(optimizer=optimizer,
                                   dynamics=dynamics,
                                   global_step=global_step)

    def _create_loss(self):
        """Compute loss defined in Eq. (8) of paper."""
        with tf.name_scope('loss'):
            #  def _create_loss(self, dynamics, samples, scale=0.1, eps=1e-4):
            num_samples = self.lattice.samples.shape[0]
            flat_samples = self.lattice.samples.reshape((num_samples, -1))
            self.x = tf.placeholder(shape=flat_samples.shape,
                                    dtype=tf.float32,
                                    name='x')
            #  self.x = tf.placeholder(shape=self.samples.shape,
            #                          dtype=tf.float32,
            #                          name='x')
            # Auxiliary variable
            z = tf.random_normal(tf.shape(self.x),
                                 dtype=tf.float32,
                                 name='z')

            x_transition = self.dynamics.apply_transition(self.x)
            _x, _, self.x_accept_prob, x_out = x_transition

            _z, _, z_accept_prob, _ = self.dynamics.apply_transition(z)

            _x = tf.mod(_x, 2 * np.pi)
            _z = tf.mod(_z, 2 * np.pi)
            x_out = tf.mod(x_out, 2 * np.pi)

            self.x_out = []
            self.x_out.append(x_out)

            #  if self.x.shape != _x.shape:
            #      self.x = tf.reshape(self.x, shape=_x.shape)
            #  if z.shape != _z.shape:
            #      z = tf.reshape(z, shape=_z.shape)

            # Add eps for numerical stability; following released
            # implementation
            if self.metric == 'cos':
                x_loss = (tf.reduce_sum(
                    (tf.math.cos(self.x) - tf.math.cos(_x))**2, axis=1
                ) * self.x_accept_prob + self.loss_eps)

                z_loss = (tf.reduce_sum(
                    (tf.math.cos(z) - tf.math.cos(_z))**2, axis=1
                ) * z_accept_prob + self.loss_eps)

            else:
                x_loss = (tf.reduce_sum((self.x - _x)**2, axis=1)
                          * self.x_accept_prob + self.loss_eps)

                z_loss = (tf.reduce_sum((z - _z)**2, axis=1)
                          * z_accept_prob + self.loss_eps)

            self.loss_op = tf.Variable(0., trainable=False, dtype=tf.float32,
                                       name='loss')
            self.loss_op = self.loss_op + (
                tf.reduce_mean((1. / x_loss + 1. / z_loss) * self.loss_scale
                               - (x_loss + z_loss) / self.loss_scale, axis=0)
            )

    def _create_optimizer(self):
        """Initialize optimizer to be used during training."""
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate
            )
            grads, vars = zip(*optimizer.compute_gradients(self.loss_op))
            grads, _ = tf.clip_by_global_norm(grads, self.clip_value)
            gradients = zip(grads, vars)
            self.train_op = optimizer.apply_gradients(
                gradients,
                global_step=self.global_step,
                name='train_op'
            )

    def _calc_observables(self, samples):
        """Calculate lattice observables from samples."""
        #  feed_dict = {self.samples_ph: samples}

        observables = self.lattice.calc_plaq_observables(samples)

        observables_arr = np.array(self.sess.run(observables)).reshape(-1, 3)

        total_actions = observables_arr[:, 0]
        avg_plaquettes = observables_arr[:, 1]
        top_charges = observables_arr[:, 2]

        self.total_actions_arr.append(total_actions)
        self.average_plaquettes_arr.append(avg_plaquettes)
        self.topological_charges_arr.append(top_charges)

        self._update_data(total_actions, avg_plaquettes, top_charges)

        return total_actions, avg_plaquettes, top_charges

    def _create_summaries(self, data):
        """Create summary objects for logging in tensorboard."""
        #  with tf.name_scope('summaries'):
        tot_actions_tensor = tf.convert_to_tensor(self.total_actions_arr,
                                                  dtype=tf.float32)
        avg_plaqs_tensor = tf.convert_to_tensor(self.average_plaquettes_arr,
                                                dtype=tf.float32)
        top_charges_tensor = tf.convert_to_tensor(self.topological_charges_arr,
                                                  dtype=tf.float32)

        _variable_summaries(tot_actions_tensor, name='total_actions')
        _variable_summaries(avg_plaqs_tensor, name='average_plaquettes')
        _variable_summaries(top_charges_tensor, name='topological_charge')

        tf.summary.scalar('loss', self.loss_op)
        tf.summary.scalar("Training_loss", data['loss'])
        tf.summary.scalar("avg_action", data['avg_action'])
        tf.summary.scalar("avg_plaquettes", data['avg_plaq'])
        tf.summary.scalar("avg_top_charge", data['avg_top_charge'])
        # pylint: disable=attribute-defined-outside-init
        self.summary_op = tf.summary.merge_all()

    def _create_params_file(self, parameters=None):
        """Create txt file for storing all current values of parameters."""
        params_file = self.files['parameters_file']

        if parameters is not None:
            with open(params_file, 'w') as f:
                f.write("Parameters:\n")
                f.write(80 * '-' + '\n')
                for key, val in parameters.items():
                    f.write(f'{key}: {val}\n')
                f.write(80 * '=')
                f.write('\n')
                f.write(80 * '=')
                f.write('\n')

        with open(params_file, 'a') as f:
            for key, val in self.__dict__.items():
                if isinstance(val, (int, float, str)):
                    f.write(f'\n{key}: {val}\n')

    def _save_variables(self):
        """Save current values of variables."""
        self._create_params_file()

        #  for name, file in self.files.items():
        #      with open(file, 'wb') as f:
        #          pickle.dump(getattr(self, name), f)

        #  _params_dict = {}
        #  for key, val in self.__dict__.items():
        #      if isinstance(val, (int, float)) or key == 'means':
        #          _params_dict[key] = val

        #  _params_file = self.info_dir + '_params.pkl'
        #  _params_file = os.path.join(self.info_dir, 'params.pkl')
        #  dict_file = os.path.join(self.info_dir, '__dict__.pkl')
        #  with open(dict_file, 'wb') as f:
        #      pickle.dump(self.__dict__, f)

        _data_file = os.path.join(self.info_dir, 'data.pkl')
        with open(_data_file, 'wb') as f:
            pickle.dump(self.data, f)

        np.save(self.info_dir + 'steps_arr.npy', np.array(self.steps_arr))
        np.save(self.info_dir + 'losses_arr.npy', np.array(self.losses_arr))

        np.save(self.files['average_plaquettes_file'],
                self.data['average_plaquettes_arr'])
        np.save(self.files['total_actions_file'],
                self.data['total_actions_arr'])
        np.save(self.files['topological_charges_file'],
                self.data['topological_charges_arr'])
        np.save(self.files['samples_file'],
                self.data['samples'])

    def _save_model(self, step_num):
        """Save model and all values of current parameters."""
        self._save_variables()
        ckpt_file = os.path.join(self.log_dir, 'model.ckpt')
        self.saver.save(self.sess, ckpt_file, global_step=step_num)
        self.writer.flush()

    def _restore_model(self, sess=None, log_dir=None):
        """Restore model from `log_dir` using tensorflow session `sess`."""
        if log_dir is None:
            log_dir = self.log_dir
        if sess is None:
            sess = self.sess

        saver = tf.train.Saver(max_to_keep=3)
        #  if not self.eager:
        #      sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring previous model from: "
                  f"{ckpt.model_checkpoint_path}")
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored.\n")
            self.global_step = tf.train.get_global_step()

            self.sess.run(tf.global_variables_initializer())

    @staticmethod
    def write_run_data(file_path, data, write_mode='a'):
        """Write information about current run data to txt file."""
        info_str = (f"\nstep: {data['step']:<3g} "
                    f"loss: {data['loss']:^6.5g} "
                    f" time/step: {data['time']:^6.5g} "
                    f" accept: {np.mean(data['accept_prob']):^6.5g} "
                    f" eps: {data['eps']:^6.5g} "
                    f" avg_action: {data['avg_action']:^6.5g} "
                    f" avg_top_charge: {data['avg_top_charge']:^6.5g} "
                    f" avg_plaq: {data['avg_plaq']:^6.5g} \n")
        with open(file_path, write_mode) as f:
            f.write('\n')
            f.write(info_str)
            f.write('\n')
            f.write('avg_plaquettes: {}\n'.format(data['avg_plaquettes']))
            f.write('topological_charges: {}\n'.format(data['top_charges']))
            f.write('total_actions: {}\n'.format(data['total_actions']))
            f.write((len(info_str) + 1)*'-')

    @staticmethod
    def write_run_parameters(file_path, parameters):
        """Write run parameters to file_path."""
        with open(file_path, 'w') as f:
            f.write('Parameters:\n')
            f.write(80 * '-' + '\n')
            for key, val in parameters.items():
                f.write(f'{key}: {val}\n')
            f.write(80 * '=')
            f.write('\n')

    @staticmethod
    def print_run_data(data):
        """Print information about current run to std out."""
        print(f"\nstep: {data['step']:<3g} loss: {data['loss']:^6.5g} "
              f" time/step: {data['time']:^6.5g} "
              f" accept: {np.mean(data['accept_prob']):^6.5g} "
              f" eps: {data['eps']:^6.5g} "
              f" avg_action: {data['avg_action']:^6.5g} "
              f" avg_top_charge: {data['avg_top_charge']:^6.5g} "
              f" avg_plaq: {data['avg_plaq']:^6.5g} \n")
        print('avg_plaquettes: {}\n'.format(data['avg_plaquettes']))

    def warmup(self, dynamics, optimizer, n_iters=1):
        """Warmup optimization to reduce overhead.
            Args:
                dynamics: Dynamics object.
                optimizer: tensorflow optimizer object.
                n_iters: Number of warmup iterations to perform.
        """
        samples = tf.random_normal(shape=self.samples.shape, dtype=tf.float32)

        for _ in range(n_iters):
            _, samples = self._train_step(dynamics,
                                          optimizer,
                                          samples)

    def build_graph(self):
        """Build the graph for our model."""
        print("Creating loss...\n")
        self._create_loss()
        print('done.')
        print("Creating optimizer...\n")
        self._create_optimizer()
        print('done.')
        print("Creating summaries...\n")
        self._create_summaries(self.data)
        #  print('done.')
        #  print("Saving variables...\n")
        #  self._save_variables()

    def train(self, num_train_steps, keep_samples=False):
        """Train the model."""
        self.saver = tf.train.Saver(max_to_keep=3)
        initial_step = 0
        #  if not self.eager:
        #      self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.global_variables_initializer())

        #  run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #  run_metadata = tf.RunMetadata()

        ckpt = tf.train.get_checkpoint_state(self.log_dir)
        time_delay = 0.
        if ckpt and ckpt.model_checkpoint_path:
            print('Restoring previous model from: '
                  f'{ckpt.model_checkpoint_path}')
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('Model restored.\n')
            self.global_step = tf.train.get_global_step()
            initial_step = self.sess.run(self.global_step)
            previous_time = self.train_times[initial_step]
            time_delay = time.time() - previous_time

        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        total_actions, avg_plaquettes, top_charges = (
            self._calc_observables(self.samples)
        )

        self._update_data(total_actions, avg_plaquettes, top_charges)

        self.print_run_data(self.data)
        self.write_run_data(self.files['run_info_file'], self.data, 'w')
        self.write_run_parameters(self.files['parameters_file'], self.params)

        start_time = time.time()
        self.train_times[initial_step] = start_time - time_delay

        # Move attribute look ups outside loop to improve performance
        loss_op = self.loss_op
        train_op = self.train_op
        x_out = self.x_out
        x_accept_prob = self.x_accept_prob
        learning_rate = self.learning_rate
        dynamics = self.dynamics

        num_samples = self.lattice.samples.shape[0]
        _samples = self.lattice.samples.reshape(num_samples, -1)
        #  _samples = self.lattice.samples

        #  tf.get_default_graph().finalize()
        #  profiler = Profiler(self.sess.graph)

        try:
            for step in range(initial_step, initial_step + num_train_steps):
                t1 = time.time()

                feed_dict = {self.x: _samples}

                out = self.sess.run([train_op,
                                     loss_op,
                                     x_out[0],
                                     x_accept_prob,
                                     learning_rate],
                                    feed_dict=feed_dict)

                _, _loss, _samples, _accept_prob, _lr = out

                if step % self.print_steps == 0:
                    total_actions, avg_plaquettes, top_charges = (
                        self._calc_observables(_samples)
                    )

                    self._update_data(total_actions,
                                      avg_plaquettes,
                                      top_charges)

                    self.data['step'] = step
                    self.data['time'] = time.time() - t1
                    self.data['eps'] = self.sess.run(dynamics.eps)
                    self.data['loss'] = _loss
                    self.data['accept_prob'] = np.mean(_accept_prob)
                    self.data['samples'] = _samples

                    self.losses_arr.append(_loss)
                    self.accept_prob_arr.append(_accept_prob)

                    if keep_samples:
                        self.samples_arr.append(_samples)

                    self.print_run_data(self.data)
                    self.write_run_data(file_path=self.files['run_info_file'],
                                        data=self.data,
                                        write_mode='a')

                if step % self.logging_steps == 0:
                    try:
                        summary_str = (
                            self.sess.run(self.summary_op,
                                          feed_dict={self.x: _samples})
                        )
                        self.writer.add_summary(summary_str, global_step=step)
                        self.writer.flush()

                    except:
                        import pdb
                        pdb.set_trace()

                if step % self.save_steps == 0:
                    self._save_model(step)

            self.writer.close()
            self.sess.close()

        except (KeyboardInterrupt, SystemExit):
            print("\nKeyboardInterrupt detected! \n"
                  "Saving current state and exiting.\n")
            #  self._save_variables()
            self._save_model(step)
            self.writer.close()
            self.sess.close()

#########################################################
#  PROFILING USING `Timeline` OBJECT WITH JSON FILES
# ------------------------------------------------------
# Create the Timeline object, and write it to a json file
#  fetched_timeline = (
#      timeline.Timeline(run_metadata.step_stats)
#  )
#
#  chrome_trace = (
#      fetched_timeline.generate_chrome_trace_format()
#  )
#
#  timeline_json_file = os.path.join(
#      self.log_dir,
#      'timeline_step_%d.json' % step
#  )
#  with open(timeline_json_file, 'w') as f:
#      f.write(chrome_trace)
##########################################################

#########################################################
#  PROFILING USING `tf.profiler` API  (WIP)
# ------------------------------------------------------
#  profiler.add_step(step, run_metadata)
#
#  # Profile the parameters of the model
#  profiler.profile_name_scope(
#      options=option_builder.ProfileOptionBuilder
#      .trainable_variables_parameter())
#########################################################
