"""
Augmented Hamiltonian Monte Carlo Sampler using the L2HMC algorithm, applied
to a U(1) lattice gauge theory model.

==============================================================================
* TODO:
-----------------------------------------------------------------------------
    * Look at thermalization times for L2HMC vs generic HMC.
    * Find out how large of a lattice is feasible for running on local laptop.

==============================================================================
* COMPLETED:
-----------------------------------------------------------------------------
==============================================================================
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# pylint: disable=no-member, too-many-arguments, invalid-name
import os
import sys
import time
import pickle
import argparse

import numpy as np
import tensorflow as tf

try:
    import horovod.tensorflow as hvd

    HAS_HOROVOD = True
    hvd.init()

except ImportError:
    HAS_HOROVOD = False

import utils.gauge_model_helpers as helpers

from lattice.lattice import GaugeLattice
from utils.tf_logging import variable_summaries

from dynamics.gauge_dynamics import GaugeDynamics

tfe = tf.contrib.eager
tf.logging.set_verbosity(tf.logging.INFO)

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
tf.set_random_seed(GLOBAL_SEED)


#  MODULE_PATH = os.path.abspath(os.path.join('..'))
#  if MODULE_PATH not in sys.path:
#      sys.path.append(MODULE_PATH)

PARAMS = {
###################### Lattice parameters ############################
    'time_size': 12,
    'space_size': 12,
    'link_type': 'U1',
    'dim': 2,
    'num_samples': 6,
    'rand': False,
    'data_format': 'channels_last',
###################### Leapfrog parameters ###########################
    'num_steps': 1,
    'eps': 0.2,
    'loss_scale': .1,
###################### Learning rate parameters ######################
    'learning_rate_init': 1e-3,
    'learning_rate_decay_steps': 500,
    'learning_rate_decay_rate': 0.96,
###################### Annealing rate parameters #####################
    'annealing': True,
    'annealing_steps': 200,
    'annealing_factor': 0.97,
    #  'beta': 2.,
    'beta_init': 2.,
    'beta_final': 6.,
###################### Training parameters ###########################
    'train_steps': 20000,
    'save_steps': 1000,
    'logging_steps': 50,
    'print_steps': 1,
    'training_samples_steps': 1000,
    'training_samples_length': 100,
###################### Model parameters ##############################
    #  'conv_net': True,
    'network_arch': 'conv3D',
    'hmc': False,
    'eps_trainable': True,
    'metric': 'l2',
    'aux': True,
    'clip_grads': False,
    'clip_value': 10.,
}


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


def check_else_make_dir(d):
    """If directory `d` doesn't exist, it is created."""
    if not os.path.isdir(d):
        log(f"Creating directory: {d}")
        #  print(f"Creating directory: {d}.")
        os.makedirs(d)


def save_params_to_pkl_file(params, out_dir):
    """Save `params` dictionary to `parameters.pkl` in `out_dir.`"""
    check_else_make_dir(out_dir)
    params_file = os.path.join(out_dir, 'parameters.pkl')
    #  print(f"Saving params to: {params_file}.")
    log(f"Saving params to: {params_file}.")
    with open(params_file, 'wb') as f:
        pickle.dump(params, f)


def tf_accept(x, _x, px):
    """Helper function for determining if x is accepted given _x."""
    mask = (px - tf.random_uniform(tf.shape(px)) > 0.)
    return tf.where(mask, _x, x)


###############################################################################
#            Perform a single update step using graph execution
###############################################################################
# pylint: disable=too-many-locals
def compute_loss(dynamics, x, beta, metric_fn, scale=.1, eps=1e-4):
    """Compute loss defined in equation (8)."""
    log("    Creating loss...")
    t0 = time.time()

    with tf.name_scope('compute_loss'):
        with tf.name_scope('DynamicsTransition'):
            with tf.name_scope('x_transition'):
                inputs = (x, beta)
                x_, _, px, x_out = dynamics(inputs)
                #  x_, _, px, x_out = dynamics.apply_transition(x, beta)
            with tf.name_scope('z_transition'):
                z = tf.random_normal(tf.shape(x))  # Auxiliary variable
                inputs = (z, beta)
                z_, _, pz, _ = dynamics(inputs)
                #  z_, _, pz, _ = dynamics.apply_transition(z, beta)

        # Add eps for numerical stability; following released impl
        with tf.name_scope('loss'):
            with tf.variable_scope('x_loss'):
                x_loss = tf.reduce_sum(metric_fn(x, x_),
                                       axis=dynamics.axes) * px + eps

            with tf.variable_scope('z_loss'):
                z_loss = tf.reduce_sum(metric_fn(z, z_),
                                       axis=dynamics.axes) * pz + eps

            loss = tf.reduce_mean((1. / x_loss + 1. / z_loss) * scale
                                  - (x_loss + z_loss) / scale, axis=0)

        t_diff = time.time() - t0
        log(f"    done. took: {t_diff:4.3g} s.")

    return loss, x_out, px


def loss_and_grads(dynamics, x, beta, metric_fn, **kwargs):
    """Obtain loss value and gradients."""
    loss_fn = kwargs.get('loss_fn', compute_loss)
    clip_value = kwargs.get('clip_value', None)

    log(f"  Creating gradient operations...")
    t0 = time.time()

    if tf.executing_eagerly():
        with tf.name_scope('grads'):
            with tf.GradientTape() as tape:
                loss, x_out, accept_prob = loss_fn(dynamics, x,
                                                   beta, metric_fn)
            grads = tape.gradient(loss, dynamics.trainable_variables)
    else:
        loss, x_out, accept_prob = loss_fn(dynamics, x, beta, metric_fn)
        with tf.name_scope('grads'):
            grads = tf.gradients(loss, dynamics.trainable_variables)
            if clip_value is not None:
                grads, _ = tf.clip_by_global_norm(grads, clip_value)

    t_diff = time.time() - t0
    log(f"  done. took: {t_diff:4.3g} s")

    return loss, grads, x_out, accept_prob


def graph_step(dynamics, optimizer, samples, beta, step, metric_fn, **kwargs):
    """Perform a single training update step using graph execution."""
    with tf.name_scope('train'):
        output = loss_and_grads(dynamics, samples, beta, metric_fn, **kwargs)

        loss, grads, samples, accept_prob = output

        with tf.name_scope('apply_grads'):
            grads_and_vars = zip(grads, dynamics.trainable_variables)
            train_op = optimizer.apply_gradients(grads_and_vars,
                                                 global_step=step,
                                                 name='train_op')

    return train_op, loss, grads, samples, accept_prob


# pylint: disable=attribute-defined-outside-init
class GaugeModel(object):
    """Wrapper class implementing L2HMC algorithm on lattice gauge models."""
    def __init__(self,
                 params=None,
                 sess=None,
                 config=None,
                 log_dir=None,
                 restore=False):
        """Initialization method."""
        #  super(GaugeModel, self).__init__()
        np.random.seed(GLOBAL_SEED)
        tf.set_random_seed(GLOBAL_SEED)
        tf.enable_resource_variables()


        # create attributes using key, value pairs in params
        self._create_params(params)
        self._create_dir_structure(log_dir)
        self._write_run_parameters(_print=True)
        self._create_lattice()
        self._create_tensors()
        self._create_dynamics()

        if restore:
            self._restore_model(log_dir, sess, config)
        else:
            if not tf.executing_eagerly():
                self.build_graph(sess, config)

    def _create_params(self, params):
        """Parse key value pairs from params and set as class attributes."""
        if params is None:
            print('Using default parameters...')
            for key, val in params.items():
                print(f'{key}: {val}')
            params = PARAMS

        self.params = params

        for key, val in params.items():
            setattr(self, key, val)

        if not self.clip_grads:
            self.clip_value = None

        self.losses_arr = []

        self.data = {
            'step': 1,
            'loss': 0.,
            'step_time': 0.,
            'accept_prob': 0.,
            'samples': [],
            'eps': params.get('eps', 0.),
            'beta_init': params.get('beta_init', 2.),
            'beta': params.get('beta_init', 2.),
            'train_steps': params.get('train_steps', 20000),
            'learning_rate': params.get('learning_rate_init', 1e-3),
        }

        self.condition1 = not self.using_hvd  # condition1: NOT using horovod
        self.condition2 = False               # condition2: Initially False
        # If we're using horovod, we have     --------------------------------
        # to make sure all file IO is done    --------------------------------
        # only from rank 0                    --------------------------------
        if self.using_hvd:                    # If we are using horovod:
            if hvd.rank() == 0:               # AND rank == 0:
                self.condition2 = True        # condition2: True


    def _create_dir_structure(self, log_dir):
        """Create self.files and directory structure."""
        if self.using_hvd:
            if hvd.rank() != 0:
                return

        if self.condition1 or self.condition2:  # DEFINED IN: _create_params
            if log_dir is None:
                dirs = helpers.create_log_dir('gauge_logs_graph')
            else:
                dirs = helpers.check_log_dir(log_dir)

            self.log_dir, self.info_dir, self.figs_dir = dirs

        self.files = {
            'parameters_file': os.path.join(self.info_dir, 'parameters.txt'),
            'run_info_file': os.path.join(self.info_dir, 'run_info.txt'),
            'data_pkl_file': os.path.join(self.info_dir, 'data.pkl'),
            'samples_pkl_file': os.path.join(self.info_dir, 'samples.pkl'),
            'parameters_pkl_file': os.path.join(self.info_dir,
                                                'parameters.pkl'),
        }

        #  self.samples_history_dir = os.path.join(self.log_dir,
        #  'samples_history')
        #  self.train_samples_dir = os.path.join(self.log_dir, 'train_samples')
        #  self.train_samples_history_dir = os.path.join(
        #      self.samples_history_dir, 'training'
        #  )

        self.eval_dir = os.path.join(self.log_dir, 'eval_info')
        self.samples_dir = os.path.join(self.eval_dir, 'samples')

        self.train_eval_dir = os.path.join(self.eval_dir, 'training')
        self.train_samples_dir = os.path.join(self.train_eval_dir, 'samples')

        def make_dirs(dirs):
            [check_else_make_dir(d) for d in dirs]

        make_dirs([self.eval_dir, self.samples_dir,
                   self.train_eval_dir, self.train_samples_dir])

    def _create_lattice(self):
        with tf.name_scope('lattice'):
            self.lattice = GaugeLattice(time_size=self.time_size,
                                        space_size=self.space_size,
                                        dim=self.dim,
                                        link_type=self.link_type,
                                        num_samples=self.num_samples,
                                        rand=self.rand)

    def _create_dynamics(self, kwargs=None):
        """Initialize dynamics object."""
        if kwargs is None:
            kwargs = {
                'eps': self.eps,
                'hmc': self.hmc,
                'network_arch': self.network_arch,
                'num_steps': self.num_steps,
                'eps_trainable': self.eps_trainable,
                'data_format': self.data_format

            }
        if self.hmc:
            kwargs['network_arch'] = None
            kwargs['data_format'] = None

        with tf.name_scope('potential_fn'):
            self.potential_fn = self.lattice.get_energy_function(self.samples)

        with tf.name_scope('dynamics'):
            self.dynamics = GaugeDynamics(lattice=self.lattice,
                                          potential_fn=self.potential_fn,
                                          **kwargs)

    def reshape_5D(self, tensor):
        """
        Reshape tensor to be compatible with tf.keras.layers.Conv3D.

        If self.data_format is 'channels_first', and input `tensor` has shape
        (N, 2, L, L), the output tensor has shape (N, 1, 2, L, L).

        If self.data_format is 'channels_last' and input `tensor` has shape 
        (N, L, L, 2), the output tensor has shape (N, 2, L, L, 1).
        """
        if self.data_format == 'channels_first':
            N, D, H, W = tensor.shape
            if isinstance(tensor, np.ndarray):
                return np.reshape(tensor, (N, 1, H, W, D))
                #  return np.expand_dims(tensor, axis=1)
            #  return tf.expand_dims(tensor, 1)
            return tf.reshape(tensor, (N, 1, D, H, W))

        elif self.data_format == 'channels_last':
            N, H, W, D = tensor.shape
            if isinstance(tensor, np.ndarray):
                #  return np.expand_dims(tensor, axis=-1)
                return np.reshape(tensor, (N, H, W, D, 1))
            #  return tf.expand_dims(tensor, -1)
            return tf.reshape(tensor, (N, H, W, D, 1))

        else:
            raise AttributeError("`self.data_format` should be one of "
                                 "'channels_first' or 'channels_last'")

    def _create_tensors(self):
        """Initialize tensors (and placeholders if executing in graph mode)."""
        self.batch_size = self.lattice.samples.shape[0]
        self.samples = tf.convert_to_tensor(self.lattice.samples,
                                            dtype=tf.float32)

        #  self.samples = self.reshape_5D(self.samples)

        if not tf.executing_eagerly():
            self.x = tf.placeholder(tf.float32, self.samples.shape, name='x')
            self.beta = tf.placeholder(tf.float32, shape=(), name='beta')
        else:
            self.beta = self.beta_init

    def _restore_checkpoint(self, log_dir):
        """Restore from `tf.train.Checkpoint`."""
        if self.using_hvd:
            if hvd.rank() != 0:
                return

        #  latest_path = tf.train.latest_checkpoint(log_dir)
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            dynamics=self.dynamics,
            global_step=self.global_step,
        )

    def _restore_model(self, log_dir, sess, config):
        """Restore model from previous run contained in `log_dir`."""
        if self.using_hvd:
            if hvd.rank() != 0:
                return

        if self.hmc:
            log(f"ERROR: self.hmc: {self.hmc}. No model to restore. Exiting.")
            sys.exit(1)

        assert os.path.isdir(log_dir), (f"log_dir: {log_dir} does not exist.")

        run_info_dir = os.path.join(log_dir, 'run_info')
        assert os.path.isdir(run_info_dir), (f"run_info_dir: {run_info_dir}"
                                             " does not exist.")

        with open(self.files['parameters_pkl_file'], 'rb') as f:
            self.params = pickle.load(f)

        with open(self.files['data_pkl_file'], 'rb') as f:
            self.data = pickle.load(f)

        with open(self.files['samples_pkl_file'], 'rb') as f:
            self.samples = pickle.load(f)

        self._create_params(self.params)
        self.global_step.assign(self.data['step'])
        self.learning_rate.assign(self.data['learning_rate'])

        kwargs = {
            'hmc': self.hmc,
            'eps': self.data['eps'],
            'network_arch': self.network_arch,
            #  'conv_net': self.conv_net,
            'beta_init': self.data['beta'],
            'num_steps': self.num_steps,
            'eps_trainable': self.eps_trainable
        }

        self._create_dynamics(kwargs)

        self.build_graph(sess, config)

        self.saver = tf.train.Saver(max_to_keep=3)

        ckpt = tf.train.get_checkpoint_state(self.log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            log('Restoring previous model from: '
                f'{ckpt.model_checkpoint_path}')
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            log('Model restored.\n', nl=False)
            self.global_step = tf.train.get_global_step()
            initial_step = self.sess.run(self.global_step)
            return initial_step

        sys.stdout.flush()

    def _save_model(self, samples=None, step=None):
        """Save run `data` to `files` in `log_dir` using `checkpointer`"""
        if HAS_HOROVOD and self.using_hvd:
            if hvd.rank() != 0:
                return

        if samples is None:
            samples = self.samples

        with open(self.files['data_pkl_file'], 'wb') as f:
            pickle.dump(self.data, f)

        if samples is not None:
            with open(self.files['samples_pkl_file'], 'wb') as f:
                pickle.dump(samples, f)

        if not tf.executing_eagerly():
            ckpt_prefix = os.path.join(self.log_dir, 'ckpt')
            ckpt_file = os.path.join(self.log_dir, 'model.ckpt')
            log(f'Saving checkpoint to: {ckpt_file}\n', nl=False)
            self.saver.save(self.sess, ckpt_file, global_step=step)
            self.writer.flush()
        else:
            saved_path = self.checkpoint.save(
                file_prefix=os.path.join(self.log_dir, 'ckpt')
            )
            log(f"\n Saved checkpoint to: {saved_path}\n", nl=False)

        if not self.hmc:
            if tf.executing_eagerly():
                self.dynamics.position_fn.save_weights(
                    os.path.join(self.log_dir, 'position_model_weights.h5')
                )
                self.dynamics.momentum_fn.save_weights(
                    os.path.join(self.log_dir, 'momentum_model_weights.h5')
                )

        self.writer.flush()

    def _write_run_parameters(self, _print=False):
        """Write model parameters out to human readable .txt file."""
        if self.using_hvd:
            if hvd.rank() != 0:
                return

        if _print:
            for key, val in self.params.items():
                log(f'{key}: {val}')


        s0 = 'Parameters'
        sep_str = 80 * '-'
        strings = []
        for key, val in self.params.items():
            strings.append(f'{key}: {val}')

        write(s0, self.files['parameters_file'], 'w')
        write(sep_str, self.files['parameters_file'], 'a')
        _ = [write(s, self.files['parameters_file'], 'a') for s in strings]
        write(sep_str, self.files['parameters_file'], 'a')

    def _create_summaries(self):
        """Create summary objects for logging in TensorBoard."""
        if self.using_hvd:
            if hvd.rank() != 0:
                return

        ld = self.log_dir
        self.summary_writer = tf.contrib.summary.create_file_writer(ld)

        grads_and_vars = zip(self.grads, self.dynamics.trainable_variables)

        with tf.name_scope('loss'):
            tf.summary.scalar('loss', self.loss_op)

        with tf.name_scope('step_size'):
            tf.summary.scalar('step_size', self.dynamics.eps)

        with tf.name_scope('summaries'):
            with tf.name_scope('position_fn'):
                for var in self.dynamics.position_fn.trainable_variables:
                    try:
                        layer, type = var.name.split('/')[-2:]
                        name = layer + '_' + type[:-2]
                    except:
                        name = var.name[:-2]

                    variable_summaries(var, name)

            with tf.name_scope('momentum_fn'):
                for var in self.dynamics.momentum_fn.trainable_variables:
                    try:
                        layer, type = var.name.split('/')[-2:]
                        name = layer + '_' + type[:-2]
                    except:
                        name = var.name[:-2]

                    variable_summaries(var, name)

            with tf.name_scope('grads'):
                for grad, var in grads_and_vars:
                    try:
                        layer, type = var.name.split('/')[-2:]
                        name = layer + '_' + type[:-2]
                    except:
                        name = var.name[:-2]

                    variable_summaries(grad, name + '_gradient')

            self.summary_op = tf.summary.merge_all(name='summary_op')

    def _create_metric_fn(self):
        """Create metric fn for measuring the distance between two samples."""
        if self.metric == 'l1':
            self.metric_fn = lambda x1, x2: tf.abs(x1 - x2)

        if self.metric == 'l2':
            self.metric_fn = lambda x1, x2: tf.square(x1 - x2)

        if self.metric == 'cos':
            self.metric_fn = lambda x1, x2: tf.abs(tf.cos(x1) - tf.cos(x2))

        if self.metric == 'cos2':
            self.metric_fn = lambda x1, x2: tf.square(tf.cos(x1) - tf.cos(x2))

        if self.metric == 'cos_diff':
            self.metric_fn = lambda x1, x2: 1. - tf.cos(x1 - x2)

    def _create_sampler(self):
        """Create operation for generating new samples using dynamics engine.

        This method is to be used when running generic HMC to create operations
        for dealing with `dynamics.apply_transition` without building
        unnecessary operations for calculating loss.
        """
        with tf.name_scope('sampler'):
            inputs = (self.x, self.beta)
            _, _, self.px, self.x_out = self.dynamics(inputs)
            #  _, _, self.px, self.x_out = self.dynamics.apply_transition(
            #      self.x,
            #      self.beta
            #  )

    def build_graph(self, sess=None, config=None):
        """Build graph for TensorFlow."""
        sep_str = 80 * '-'
        s = f"Building graph... (started at: {time.ctime()})"
        log(sep_str)
        log(s)

        if config is None:
            self.config = tf.ConfigProto()
        else:
            self.config = config

        if sess is None:
            self.sess = tf.Session(config=self.config)
        else:
            self.sess = sess

        # if running generic HMC, all we need is the sampler
        if self.hmc:
            self._create_sampler()
            self.sess.run(tf.global_variables_initializer())
            return

        with tf.name_scope('global_step'):
            self.global_step = tf.train.get_or_create_global_step()
            self.global_step.assign(1)

        self.learning_rate = tf.train.exponential_decay(
            self.learning_rate_init,
            self.global_step,
            self.learning_rate_decay_steps,
            self.learning_rate_decay_rate,
            staircase=True,
            name='learning_rate'
        )

        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate,
                name='AdamOptimizer'
            )

            if self.using_hvd:
                self.optimizer = hvd.DistributedOptimizer(self.optimizer)

        self._create_metric_fn()
        start_time = time.time()

        kwargs = {
            'loss_fn': compute_loss,
            'clip_value': self.clip_value
        }
        outputs = graph_step(self.dynamics,
                             self.optimizer,
                             self.x,
                             self.beta,
                             self.global_step,
                             self.metric_fn,
                             **kwargs)

        self.train_op, self.loss_op, self.grads, self.x_out, self.px = outputs

        log("  Creating summaries...")
        t0 = time.time()

        self._create_summaries()

        t_diff = time.time() - t0
        log(f'  done. took: {t_diff:4.3g} s to create.')
        log(f'done. took: {time.time() - start_time:4.3g} s to create.')
        log(sep_str)
        #  log(80 * '-' + '\n', nl=False)

        if self.condition1 or self.condition2:
            write(sep_str, self.files['run_info_file'], 'a')
            write(s, self.files['run_info_file'], 'a')
            dt = time.time() - start_time
            s0 = f'Summaries took: {t_diff:4.3g} s to build.\n'
            s1 = f'Graph took: {dt:4.3g} s to build.\n'
            write(s0, self.files['run_info_file'], 'a')
            write(s1, self.files['run_info_file'], 'a')
            write(sep_str, self.files['run_info_file'], 'a')

    def update_beta(self, step):
        """Returns new beta to follow annealing schedule."""
        temp = ((1. / self.beta_init - 1. / self.beta_final)
                * (1. - step / float(self.train_steps))
                + 1. / self.beta_final)
        new_beta = 1. / temp

        return new_beta

    def pre_train(self):
        """Set up training for the model."""
        if self.condition1 or self.condition2:
            self.saver = tf.train.Saver(max_to_keep=3)

        self.samples_file_path = os.path.join(self.eval_dir,
                                              'training',
                                              'intermediate_beta')
        check_else_make_dir(self.samples_file_path)

        self.sess.run(tf.global_variables_initializer())

        if self.using_hvd:
            self.sess.run(hvd.broadcast_global_variables(0))

        if self.condition1 or self.condition2:
            ckpt = tf.train.get_checkpoint_state(self.log_dir)
            #  time_delay = 0.
            if ckpt and ckpt.model_checkpoint_path:
                log('Restoring previous model from: '
                    f'{ckpt.model_checkpoint_path}')
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                log('Model restored.\n', nl=False)
                self.global_step = tf.train.get_global_step()
                #  initial_step = self.sess.run(self.global_step)

            self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        self.sess.graph.finalize()

    # pylint: disable=too-many-statements
    def train(self,
              train_steps,
              samples_init=None,
              beta_init=None,
              pre_train=True,
              trace=False):
        """Train the L2HMC sampler for `train_steps`.

        Args:
            train_steps: Integer specifying the number of training steps to
                perform.
            pre_train: Boolean that when True, creates `self.saver`, and 
                `self.writer` objects and finalizes the graph to ensure no
                additional operations are created during training.
            trace: Boolean that when True performs a full trace of the training
                procedure.
        """

        # Move attribute look ups outside loop to improve performance
        #  loss_op = self.loss_op
        #  train_op = self.train_op
        #  summary_op = self.summary_op
        #  x_out = self.x_out
        #  px = self.px
        #  learning_rate = self.learning_rate
        #  dynamics = self.dynamics
        #  x = self.x
        #  dynamics_beta = self.dynamics.beta

        start_time = time.time()
        if pre_train:
            self.pre_train()

        initial_step = self.data['step']

        #  tsl = self.training_samples_length
        norm_factor = self.num_steps * self.batch_size * self.lattice.num_links

        data_header = helpers.data_header()

        self.data['learning_rate'] = self.sess.run(self.learning_rate)
        lr_np = self.data['learning_rate']

        if beta_init is None:
            beta_np = self.beta_init
        else:
            beta_np = beta_init

        if samples_init is None:
            samples_np = np.array(self.lattice.samples, dtype=np.float32)
        else:
            samples_np = samples_init
        #  samples_np = self.reshape_5D(samples_np)

        try:
            log(data_header)
            if self.condition1 or self.condition2:
                helpers.write_run_data(
                    self.files['run_info_file'],
                    self.data,
                    header=True
                )
            for step in range(initial_step, initial_step + train_steps):
                start_step_time = time.time()

                beta_np = self.update_beta(beta_np)

                fd = {self.x: samples_np,
                      self.beta: beta_np}

                _, loss_np, samples_np, px_np, eps_np = self.sess.run([
                    self.train_op,
                    self.loss_op,
                    self.x_out,
                    self.px,
                    self.dynamics.eps
                ], feed_dict=fd)

                if step % self.learning_rate_decay_steps == 0:
                    lr_np = self.sess.run(self.learning_rate)

                if step % self.print_steps == 0:
                    self.data['step'] = step
                    self.data['loss'] = loss_np
                    self.data['accept_prob'] = px_np
                    self.data['eps'] = eps_np
                    self.data['beta'] = beta_np
                    self.data['learning_rate'] = lr_np
                    self.data['step_time_norm'] = (
                        (time.time() - start_step_time) / norm_factor
                    )
                    self.data['step_time'] = (
                        time.time() - start_step_time
                    )
                    self.losses_arr.append(loss_np)

                    #  if (step + 1) % 10 == 0:
                    if self.condition1 or self.condition2:
                        helpers.print_run_data(self.data)
                        helpers.write_run_data(self.files['run_info_file'],
                                               self.data)

                #  if step % self.annealing_steps == 0:
                #      new_beta = beta_np / self.annealing_factor
                #
                #      if new_beta < self.beta_final:
                #          beta_np = new_beta
                #      else:
                #          train_time = time.time() - start_time
                #          log("Annealing schedule finished!")
                #          log("Time to complete training: {train_time:.3g}.")
                #          log("Saving model and exiting...")
                #          beta_np = self.beta_final
                #          step = self.sess.run(self.global_step)
                #          self.data['step'] = step
                #          self.data['eps'] = eps_np
                #          self.data['beta_np'] = beta_np
                #          self._save_model(samples=samples_np, step=step)
                #          if self.condition1 or self.condition2:
                #              helpers.write_run_data(self.files['run_info_file'],
                #                                     self.data)
                #              sys.stdout.flush()
                #
                #          return 0

                # Intermittently run sampler and save samples to pkl file.
                # We can calculate observables from these samples to
                # evaluate the samplers performance while we continue training.
                if (step + 1) % self.training_samples_steps == 0:
                    if self.condition1 or self.condition2:
                        t0 = time.time()
                        log(80 * '-')
                        self.run(self.training_samples_length,
                                 current_step=step+1,
                                 beta=self.beta_final)
                        log(f"  done. took: {time.time() - t0}.")
                        log(80 * '-')
                        log(data_header)

                if step % self.save_steps == 0:
                    if self.condition1 or self.condition2:
                        self._save_model(samples=samples_np, step=step-2)
                        helpers.write_run_data(self.files['run_info_file'],
                                               self.data)

                if step % self.logging_steps == 0:
                    if self.using_hvd:
                        if hvd.rank() != 0:
                            continue

                    #  samples_file_name =
                    #  (f'training_samples_{step}_train_steps'
                    #  f'_beta_{beta_np:.1g}.pkl')
                    #  out_file = os.path.join(self.samples_file_path,
                    #                          samples_file_name)
                    #  with open(out_file, 'wb') as f:
                    #      pickle.dump(samples_np, f)

                    if trace:
                        options = tf.RunOptions(
                            trace_level=tf.RunOptions.FULL_TRACE
                        )
                        run_metadata = tf.RunMetadata()

                    else:
                        options = None
                        run_metadata = None

                    #  if self.condition1 or self.condition2:
                    summary_str = self.sess.run(
                        self.summary_op, feed_dict={
                            self.x: samples_np,
                            self.beta: beta_np
                        }, options=options, run_metadata=run_metadata
                    )
                    self.writer.add_summary(summary_str,
                                            global_step=step)
                    if trace:
                        self.writer.add_run_metadata(run_metadata,
                                                     global_step=step)
                    self.writer.flush()

            train_time = time.time() - start_time
            log("Training complete!")
            log(f"Time to complete training: {train_time:.3g}.")
            step = self.sess.run(self.global_step)
            self._save_model(samples=samples_np, step=step)

            if self.condition1 or self.condition2:
                helpers.write_run_data(self.files['run_info_file'], self.data)
                sys.stdout.flush()

        except (KeyboardInterrupt, SystemExit):
            log("\nKeyboardInterrupt detected! \n", nl=False)
            log("Saving current state and exiting.\n", nl=False)
            step = self.sess.run(self.global_step)
            self._save_model(samples=samples_np, step=step)

    # pylint: disable=inconsistent-return-statements
    def run(self, run_steps, ret=False, current_step=None, beta=None):
        """Run the simulation to generate samples and calculate observables.

        Args:
            run_steps: Number of steps to run the sampler for.
            ret: Boolean value indicating if the generated samples should be
                returned. If ret is False, the samples are saved to a `.pkl`
                file and then deleted.
            current_step: Integer passed when the sampler is ran intermittently
                during the training procedure, as a way to create unique file
                names each time the sampler is ran. By running the sampler
                during the training procedure, we are able to monitor the
                performance during training.
            beta: Float value indicating the inverse coupling constant that the
                sampler should be ran at.
        Returns:
            If `ret` is True, return the chain of generated samples. 
            Otherwise, save the samples to a `.pkl` file and free up memory by
            deleting them.
        """
        if self.using_hvd:        # if using horovod, make sure we only perform
            if hvd.rank() != 0:   # file IO on rank 0.
                return

        if beta is None:
            beta = self.beta_final

        # start with randomly generated samples
        samples = np.random.randn(*self.samples.shape)
        samples_history = []

        log(f"Running sampler for {run_steps} steps at beta = {beta}...")

        if current_step is None:                     # running AFTER training
            txt_file = f'eval_info_steps_{run_steps}_beta_{beta}.txt'
            pkl_file = f'samples_history_steps_{run_steps}_beta_{beta}.pkl'

            eval_file = os.path.join(self.eval_dir, txt_file)
            out_file = os.path.join(self.samples_dir, pkl_file)

        else:                                        # running DURING training
            txt_file = (f'eval_info_{current_step}_TRAIN_'
                        f'steps_{run_steps}_beta_{beta}.txt')
            pkl_file = (f'samples_history_{current_step}_TRAIN_'
                        f'steps_{run_steps}_beta_{beta}.pkl')

            eval_file = os.path.join(self.train_eval_dir, txt_file)
            out_file = os.path.join(self.train_samples_dir, pkl_file)

        eps = self.sess.run(self.dynamics.eps)

        start_time = time.time()
        for step in range(run_steps):
            t0 = time.time()
            samples, px = self.sess.run(
                [self.x_out, self.px], feed_dict={self.x: samples,
                                                  self.beta: beta}
            )

            samples_history.append(np.squeeze(samples))

            if step % 10 == 0:
                tt = (time.time() - t0)# / (norm_factor)
                eval_str = (f'step: {step:>6.4g}/{run_steps:<6.4g} '
                            f'beta: {beta:^6.4g} '
                            f'accept prob (avg): {np.mean(px):^9.4g} '
                            f'step size: {eps:^6.4g} '
                            f'  time/step: {tt:^6.4g}\n')

                log(eval_str)
                #log('accept prob: ', nl=False)
                #log(str(px))
                #log('\n')

                write(eval_str, eval_file, 'a')
                write('accept_prob:', eval_file, 'a', nl=False)
                write(str(px), eval_file, 'a', nl=True)
                write('', eval_file, 'a')

        with open(out_file, 'wb') as f:
            pickle.dump(samples_history, f)

        log(f'\nSamples saved to: {out_file}.')
        log(f'\n Time to complete run: {time.time() - start_time} seconds.')
        log(80*'-' + '\n', nl=False)

        if ret:
            return samples_history

        del samples_history  # free up some memory


# pylint: disable=too-many-statements
def main(flags):
    """Main method for creating/training U(1) gauge model from command line."""
    params = PARAMS  # use default parameters if no command line args passed

########################### Lattice parameters ###############################
    params['time_size'] = flags.time_size
    params['space_size'] = flags.space_size
    params['link_type'] = flags.link_type
    params['dim'] = flags.dim
    params['num_samples'] = flags.num_samples
########################### Leapfrog parameters ##############################
    params['num_steps'] = flags.num_steps
    params['eps'] = flags.eps
    params['loss_scale'] = flags.loss_scale
    params['loss_eps'] = 1e-4
########################### Learning rate parameters #########################
    params['learning_rate_init'] = flags.learning_rate_init
    params['learning_rate_decay_rate'] = flags.learning_rate_decay_rate
    params['learning_rate_decay_steps'] = flags.learning_rate_decay_steps
########################### Annealing parameters #############################
    params['annealing'] = flags.annealing
    #  params['annealing_steps'] = flags.annealing_steps
    #  params['annealing_factor'] = flags.annealing_factor
    params['beta_init'] = flags.beta_init
    params['beta_final'] = flags.beta_final
########################### Training parameters ##############################
    params['train_steps'] = flags.train_steps
    params['save_steps'] = flags.save_steps
    params['logging_steps'] = flags.logging_steps
    params['print_steps'] = flags.print_steps
    params['training_samples_steps'] = flags.training_samples_steps
    params['training_samples_length'] = flags.training_samples_length
########################### Model parameters #################################
    params['network_arch'] = flags.network_arch
    params['hmc'] = flags.hmc
    params['eps_trainable'] = flags.eps_trainable
    params['metric'] = flags.metric
    params['aux'] = flags.aux
    params['clip_grads'] = flags.clip_grads
    params['clip_value'] = flags.clip_value
    params['using_hvd'] = flags.horovod
##############################################################################

    if flags.beta != flags.beta_init:
        if flags.annealing:
            params['beta'] = flags.beta_init

    if flags.hmc:
        params['eps_trainable'] = False

    config = tf.ConfigProto()

    if flags.gpu:
        print("Using gpu for training.")
        params['data_format'] = 'channels_first'
        os.environ["KMP_BLOCKTIME"] = str(0)
        os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        #  config.intra_op_parallelism_threads = flags.num_intra_threads
        #  config.inter_op_parallelism_threads = flags.num_inter_threads
    else:
        params['data_format'] = 'channels_last'

    if flags.theta:
        log("Training on Theta @ ALCF...")
        params['data_format'] = 'channels_first'
        os.environ["KMP_BLOCKTIME"] = str(0)
        os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
        config.allow_soft_placement = True
        config.intra_op_parallelism_threads = 62


    model = GaugeModel(params=params,
                       config=config,
                       sess=None,
                       log_dir=flags.log_dir,
                       restore=flags.restore)

    if flags.horovod:
        if hvd.rank() == 0:
            save_params_to_pkl_file(params, model.info_dir)

        log('Number of CPUs: %d' % hvd.size())

    log(f"Training began at: {time.ctime()}")

    model.train(flags.train_steps, pre_train=True, trace=False)

    try:
        run_steps_grid = [100, 500, 1000, 2500, 5000, 10000]
        for steps in run_steps_grid:
            model.run(steps)

    except (KeyboardInterrupt, SystemExit):
        log("\nKeyboardInterrupt detected! \n")

        import pdb

        pdb.set_trace()

###############################################################################
#  * NOTE:
#      - if action == 'store_true':
#          The argument is FALSE by default. Passing this flag will cause the
#          argument to be ''stored true''.
#      - if action == 'store_false':
#          The argument is TRUE by default. Passing this flag will cause the
#          argument to be ''stored false''.
###############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('L2HMC model using U(1) lattice gauge theory for target '
                     'distribution.')
    )
########################### Lattice parameters ###############################
    parser.add_argument("-s", "--space_size", type=int,
                        default=8, required=False, dest="space_size",
                        help="Spatial extent of lattice. (Default: 8)")

    parser.add_argument("-t", "--time_size", type=int,
                        default=8, required=False, dest="time_size",
                        help="Temporal extent of lattice. (Default: 8)")

    parser.add_argument("--link_type", type=str, required=False,
                        default='U1', dest="link_type",
                        help="Link type for gauge model. (Default: U1)")

    parser.add_argument("--dim", type=int, required=False,
                        default=2, dest="dim",
                        help="Dimensionality of lattice (Default: 2)")

    parser.add_argument("-N", "--num_samples", type=int,
                        default=2, required=False, dest="num_samples",
                        help=("Number of samples (batch size) to use for "
                              "training. (Default: 2)"))

    parser.add_argument("--rand", action="store_true",
                        required=False, dest="rand",
                        help=("Start lattice from randomized initial "
                              "configuration. (Default: False)"))

########################### Leapfrog parameters ##############################

    parser.add_argument("-n", "--num_steps", type=int,
                        default=5, required=False, dest="num_steps",
                        help=("Number of leapfrog steps to use in (augmented) "
                              "HMC sampler. (Default: 5)"))

    parser.add_argument("--eps", type=float, default=0.1,
                        required=False, dest="eps",
                        help=("Step size to use in leapfrog integrator. "
                              "(Default: 0.1)"))

    parser.add_argument("--loss_scale", type=float, default=0.1,
                        required=False, dest="loss_scale",
                        help=("Scaling factor to be used in loss function. "
                              "(lambda in Eq. 7 of paper). (Default: 0.1)"))

########################### Learning rate parameters ##########################

    parser.add_argument("--learning_rate_init", type=float, default=1e-3,
                        required=False, dest="learning_rate_init",
                        help=("Initial value of learning rate. "
                              "(Deafult: 1e-3)"))

    parser.add_argument("--learning_rate_decay_steps", type=int, default=500,
                        required=False, dest="learning_rate_decay_steps",
                        help=("Number of steps after which to decay learning "
                              "rate. (Default: 500)"))

    parser.add_argument("--learning_rate_decay_rate", type=float, default=0.96,
                        required=False, dest="learning_rate_decay_rate",
                        help=("Learning rate decay rate to be used during "
                              "training. (Default: 0.96)"))

########################### Annealing rate parameters ########################

    parser.add_argument("--annealing", action="store_true",
                        required=False, dest="annealing",
                        help=("Flag that when passed will cause the model "
                              "to perform simulated annealing during "
                              "training. (Default: False)"))

    parser.add_argument("--annealing_steps", type=float, default=200,
                        required=False, dest="annealing_steps",
                        help=("Number of steps after which to anneal beta."))

    parser.add_argument("--annealing_factor", type=float, default=0.97,
                        required=False, dest="annealing_factor",
                        help=("Factor by which to anneal beta."))

    parser.add_argument("-b", "--beta", type=float,
                        required=False, dest="beta",
                        help=("Beta (inverse coupling constant) used in "
                              "gauge model. (Default: 8.)"))

    parser.add_argument("--beta_init", type=float, default=1.,
                        required=False, dest="beta_init",
                        help=("Initial value of beta (inverse coupling "
                              "constant) used in gauge model when annealing. "
                              "(Default: 1.)"))

    parser.add_argument("--beta_final", type=float, default=8.,
                        required=False, dest="beta_final",
                        help=("Final value of beta (inverse coupling "
                              "constant) used in gauge model when annealing. "
                              "(Default: 8.)"))

########################### Training parameters ##############################

    parser.add_argument("--train_steps", type=int, default=1000,
                        required=False, dest="train_steps",
                        help=("Number of training steps to perform. "
                              "(Default: 1000)"))

    parser.add_argument("--save_steps", type=int, default=50,
                        required=False, dest="save_steps",
                        help=("Number of steps after which to save the model "
                              "and current values of all parameters. "
                              "(Default: 50)"))

    parser.add_argument("--print_steps", type=int, default=1,
                        required=False, dest="print_steps",
                        help=("Number of steps after which to display "
                              "information about the loss and various "
                              "other quantities (Default: 1)"))

    parser.add_argument("--logging_steps", type=int, default=50,
                        required=False, dest="logging_steps",
                        help=("Number of steps after which to write logs for "
                              "tensorboard. (Default: 50)"))

    parser.add_argument("--training_samples_steps", type=int, default=500,
                        required=False, dest="training_samples_steps",
                        help=("Number of intermittent steps after which "
                              "the sampler is evaluated at `beta_final`. "
                              "This allows us to monitor the performance of "
                              "the sampler during training. (Default: 500)"))

    parser.add_argument("--training_samples_length", type=int, default=100,
                        required=False, dest="training_samples_length",
                        help=("Number of steps to run sampler for when "
                              "evaluating the sampler during training. "
                              "(Default: 100)"))

########################### Model parameters ################################

    parser.add_argument('--network_arch', type=str, default='conv3D',
                        required=False, dest='network_arch',
                        help=("String specifying the architecture to use for "
                              "the neural network. Must be one of: "
                              "`'conv3D', 'conv2D', 'generic'`. "
                              "(Default: conv3D)"))
    #  parser.add_argument("--conv_net", action="store_true",
    #                      required=False, dest="conv_net",
    #                      help=("Whether or not to use convolutional "
    #                            "neural network for pre-processing lattice "
    #                            "configurations (prepended to generic FC net "
    #                            "as outlined in paper). (Default: False)"))

    parser.add_argument("--hmc", action="store_true",
                        required=False, dest="hmc",
                        help=("Use generic HMC (without augmented leapfrog "
                              "integrator described in paper). Used for "
                              "comparing against L2HMC algorithm. "
                              "(Default: False)"))

    parser.add_argument("--eps_trainable", action="store_true",
                        required=False, dest="eps_trainable",
                        help=("Flag that when passed will allow the step size "
                              "`eps` to be a trainable parameter."))

    parser.add_argument("--metric", type=str, default="l2",
                        required=False, dest="metric",
                        help=("Metric to use in loss function. "
                              "(Default: `l2`, choices: [`l2`, `l1`, `cos`])"))

    parser.add_argument("--aux", action="store_true",
                        required=False, dest="aux",
                        help=("Include auxiliary function `q` for calculating "
                              "expected squared jump distance conditioned on "
                              "initialization distribution. (Default: False)"))

    parser.add_argument("--clip_grads", action="store_true",
                        required=False, dest="clip_grads",
                        help=("Flag that when passed will clip gradients by "
                              "global norm using `--clip_value` command line "
                              "argument. If `--clip_value` is not passed, "
                              "it defaults to 100."))

    parser.add_argument("--clip_value", type=int, default=100,
                        required=False, dest="clip_value",
                        help=("Clip value, used for clipping value of "
                              "gradients by global norm. (Default: 100)"))

    parser.add_argument("--log_dir", default=None,
                        required=False, dest="log_dir",
                        help=("Log directory to use from previous run. "
                              "If this argument is not passed, a new "
                              "directory will be created. (Default: None)"))

    parser.add_argument("--restore", action="store_true",
                        required=False, dest="restore",
                        help=("Restore model from previous run. "
                              "If this argument is passed, a `log_dir` "
                              "must be specified and passed to `--log_dir` "
                              "argument. (Default: False)"))

    parser.add_argument("--gpu", action="store_true",
                        required=False, dest="gpu",
                        help=("Flag that when passed indicates we're training "
                              "using an NVIDIA GPU."))

    parser.add_argument("--theta", action="store_true",
                        required=False, dest="theta",
                        help=("Flag that when passed indicates we're training "
                              "on theta @ ALCf."))

    parser.add_argument("--horovod", action="store_true",
                        required=False, dest="horovod",
                        help=("Flag that when passed uses Horovod for "
                              "distributed training on multiple nodes."))

    parser.add_argument("--num_intra_threads", default=0,
                        required=False, dest="num_intra_threads",
                        help=("Number of intra op threads to use for "
                              "tf.ConfigProto.intra_op_parallelism_threads"))

    parser.add_argument("--num_inter_threads", default=0,
                        required=False, dest="num_intra_threads",
                        help=("Number of intra op threads to use for "
                              "tf.ConfigProto.intra_op_parallelism_threads"))

    args = parser.parse_args()

    main(args)
