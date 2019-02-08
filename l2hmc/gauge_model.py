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

# pylint: disable=wildcard-import, no-member, too-many-arguments, invalid-name
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
except ImportError:
    HAS_HOROVOD = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from lattice.lattice import GaugeLattice, u1_plaq_exact
from dynamics.gauge_dynamics import GaugeDynamics
import utils.gauge_model_helpers as helpers
from utils.tf_logging import variable_summaries


tfe = tf.contrib.eager
tf.logging.set_verbosity(tf.logging.INFO)

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
tf.set_random_seed(GLOBAL_SEED)


#  MODULE_PATH = os.path.abspath(os.path.join('..'))
#  if MODULE_PATH not in sys.path:
#      sys.path.append(MODULE_PATH)

PARAMS = {
#--------------------- Lattice parameters ----------------------------
    'time_size': 12,
    'space_size': 12,
    'link_type': 'U1',
    'dim': 2,
    'num_samples': 6,
    'rand': False,
    'data_format': 'channels_last',
#--------------------- Leapfrog parameters ---------------------------
    'num_steps': 5,
    'eps': 0.2,
    'loss_scale': 1.,
    'loss_eps': 1e-4,
#--------------------- Learning rate parameters ----------------------
    'learning_rate_init': 1e-3,
    'learning_rate_decay_steps': 500,
    'learning_rate_decay_rate': 0.96,
#--------------------- Annealing rate parameters ---------------------
    'annealing': True,
    'annealing_steps': 200,
    'annealing_factor': 0.97,
    #  'beta': 2.,
    'beta_init': 2.,
    'beta_final': 8.,
#--------------------- Training parameters ---------------------------
    'train_steps': 10000,
    'save_steps': 1000,
    'logging_steps': 50,
    'print_steps': 1,
    'training_samples_steps': 1000,
    'training_samples_length': 100,
#--------------------- Model parameters ------------------------------
    'conv_net': True,
    'hmc': False,
    'eps_trainable': True,
    'metric': 'l2',
    'aux': True,
    'clip_grads': False,
    'clip_value': 10.,
}

def write(s, f, mode='a', nl=True):
    if HAS_HOROVOD and hvd.rank() != 0:
        return
    with open(f, mode) as f:
        f.write(s)
        if nl:
            f.write('\n')

def log(s, nl=True):
    if HAS_HOROVOD and hvd.rank() != 0:
        return
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
    mask = (px - tf.random_uniform(tf.shape(px)) > 0.)
    return tf.where(mask, _x, x)

def graph_step(dynamics, optimizer, samples, beta, step, 
               aux=True, out_file=None):
    with tf.name_scope('train'):
        loss, grads, samples, accept_prob = loss_and_grads(
            dynamics, samples, beta, loss_fn=compute_loss,
            aux=aux, out_file=out_file
        )
        train_op = optimizer.apply_gradients(zip(grads, dynamics.variables),
                                             global_step=step,
                                             name='train_op')

    return train_op, loss, grads, samples, accept_prob

# Loss function
def compute_loss(dynamics, x, beta, 
                 aux=True, scale=.1, eps=1e-4, out_file=None):
    """Compute loss defined in equation (8)."""
    log("    Creating loss...")
    t0 = time.time()

    x_, _, px, x_out = dynamics.apply_transition(x, beta)
    if aux:
        z = tf.random_normal(tf.shape(x))  # Auxiliary variable
        z_, _, pz, _ = dynamics.apply_transition(z, beta)

    # Add eps for numerical stability; following released impl
    with tf.name_scope('loss'):
        with tf.name_scope('x_loss'):
            x_loss = tf.reduce_sum((x - x_)**2, axis=dynamics.axes) * px + eps
        if aux:
            with tf.name_scope('z_loss'):
                z_loss = (
                    tf.reduce_sum((z - z_)**2, axis=dynamics.axes) * pz + eps
                )

            loss = tf.reduce_mean(
                (1. / x_loss + 1. / z_loss) * scale - (x_loss + z_loss) / scale,
                axis=0
            )
        else:
            loss = tf.reduce_mean(scale  / x_loss - x_loss / scale, axis=0)

    t_diff = time.time() - t0
    log(f"    done. took: {t_diff:4.3g} s.")

    if out_file is not None:
        s = f'Loss took: {t_diff:4.3g} s to create.'
        write(s, out_file, 'a')
        #  with open(out_file, 'a') as f:
        #      f.write(f'Loss took: {t_diff:4.3g} s to create.\n')


    return loss, x_out, px

def loss_and_grads(dynamics, x, beta, 
                   loss_fn=compute_loss, aux=True, out_file=None):
    """Obtain loss value and gradients."""
    log(f"  Creating gradient operations...")
    t0 = time.time()

    with tf.name_scope('grads'):
        with tf.GradientTape() as tape:
            loss_val, out, accept_prob = loss_fn(dynamics, x, beta, aux)
        grads = tape.gradient(loss_val, dynamics.trainable_variables)

    t_diff = time.time() - t0
    log(f"  done. took: {t_diff:4.3g} s")

    if out_file is not None:
        s = f'Gradient operations took: {t_diff:4.3g} s to create.\n'
        write(s, out_file, 'a')
        #  with open(out_file, 'a') as f:
        #      f.write(f'Gradient operations took: {t_diff:4.3g} s to create.\n')

    return loss_val, grads, out, accept_prob


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

        if config is None:
            config = tf.ConfigProto()

        if log_dir is None:
            dirs = helpers.create_log_dir('gauge_logs_graph')
        else:
            dirs = helpers.check_log_dir(log_dir)

        self.log_dir, self.info_dir, self.figs_dir = dirs

        self.summary_writer = (
            tf.contrib.summary.create_file_writer(self.log_dir)
        )

        self.data = {}
        self.train_samples = {}
        self.losses_arr = []
        self.steps_arr = []
        self.samples_arr = []
        self.accept_prob_arr = []
        self.step_times_arr = []

        # create attributes using key, value pairs in params
        self._init_params(params)

        with tf.name_scope('input'):
            with tf.name_scope('lattice'):
                self.lattice = GaugeLattice(time_size=self.time_size,
                                            space_size=self.space_size,
                                            dim=self.dim,
                                            link_type=self.link_type,
                                            num_samples=self.num_samples,
                                            rand=self.rand,
                                            data_format=self.data_format)


            self.batch_size = self.lattice.samples.shape[0]
            self.samples = tf.convert_to_tensor(
                self.lattice.samples, dtype=tf.float32
            )
            if not tf.executing_eagerly():
                self.x = tf.placeholder(dtype=tf.float32,
                                        shape=self.samples.shape,
                                        name='x')

                self.beta = tf.placeholder(tf.float32, shape=(), name='beta')
            else:
                self.beta = self.beta_init

        with tf.name_scope('potential_fn'):
            self.potential_fn = self.lattice.get_energy_function(self.samples)

        if restore:
            self._restore_model(self.log_dir)

        else:
            kwargs = {
                'beta_init': self.beta_init,
                'num_steps': self.num_steps,
                'eps': self.eps,
                'conv_net': self.conv_net,
                'hmc': self.hmc,
                'eps_trainable': self.eps_trainable
            }
            with tf.name_scope('dynamics'):
                self.dynamics = GaugeDynamics(lattice=self.lattice,
                                              potential_fn=self.potential_fn,
                                              **kwargs)
                                              #  beta_init=self.beta_init,
                                              #  num_steps=self.num_steps,
                                              #  eps=self.eps,
                                              #  conv_net=self.conv_net,
                                              #  hmc=self.hmc,
                                              #  eps_trainable=self.eps_trainable)


            # if running generic HMC, all we need is self.x_out to sample
            if self.hmc:
                self._create_sampler()
            else:
                self.build_graph()

            if sess is None:
                self.sess = tf.Session(config=config)
            else:
                self.sess = sess

            if self.hmc:
                self.sess.run(tf.global_variables_initializer())

    def _init_params(self, params=None):
        """Parse key value pairs from params and set as class attributes."""
        if params is None:
            params = PARAMS

        for key, val in params.items():
            setattr(self, key, val)

        #  if self.annealing:
            #  self.beta = params.get('beta_init', 1.,)
        #  else:
            #  self.beta = params.get('beta_final')

        self.params = params

        self.data = {
            'step': 0,
            'loss': 0.,
            'step_time': 0.,
            'accept_prob': 0.,
            'samples': [],
            'eps': params.get('eps', 0.),
            'beta_init': params.get('beta_init', 1.),
            'beta': params.get('beta_init', 1.),
            'train_steps': params.get('train_steps', 1000),
            'learning_rate': params.get('learning_rate_init', 1e-4),
        }

        self._create_dir_structure()

        self._write_run_parameters(_print=True)

        if not self.clip_grads:
            self.clip_value = None

    def _create_dir_structure(self):
        """Create self.files and directory structure."""
        self.files = {
            'parameters_file': os.path.join(self.info_dir, 'parameters.txt'),
            'run_info_file': os.path.join(self.info_dir, 'run_info.txt'),
            'data_pkl_file': os.path.join(self.info_dir, 'data.pkl'),
            'samples_pkl_file': os.path.join(self.info_dir, 'samples.pkl'),
            'parameters_pkl_file': os.path.join(self.info_dir,
                                                'parameters.pkl'),
        }

        self.samples_history_dir = os.path.join(self.log_dir, 'samples_history')
        check_else_make_dir(self.samples_history_dir)

        self.train_samples_history_dir = os.path.join(
            self.samples_history_dir, 'training'
        )
        check_else_make_dir(self.train_samples_history_dir)

        self.train_samples_dir = os.path.join(self.log_dir, 'train_samples')
        check_else_make_dir(self.train_samples_dir)

    def _restore_checkpoint(self, log_dir):
        """Restore from `tf.train.Checkpoint`."""
        latest_path = tf.train.latest_checkpoint(log_dir)
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            dynamics=self.dynamics,
            global_step=self.global_step,
        )

    def _restore_model(self, log_dir):
        """Restore model from previous run contained in `log_dir`."""
        if self.hmc:
            log(f"ERROR: self.hmc: {self.hmc}. No model to restore. Exiting.")
            sys.exit(1)

        assert os.path.isdir(log_dir), (f"log_dir: {log_dir} does not exist.")

        run_info_dir = os.path.join(log_dir, 'run_info')
        assert os.path.isdir(run_info_dir), (f"run_info_dir: {run_info_dir}"
                                             " does not exist.")

        with open(self.files['parameters_pkl_file'], 'rb') as f:
            self.params = pickle.load(f)

        self._init_params(self.params)

        with open(self.files['data_pkl_file'], 'rb') as f:
            self.data = pickle.load(f)

        with open(self.files['samples_pkl_file'], 'rb') as f:
            self.samples = pickle.load(f)

        self.dynamics = GaugeDynamics(
            lattice=self.lattice,
            potential_fn=self.potential_fn,
            beta_init=self.data['beta'], # use previous value of beta from data
            num_steps=self.num_steps,
            eps=self.data['eps'],
            conv_net=self.conv_net,
            hmc=self.hmc,
            eps_trainable=self.eps_trainable
        )

        self.global_step = tf.train.get_or_create_global_step()
        #  self.global_step.assign(self.data['step'])
        #  tf.add_to_collection('global_step', self.global_step)

        self.learning_rate = tf.train.exponential_decay(
            self.data['learning_rate'],
            self.data['step'],
            self.learning_rate_decay_steps,
            self.learning_rate_decay_rate,
            staircase=True
        )

        self.summary_writer = tf.contrib.summary.create_file_writer(log_dir)
        if not tf.executing_eagerly():
            self.build_graph()
            try:
                self.sess = tf.Session(config=self.config)
            except AttributeError:
                self.config = tf.ConfigProto()
                self.sess = tf.Session(config=self.config)
            self.saver = tf.train.Saver(max_to_keep=3)
            self.sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                log("Restoring previous model from: "
                      f"{ckpt.model_checkpoint_path}")
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                #  self.sess.run(tf.global_variables_initializer())
                log("Model restored.")
                self.global_step = tf.train.get_global_step()
        else:
            latest_path = tf.train.latest_checkpoint(log_dir)
            self.checkpoint.restore(latest_path)
            log("Restored latest checkpoint from:\"{}\"".format(latest_path))

        if not self.hmc:
            if tf.executing_eagerly():
                self.dynamics.position_fn.load_weights(
                    os.path.join(self.log_dir, 'position_model_weights.h5')
                )

                self.dynamics.momentum_fn.load_weights(
                    os.path.join(self.log_dir, 'momentum_model_weights.h5')
                )

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

        #  with open(self.files['parameters_file'], 'w') as f:
        #      f.write('Parameters:\n')
        #      f.write(80 * '-' + '\n')
        #      for key, val in self.params.items():
        #          f.write(f'{key}: {val}\n')
        #      f.write(80*'=')
        #      f.write('\n')

    def _create_summaries(self):
        """Create summary objects for logging in TensorBoard."""
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss_op)
            tf.summary.scalar('step_size', self.dynamics.eps)

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
                for grad, var in zip(self.grads,
                                     self.dynamics.trainable_variables):
                    try:
                        layer, type = var.name.split('/')[-2:]
                        name = layer + '_' + type[:-2]
                    except:
                        name = var.name[:-2]

                    variable_summaries(grad, name + '_gradient')

            self.summary_op = tf.summary.merge_all(name='summary_op')


    def _create_optimizer(self):
        """Create optimizer to use during training."""
        with tf.name_scope('train'):
            #  self.grads = tf.gradients(self.loss_op,
            #                            self.dynamics.trainable_variables)
            self.global_step = tf.train.get_or_create_global_step()
            self.global_step.assign(1)
            tf.add_to_collection('global_step', self.global_step)

            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate_init,
                self.global_step,
                self.learning_rate_decay_steps,
                self.learning_rate_decay_rate,
                staircase=True
            )

            #  clip_value = self.params['clip_value']
            self.grads = tf.gradients(self.loss_op,
                                      self.dynamics.trainable_variables)
            if self.clip_grads:
                self.grads, _ = tf.clip_by_global_norm(
                    self.grads,
                    self.clip_value,
                    name='clipped_grads'
                )

            #  self.grads_and_vars = list(
            #      zip(self.grads, self.dynamics.trainable_variables)
            #  )

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate,
                name='AdamOptimizer'
            )
            self.train_op = self.optimizer.apply_gradients(
                zip(self.grads, self.dynamics.trainable_variables),
                global_step=self.global_step,
                name='train_op'
            )
            #  else:
            #      self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            #      self.train_op = self.optimizer.minimize(
            #          self.loss_op,
            #          global_step=self.global_step
            #      )

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

    def _create_loss(self):
        """Define loss function to minimize during training."""
        scale = self.params['loss_scale']

        self._create_metric_fn()


        with tf.name_scope('loss'):
            self._x, _, self.px, self.x_out = (
                self.dynamics.apply_transition(self.x, self.beta)
            )

            with tf.name_scope('x_loss'):
                self.x_loss = ((tf.reduce_sum(
                    self.metric_fn(self.x, self._x),
                    axis=self.dynamics.axes,
                    name='x_loss'
                ) * self.px) + 1e-4)

            if self.aux:
                z = tf.random_normal(tf.shape(self.x), name='z')
                _z, _, pz, _ = self.dynamics.apply_transition(z, self.beta)
                with tf.name_scope('z_loss'):
                    z_loss = ((tf.reduce_sum(
                        self.metric_fn(z, _z),
                        axis=self.dynamics.axes,
                        name='z_loss'
                    ) * pz) + 1e-4)

                # Squared jump distance
                with tf.name_scope('total_loss'):
                    self.loss_op = scale * tf.reduce_mean(
                        (1. / self.x_loss + 1. / z_loss) * scale
                        - (self.x_loss + z_loss) / scale, axis=0, name='loss_op'
                    )

            else:
                with tf.name_scope('total_loss'):
                    self.loss_op = tf.reduce_mean(scale / self.x_loss
                                                  - self.x_loss / scale,
                                                  axis=0, name='loss_op')

    def _create_sampler(self):
        """Create operation for generating new samples using dynamics engine.


        This method is to be used when running generic HMC to create operations
        for dealing with `dynamics.apply_transition` without building
        unnecessary operations for calculating loss.
        """
        with tf.name_scope('sampler'):
            _, _, self.px, self.x_out = self.dynamics.apply_transition(
                self.x,
                self.beta
            )


    def build_graph(self):
        """Build graph for TensorFlow."""
        #  def _write_strs_to_file(strings, last=False):
        #      with open(self.files['run_info_file'], 'a') as f:
        #          for s in strings:
        #              f.write(s + '\n')
        #          if last:
        #              f.write(80 * '-' + '\n')

        #  str0 = f"Building graph... (started at: {time.ctime()})\n"
        #  str1 = "  Creating loss...\n"
        #  str3 = "  Creating optimizer...\n"
        #  str5 = "  Creating summaries...\n"
        #  t_diff_str = lambda ti, tf : f"    took: {tf - ti} seconds."
        #  t_diff_str1 = lambda t : f"Time to build graph: {time.time() - t}."

        #  print(80*'-' + '\n')
        #  print(str0 + str1)
        #  _write_strs_to_file([str0, str1])
        #  t0 = time.time()

        self.global_step = tf.train.get_or_create_global_step()
        self.learning_rate = tf.train.exponential_decay(
            self.learning_rate_init,
            self.global_step,
            self.learning_rate_decay_steps,
            self.learning_rate_decay_rate,
            staircase=True
        )
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            name='AdamOptimizer'
        )
        if self.using_hvd:
            opt = hvd.DistributedOptimizer(self.optimizer)

        log(80 * '-' + '\n', nl=False)
        log(f"Building graph... (started at: {time.ctime()})")
        start_time = time.time()
        #  str0 = f"Building graph... (started at: {time.ctime()})\n"
        outputs = graph_step(self.dynamics,
                             self.optimizer,
                             self.x,
                             self.beta,
                             self.global_step,
                             aux=self.aux,
                             out_file=self.files['run_info_file'])

        self.train_op, self.loss_op, self.grads, self.x_out, self.px = outputs

        s = f"Building graph... (started at: {time.ctime()})\n"
        write(s, self.files['run_info_file'], 'a')

        #  with open(self.files['run_info_file'], 'a') as f:
        #      f.write(f"Building graph... (started at: {time.ctime()})\n")

        #  self._create_loss()

        #  t1 = time.time()
        #  print(t_diff_str(t0, t1) + '\n' + str3 + '\n')
        #  _write_strs_to_file([t_diff_str(t0, t1), str3])

        #  self._create_optimizer()

        #  t2 = time.time()
        #  print(t_diff_str(t1, t2) + '\n' + str5 + '\n')
        #  _write_strs_to_file([t_diff_str(t1, t2), str5])

        log("  Creating summaries...")
        t0 = time.time()
        self._create_summaries()
        t_diff = time.time() - t0
        log(f'  done. took: {t_diff:4.3g} s to create.')
        log(f'done. took: {time.time() - start_time:4.3g} s to create.\n',False)
        log(80 * '-' + '\n', nl=False)

        dt = time.time() - start_time
        s0 = f'Summaries took: {t_diff:4.3g} s to create.\n'
        s1 = f'Graph took: {dt:4.3g} s to build.\n'
        sep_str = 80 * '-'
        write(s0, self.files['run_info_file'], 'a')
        write(s1, self.files['run_info_file'], 'a')
        write(sep_str, self.files['run_info_file'], 'a')

        #  with open(self.files['run_info_file'], 'a') as f:
        #      f.write(f'Summaries took: {t_diff:4.3g} s to create.\n')
        #      f.write(f'Graph took: {dt:4.3g} s to build.\n')
        #      f.write(80 * '-' + '\n')

        #  print(t_diff_str(t2, time.time()) + '\n' + t_diff_str1(t0) + '\n')
        #  _write_strs_to_file(
        #      [t_diff_str(t2, time.time()), t_diff_str1(t0)], last=True
        #  )

    def pre_train(self):
        """Set up training for the model."""
        self.saver = tf.train.Saver(max_to_keep=3)
        self.sess.run(tf.global_variables_initializer())
        if self.using_hvd:
            self.sess.run(hvd.broadcast_global_variables(0))
        ckpt = tf.train.get_checkpoint_state(self.log_dir)
        time_delay = 0.
        if ckpt and ckpt.model_checkpoint_path:
            log('Restoring previous model from: '
                  f'{ckpt.model_checkpoint_path}')
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            log('Model restored.\n', nl=False)
            self.global_step = tf.train.get_global_step()
            initial_step = self.sess.run(self.global_step)

        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        self.sess.graph.finalize()


    #pylint: disable=too-many-statements
    def train(self, num_train_steps, pre_train=True, kill_sess=True,
              trace=False):
        start_time = time.time()
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
        if pre_train:
            self.pre_train()

        initial_step = self.data['step']

        tsl = self.training_samples_length
        norm_factor = self.num_steps * self.batch_size * self.lattice.num_links

        data_header = helpers.data_header()

        self.data['learning_rate'] = self.sess.run(self.learning_rate)
        lr_np = self.data['learning_rate']

        beta_np = self.beta_init
        samples_np = np.array(self.lattice.samples, dtype=np.float32)
        try:
            log(data_header)
            helpers.write_run_data(
                self.files['run_info_file'],
                self.data,
                header=True
            )
            for step in range(initial_step, initial_step + num_train_steps):
                start_step_time = time.time()

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
                    helpers.print_run_data(self.data)
                    helpers.write_run_data(self.files['run_info_file'],
                                           self.data)

                if self.annealing:
                    _beta_np = beta_np / self.annealing_factor

                    if _beta_np < self.beta_final:
                        beta_np = _beta_np
                    else:
                        log("Annealing schedule finished!")
                        log("Saving model and exiting...")
                        beta_np = self.beta_final
                        step = self.sess.run(self.global_step)
                        self._save_model(samples=samples_np, step=step)
                        helpers.write_run_data(self.files['run_info_file'],
                                               self.data)
                        sys.stdout.flush()
                        return 0

                # Intermittently run sampler and save samples to pkl file.
                # We can calculate observables from these samples to
                # evaluate the samplers performance while we continue training.
                if (step + 1) % self.training_samples_steps == 0:
                    if hvd.rank() == 0:
                        t0 = time.time()
                        log(80 * '-')
                        log(f"\nEvaluating sampler for {tsl} steps"
                              f" at beta = {self.beta_final}.")
                        self.run(self.training_samples_length,
                                 current_step=step+1,
                                 beta=self.beta_final)
                        log(f"  done. took: {time.time() - t0}.")
                        log(80 * '-')
                        log(data_header)

                if (step + 2) % self.save_steps == 0:
                    self._save_model(samples=samples_np, step=step-2)
                    helpers.write_run_data(self.files['run_info_file'],
                                           self.data)

                if step % self.logging_steps == 0:
                    if trace:
                        options = tf.RunOptions(
                            trace_level=tf.RunOptions.FULL_TRACE
                        )
                        run_metadata = tf.RunMetadata()
                    else:
                        options = None
                        run_metadata = None

                    summary_str = self.sess.run(
                        self.summary_op, feed_dict={
                            self.x: samples_np,
                            self.beta: beta_np
                        }, options=options, run_metadata=run_metadata
                    )
                    cond1 = self.using_hvd and hvd.rank() == 0
                    cond2 = not self.using_hvd
                    if cond1 or cond2:
                        self.writer.add_summary(summary_str,
                                                global_step=step)
                    if trace:
                        if cond1 or cond2:
                            self.writer.add_run_metadata(run_metadata,
                                                         global_step=step)
                    self.writer.flush()


            log("Training complete!")
            step = self.sess.run(self.global_step)
            self._save_model(samples=samples_np, step=step)

            helpers.write_run_data(self.files['run_info_file'], self.data)
            if kill_sess:
                self.writer.close()
                self.sess.close()
            sys.stdout.flush()

            return 0

        except (KeyboardInterrupt, SystemExit):
            log("\nKeyboardInterrupt detected! \n"
                  "Saving current state and exiting.\n", nl=False)
            step = self.sess.run(self.global_step)
            self._save_model(samples=samples_np, step=step)
            if kill_sess:
                self.writer.close()
                self.sess.close()

            return -1

    def run(self, run_steps, ret=False, current_step=None, beta=None):
        """Run the simulation to generate samples and calculate observables."""
        if beta is None:
            beta = self.beta_final

        samples = np.random.randn(*self.samples.shape)
        samples_history = []
        px_history = []

        if self.hmc:
            log(f"Running generic HMC sampler for {run_steps} steps...")
        else:
            log(f"Running (trained) L2HMC sampler for {run_steps} steps...")

        if current_step is None:
            eval_file = os.path.join(
                self.info_dir,
                f'eval_info_{run_steps}.txt'
            )
        else:
            eval_file = os.path.join(
                self.info_dir,
                f'eval_info_{current_step}_TRAIN_{run_steps}.pkl'
            )

        eps = self.sess.run(self.dynamics.eps)

        start_time = time.time()
        for step in range(run_steps):
            t0 = time.time()
            samples, px = self.sess.run(
                [self.x_out, self.px], feed_dict={self.x: samples,
                                                  self.beta: beta}
            )

            samples_history.append(samples)
            px_history.append(px)

            if step % 10 == 0:
                tt = (time.time() - t0)# / (norm_factor)
                eval_str = (f'step: {step:>6.4g}/{run_steps:<6.4g} '
                            f'accept prob (avg): {np.mean(px):^9.4g} '
                            f'step size: {eps:^6.4g} '
                            f'\t time/step: {tt:^6.4g}\n')

                log(eval_str)
                log('accept prob: ', px)
                log('\n')

                try:
                    write(eval_str, eval_file, 'a')
                    write('accept_prob:', eval_file, 'a', nl=False)
                    write(px, eval_file, 'a', nl=True)
                    write('', eval_file, 'a')
                    #  with open(eval_file, 'a') as f:
                    #      f.write(eval_str)
                    #      f.write('accept_prob: ', px)
                    #      f.write('\n')
                except:
                    continue


        if current_step is None:
            out_file = os.path.join(
                self.samples_history_dir,
                f'samples_history_{run_steps}.pkl'
            )
            px_file = os.path.join(
                self.samples_history_dir,
                f'accept_prob_history_{run_steps}.pkl'
            )

        else:
            out_file = os.path.join(
                self.train_samples_history_dir,
                f'samples_history_{current_step}_TRAIN_{run_steps}.pkl'
            )
            px_file = os.path.join(
                self.train_samples_history_dir,
                f'accept_prob_history_{current_step}_TRAIN_{run_steps}.pkl'
            )

        with open(out_file, 'wb') as f:
            pickle.dump(samples_history, f)
        with open(px_file, 'wb') as f:
            pickle.dump(px_history, f)

        log(f'\nSamples saved to: {out_file}.')
        log(f'Accept probabilities saved to: {px_file}.')
        log(f'\n Time to complete run: {time.time() - start_time} seconds.')
        log(80*'-' + '\n', nl=False)

        if ret:
            return samples_history

        del samples_history  # free up some memory


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
    params['annealing_steps'] = flags.annealing_steps
    params['annealing_factor'] = flags.annealing_factor
    params['beta_init'] = flags.beta_init
    params['beta_final'] = flags.beta_final
########################### Training parameters ##############################
    params['train_steps'] = flags.train_steps
    params['save_steps'] = flags.save_steps
    params['logging_steps'] = flags.logging_steps
    params['print_steps'] = flags.print_steps
    params['training_samples_steps'] = flags.training_samples_steps
    params['training_samples_length'] = flags.training_samples_length
########################### Model parameters ################################
    params['conv_net'] = flags.conv_net
    params['hmc'] = flags.hmc
    params['eps_trainable'] = flags.eps_trainable
    params['metric'] = flags.metric
    params['aux'] = flags.aux
    params['clip_grads'] = flags.clip_grads
    params['clip_value'] = flags.clip_value

    if flags.beta != flags.beta_init:
        if flags.annealing:
            params['beta'] = flags.beta_init

    eps_trainable = True

    if flags.hmc:
        eps_trainable = False

    if flags.horovod:
        params['using_hvd'] = True
        params['train_steps'] = params['train_steps'] // hvd.size()
        hvd.init()

    config = tf.ConfigProto()

    if flags.gpu:
        print("Using gpu for training.")
        params['data_format'] = 'channels_first'
        os.environ["KMP_BLOCKTIME"] = str(0)
        os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
        config.allow_soft_placement = True
        #  config.intra_op_parallelism_threads = flags.num_intra_threads
        #  config.inter_op_parallelism_threads = flags.num_inter_threads

    if flags.theta:
        print("Training on Theta @ ALCF...")
        params['data_format'] = 'channels_last'
        os.environ["KMP_BLOCKTIME"] = str(0)
        os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
        config.allow_soft_placement = True
        config.intra_op_parallelism_threads = 62


    else:
        params['data_format'] = 'channels_last'


    model = GaugeModel(params=params,
                       config=config,
                       sess=None,
                       log_dir=flags.log_dir,
                       restore=flags.restore)


    save_params_to_pkl_file(params, model.info_dir)

    #  start_time_str = time.strftime("%a, %d %b %Y %H:%M:%S",
    #                                 time.ctime(time.time()))

    print(f"Training began at: {time.ctime()}")

    model.train(flags.train_steps, kill_sess=False)

    try:
        run_steps_grid = [100, 500, 1000, 2500, 5000, 10000]
        for steps in run_steps_grid:
            model.run(steps)

        #  _ = model.run(flags.run_steps)
    except (KeyboardInterrupt, SystemExit):
        print("\nKeyboardInterrupt detected! \n")
        import pdb
        pdb.set_trace()

        model.sess.close()



#############################################################################
#  NOTE:
#----------------------------------------------------------------------------
#   * if action = 'store_true':
#       The argument is FALSE by default. Passing this flag will cause the
#       argument to be ''stored true''.
#   * if action = 'store_false':
#       The argument is TRUE by default. Passing this flag will cause the
#       argument to be ''stored false''.
#############################################################################
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

    parser.add_argument("--conv_net", action="store_true",
                        required=False, dest="conv_net",
                        help=("Whether or not to use convolutional "
                              "neural network for pre-processing lattice "
                              "configurations (prepended to generic FC net "
                              "as outlined in paper). (Default: False)"))

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
