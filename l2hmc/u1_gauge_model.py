"""
Module for running either generic HMC or L2HMC algorithms for gauge models
defined on the lattice.

Capable of parsing command line options specifying lattice attributes and
hyperparameters for training L2HMC.

Author: Sam Foreman
"""
# pylint: disable=wildcard-import, no-member, too-many-arguments
import os
import sys
import numpy as np
import tensorflow as tf
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from l2hmc_eager import gauge_dynamics_eager as gauge_dynamics
from l2hmc_eager.neural_nets import *
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
    'dim': 2,
    'beta': 2.,
    'num_samples': 5,
    'rand': False,
    'link_type': 'U1',
    'n_lf_steps': 10,
    'lf_step_size': 0.05,
    'train_iters': 500,
    'record_loss_every': 50,
    'save_steps': 50,
    'lr_init': 1e-3,
    'lr_decay_steps': 1000,
    'lr_decay_rate': 0.96,
}

def train_one_iter(dynamics, beta, x, optimizer,
                   loss_fn=gauge_dynamics.compute_loss, global_step=None):
    dynamics_out = gauge_dynamics.loss_and_grads(dynamics, x, loss_fn=loss_fn)
    loss, grads, out, accept_prob = dynamics_out

    optimizer.apply_gradients(zip(grads, dynamics.trainable_variables),
                              global_step=global_step)
    return loss, out, accept_prob

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


def step(dynamics, optimizer, samples):
    """Single training step for use with tensorflow's eager execution.
    
    To be defunnable, the funciton cannot return an Operation, so the `step`
    method is used for defun or eager.
    """
    loss, grads, samples, _ = gauge_dynamics.loss_and_grads(
        dynamics, samples, loss_fn=gauge_dynamics.compute_loss
    )
    optimizer.apply_gradients(zip(grads, dynamics.variables))

    return loss, samples


def graph_step(dynamics, optimizer, samples):
    """Single training step for use with a tensorflow graph object."""

    loss, grads, samples, _ = gauge_dynamics.loss_and_grads(
        dynamics, samples, loss_fn=gauge_dynamics.compute_loss
    )
    train_op = optimizer.apply_gradients(zip(grads, dynamics.variables))

    return train_op, loss, samples


# pylint: ignore  too-many-instance-attributes
class GaugeModel(object):
    """Class for implementing (L2)HMC for a pure gauge model on the lattice."""
    def __init__(self, 
                 params, 
                 config, 
                 log_dir=None, 
                 restore=False, 
                 eager=True):
        """Initialize GaugeModel object.

            Args:
                params: Parameters for model.
                config: Tensorflow config object.
                log_dir: Directory tu use for saving run information. If
                    log_dir is None, one will be created.
                restore: Flag for restoring model from previous training
                    session.
                eager: Flag specifying whether or not to use tensorflow's eager
                    execution.
        """
        self._init_params(params)

        if log_dir:
            dirs = check_log_dir(log_dir)
        else:
            dirs = create_log_dir()

        self.log_dir, self.info_dir, self.figs_dir = dirs
        self.summary_writer = tf.contrib.summary.create_file_writer(log_dir)

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

        self.dynamics = self._create_dynamics(lattice=self.lattice,
                                              num_lf_steps=self.n_lf_steps,
                                              eps=self.lf_step_size,
                                              potential_fn=self.potential_fn)

        #  self.optimizer = tf.train.AdamOptimizer(
        #      learning_rate=self.lr_init
        #  )

        self.global_step = tf.train.get_or_create_global_step()
        self.global_step.assign(1)

        if eager:
            self.step_fn = step
            #  self.step_fn = step(self.dynamics, self.optimizer, self.samples)

        else:
            self.sess = tf.Session(config=config)
            #  self.step_fn = graph_step(self.dynamics, self.optimizer,
            #                            self.samples)
            #  self.step_fn = graph_step
            #  self.loss_op, x_out, x_accept_prob = self._create_loss()

        if restore:
            self.restore_model()

    def _init_params(self, params=None):
        """Parse key value pairs from params and set as as class attributes.

        Args:
            params: Dictionary containing model parameters.

        Returns:
            None
        """
        self.train_times = {}
        self.actions_arr = []
        self.avg_plaquettes_arr = []
        self.losses_arr = []

        if params is None:
            params = PARAMS

        for key, val in params.items():
            setattr(self, key, val)

    def _create_params_file(self):
        """Create txt file for storing all current values of parameters."""
        params_txt_file = self.info_dir + 'parameters.txt'
        with open(params_txt_file, 'w') as f:
            for key, val in self.__dict__.items():
                if isinstance(val, (int, float, str)):
                    f.write(f'\n{key}: {val}\n')

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
    def _create_dynamics(lattice, num_lf_steps, eps, potential_fn):
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
        return gauge_dynamics.GaugeDynamics(
            lattice,
            n_steps=num_lf_steps,
            eps=eps,
            minus_loglikelihood_fn=potential_fn
        )

    @staticmethod
    def _create_checkpointer(optimizer, dynamics, global_step):
        """Create checkpointer object."""
        return tf.train.Checkpoint(optimizer=optimizer,
                                   dynamics=dynamics,
                                   global_step=global_step)

    @staticmethod
    def _create_loss(dynamics, x, scale=0.1, eps=1e-4):
        """Compute loss defined in Eq. (8) of paper."""
        with tf.name_scope('loss'):
            z = tf.random_normal(tf.shape(x))  # Auxiliary variable
            _x, _, x_accept_prob, x_out = dynamics.apply_transition(x)
            _z, _, z_accept_prob, _ = dynamics.apply_transition(z)

            if x.shape != _x.shape:
                x = tf.reshape(x, shape=_x.shape)
            if z.shape != _z.shape:
                z = tf.reshape(z, shape=_z.shape)

             # Add eps for numerical stability; following released implementation
            x_loss = (tf.reduce_sum(
                (tf.math.cos(x) - tf.math.cos(_x))**2, axis=1
            ) * x_accept_prob + eps)

            z_loss = (tf.reduce_sum(
                (tf.math.cos(z) - tf.math.cos(_z))**2, axis=1
            ) * z_accept_prob + eps)

            loss_op = tf.reduce_mean((1. / x_loss + 1. / z_loss) * scale
                                     - (x_loss + z_loss) / scale, axis=0)

        return loss_op, x_out, x_accept_prob

    @staticmethod
    def _compute_loss_and_grads(dynamics):
        """Obtain loss value and gradients."""
        with tf.GradientTape() as tape:
            loss_val, out, accept_prob = loss_fn(dynamics, x)
        grads = tape.gradient(loss_val, dynamics.trainable_variables)

        return loss_val, grads, out, accept_prob

    @staticmethod
    def _create_optimizer(learning_rate, loss_op, global_step):
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(loss_op,
                                          global_step=global_step,
                                         name='train_op')
        return train_op, optimizer

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

    def save_model(self, saver, writer, step):
        """Save model and all values of current parameters."""
        pass

    def restore_model(self, sess=None, log_dir=None):
        """Restore model from `log_dir` using tensorflow session `sess`."""
        if log_dir is None:
            log_dir = self.log_dir
        if sess is None:
            sess = self.sess

        saver = tf.train.Saver(max_to_keep=3)
        if not self.eager:
            sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring previous model from: "
                  f"{ckpt.model_checkpoint_path}")
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored.\n")
            self.global_step = tf.train.get_global_step()


            self.sess.run(tf.global_variables_initializer())

    def train(self, num_train_steps):
        """Train the model."""
        saver = tf.train.Saver(max_to_keep=3)
        initial_step = 0
        x = tf.placeholder(dtype=tf.float32,
                           shape=self.samples.shape,
                           name='x')
        loss_op, x_out, x_accept_prob = self._create_loss(self.dynamics,
                                                          x,
                                                          scale=0.1,
                                                          eps=1e-4)
        train_op, optimizer = self._create_optimizer(loss_op, step)

        self.ckptr = self._create_checkpointer(optimizer, self.dynamics,
                                               self.global_step)

        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(self.log_dir)
        time_delay = 0.
        if ckpt and ckpt.model_checkpoint_path:
            print('Restoring previous model from: '
                  f'{ckpt.model_checkpoint_path}')
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('Model restored.\n')
            self.global_step = tf.train.get_global_step()
            initial_step = self.sess.run(self.global_step)
            previous_time = self.train_times[initial_step]
            time_delay = time.time() - previous_time
        writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        try:
            #  train_op, loss, _ = self.step_fn(self.dynamics,
            #                                   self.optimizer,
            #                                   self.samples)
            learning_rate = self.learning_rate
            _samples = self.samples
            self.actions_arr.extend(self.lattice.total_action(_samples))
            self.avg_plaquettes_arr.extend(
                self.lattice.average_plaquette(_samples)
            )

            # Warmup to reduce initialization effect when timing
            #  for _ in range(1):
            #      _, _ = sess.run([train_op, loss])

            start_time = time.time()
            for step in range(initial_step, initial_step + num_train_steps):
                t1 = time.time()
                self.train_times[initial_step] = start_time - time_delay


                _, _loss, _samples, _accept_prob, _lr = self.sess.run([
                    train_op,
                    loss_op,
                    x_out,
                    x_accept_prob,
                ], feed_dict={x: _samples})

                total_actions = self.lattice.total_action(_samples)
                avg_plaquettes = self.lattice.average_plaquette(_samples)

                self.actions_arr.extend(total_actions)
                self.avg_plaquettes_arr.extend(avg_plaquettes)
                self.losses_arr.append(_loss)

                print(f"Iteration: {step}, "
                      f"loss: {_loss:^.4f}, "
                      f"x_accept: {np.mean(_accept_prob):^.4f}, "
                      f"eps: {self.dynamics.eps:^.4f}, "
                      f"avg_action: {np.mean(total_actions):^.4f}, "
                      f"avg_plaq: {np.mean(avg_plaquettes):^.4f}, "
                      f"step time: {time.time() - t1}")

        except (KeyboardInterrupt, SystemExit):
            print("\nKeyboardInterrupt detected! \n"
                  " Saving current state and exiting.\n")
            self._save_variables()
            self._save_model(saver, writer, step)
            writer.close()
            self.sess.close()

