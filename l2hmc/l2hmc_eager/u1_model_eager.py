"""
L2HMC Algorithm as applied to the U(1) lattice gauge theory model.

NOTE: We are using eager execution.
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

from scipy.special import i0, i1

#  from .dynamics_eager  import Dynamics
#  import .dynamics_eager as dynamics

#  import .gauge_dynamics_eager as gde
from . import gauge_dynamics_eager as gde
#  from .gauge_dynamics_eager import GaugeDynamicsEager, \
#          compute_loss, loss_and_grads

from .neural_nets import ConvNet, GenericNet

from definitions import ROOT_DIR

from lattice.gauge_lattice import GaugeLattice, u1_plaq_exact

from utils.tf_logging import variable_summaries, get_run_num, make_run_dir

if not tf.executing_eagerly():
    tf.enable_eager_execution()

tfe = tf.contrib.eager

#  MODULE_PATH = os.path.abspath(os.path.join('..'))
#  if MODULE_PATH not in sys.path:
#      sys.path.append(MODULE_PATH)

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
    'metric': 'l2',
    'conv_net': True,
    'hmc': False
}

##############################################################################
# Helper functions etc.
##############################################################################


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


def create_log_dir(base_name='gauge_logs'):
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


def write_run_data(file_path, data, write_mode='a'):
    """Write run `data` to human-readable file at `file_path`."""
    with open(file_path, write_mode) as f:
        f.write('\n')
        info_str = (f"step: {data['step']:<3g} "
                    f"loss: {data['loss']:^6.5g} "
                    f" time/step: {data['step_time']:^6.5g} "
                    f" accept: {data['accept_prob']:^6.5g} "
                    f" eps: {data['eps']:^6.5g} "
                    f" avg_action: {data['avg_action']:^6.5g} "
                    f" avg_top_charge: {data['avg_top_charge']:^6.5g} "
                    f" avg_plaq: {data['avg_plaq']:^6.5g} \n")
        f.write(info_str)
        f.write('\n')
        f.write('avg_plaquettes: {}\n'.format(data['avg_plaquettes']))
        f.write('topological_charges: {}\n'.format(data['top_charges']))
        f.write('total_actions: {}\n'.format(data['total_actions']))
        f.write((len(info_str) + 1)*'-')


def write_run_parameters(file_path, parameters):
    """Write `parameters` to human-readable file at `file_path`."""
    with open(file_path, 'w') as f:
        f.write('Parameters:\n')
        f.write(80 * '-' + '\n')
        for key, val in parameters.items():
            f.write(f'{key}: {val}\n')
        f.write(80*'=')
        f.write('\n')


def print_run_data(data):
    """Print information about current run to std out."""
    print("\n")
    print(f"step: {data['step']:<3g} "
          f"loss: {data['loss']:^6.5g} "
          f" time/step: {data['step_time']:^6.5g} "
          f" accept: {np.mean(data['accept_prob']):^6.5g} "
          f" eps: {data['eps']:^6.5g} "
          f" avg_action: {data['avg_action']:^6.5g} "
          f" avg_top_charge: {data['avg_top_charge']:^6.5g} "
          f" avg_plaq: {data['avg_plaq']:^6.5g} \n")


def save_run_data(checkpointer, log_dir, files, data):
    """Save run `data` to `files` in `log_dir` using `checkpointer`"""
    saved_path = checkpointer.save(file_prefix=os.path.join(log_dir, "ckpt"))
    print(f"Saved checkpoint to: {saved_path}")
    np.save(files['average_plaquettes_file'], data['average_plaquettes_arr'])
    np.save(files['total_actions_file'], data['total_actions_arr'])
    np.save(files['topological_charges_file'], data['topological_charges_arr'])
    np.save(files['samples_file'], data['samples'])
    print('avg_plaquettes: {}\n'.format(data['avg_plaquettes']))


def write_summaries(summary_writer, data):
    """Write `summaries` using `summary_writer` for use in TensorBoard."""
    with summary_writer.as_default():
        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar("Training_loss", data['loss'],
                                      step=data['step'])
            tf.contrib.summary.scalar("avg_plaquette", data['avg_plaq'],
                                      step=data['step'])
            tf.contrib.summary.scalar("avg_action", data['avg_action'],
                                      step=data['step'])
            tf.contrib.summary.scalar("avg_top_charge", data['avg_top_charge'],
                                      step=data['step'])


# pylint: disable=too-many-locals
def train_one_iter(dynamics, samples, optimizer, 
                   loss_fn, params, global_step=None):
    """
    Peform a single training step for the `dynamics` engine.

    Args:
        dynamics: Main dynamics engine responsible for implementing L2HMC alg.
        samples (tf.Tensor): Batch of training samples.
        optimizer: Tensorflow optimizer (e.g. tf.train.AdamOptimizer)
        loss_fn (function): Function that computes loss from network output.
        params (dict): Dictionary of parameters. 
            We are interested in:
                * params['loss_eps']: Small constant for numerical stability.
                * params['metric']: Metric used in calculating loss. 
                * params['loss_scale']: Scaling factor (lambda) used in
                    calculating the loss.
                * params['clip_value']: Value used for clipping the gradients
                    by global norm during training.
        global_step (int): Current value of global step.

    Returns:
        loss (float): `loss` value output from network.
        samples_out: Output from Metropolis Hastings accept/reject step.
        accept_prob (float): Probability that proposed states were accepted.
        gradients: Resulting gradient values from this training step.
    """
    clip_value = params.get('clip_value', 10)

    loss, samples_out, accept_prob, grads = gde.loss_and_grads(
        dynamics=dynamics,
        x=samples,
        params=params,
        loss_fn=loss_fn
    )

    gradients, _ = tf.clip_by_global_norm(grads, clip_value)

    optimizer.apply_gradients(
        zip(grads, dynamics.trainable_variables), global_step=global_step
    )

    return loss, samples_out, accept_prob, gradients


###############################################################################
# `GaugeeModelEager` class definition below.
###############################################################################

# pylint: disable=too-many-instance-attributes
class GaugeModelEager(object):
    """
    Wrapper class implementing L2HMC algorithm on lattice gauge models.

    NOTE:
        * Throughout the documentation I will use:
            - `T`: To represent the `time_size` of the lattice.
            - `X`: To represent the `space_size` of the lattice.
            - `D`: To represent the `dim` (dimension) of the lattice.
            - `N`: The number of `samples` contained in each `mini-batch`,
                where each `sample` is a tensor of link variables. 

        * For 2D U(1), the lattice's link variables will be a tensor of shape:
            (T, X, 2)

        * Because of this, a batch of `samples` will have shape:
            (N, T, X, 2)
    """
    def __init__(self, 
                 params=None,
                 log_dir=None, 
                 restore=False,
                 use_defun=True):
        """Initialization method."""
        _t0 = time.time()

        self.params = params
        self.data = {}
        self._defun = use_defun

        self.conv_net = params.get('conv_net', True)

        self._init_params(params, log_dir)

        print('Creating lattice...')
        self.lattice = self._create_lattice(self.params)

        # batch size is equivalent to the number of samples (call it `N`)
        self.batch_size = self.lattice.samples.shape[0]
        # reshape lattice.links from (N, T, X, 2) --> (N, T * X * 2)
        self.samples = tf.convert_to_tensor(
            self.lattice.samples.reshape((self.batch_size, -1)),
            dtype=tf.float32
        )

        self.potential_fn = self.lattice.get_energy_function(self.samples)
        print('done.')

        print("Creating dynamics...")
        self.dynamics = self._create_dynamics(lattice=self.lattice,
                                              potential_fn=self.potential_fn,
                                              params=self.params)
        print('done.')

        self.global_step = tf.train.get_or_create_global_step()
        self.global_step.assign(1)

        self.learning_rate = tf.train.exponential_decay(
            self.learning_rate_init,
            self.global_step,
            self.learning_rate_decay_steps,
            self.learning_rate_decay_rate,
            staircase=True
        )
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        self.checkpointer = tf.train.Checkpoint(optimizer=self.optimizer,
                                                dynamics=self.dynamics,
                                                global_step=self.global_step)

        if use_defun:  # Use `tfe.defun` to boost performance
            self.loss_fn = tfe.defun(gde.compute_loss)
        else:
            self.loss_fn = gde.compute_loss

        if log_dir is None:
            self.summary_writer = (  # using newly created `self.log_dir`
                tf.contrib.summary.create_file_writer(self.log_dir)
            )

        else:
            if restore:
                self._restore_model(log_dir)

        print(f"total initialization time: {time.time() - _t0}\n")

    def _init_params(self, params=None, log_dir=None):
        """
        Parse key value pairs from params and set as as class attributes.

        Args:
            params: Dictionary containing model parameters.
        """
        if params is None:
            params = PARAMS

        for key, val in params.items():
            setattr(self, key, val)

        self.params = params

        if log_dir is None:
            dirs = create_log_dir('gauge_logs_eager')
        else:
            dirs = check_log_dir(log_dir)

        self.log_dir, self.info_dir, self.figs_dir = dirs

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
            'step_time': 0.,
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
            'samples': None,
        }

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

        write_run_parameters(self.files['parameters_file'], self.params)

    def _create_lattice(self, params=None):
        """Delegate creation logic to private methods."""
        if params is None:
            params = self.params

        return GaugeLattice(time_size=params.get('time_size', 8),
                            space_size=params.get('space_size', 8),
                            dim=params.get('dim', 2),
                            beta=params.get('beta', '2.5'),
                            link_type=params.get('link_type', 'U1'),
                            num_samples=params['num_samples'],
                            rand=params['rand'])

    def _create_dynamics(self, lattice=None, potential_fn=None, params=None):
        """
        Delegate creation logic to private methods.

        Args:
            lattice (:obj:GaugeLattice, optional): Lattice object containing
                link variables of interest.
            potential_fn (function, optional): The minus-loglikelihood function
                describing the target distribution.
            params (dict, optional): Dictionary containing parameters for
                initializing the `dynamics` object.
        Returns:
            `GaugeDynamicsEager` object.
        """
        if lattice is None:
            lattice = self.lattice
        if potential_fn is None:
            potential_fn = self.potential_fn
        if params is None:
            params = self.params

        return gde.GaugeDynamicsEager(lattice=lattice,
                                      n_steps=params.get('n_steps', 10),
                                      eps=params.get('eps', 0.1),
                                      minus_loglikelihood_fn=potential_fn,
                                      conv_net=params.get('conv_net', True),
                                      hmc=params['hmc'])

    def calc_observables(self, samples, _print=True, _write=True):
        """
        Calculate observables of interest for each sample in `samples`.
        
         NOTE: `observables` is an array containing `total_actions`,
         `avg_plaquettes`, and `top_charges` for each sample in batch, with
         one sample per row, i.e. for `M` observations:

             observables = [[total_actions_1, avg_plaqs_1, top_charges_1],
                            [total_actions_2, avg_plaqs_2, top_charges_2],
                            [     ...            ...            ...     ],
                            [total_actions_M, avg_plaqs_M, top_charges_M],
         """
        observables = np.array(self.lattice.calc_plaq_observables(samples)).T

        if tf.executing_eagerly():
            total_actions, avg_plaquettes, top_charges = observables
        else:
            observables = observables.reshape((-1, 3))
            total_actions = observables[:, 0]
            avg_plaquettes = observables[:, 1]
            top_charges = observables[:, 2]

        self._update_data(total_actions, avg_plaquettes, top_charges)

        if _print:
            print_run_data(self.data)

        if _write:
            if self.global_step.numpy() > 1:
                write_mode = 'a'
            else:
                write_mode = 'w'

            write_run_data(self.files['run_info_file'], self.data, write_mode)

        return total_actions, avg_plaquettes, top_charges

    def _update_data(self, total_actions, avg_plaquettes, top_charges):
        """Update `self.data` with new values of physical observables."""
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

    def _restore_model(self, log_dir):
        assert os.path.isdir(log_dir), (f"log_dir: {log_dir} does not exist.")

        self.summary_writer = tf.contrib.summary.create_file_writer(log_dir)

        latest_path = tf.train.latest_checkpoint(log_dir)
        self.checkpointer.restore(latest_path)
        print("Restored latest checkpoint from:\"{}\"".format(latest_path))
        sys.stdout.flush()

    def train(self, num_train_steps, keep_samples=False):
        """Run the training procedure of the L2HMC algorithm."""
        start_step = self.global_step.numpy()
        #  start_time = time.time()

        samples = self.samples

        for step in range(start_step, num_train_steps):
            start_step_time = time.time()

            loss, samples, accept_prob, _ = train_one_iter(
                dynamics=self.dynamics,
                samples=samples,
                optimizer=self.optimizer,
                loss_fn=self.loss_fn,
                global_step=self.global_step,
                params=self.params
            )

            self.data['step'] = step
            self.data['loss'] = loss.numpy()
            self.data['step_time'] = time.time() - start_step_time
            self.data['accept_prob'] = accept_prob.numpy().mean()
            self.data['eps'] = self.dynamics.eps.numpy()

            if keep_samples:
                self.data['samples'] = samples.numpy()

            # pylint: disable=bad-option-value
            _ = self.calc_observables(samples,
                                      _print=True,
                                      _write=True)

            if step % self.logging_steps == 0:
                write_summaries(self.summary_writer, self.data)

            if step % self.save_steps == 0:
                save_run_data(self.checkpointer,
                              self.log_dir,
                              self.files,
                              self.data)

        print("Training complete.")
        sys.stdout.flush()

        save_run_data(self.checkpointer, self.log_dir, self.files, self.data)
