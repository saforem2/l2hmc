"""
Augmented Hamiltonian Monte Carlo Sampler using the L2HMC algorithm, applied to
a U(1) lattice gauge theory model.

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
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from scipy.special import i0, i1 # pylint: disable=no-name-in-module

from l2hmc_eager import gauge_dynamics_eager as gde
from l2hmc_eager.neural_nets import ConvNet, GenericNet
from definitions import ROOT_DIR
from lattice.gauge_lattice import GaugeLattice, u1_plaq_exact
from utils.tf_logging import variable_summaries, get_run_num, make_run_dir
import utils.gauge_model_helpers as helpers
#  from utils.gauge_model_helpers import *


tfe = tf.contrib.eager

#  MODULE_PATH = os.path.abspath(os.path.join('..'))
#  if MODULE_PATH not in sys.path:
#      sys.path.append(MODULE_PATH)

PARAMS = {
    'time_size': 8,
    'space_size': 8,
    'link_type': 'U1',
    'dim': 2,
    'beta': 8.,
    'num_samples': 2,
    'num_steps': 5,
    'eps': 0.05,
    'loss_scale': 0.1,
    'loss_eps': 1e-4,
    'learning_rate_init': 1e-4,
    'learning_rate_decay_steps': 100,
    'learning_rate_decay_rate': 0.96,
    'train_iters': 1000,
    'record_loss_every': 50,
    'data_steps': 10,
    'save_steps': 50,
    'print_steps': 1,
    'logging_steps': 50,
    'clip_value': 100,
    'rand': False,
    'metric': 'l2',
}

##############################################################################
# Helper functions etc.
##############################################################################
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
                   loss_fn, params, global_step=None, hmc=False):
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

    #  print("Computing loss_and_grads...")
    loss, samples_out, accept_prob, grads = gde.loss_and_grads(
        dynamics=dynamics,
        x=samples,
        params=params,
        loss_fn=loss_fn,
        hmc=hmc
    )

    if not hmc:
        grads, _ = tf.clip_by_global_norm(grads, clip_value)

        #  print("Applying gradients...")
        optimizer.apply_gradients(
            zip(grads, dynamics.trainable_variables), global_step=global_step
        )
        #  print("done.")

    return loss, samples_out, accept_prob, grads

def graph_step(dynamics, samples, optimizer, loss_fn, 
               params,  global_step=None, hmc=False):
    """Perform a single training step using a compiled TensorFlow graph."""

    loss, samples_out, accept_prob, grads = gde.loss_and_grads(
        dynamics=dynamics,
        x=samples,
        params=params,
        loss_fn=loss_fn,
        hmc=hmc
    )

    clip_value = params.get('clip_value', 100)
    grads, _ = tf.clip_by_global_norm(grads, clip_value)
    train_op = optimizer.apply_gradients(
        zip(grads, dynamics.trainable_variables), global_step=global_step
    )

    return train_op, loss, samples_out, accept_prob, grads

class GaugeModel(object):
    """Wrapper class implementing L2HMC algorithm on lattice gauge models. """
    def __init__(self,
                 params=None,
                 conv_net=True,
                 hmc=False,
                 log_dir=None,
                 restore=False):

        tf.enable_resource_variables()

        if log_dir is None:
            dirs = helpers.create_log_dir('gauge_logs_graph')
        else:
            dirs = helpers.check_log_dir(log_dir)

        self.log_dir, self.info_dir, self.figs_dir = dirs

        if restore:
            self.restore_model(log_dir)
        else:
            self.summary_writer = (
                tf.contrib.summary.create_file_writer(self.log_dir)
            )
            self.params = params
            self.data = {}
            self.conv_net = conv_net
            self.hmc = hmc

            self._init_params(params, log_dir)

        self.lattice = self._create_lattice(self.params)
        self.batch_size = self.lattice.samples.shape[0]

        if self.conv_net:
            self.samples_np = np.array(self.lattice.samples, dtype=np.float32)
            #  self.samples = tf.convert_to_tensor(
            #      self.samples_np, dtype=tf.float32
            #  )
            #  self.samples = tf.convert_to_tensor(
            #      self.lattice.samples, dtype=tf.float32
            #  )
        else:
            self.samples_np = np.array(
                self.lattice.samples.reshape((self.batch_size, -1)),
                dtype=np.float32
            )
            #  self.samples = tf.convert_to_tensor(
            #      self.lattice.samples.reshape((self.batch_size, -1)),
            #      dtype=tf.float32
            #  )
        self.samples = tf.convert_to_tensor(self.samples_np, dtype=tf.float32)

        self.potential_fn = self.lattice.get_energy_function(self.samples)

        self.dynamics = self._create_dynamics(lattice=self.lattice,
                                              potential_fn=self.potential_fn,
                                              conv_net=self.conv_net,
                                              hmc=self.hmc,
                                              params=self.params)

        self.global_step = tf.train.get_or_create_global_step()
        self.global_step.assign(1)

        self.learning_rate = tf.train.exponential_decay(
            self.learning_rate_init,
            self.global_step,
            self.learning_rate_decay_steps,
            self.learning_rate_decay_rate,
            staircase=True
        )

        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        self.loss_fn = gde.compute_loss

        print(80*'#')
        print('Model parameters:')
        for key, val in self.__dict__.items():
            if isinstance(val, (int, float, str)):
                print(f'{key}: {val}\n')
        print(80*'#')
        print('\n')

    def _init_params(self, params=None, log_dir=None):
        """Parse key value pairs from params and set as class attributes."""
        if params is None:
            params = PARAMS

        for key, val in params.items():
            setattr(self, key, val)

        self.params = params

        self.train_times = {}
        self.total_actions_arr = []
        self.average_plaquettes_arr = []
        self.topological_charges_arr = []
        self.losses_arr = []
        self.steps_arr = []
        self.samples_arr = []
        self.accept_prob_arr = []
        self.step_times_arr = []

        self.data = {
            'step': 0,
            'loss': 0.,
            'step_time': 0.,
            'accept_prob': 0.,
            'eps': params.get('eps', 0.),
            'train_steps': params.get('train_steps', 1000),
            'learning_rate': params.get('learning_rate_init', 1e-4),
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
            'samples': [],
        }

        self.files = {
            'parameters_file': os.path.join(self.info_dir, 'parameters.txt'),
            'samples_file': os.path.join(self.info_dir, 'samples.npy'),
            'run_info_file': os.path.join(self.info_dir, 'run_info.txt'),
            'data_pkl_file': os.path.join(self.info_dir, 'data.pkl'),
            'parameters_pkl_file': (
                os.path.join(self.info_dir, 'parameters.pkl')
            ),
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

        helpers.write_run_parameters(self.files['parameters_file'],
                                     self.params)

    def _create_lattice(self, params=None):
        if params is None:
            params = self.params

        return GaugeLattice(time_size=params.get('time_size', 8),
                            space_size=params.get('space_size', 8),
                            dim=params.get('dim', 2),
                            beta=params.get('beta', '2.5'),
                            link_type=params.get('link_type', 'U1'),
                            num_samples=params['num_samples'],
                            rand=params['rand'])

    def _create_dynamics(self, lattice=None, conv_net=True, 
                         hmc=False, potential_fn=None, params=None):
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
                                      num_steps=params.get('num_steps', 5),
                                      eps=params.get('eps', 0.1),
                                      minus_loglikelihood_fn=potential_fn,
                                      conv_net=conv_net,
                                      hmc=hmc)

    def calc_observables(self, samples):
        """Calculate observables of interest for each sample in `samples`.
        
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

    def restore_model(self, log_dir=None):
        if log_dir is not None:
            assert os.path.isdir(log_dir), (f"log_dir: {log_dir} "
                                            "does not exist.")

            run_info_dir = os.path.join(log_dir, 'run_info')
            assert os.path.isdir(run_info_dir), ("run_info_dir: "
                                                 f"{run_info_dir} "
                                                 "does not exist.")

        else:
            log_dir = self.log_dir

        self.summary_writer = tf.contrib.summary.create_file_writer(log_dir)

        saver = tf.train.Saver(max_to_keep=3)
        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring previous model from: "
                  f"{ckpt.model_checkpoint_path}")
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Model restored.")
            self.global_step = tf.train.get_global_step()

        params_pkl_file = os.path.join(run_info_dir, 'parameters.pkl')
        with open(params_pkl_file, 'rb') as f:
            self.params = pickle.load(f)

        self._init_params(self.params)

        data_pkl_file = os.path.join(run_info_dir, 'data.pkl')
        with open(data_pkl_file, 'rb') as f:
            self.data = pickle.load(f)

        self.total_actions_arr = list(
            np.load(os.path.join(run_info_dir, 'total_actions.npy'))
        )
        self.average_plaquettes_arr = list(
            np.load(os.path.join(run_info_dir, 'average_plaquettes.npy'))
        )
        self.topological_charges_arr = list(
            np.load(os.path.join(run_info_dir, 'topological_charges.npy'))
        )

    def save_model(self, saver, writer, step):
        """Save current graph and all quantities used in training."""
        ckpt_file = os.path.join(self.log_dir, 'model.ckpt')
        saver.save(self.sess, ckpt_file, global_step=step)
        print(f"Model saved to: {ckpt_file}.")

        np.save(self.files['average_plaquettes_file'],
                self.data['average_plaquettes_arr'])
        np.save(self.files['total_actions_file'],
                self.data['total_actions_arr'])
        np.save(self.files['topological_charges_file'],
                self.data['topological_charges_arr'])

        np.save(self.files['samples_file'], self.data['samples'])

        with open(self.files['data_pkl_file'], 'wb') as f:
            pickle.dump(self.data, f)
        with open(self.files['parameters_pkl_file'], 'wb') as f:
            pickle.dump(self.params, f)

        writer.flush()

    def train(self, num_train_steps, keep_samples=False, config=None):
        """Train the model."""
        saver = tf.train.Saver(max_to_keep=3)

        samples_np = self.samples_np
        samples = self.samples
        initial_step = 0

        print("Building graph...")
        x = tf.placeholder(tf.float32, shape=self.samples.shape)

        loss, _, _ = self.loss_fn(self.dynamics, x, self.params)


        train_op, loss, samples_out, accept_prob, _ = graph_step(
            dynamics=self.dynamics,
            samples=x,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            params=self.params,
            global_step=self.global_step,
            hmc=self.hmc
        )
        print('done.\n')

        if config is None:
            config = tf.ConfigProto()

        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(self.log_dir)
        time_delay = 0.
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring previous model from: "
                  f"{ckpt.model_checkpoint_path}")
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Model restored.\n")
            self.global_step = tf.train.get_global_step()
            initial_step = self.sess.run(self.global_step)
            previous_time = self.train_times[initial_step]
            time_delay = time.time() - previous_time

        writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        start_time = time.time()
        self.train_times[initial_step] = start_time - time_delay

        try:
            print(helpers.data_header())
            helpers.write_run_data(self.files['run_info_file'], self.data, 'w')

            print("Starting training...\n")
            for step in range(initial_step, initial_step + num_train_steps):
                start_step_time = time.time()

                _, loss_np, samples_np, accept_prob_np, eps_np = self.sess.run(
                    [train_op, loss, samples_out,
                     accept_prob, self.dynamics.eps],
                    feed_dict={x: samples_np}
                )

                self.data['step'] = step
                self.data['loss'] = loss_np
                self.data['accept_prob'] = accept_prob_np
                self.data['eps'] = eps_np

                self.losses_arr.append(loss_np)

                if keep_samples:
                    self.data['samples'] = samples_np

                if (step + 1) % self.save_steps == 0:
                    self.train_times[step+1] = time.time() - time_delay
                    print(helpers.data_header())
                    self.save_model(saver, writer, step)
                    helpers.write_run_data(self.files['run_info_file'],
                                           self.data,
                                           'a')

                if (step + 1) % self.data_steps == 0:
                    observables = self.calc_observables(samples_np)
                    total_actions, avg_plaquettes, top_charges = observables

                    _actions_arr = []
                    _plaqs_arr = []
                    _charges_arr = []
                    for i in range(self.batch_size):
                        action_np, plaqs_np, charges_np = self.sess.run([
                            total_actions[i],
                            avg_plaquettes[i],
                            top_charges[i]
                        ])
                        _actions_arr.append(action_np)
                        _plaqs_arr.append(plaqs_np)
                        _charges_arr.append(charges_np)

                    self._update_data(_actions_arr,
                                      _plaqs_arr,
                                      _charges_arr)

                    helpers.print_run_data(self.data)

                    helpers.write_run_data(self.files['run_info_file'],
                                           self.data,
                                           'a')

                if (step + 1) % self.logging_steps == 0:
                    write_summaries(self.summary_writer, self.data)

                self.data['step_time'] = time.time() - start_step_time

            print("Training complete!")
            sys.stdout.flush()
            self.save_model(saver, writer, step)
            helpers.write_run_data(self.files['run_info_file'],
                                   self.data,
                                   'a')

            writer.close()
            self.sess.close()

        except (KeyboardInterrupt, SystemExit):
            print("\nKeyboardInterrupt detected! \n"
                  "Saving current state and exiting.\n")

            self.save_model(saver, writer, step)
            writer.close()
            self.sess.close()



def main(args):
    """Main method for creating/training U(1) gauge model from command line."""
    #  tf.enable_eager_execution()

    params = PARAMS  # use default parameters if no command line args passed

    params['time_size'] = args.time_size
    params['space_size'] = args.space_size
    params['link_type'] = args.link_type
    params['dim'] = args.dim
    params['beta'] = args.beta
    params['num_samples'] = args.num_samples
    params['num_steps'] = args.num_steps
    params['eps'] = args.eps
    params['loss_scale'] = args.loss_scale
    params['loss_eps'] = args.loss_eps
    params['learning_rate_init'] = args.learning_rate_init
    params['learning_rate_decay_rate'] = args.learning_rate_decay_rate
    params['train_steps'] = args.train_steps
    params['record_loss_every'] = args.record_loss_every
    params['data_steps'] = args.data_steps
    params['save_steps'] = args.save_steps
    params['print_steps'] = args.print_steps
    params['logging_steps'] = args.logging_steps
    params['clip_value'] = args.clip_value
    params['rand'] = args.rand
    params['metric'] = args.metric

    model = GaugeModel(params=params,
                       conv_net=args.conv_net,
                       hmc=args.hmc,
                       log_dir=args.log_dir,
                       restore=args.restore,
                       defun=args.defun)

    #  import pdb
    #  pdb.set_trace()

    #  observables = model.calc_observables(model.samples,
    #                                       _print=True,
    #                                       _write=True)


    start_time_str = time.strftime("%a, %d %b %Y %H:%M:%S",
                                   time.gmtime(time.time()))
    print(f"Training began at: {start_time_str}.")
    model.train(args.train_steps, args.keep_samples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('L2HMC model using Mixture of Gaussians '
                     'for target distribution')
    )
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

    parser.add_argument("-b", "--beta", type=float,
                        required=False, dest="beta",
                        help=("Beta (inverse coupling constant) used in "
                              "gauge model. (Default: 8.)"))

    parser.add_argument("-N", "--num_samples", type=int,
                        default=2, required=False, dest="num_samples",
                        help=("Number of samples (batch size) to use for "
                              "training. (Default: 2)"))

    parser.add_argument("-n", "--num_steps", type=int,
                        default=5, required=False, dest="num_steps",
                        help=("Number of leapfrog steps to use in (augmented) "
                              "HMC sampler. (Default: 5)"))

    parser.add_argument("--eps", type=float, default=0.1,
                        required=False, dest="eps",
                        help=("Step size to use in leapfrog integrator. "
                              "(Default: 0.1)"))

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

    parser.add_argument("--defun", action="store_false",
                        required=False, dest="defun",
                        help=("Whether or not to use `tfe.defun` to compile "
                              "functions as a graph to speed up computations. "
                              "(Default: True)"))

    parser.add_argument("--loss_scale", type=float, default=0.1,
                        required=False, dest="loss_scale",
                        help=("Scaling factor to be used in loss function. "
                              "(lambda in Eq. 7 of paper). (Default: 0.1)"))

    parser.add_argument("--loss_eps", type=float, default=1e-4,
                        required=False, dest="loss_eps",
                        help=("Small value added at the end of Eq. 7 in the "
                              "paper used to prevent division by zero errors. "
                              "(Default: 1e-4)"))

    parser.add_argument("--learning_rate_init", type=float, default=1e-4,
                        required=False, dest="learning_rate_init",
                        help=("Initial value of learning rate. "
                              "(Deafult: 1e-4)"))

    parser.add_argument("--learning_rate_decay_rate", type=float, default=0.96,
                        required=False, dest="learning_rate_decay_rate",
                        help=("Learning rate decay rate to be used during "
                              "training. (Default: 0.96)"))

    parser.add_argument("--learning_rate_decay_steps", type=int, default=100,
                        required=False, dest="learning_rate_decay_steps",
                        help=("Number of steps after which to decay learning "
                              "rate. (Default: 100)"))

    parser.add_argument("--train_steps", type=int, default=1000,
                        required=False, dest="train_steps",
                        help=("Number of training steps to perform. "
                              "(Default: 1000)"))

    parser.add_argument("--record_loss_every", type=int, default=20,
                        required=False, dest="record_loss_every",
                        help=("Number of steps after which to record loss "
                              "value (Default: 20)"))
    parser.add_argument("--data_steps", type=int, default=10,
                        required=False, dest="data_steps",
                        help=("Number of steps after which to compute and "
                              "record data (including physical observables) "
                              "(Default: 10)"))

    parser.add_argument("--save_steps", type=int, default=50,
                        required=False, dest="save_steps",
                        help=("Number of steps after which to save the model "
                              "and current values of all parameters. "
                              "(Default: 50)"))

    parser.add_argument("--print_steps", type=int, default=1,
                        required=False, dest="print_steps",
                        help=("Number of steps after which to print "
                              "information about current run. (Default: 1)"))

    parser.add_argument("--logging_steps", type=int, default=50,
                        required=False, dest="logging_steps",
                        help=("Number of steps after which to write logs for "
                              "tensorboard. (Default: 50)"))

    parser.add_argument("--clip_value", type=int, default=100,
                        required=False, dest="clip_value",
                        help=("Clip value, used for clipping value of "
                              "gradients by global norm. (Default: 100)"))

    parser.add_argument("--rand", action="store_true",
                        required=False, dest="rand",
                        help=("Start lattice from randomized initial "
                              "configuration. (Default: False)"))

    parser.add_argument("--metric", type=str, default="l2",
                        required=False, dest="metric",
                        help=("Metric to use in loss function. "
                              "(Default: `l2`, choices: [`l2`, `l1`, `cos`])"))

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

    parser.add_argument("--keep_samples", action="store_true",
                        required=False, dest="keep_samples",
                        help=("Keep samples output from L2HMC algorithm "
                              "during training iterations. (Default: False)"))

    args = parser.parse_args()

    main(args)
