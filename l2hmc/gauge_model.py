"""
Augmented Hamiltonian Monte Carlo Sampler using the L2HMC algorithm, applied
to a U(1) lattice gauge theory model.

==============================================================================
* TODO:
-----------------------------------------------------------------------------
    (!!)  *

==============================================================================
* COMPLETED:
-----------------------------------------------------------------------------
    (x)  * Implement model with pair of Gaussians both separated along
         a single axis, and separated diagonally across all
         dimensions.
    (x)  * Look at replacing self.params['...'] with setattr for
         initalization.
    (x)  * Go back to 2D case and look at different starting
         temperatures
    (x)  * Make trajectory length go with root T, go with higher
         temperature
    (x)  * In 2D start with higher initial temp to get around 50%
         acceptance rate.
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
import utils.gauge_model_helpers as helpers

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


from l2hmc_eager import gauge_dynamics_eager as gde
#  from l2hmc_eager.neural_nets import ConvNet, GenericNet
#  from definitions import ROOT_DIR
from lattice.gauge_lattice import GaugeLattice, u1_plaq_exact
#  from utils.tf_logging import variable_summaries, get_run_num, make_run_dir
from utils.plot_helper import plot_broken_xaxis
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
    """Wrapper class implementing L2HMC algorithm on lattice gauge models."""
    def __init__(self,
                 params=None,
                 sess=None,
                 config=None,
                 conv_net=True,
                 hmc=False,
                 log_dir=None,
                 restore=False,
                 eps_trainable=True):
        """Initialization method."""
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
        self.params = params
        self.data = {}
        self.conv_net = conv_net
        self.hmc = hmc

        self._init_params(params)

        self.lattice = GaugeLattice(time_size=self.time_size,
                                    space_size=self.space_size,
                                    dim=self.dim,
                                    beta=self.beta,
                                    link_type=self.link_type,
                                    num_samples=self.num_samples,
                                    rand=self.rand)

        self.batch_size = self.lattice.samples.shape[0]

        self.samples_np = np.array(self.lattice.samples, dtype=np.float32)
        self.samples = tf.convert_to_tensor(
            self.lattice.samples, dtype=tf.float32
        )
        self.x = tf.placeholder(dtype=tf.float32,
                                shape=self.samples.shape,
                                name='x')

        self.potential_fn = self.lattice.get_energy_function(self.samples)

        self.dynamics = gde.GaugeDynamicsEager(
            lattice=self.lattice,
            num_steps=self.num_steps,
            eps=self.eps,
            minus_loglikelihood_fn=self.potential_fn,
            conv_net=self.conv_net,
            hmc=self.hmc,
            eps_trainable=eps_trainable
        )

        if restore:
            self._restore_model(self.log_dir)

        else:
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

            self.build_graph()
            if sess is None:
                self.sess = tf.Session(config=config)
            else:
                self.sess = sess

    def _init_params(self, params=None):
        """Parse key value pairs from params and set as class attributes."""
        if params is None:
            params = PARAMS

        for key, val in params.items():
            setattr(self, key, val)

        self.params = params

        #  self.train_times = {}
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
            'run_info_file': os.path.join(self.info_dir, 'run_info.txt'),
            'data_pkl_file': os.path.join(self.info_dir, 'data.pkl'),
            'samples_pkl_file': os.path.join(self.info_dir, 'samples.pkl'),
            'samples_history_file': (
                os.path.join(self.info_dir, 'samples_history.pkl')
            ),
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
        self._params = {}
        for key, val in self.__dict__.items():
            if isinstance(val, (int, float, str)):
                self._params[key] = val
            if isinstance(val, (int, float, str)):
                self._params[key] = val

        self._write_run_parameters(_print=True)

    def calc_observables(self, samples, update=True):
        """Calculate observables of interest for each sample in `samples`.
        
         NOTE: 
             `observables` is an array containing `total_actions`,
             `avg_plaquettes`, and `top_charges` for each sample in batch, with
             one sample per row, i.e. for `M` observations:
                 observables = [[total_actions_1, avg_plaqs_1, top_charges_1],
                                [total_actions_2, avg_plaqs_2, top_charges_2],
                                [     ...            ...            ...     ],
                                [total_actions_M, avg_plaqs_M, top_charges_M],
        """
        with tf.name_scope('observables'):
            observables = self.lattice.calc_plaq_observables(samples)
            _actions, _plaqs, _charges = observables
            #  observables = np.array(observables).reshape((-1, 3))

        #  if tf.executing_eagerly():
        #      total_actions, avg_plaquettes, top_charges = observables
        #  else:
        #      #  observables = observables.reshape((-1, 3))
        #      total_actions = observables[:, 0]
        #      avg_plaquettes = observables[:, 1]
        #      top_charges = observables[:, 2]

        if update:
            self._update_data(_actions, _plaqs, _charges)

        return _actions, _plaqs, _charges

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
        """Restore model from previous run contained in `log_dir`."""
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

        total_actions_arr = np.load(os.path.join(run_info_dir,
                                                 'total_actions.npy'))
        average_plaquettes_arr = np.load(
            os.path.join(run_info_dir, 'average_plaquettes.npy')
        )
        topological_charges_arr = np.load(
            os.path.join(run_info_dir, 'topological_charges.npy')
        )
        self._update_data(total_actions_arr,
                          average_plaquettes_arr,
                          topological_charges_arr)

        self.global_step = tf.train.get_or_create_global_step()
        self.global_step.assign(1)

        self.global_step = tf.train.get_or_create_global_step()
        self.learning_rate = tf.train.exponential_decay(
            self.learning_rate_init,
            self.global_step,
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
            saver = tf.train.Saver(max_to_keep=3)
            self.sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("Restoring previous model from: "
                      f"{ckpt.model_checkpoint_path}")
                saver.restore(self.sess, ckpt.model_checkpoint_path)
                print("Model restored.")
                self.global_step = tf.train.get_global_step()
        else:
            latest_path = tf.train.latest_checkpoint(log_dir)
            self.checkpointer.restore(latest_path)
            print("Restored latest checkpoint from:\"{}\"".format(latest_path))

        #  _, _, _, self.samples = self.dynamics.apply_transition(self.samples)

        if not self.hmc:
            if tf.executing_eagerly():
                self.dynamics.position_fn.load_weights(
                    os.path.join(self.log_dir, 'position_model_weights.h5')
                )

                self.dynamics.momentum_fn.load_weights(
                    os.path.join(self.log_dir, 'momentum_model_weights.h5')
                )
        sys.stdout.flush()

    def _save_model(self, samples=None, saver=None, writer=None, step=None):
        """Save run `data` to `files` in `log_dir` using `checkpointer`"""
        if samples is None:
            samples = self.samples

        if not tf.executing_eagerly():
            ckpt_file = os.path.join(self.log_dir, 'model.ckpt')
            print(f'Saving checkpoint to: {ckpt_file}\n')
            saver.save(self.sess, ckpt_file, global_step=step)
            writer.flush()
        else:
            saved_path = self.checkpointer.save(
                file_prefix=os.path.join(self.log_dir, 'ckpt')
            )
            print('\n')
            print(f"Saved checkpoint to: {saved_path}")
            print('\n')

        if not self.hmc:
            if tf.executing_eagerly():
                self.dynamics.position_fn.save_weights(
                    os.path.join(self.log_dir, 'position_model_weights.h5')
                )
                self.dynamics.momentum_fn.save_weights(
                    os.path.join(self.log_dir, 'momentum_model_weights.h5')
                )

        np.save(self.files['average_plaquettes_file'],
                self.data['average_plaquettes_arr'])
        np.save(self.files['total_actions_file'],
                self.data['total_actions_arr'])
        np.save(self.files['topological_charges_file'],
                self.data['topological_charges_arr'])

        with open(self.files['data_pkl_file'], 'wb') as f:
            pickle.dump(self.data, f)
        with open(self.files['parameters_pkl_file'], 'wb') as f:
            pickle.dump(self.params, f)
        with open(self.files['samples_pkl_file'], 'wb') as f:
            pickle.dump(samples, f)



        writer.flush()

    def _write_run_parameters(self, _print=False):
        """Write model parameters out to human readable .txt file."""
        if _print:
            for key, val in self._params.items():
                print(f'{key}: {val}')

        with open(self.files['parameters_file'], 'w') as f:
            f.write('Parameters:\n')
            f.write(80 * '-' + '\n')
            for key, val in self._params.items():
                f.write(f'{key}: {val}\n')
            #  for key, val in parameters.items():
            #      f.write(f'{key}: {val}\n')
            f.write(80*'=')
            f.write('\n')

    def _create_summaries(self):
        """"Create summary objects for logging in TensorBoard."""
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss_op)
            #  tf.summary.histogram('histogram_loss', self.loss_op)
            #  variable_summaries(self.loss)
            self.summary_op = tf.summary.merge_all()

    def _create_optimizer(self):
        """Create optimizer to use during training."""
        with tf.name_scope('train'):
            if not self.hmc:
                clip_value = self.params['clip_value']
                grads, _ = tf.clip_by_global_norm(
                    tf.gradients(self.loss_op,
                                 self.dynamics.trainable_variables),
                    clip_value
                )
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate
                )
                self.train_op = optimizer.apply_gradients(
                    zip(grads, self.dynamics.trainable_variables),
                    global_step=self.global_step,
                    name='train_op'
                )
            else:
                self.train_op = tf.no_op(name='train_op')  # dummy operation

    def _create_loss(self):
        """Define loss function to minimize during training."""
        scale = self.params['loss_scale']
        with tf.name_scope('loss'):
            #  self.z = tf.random_normal(tf.shape(self.x), name='z')
            #  z = tf.random_normal(tf.shape(self.x))  # Auxiliary variable
            _x, _, self.px, self.x_out = self.dynamics.apply_transition(self.x)
            #  outputs_z = self.dynamics.apply_transition(z)

            #  z_proposed, _, pz, z_out = outputs_z

                    idx_top = (tl + bl, k + tr - tl, tr + br)
            _x = tf.mod(_x, 2*np.pi)
            self.x_out = tf.mod(self.x_out, 2*np.pi)

            #  z_proposed = tf.mod(z_proposed, 2*np.pi)
            #  z_out = tf.mod(z_out, 2*np.pi)

            #  outputs = self.dynamics.apply_transition(z)
            #  z_proposed, _, pz, z_out = outputs

            #  self.loss_op = tf.Variable(0., trainable=False, name='loss')

            # Squared jump distance
            x_loss = ((tf.reduce_sum(
                tf.square(self.x - _x),
                axis=self.dynamics.axes,
                name='x_loss'
            ) * self.px) + 1e-4)

            #  z_loss = ((tf.reduce_sum(
            #      tf.square(z - z_proposed),
            #      axis=self.dynamics.axes,
            #      name='z_loss'
            #  ) * pz) + 1e-4)

            #  self.loss_op = tf.reduce_mean(
            #      (1. / x_loss + 1. / z_loss) * scale
            #      - (x_loss - z_loss) / scale, axis=0
            #  )
            self.loss_op = tf.reduce_mean(scale / x_loss - x_loss / scale,
                                           name='loss')

    def build_graph(self):
        """Build graph for TensorFlow."""
        #  start_time_str = time.strftime("%a, %d %b %Y %H:%M:%S",
        #                                 time.ctime(time.time()))
        str0 = f"Building graph... (started at: {time.ctime()})"
        str1 = "  Creating loss..."

        print(str0)
        print(str1)
        with open(self.files['run_info_file'], 'a') as f:
            f.write(str0 + '\n')
            f.write(str1 + '\n')

        t0 = time.time()
        self._create_loss()
        t1 = time.time()

        str2 = f"    took: {t1 - t0} seconds."
        str3 = "  Creating optimizer..."

        print(str2)
        print(str3)
        with open(self.files['run_info_file'], 'a') as f:
            f.write(str2 + '\n')
            f.write(str3 + '\n')

        self._create_optimizer()
        t2 = time.time()

        str4 = f"    took: {t2 - t1} seconds."
        str5 = "  Creating summaries..."
        print(str4)
        print(str5)
        with open(self.files['run_info_file'], 'a') as f:
            f.write(str4 + '\n')
            f.write(str5 + '\n')

        self._create_summaries()

        str6 = f"    took: {time.time() - t2} seconds."
        str7 = "done."
        str8 = f"Time to build graph: {time.time() - t0} seconds."
        print(str6)
        print(str7)
        print(str8)
        with open(self.files['run_info_file'], 'a') as f:
            f.write(str6 + '\n')
            f.write(str7 + '\n')
            f.write(str8 + '\n')
            f.write(80 * '-' + '\n')

    #pylint: disable=too-many-statements
    def train(self, num_train_steps, kill_sess=True):
        """Train the model."""
        saver = tf.train.Saver(max_to_keep=3)
        initial_step = 0
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
            #  previous_time = self.train_times[initial_step-1]
            #  time_delay = time.time() - previous_time

        writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        start_time = time.time()
        #  self.train_times[initial_step] = start_time - time_delay

        # Move attribute look ups outside loop to improve performance
        loss_op = self.loss_op
        train_op = self.train_op
        summary_op = self.summary_op
        x_out = self.x_out
        px = self.px
        learning_rate = self.learning_rate
        dynamics = self.dynamics
        samples_np = self.samples_np
        self.sess.graph.finalize()
        try:
            print(helpers.data_header())
            helpers.write_run_data(self.files['run_info_file'], self.data,
                                   header=True)
            for step in range(initial_step, initial_step + num_train_steps):
                start_step_time = time.time()

                _, loss_np, samples_np, px_np, lr_np, eps_np = self.sess.run([
                    train_op,
                    loss_op,
                    x_out,
                    px,
                    learning_rate,
                    dynamics.eps
                ], feed_dict={self.x: samples_np})

                self.data['step'] = step
                self.data['loss'] = loss_np
                self.data['accept_prob'] = px_np
                self.data['eps'] = eps_np
                self.data['step_time'] = (
                    (time.time() - start_step_time)
                    / (self.num_steps * self.batch_size)
                )

                helpers.print_run_data(self.data)
                helpers.write_run_data(self.files['run_info_file'], self.data)

                if (step + 1) % self.save_steps == 0:
                    #  self.train_times[step+1] = time.time() - time_delay
                    tt = time.time()
                    self._save_model(samples=samples_np,
                                     saver=saver,
                                     writer=writer,
                                     step=step)
                    helpers.write_run_data(self.files['run_info_file'],
                                           self.data)
                    save_str = (
                        f"Time to complete saving: {time.time() - tt:^6.4g}\n"
                    )
                    print(save_str)
                    print(helpers.data_header())

                if (step + 1) % self.logging_steps == 0:
                    tt = time.time()
                    summary_str = self.sess.run(summary_op,
                                                feed_dict={self.x: samples_np})
                    writer.add_summary(summary_str, global_step=step)
                    writer.flush()
                    log_str = (
                        f"Time to complete logging: {time.time() - tt:^6.4g}\n"
                    )
                    print(log_str)

            print("Training complete!")
            self._save_model(samples=samples_np,
                             saver=saver,
                             writer=writer,
                             step=step)

            helpers.write_run_data(self.files['run_info_file'], self.data)
            if kill_sess:
                writer.close()
                self.sess.close()
            sys.stdout.flush()

        except (KeyboardInterrupt, SystemExit):
            print("\nKeyboardInterrupt detected! \n"
                  "Saving current state and exiting.\n")
            self._save_model(samples=samples_np,
                             saver=saver,
                             writer=writer,
                             step=step)
            if kill_sess:
                writer.close()
                self.sess.close()

    def run(self, num_steps):
        """Run the simulation to generate samples and calculate observables."""
        samples = np.random.randn(*self.samples.shape)
        samples_history = []

        print(f"Running (trained) L2HMC sampler for {num_steps} steps...")
        for step in range(num_steps):
            t0 = time.time()
            samples = self.sess.run(self.x_out, feed_dict={self.x: samples})
            print(f'step: {step:^6.4g}  '
                  f'model invariant time / step: {time.time() - t0:^6.4g}')
            samples_history.append(samples)

        with open(self.files['samples_history_file'], 'wb') as f:
            pickle.dump(samples_history, f)

        print(f"done. Samples saved to: {self.files['samples_history_file']}.")

        return samples_history


def main(flags):
    """Main method for creating/training U(1) gauge model from command line."""
    params = PARAMS  # use default parameters if no command line args passed

    params['time_size'] = flags.time_size
    params['space_size'] = flags.space_size
    params['link_type'] = flags.link_type
    params['dim'] = flags.dim
    params['beta'] = flags.beta
    params['num_samples'] = flags.num_samples
    params['num_steps'] = flags.num_steps
    params['eps'] = flags.eps
    params['loss_scale'] = flags.loss_scale
    params['learning_rate_init'] = flags.learning_rate_init
    params['learning_rate_decay_rate'] = flags.learning_rate_decay_rate
    params['train_steps'] = flags.train_steps
    params['data_steps'] = flags.data_steps
    params['save_steps'] = flags.save_steps
    params['logging_steps'] = flags.logging_steps
    params['clip_value'] = flags.clip_value
    params['rand'] = flags.rand
    params['metric'] = flags.metric

    config = tf.ConfigProto()

    eps_trainable = True

    if flags.hmc:
        eps_trainable = False

    model = GaugeModel(params=params,
                       config=config,
                       sess=None,
                       conv_net=flags.conv_net,
                       hmc=flags.hmc,
                       log_dir=flags.log_dir,
                       restore=flags.restore,
                       eps_trainable=eps_trainable)

    #  start_time_str = time.strftime("%a, %d %b %Y %H:%M:%S",
    #                                 time.ctime(time.time()))

    print(f"Training began at: {time.ctime()}")

    model.train(flags.train_steps, kill_sess=False)

    _ = model.run(500)

    model.sess.close()


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

    parser.add_argument("--loss_scale", type=float, default=0.1,
                        required=False, dest="loss_scale",
                        help=("Scaling factor to be used in loss function. "
                              "(lambda in Eq. 7 of paper). (Default: 0.1)"))

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

    parser.add_argument("--logging_steps", type=int, default=50,
                        required=False, dest="logging_steps",
                        help=("Number of steps after which to write logs for "
                              "tensorboard. (Default: 50)"))

    parser.add_argument("--clip_value", type=int, default=100,
                        required=False, dest="clip_value",
                        help=("Clip value, used for clipping value of "
                              "gradients by global norm. (Default: 100)"))

    parser.add_argument("--rand", action="store_false",
                        required=False, dest="rand",
                        help=("Start lattice from randomized initial "
                              "configuration. (Default: True)"))

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

    args = parser.parse_args()

    main(args)
