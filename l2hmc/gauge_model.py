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
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from l2hmc_eager import gauge_dynamics_eager as gde
from lattice.gauge_lattice import GaugeLattice, u1_plaq_exact
from utils.plot_helper import plot_broken_xaxis
import utils.gauge_model_helpers as helpers
from utils.tf_logging import variable_summaries


tfe = tf.contrib.eager

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
tf.set_random_seed(GLOBAL_SEED)

tf.enable_resource_variables()

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

# pylint: disable=attribute-defined-outside-init
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
                 eps_trainable=True,
                 aux=False):
        """Initialization method."""

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
        self.aux = aux
        self.eps_trainable = eps_trainable

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

        if restore:
            self._restore_model(self.log_dir)

        else:
            self.dynamics = gde.GaugeDynamicsEager(
                lattice=self.lattice,
                num_steps=self.num_steps,
                eps=self.eps,
                minus_loglikelihood_fn=self.potential_fn,
                conv_net=self.conv_net,
                hmc=self.hmc,
                eps_trainable=eps_trainable
            )
            if eps_trainable:
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

        self.train_samples = {}
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
            'samples': [],
            'eps': params.get('eps', 0.),
            'train_steps': params.get('train_steps', 1000),
            'learning_rate': params.get('learning_rate_init', 1e-4),
        }

        self.files = {
            'parameters_file': os.path.join(self.info_dir, 'parameters.txt'),
            'run_info_file': os.path.join(self.info_dir, 'run_info.txt'),
            'data_pkl_file': os.path.join(self.info_dir, 'data.pkl'),
            'samples_pkl_file': os.path.join(self.info_dir, 'samples.pkl'),
            'train_samples_pkl_file': (
                os.path.join(self.info_dir, 'train_samples.pkl')
            ),
            'parameters_pkl_file': (
                os.path.join(self.info_dir, 'parameters.pkl')
            ),
        }

        self.samples_history_dir = os.path.join(self.log_dir, 'samples_history')
        if not os.path.isdir(self.samples_history_dir):
            os.makedirs(self.samples_history_dir)

        self._params = {}
        for key, val in self.__dict__.items():
            if isinstance(val, (int, float, str)):
                self._params[key] = val
            if isinstance(val, (int, float, str)):
                self._params[key] = val

        self._write_run_parameters(_print=True)

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

        self.dynamics = gde.GaugeDynamicsEager(
            lattice=self.lattice,
            num_steps=self.num_steps,
            eps=self.data['eps'],
            minus_loglikelihood_fn=self.potential_fn,
            conv_net=self.conv_net,
            hmc=self.hmc,
            eps_trainable=self.eps_trainable
        )

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
            #  self.sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("Restoring previous model from: "
                      f"{ckpt.model_checkpoint_path}")
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                self.sess.run(tf.global_variables_initializer())
                print("Model restored.")
                self.global_step = tf.train.get_global_step()
        else:
            latest_path = tf.train.latest_checkpoint(log_dir)
            self.checkpoint.restore(latest_path)
            print("Restored latest checkpoint from:\"{}\"".format(latest_path))

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
        if samples is None:
            samples = self.samples

        with open(self.files['data_pkl_file'], 'wb') as f:
            pickle.dump(self.data, f)
        with open(self.files['parameters_pkl_file'], 'wb') as f:
            pickle.dump(self.params, f)
        with open(self.files['samples_pkl_file'], 'wb') as f:
            pickle.dump(samples, f)
        with open(self.files['train_samples_pkl_file'], 'wb') as f:
            pickle.dump(self.train_samples, f)

        if not tf.executing_eagerly():
            ckpt_prefix = os.path.join(self.log_dir, 'ckpt')
            ckpt_file = os.path.join(self.log_dir, 'model.ckpt')
            print(f'Saving checkpoint to: {ckpt_file}\n')
            self.saver.save(self.sess, ckpt_file, global_step=step)
            #  self.checkpoint.save(file_prefix=ckpt_prefix, session=self.sess)
            self.writer.flush()
        else:
            saved_path = self.checkpoint.save(
                file_prefix=os.path.join(self.log_dir, 'ckpt')
            )
            print(f"\n Saved checkpoint to: {saved_path}\n")

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
            for key, val in self._params.items():
                print(f'{key}: {val}')

        with open(self.files['parameters_file'], 'w') as f:
            f.write('Parameters:\n')
            f.write(80 * '-' + '\n')
            for key, val in self._params.items():
                f.write(f'{key}: {val}\n')
            f.write(80*'=')
            f.write('\n')

    def _create_summaries(self):
        """"Create summary objects for logging in TensorBoard."""
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss_op)
            with tf.name_scope('position_fn'):
                for var in self.dynamics.position_fn.trainable_variables:
                    try:
                        layer, type = var.name.split('/')[-2:]
                        name = layer + '_' + type[:-2]
                    except:
                        name = var.name[:-2]
                    try:
                        variable_summaries(var, name)
                    except:
                        import pdb
                        pdb.set_trace()

            with tf.name_scope('momentum_fn'):
                for var in self.dynamics.momentum_fn.trainable_variables:
                    try:
                        layer, type = var.name.split('/')[-2:]
                        name = layer + '_' + type[:-2]
                    except:
                        name = var.name[:-2]
                    try:
                        variable_summaries(var, name)
                    except:
                        import pdb
                        pdb.set_trace()

            #  tf.summary.histogram('histogram_loss', self.loss_op)
            #  variable_summaries(self.loss)
            self.summary_op = tf.summary.merge_all(name='summary_op')

    def _create_optimizer(self):
        """Create optimizer to use during training."""
        with tf.name_scope('train'):
            if not self.hmc:
                clip_value = self.params['clip_value']
                grads, _ = tf.clip_by_global_norm(
                    tf.gradients(self.loss_op,
                                 self.dynamics.trainable_variables),
                    clip_value,
                    name='clip_grads'
                )
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate,
                    name='AdamOptimizer'
                )
                self.train_op = self.optimizer.apply_gradients(
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
            self._x, _, self.px, self.x_out = (
                self.dynamics.apply_transition(self.x)
            )

            self._x = tf.mod(self._x, 2*np.pi)
            self.x_out = tf.mod(self.x_out, 2*np.pi, name='x_out')

            self.x_loss = ((tf.reduce_sum(
                tf.square(self.x - self._x),
                axis=self.dynamics.axes,
                name='x_loss'
            ) * self.px) + 1e-4)

            if self.aux:
                z = tf.random_normal(tf.shape(self.x), name='z')
                _z, _, pz, z_out = self.dynamics.apply_transition(z)
                _z = tf.mod(_z, 2*np.pi)
                z_out = tf.mod(z_out, 2*np.pi)
                z_loss = ((tf.reduce_sum(
                    tf.square(z - _z),
                    axis=self.dynamics.axes,
                    name='z_loss'
                ) * pz) + 1e-4)

                # Squared jump distance
                self.loss_op = tf.reduce_mean(
                    (1. / self.x_loss + 1. / z_loss) * scale
                    - (self.x_loss + z_loss) / scale, axis=0, name='loss'
                )

            else:
                self.loss_op = tf.reduce_mean(scale * (1. / self.x_loss)
                                              - self.x_loss / scale,
                                              axis=0, name='loss')

    def _create_sampler(self):
        """Create operation for generating new samples using dynamics engine.

        This method is to be used when running generic HMC to create operations
        for dealing with `dynamics.apply_transition` without building
        unnecessary operations for calculating loss.
        """
        scale = self.params['loss_scale']
        _, _, self.px, self.x_out = self.dynamics.apply_transition(self.x)
        self.x_out = tf.mod(self.x_out, 2 * np.pi, name='x_out')

    def build_graph(self):
        """Build graph for TensorFlow."""
        def _write_strs_to_file(strings, last=False):
            with open(self.files['run_info_file'], 'a') as f:
                for s in strings:
                    f.write(s + '\n')
                if last:
                    f.write(80 * '-' + '\n')

        str0 = f"Building graph... (started at: {time.ctime()})"
        str1 = "  Creating loss..."
        str3 = "  Creating optimizer..."
        str5 = "  Creating summaries..."
        t_diff_str = lambda ti, tf : f"    took: {tf - ti} seconds."
        t_diff_str1 = lambda t : f"Time to build graph: {time.time() - t}."

        print(80*'-' + '\n')
        print(str0 + '\n' + str1 + '\n')
        _write_strs_to_file([str0, str1])
        t0 = time.time()

        if not self.hmc:
            self._create_loss()
        else:
            self._create_sampler()
            # if running generic HMC, all we need is self.x_out to sample
            self.sess.run(tf.global_variables_initializer())
            return 0

        t1 = time.time()
        print(t_diff_str(t0, t1) + '\n' + str3 + '\n')
        _write_strs_to_file([t_diff_str(t0, t1), str3])

        self._create_optimizer()

        t2 = time.time()
        print(t_diff_str(t1, t2) + '\n' + str5 + '\n')
        _write_strs_to_file([t_diff_str(t1, t2), str5])

        self._create_summaries()

        #  self.checkpoint = tf.train.Checkpoint(
        #      optimizer=self.optimizer,
        #      dynamics=self.dynamics,
        #      global_step=self.global_step,
        #  )

        print(t_diff_str(t2, time.time()) + '\n' + t_diff_str1(t0) + '\n')
        _write_strs_to_file(
            [t_diff_str(t2, time.time()), t_diff_str1(t0)], last=True
        )

    #pylint: disable=too-many-statements
    def train(self, num_train_steps, kill_sess=True):
        """Train the model."""
        self.saver = tf.train.Saver(max_to_keep=3)
        initial_step = 0
        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(self.log_dir)
        time_delay = 0.
        if ckpt and ckpt.model_checkpoint_path:
            print('Restoring previous model from: '
                  f'{ckpt.model_checkpoint_path}')
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('Model restored.\n')
            self.global_step = tf.train.get_global_step()
            initial_step = self.sess.run(self.global_step)
            #  previous_time = self.train_times[initial_step-1]
            #  time_delay = time.time() - previous_time

        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
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
        x = self.x
        self.sess.graph.finalize()
        try:
            print(helpers.data_header())
            helpers.write_run_data(
                self.files['run_info_file'],
                self.data,
                header=True
            )
            for step in range(initial_step, initial_step + num_train_steps):
                start_step_time = time.time()

                _, loss_np, samples_np, px_np, lr_np, eps_np = self.sess.run([
                    train_op,
                    loss_op,
                    x_out,
                    px,
                    learning_rate,
                    dynamics.eps
                ], feed_dict={x: samples_np})

                self.train_samples[step] = samples_np
                self.data['step'] = step
                self.data['loss'] = loss_np
                self.data['accept_prob'] = px_np
                self.data['eps'] = eps_np
                self.data['learning_rate'] = lr_np
                self.data['step_time'] = (
                    (time.time() - start_step_time)
                    / (self.num_steps * self.batch_size *
                       self.lattice.num_links)
                )  # pylint: disable=too-many-locals
                self.losses_arr.append(loss_np)

                #  if (step + 1) % 10 == 0:
                helpers.print_run_data(self.data)
                helpers.write_run_data(self.files['run_info_file'],
                                       self.data)

                if (step + 1) % self.save_steps == 0:
                    tt = time.time()
                    self._save_model(samples=samples_np, step=step)
                    helpers.write_run_data(self.files['run_info_file'],
                                           self.data)
                    save_str = (
                        f"Time to complete saving: {time.time() - tt:^6.4g}\n"
                    )
                    print(save_str)
                    print(helpers.data_header())

                if (step + 1) % self.logging_steps == 0 or step == 1:
                    tt = time.time()
                    summary_str = self.sess.run(summary_op,
                                                feed_dict={self.x: samples_np})
                    self.writer.add_summary(summary_str, global_step=step)
                    self.writer.flush()
                    log_str = (
                        f"Time to complete logging: {time.time() - tt:^6.4g}\n"
                    )
                    print(log_str)

            print("Training complete!")
            step = self.sess.run(self.global_step)
            self._save_model(samples=samples_np, step=step)

            helpers.write_run_data(self.files['run_info_file'], self.data)
            if kill_sess:
                self.writer.close()
                self.sess.close()
            sys.stdout.flush()

        except (KeyboardInterrupt, SystemExit):
            print("\nKeyboardInterrupt detected! \n"
                  "Saving current state and exiting.\n")
            step = self.sess.run(self.global_step)
            self._save_model(samples=samples_np, step=step)
            if kill_sess:
                self.writer.close()
                self.sess.close()

    def run(self, run_steps, _return=False):
        """Run the simulation to generate samples and calculate observables."""
        samples = np.random.randn(*self.samples.shape)
        samples_history = []

        # Move attribute lookup outside of loop to improve performance
        x_out = self.x_out
        x = self.x
        num_links = self.lattice.num_links
        num_steps = self.num_steps
        batch_size = self.batch_size
        print(f"Running (trained) L2HMC sampler for {run_steps} steps...")
        for step in range(run_steps):
            t0 = time.time()
            samples = self.sess.run(x_out, feed_dict={x: samples})
            tt = (time.time() - t0) / (num_links * num_steps * batch_size)
            #  tt /= (self.lattice.num_links * self.num_steps * self.batch_size)
            print(f'step: {step:^6.4g}/{run_steps} '
                  f'time/step/sample/link: {tt:^6.4g}')
            samples_history.append(samples)

        out_file = os.path.join(
            self.samples_history_dir, f'samples_history_{run_steps}.pkl'
        )
        with open(out_file, 'wb') as f:
            pickle.dump(samples_history, f)
        print('done.')
        print(f'Samples saved to: {out_file}.')
        print(80*'-' + '\n')

        if _return:
            return samples_history

        del samples_history  # free up some memory


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
                       eps_trainable=eps_trainable,
                       aux=flags.aux)

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

    parser.add_argument("--run_steps", type=int, default=1000,
                        required=False, dest="run_steps",
                        help=("Number of evaluation steps for generating "
                              "samples from trained sampler. "
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

    parser.add_argument("--rand", action="store_true",
                        required=False, dest="rand",
                        help=("Start lattice from randomized initial "
                              "configuration. (Default: False)"))

    parser.add_argument("--aux", action="store_true",
                        required=False, dest="aux",
                        help=("Include auxiliary function `q` for calculating "
                              "expected squared jump distance conditioned on "
                              "initialization distribution. (Default: False)"))

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
