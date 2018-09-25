"""
Dynamics object
"""

##############################################################################
#  TODO:
# ============================================================================
#    * Implement correct updates for LatticeDynamics._forward_step and
#        _backward_step, using Gauge transformations to update link variables.
# ----------------------------------------------------------------------------
#    * Finish implementing:
#        - LatticeDynamics._backward_step
#        - LatticeDynamics.forward
#        - LatticeDynamics.backward
# ----------------------------------------------------------------------------
#    * Implement `lattice_propose` in lattice_sampler.py using the
#        LatticeDynamics code.
##############################################################################

import tensorflow as tf
import numpy as np
from functools import reduce
from lattice.gauge_lattice import GaugeLattice

TF_FLOAT = tf.float32
NP_FLOAT = np.float32

NUM_AUX_FUNCS = 3

def safe_exp(x, name=None):
    return tf.exp(x)
    return tf.check_numerics(tf.exp(x), message=f'{name} is NaN')

def cast_f32(tensor):
    return tf.cast(tensor, TF_FLOAT)

class GaugeDynamics(object):
    def __init__(self,
                 lattice,
                 trajectory_length=10,
                 eps=0.1,
                 batch_size=10,
                 hmc=False,
                 net_factory=None,
                 eps_trainable=True,
                 use_temperature=False):
        """Initialization method."""
        self.lattice = lattice
        self.batch_size = batch_size

        try:
            self._energy_fn = lattice.get_energy_function()
        except AttributeError:
            raise AttributeError("lattice has no `get_energy_function`"
                                 " method. Exiting.")

        self.use_temperature = use_temperature
        self.temperature = tf.placeholder(TF_FLOAT, shape=(),
                                          name='temperature')

        if not hmc:
            alpha = tf.get_variable('alpha',
                                    initializer=tf.log(tf.constant(eps)),
                                    trainable=eps_trainable)
        else:
            alpha = tf.log(tf.constant(eps, dtype=TF_FLOAT))

        self.eps = safe_exp(alpha, name='alpha')
        self.trajectory_length = int(trajectory_length)
        self.hmc = hmc

        self._init_mask()

        # if HMC we just return all zeros
        if hmc:
            #  z = lambda x, *args, **kwargs: tf.zeros_like(x)
            self.XNet = lambda inp: [tf.zeros_like(inp[0])
                                     for t in range(NUM_AUX_FUNCS)]
            self.VNet = lambda inp: [tf.zeros_like(inp[0])
                                     for t in range(NUM_AUX_FUNCS)]
        else:
            self.XNet = net_factory(self.lattice.num_links,
                                    scope='XNet', factor=2.0)
            self.VNet = net_factory(self.lattice.num_links,
                                    scope='VNet', factor=1.0)

    def _init_mask(self):
        """Initialize mask to randomly select half of variables to update."""
        mask_per_step = []
        for t in range(self.trajectory_length):
            arr = np.arange(self.lattice.num_links)
            ind = np.random.permutation(arr)[:int(self.lattice.num_sites / 2)]

            m = np.zeros((self.lattice.num_links,))
            m[ind] = 1
            m = m.reshape(self.lattice.link_idxs)

            mask_per_step.append(m)

        self.mask = tf.constant(np.stack(mask_per_step), dtype=TF_FLOAT)

    def _get_mask(self, step):
        """Get the mask for a particular `step`."""
        m = tf.gather(self.mask, tf.cast(step, dtype=tf.int32))
        return m, 1.-m

    def _format_time(self, t, tile=1):
        trig_t = tf.squeeze([
            tf.cos(2 * np.pi * t / self.trajectory_length),
            tf.sin(2 * np.pi * t / self.trajectory_length),
        ])
        return tf.tile(tf.expand_dims(trig_t, 0), (tile, 1))

    def kinetic(self, v):
        """Calculate the kinetic energy from the MD `velocity`."""
        _axis = np.arange(1, len(self.lattice.links.shape) + 1)
        return 0.5 * tf.reduce_sum(tf.square(v), axis=_axis)
        #  return 0.5 * tf.reduce_sum(tf.square(v), axis=1)

    def clip_with_grad(self, u, min_u=-32., max_u=32.):
        u = u - tf.stop_gradient(tf.nn.relu(u - max_u))
        u = u + tf.stop_gradient(tf.nn.relu(min_u - u))
        return u

    def _forward_step(self, x, v, step, aux=None):
        """Implement a single MD step in the forward direction."""
        t = self._format_time(step, tile=tf.shape(x)[0])

        grad1 = self.grad_energy(x, aux=aux)
        S1 = self.VNet([x, grad1, t, aux])

        sv1 = 0.5 * self.eps * S1[0]
        tv1 = S1[1]
        fv1 = self.eps * S1[2]

        v_h = (tf.multiply(v, safe_exp(sv1, name='sv1F'))
               + 0.5 * self.eps * (-tf.multiply(safe_exp(fv1, name='fv1F'),
                                                grad1) + tv1))

        m, mb = self._get_mask(step)

        X1 = self.XNet([v_h, m * x, t, aux])
        sx1 = (self.eps * X1[0])
        tx1 = X1[1]
        fx1 = self.eps * X1[2]

        y = (m * x + mb * (tf.multiply(x, safe_exp(sx1, name='sx1F'))
                           + self.eps * (tf.multiply(safe_exp(fx1,
                                                              name='fx1F'),
                                                     v_h) + tx1)))

        X2 = self.XNet([v_h, mb * y, t, aux])

        sx2 = (self.eps * X2[0])
        tx2 = X2[1]
        fx2 = self.eps * X2[2]

        x_o = (mb * y + m * (tf.multiply(y, safe_exp(sx2, name='sx2F'))
                             + self.eps * (tf.multiply(safe_exp(fx2,
                                                                name='fx2F'),
                                                       v_h) + tx2)))

        S2 = self.VNet([x_o, self.grad_energy(x_o, aux=aux), t, aux])
        sv2 = (0.5 * self.eps * S2[0])
        tv2 = S2[1]
        fv2 = self.eps * S2[2]

        grad2 = self.grad_energy(x_o, aux=aux)
        v_o = (tf.multiply(v_h, safe_exp(sv2, name='sv2F'))
               + 0.5 * self.eps * (-tf.multiply(safe_exp(fv2, name='fv2F'),
                                                grad2) + tv2))

        log_jac_contrib = tf.reduce_sum(sv1 + sv2 + mb * sx1 + m * sx2, axis=1)

        return x_o, v_o, log_jac_contrib


    def _backward_step(self, x_o, v_o, step, aux=None):
        t = self._format_time(step, tile=tf.shape(x_o)[0])

        grad1 = self.grad_energy(x_o, aux=aux)

        S1 = self.VNet([x_o, grad1, t, aux])

        sv2 = (-0.5 * self.eps * S1[0])
        tv2 = S1[1]
        fv2 = self.eps * S1[2]

        exp_fv2 = safe_exp(fv2, name='fv2B')
        exp_sv2 = safe_exp(sv2, name='sv2B')
        prod_fv2 = tf.multiply(exp_fv2, grad1)
        v_h = tf.multiply((v_o - 0.5 * self.eps * (-prod_fv2 + tv2)), exp_sv2)

        #  v_h = (tf.multiply((v_o - 0.5 * self.eps
        #                      * (-tf.multiply(safe_exp(fv2, name='fv2B'), grad1)
        #                         + tv2)), safe_exp(sv2, name='sv2B')))

        m, mb = self._get_mask(step)

        # m, mb = self._gen_mask(x_o)

        X1 = self.XNet([v_h, mb * x_o, t, aux])

        sx2 = (-self.eps * X1[0])
        tx2 = X1[1]
        fx2 = self.eps * X1[2]

        exp_sx2 = safe_exp(sx2, name='sx2B')
        exp_fx2 = safe_exp(fx2, name='fx2B')
        prod_fx2 = tf.multiply(exp_fx2, v_h)
        prod_sx2 = tf.multiply(exp_sx2, (x_o - self.eps * (prod_fx2 + tx2)))
        y = mb * x_o + m * prod_sx2

        #  y = (mb * x_o + m * tf.multiply(safe_exp(sx2, name='sx2B'),
        #                                  (x_o - self.eps * (tf.multiply(
        #                                      safe_exp(fx2, name='fx2B'), v_h)
        #                                      + tx2))))

        X2 = self.XNet([v_h, m * y, t, aux])

        sx1 = (-self.eps * X2[0])
        tx1 = X2[1]
        fx1 = self.eps * X2[2]

        exp_sx = safe_exp(sx1, name='sx1B')
        exp_fx = safe_exp(fx1, name='fx1B')
        prod_fx = tf.multiply(exp_fx, v_h)
        prod_sx = tf.multiply(exp_sx, (y - self.eps * prod_fx + tx1))
        x = m * y + mb * prod_sx

        #  x = m * y + mb * tf.multiply(safe_exp(sx1, name='sx1B'),
        #                               (y - self.eps * (tf.multiply(
        #                                   safe_exp(fx1, name='fx1B'), v_h)
        #                                   + tx1)))

        grad2 = self.grad_energy(x, aux=aux)
        S2 = self.VNet([x, grad2, t, aux])

        sv1 = (-0.5 * self.eps * S2[0])
        tv1 = S2[1]
        fv1 = self.eps * S2[2]

        exp_sv = safe_exp(sv1, name='sv1B')
        exp_fv = safe_exp(fv1, name='fv1B')
        prod_fv = tf.multiply(exp_fv, grad2)
        v = tf.multiply(exp_sv, (v_h - 0.5 * self.eps * (prod_fv + tv1)))

        #  v = tf.multiply(safe_exp(sv1, name='sv1B'),
        #                  (v_h - 0.5 * self.eps * (
        #                      -tf.multiply(safe_exp(fv1, name='fv1B'), grad2)
        #                      + tv1
        #                  )))

        return x, v, tf.reduce_sum(sv1 + sv2 + mb * sx1 + m * sx2, axis=1)
        #return x, v, tf.reduce_sum(sv1 + sv2 + mb * sx1 + m * sx2, axis=1)

    def energy(self, x, batch_size):
        if self.use_temperature:
            T = self.temperature
        else:
            T = tf.constant(1.0, dtype=TF_FLOAT)

        #  if T.dtype != x.dtype:
        #      T = tf.cast(T, x.dtype)

        #  if aux is not None:
            #  return self._energy_fn(x, aux=aux) / T
        return tf.cast(self._energy_fn(x, batch_size), TF_FLOAT) / T
        #  else:
        #      #  return self._energy_fn(x) / T)
        #      return tf.cast(self._energy_fn(x), TF_FLOAT) / T

    def hamiltonian(self, x, v, aux=None):
        return self.energy(x, self.batch_size) + self.kinetic(v)

    def grad_energy(self, x, aux=None):
        #  grad_ys = tf.constant(1.0, dtype=tf.complex64, shape=())
        return tf.gradients(self.energy(x, aux=aux), x)[0]
        #  return tf.gradients(self.energy(x, aux=aux), x, grad_ys=grad_ys)[0]
        #  try:
        #      _energy = tf.cast(self.energy(x, aux=aux), TF_FLOAT)
        #      return tf.gradients(_energy, x)[0]
        #  except TypeError:
        #      import pdb
        #      pdb.set_trace()
        #  return tf.cast(tf.gradients(tf.cast(self.energy(x, aux=aux),
        #                                      tf.float32), x)[0],
                       #  dtype=TF_FLOAT)

    def _gen_mask(self, x):
        #  dX = x.get_shape().as_list()[1]
        b = np.zeros(self.lattice.num_links)
        for i in range(self.lattice.num_links):
            if i % 2 == 0:
                b[i] = 1
                b = b.astype('bool')
                nb = np.logical_not(b)
        return b.astype(NP_FLOAT), nb.astype(NP_FLOAT)

    def forward(self, x, init_v=None, aux=None, log_path=False, log_jac=False):
        if init_v is None:
            v = tf.random_normal(tf.shape(x))
        else:
            v = init_v

        dN = tf.shape(x)[0]
        t = tf.constant(0., dtype=TF_FLOAT)
        j = tf.zeros((dN,))

        def body(x, v, t, j):
            new_x, new_v, log_j = self._forward_step(x, v, t, aux=aux)
            return new_x, new_v, t+1, j+log_j

        def cond(x, v, t, j):
            return tf.less(t, self.trajectory_length)

        X, V, t, log_jac_ = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[x, v, t, j]
        )

        if log_jac:
            return X, V, log_jac_

        return X, V, self.p_accept(x, v, X, V, log_jac_, aux=aux)

    def backward(self, x, init_v=None, aux=None, log_jac=False):
        if init_v is None:
            v = tf.random_normal(tf.shape(x))
        else:
            v = init_v

        dN = tf.shape(x)[0]
        t = tf.constant(0., name='step_backward', dtype=TF_FLOAT)
        j = tf.zeros((dN,), name='acc_jac_backward')

        def body(x, v, t, j):
            new_x, new_v, log_j = self._backward_step(x, v,
                                                      (self.trajectory_length -
                                                       t - 1), aux=aux)
            return new_x, new_v, t+1, j+log_J

        def cond(x, v, t, j):
            return tf.less(t, self.trajectory_length)

        X, V, t, log_jac_ = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[x, v, t,j]
        )

        if log_jac:
            return X, V, log_jac_

        return X, V, self.p_accept(x, v, X, V, log_jac_, aux=aux)

    def p_accept(self, x0, v0, x1, v1, log_jac, aux=None):
        e_new = self.hamiltonian(x1, v1, aux=aux)
        e_old = self.hamiltonian(x0, v0, aux=aux)

        v = e_old - e_new + log_jac
        p = tf.exp(tf.minimum(v, 0.0))

        return tf.where(tf.is_finite(p), p, tf.zeros_like(p))
