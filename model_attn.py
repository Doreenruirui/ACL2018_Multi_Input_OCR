from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import tanh
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops


def _linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".

    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with vs.variable_scope(scope or "Linear"):
        matrix = vs.get_variable(
            "Matrix", [total_arg_size, output_size], dtype=dtype)
        if len(args) == 1:
            res = math_ops.matmul(args[0], matrix)
        else:
            res = math_ops.matmul(array_ops.concat(1, args), matrix)
        if not bias:
            return res
        bias_term = vs.get_variable(
            "Bias", [output_size],
            dtype=dtype,
            initializer=init_ops.constant_initializer(
                bias_start, dtype=dtype))
    return res + bias_term


class GRUCellAttn(rnn_cell.GRUCell):
    def __init__(self, num_units, enc_len, encoder_output, encoder_mask,
                 decode, scope=None):
        # len_inp * batch_size * (2 * num_units) / num_wit * len_inp * batch_size * (2 * num_units)
        self.hs = encoder_output
        # len_inp * batach_ize   /  num_wit * len_inp * batch_size(1)
        self.mask = tf.cast(encoder_mask, tf.bool)
        self.enc_len = enc_len
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Attn1"):
                # (len_inp * batch_size) * (2 * num_units) / (num_wit * len_inp * batch_size) * (2 * num_units)
                hs2d = array_ops.reshape(self.hs, [-1, 2 * num_units])
                # (len_inp * batch_size) * num_units  /  (num_wit * len_inp * batch_size) * num_units
                phi_hs2d = tanh(_linear(hs2d, num_units, True, 1.0))
                # len_inp * batch_size * num_units
                self.phi_hs = array_ops.reshape(phi_hs2d,
                                                [self.enc_len, -1, num_units])
        super(GRUCellAttn, self).__init__(num_units)

    def __call__(self, inputs, state, scope=None):
        gru_out, gru_state = super(GRUCellAttn, self).__call__(inputs, state, scope)
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Attn2"):
                # batch_size * num_units
                gamma_h = tanh(_linear(gru_out, self._num_units, True, 1.0))
            # len_inp * batch_size * num_units / batch_size * num_units => len_inp * batch_size
            weights = tf.reduce_sum(self.phi_hs * gamma_h, axis=2)
            # mask: len_inp * batch_size
            weights = tf.where(self.mask, weights,
                               tf.ones_like(weights) * (-2 ** 32 + 1))
            # len_inp * batch_size * 1
            weights = tf.expand_dims(
                tf.transpose(tf.nn.softmax(tf.transpose(weights))), -1)
            # hs: len_inp * batch_size * (2 * size) / weights: len_inp * batch_size * 1  => batch_size * (2 * size)
            context = tf.reduce_sum(self.hs * weights, axis=0)
            with vs.variable_scope("AttnConcat"):
                out = tf.nn.relu(_linear(tf.concat([context, gru_out], -1),
                                         self._num_units, True, 1.0))
            return (out, out)

    def beam_single(self, inputs, state, beam_size, scope=None):
        gru_out, gru_state = super(GRUCellAttn, self).__call__(inputs, state, scope)
        with vs.variable_scope(scope or type(self).__name__, reuse=tf.AUTO_REUSE):
            with vs.variable_scope("Attn2"):
                # beam_size * num_units
                gamma_h = tanh(_linear(gru_out, self._num_units, True, 1.0))
            # len_inp * batch_size(1) * num_units  / beam_size * num_units => len_inp * beam_size
            weights = tf.reduce_sum(self.phi_hs * gamma_h, axis=2)
            # len_inp * batch_size(1) => len_inp * beam_size
            mask = tf.tile(self.mask, [1, beam_size])
            # len_inp * beam_size => len_inp * beam_size
            weights = tf.where(mask, weights,
                               tf.ones_like(weights) * (-2 ** 32 + 1))
            # len_inp * beam_size * 1
            weights = tf.expand_dims(
                tf.transpose(tf.nn.softmax(tf.transpose(weights))), -1)
            # hs: len_inp * 1 * (2 * size)   weights: len_inp * beam_size * 1  =>  beam_size * (2 * size)
            context = tf.reduce_sum(self.hs * weights, axis=0)
            with vs.variable_scope("AttnConcat"):
                out = tf.nn.relu(_linear(tf.concat([context, gru_out], -1),
                                         self._num_units, True, 1.0))
            return (out, out)

    def beam_average(self, inputs, state, beam_size, scope=None):
        gru_out, gru_state = super(GRUCellAttn, self).__call__(inputs, state, scope)
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Attn2"):
                # beam_size * num_units
                gamma_h = tanh(_linear(gru_out, self._num_units, True, 1.0))
            # num_wit * len_inp * batch_size(1) * num_units
            phi_hs = array_ops.reshape(self.phi_hs,
                                       [-1, self.enc_len, 1, self._num_units])
            hs = array_ops.reshape(self.hs,
                                   [-1, self.enc_len, 1, 2 * self._num_units])
            # num_wit * len_inp * batch_size(1) * num_units / beam_size * num_units
            # => num_wit * len_inp * beam_size
            weights = tf.reduce_sum(phi_hs * gamma_h, axis=3)
            # num_wit * len_inp * batch_size(1) => num_wit * len_inp * beam_size
            mask = tf.tile(tf.reshape(self.mask, [-1, self.enc_len, 1]), [1, 1, beam_size])
            # num_wit * len_inp * beam_size
            weights = tf.where(mask, weights, tf.ones_like(weights) * (-2 ** 32 + 1))
            weights = tf.reshape(tf.transpose(weights,
                                              [0, 2, 1]),
                                 [-1, self.enc_len])
            # (num_wit * beam_size) * len_inp
            weights = tf.nn.softmax(weights)
            # num_wit * len_inp * beam_size * 1
            weights = tf.transpose(tf.reshape(weights,
                                              [-1, beam_size, self.enc_len, 1]),
                                   [0, 2, 1, 3])
            # num_wit * len_inp * batch_size (1) * (2 * num_units) / num_weights * len_inp * beam_size * 1
            # => num_wit * beam_size * (2 * num_units)
            context = tf.reduce_sum(hs * weights, axis=1)
            # beam_size * (2 * num_units)
            context = tf.reshape(tf.reduce_mean(context, axis=0),
                                 [beam_size, 2 * self._num_units])
            with vs.variable_scope("AttnConcat"):
                out = tf.nn.relu(_linear(tf.concat([context, gru_out], -1),
                                                  self._num_units, True, 1.0))
            return (out, out)

    def beam_weighted(self, inputs, state, beam_size, scope=None):
        gru_out, gru_state = super(GRUCellAttn, self).__call__(inputs, state,
                                                               scope)
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Attn2"):
                # beam_size * num_units
                gamma_h = tanh(_linear(gru_out,
                                                self._num_units, True, 1.0))

            phi_hs = array_ops.reshape(self.phi_hs,
                                       [-1, self.enc_len, 1, self._num_units])
            hs = array_ops.reshape(self.hs,
                                   [-1, self.enc_len, 1, 2 * self._num_units])
            # num_wit * len_inp * batch_size (1) * num_units / beam_size * num_units
            # => num_wit * len_inp * beam_size
            weights = tf.reduce_sum(phi_hs * gamma_h, axis=3)
            # num_wit * len_inp * batch_size (1) => num_wit * len_inp * beam_size
            mask = tf.tile(tf.reshape(self.mask, [-1, self.enc_len, 1]),
                           [1, 1, beam_size])
            # num_wit * len_inp * beam_size
            weights = tf.where(mask, weights,
                                tf.ones_like(weights) * (-2 ** 32 + 1))
            # (num_wit * beam_size) * len_inp
            weights = tf.reshape(tf.transpose(weights,
                                              [0, 2, 1]),
                                 [-1, self.enc_len])
            weights = tf.nn.softmax(weights)
            # num_wit * len_inp * beam_size * 1
            weights = tf.transpose(tf.reshape(weights,
                                              [-1, beam_size,
                                               self.enc_len, 1]),
                                   [0, 2, 1, 3])
            # num_wit * len_inp * batch_size (1) * (2 * num_units) / num_wit * len_inp * beam_size * 1
            # => num_wit * beam_size * (2 * num_units)
            context = tf.reduce_sum(hs * weights, axis=1)
            # num_wit * len_inp * batch_size(1) * num_units / num_wit * len_inp * beam_size * 1
            # num_wit * beam_size * num_units
            context_w1 = tf.reduce_sum(phi_hs * weights, axis=1)
            # num_wit * beam_size * num_units / beam_size * num_units => num_wit * beam_size
            weights_ctx = tf.reduce_sum(context_w1 * gamma_h, axis=2)
            weights_ctx = tf.expand_dims(
                tf.transpose(tf.nn.softmax(tf.transpose(weights_ctx))), -1)
            # num_wit * beam_size * (2 * num_units) / num_wit * beam_size * 1
            # beam_size * (2 * num_units)
            context_w = tf.reshape(tf.reduce_sum(context * weights_ctx,
                                                 axis=0),
                                   [beam_size, 2 * self._num_units])
            with vs.variable_scope("AttnConcat"):
                out = tf.nn.relu(_linear(tf.concat([context, gru_out], -1),
                                         self._num_units, True, 1.0))
            return (out, out)

    def beam_flat(self, inputs, state, beam_size, scope=None):
        gru_out, gru_state = super(GRUCellAttn, self).__call__(inputs, state, scope)
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Attn2", reuse=True):
                # beam_size * num_units
                gamma_h = tanh(_linear(gru_out, self._num_units, True, 1.0))
            phi_hs = array_ops.reshape(self.phi_hs,
                                       [-1, self.enc_len, 1, self._num_units])
            hs = array_ops.reshape(self.hs,
                                   [-1, self.enc_len, 1, 2 * self._num_units])
            # num_wit * len_inp * batch_size (1) * num_units / beam_size * num_units
            # => num_wit * len_inp * beam_size
            weights = tf.reduce_sum(phi_hs * gamma_h, axis=3)
            # num_wit * len_inp * batch_size (1) => num_wit * len_inp * beam_size
            mask = tf.tile(tf.reshape(self.mask, [-1, self.enc_len, 1]),
                           [1, 1, beam_size])
            # num_wit * len_inp * beam_size
            weights = tf.where(mask, weights,
                                tf.ones_like(weights) * (-2 ** 32 + 1))
            # beam_size * (num_wit * len_inp)
            weights = tf.transpose(tf.reshape(weights, [-1, beam_size]))
            weights = tf.nn.softmax(weights)
            # num_wit * len_inp * beam_size * 1
            weights = tf.reshape(tf.transpose(weights),
                                 [-1, self.enc_len, beam_size, 1])
            # num_wit * len_inp * batch_size (1) * (2 * num_units) / num_wit * len_inp * beam_size * 1
            # => num_wit * beam_size * (2 * num_units)
            context = tf.reduce_sum(hs * weights, axis=1)
            # beam_size * (2 * num_units)
            context = tf.reduce_sum(context, axis=0)
            with vs.variable_scope("AttnConcat"):
                out = tf.nn.relu(_linear(tf.concat([context, gru_out], -1),
                                         self._num_units, True, 1.0))
            return (out, out)

