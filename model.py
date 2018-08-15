# Copyright 2016 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from model_attn import GRUCellAttn, _linear
import util


def label_smooth(labels, num_class):
    labels = tf.one_hot(labels, depth=num_class)
    return 0.9 * labels + 0.1 / num_class


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert(False)
    return optfn


class Model(object):
    def __init__(self, size, voc_size, num_layers, max_gradient_norm,
                 learning_rate, learning_rate_decay,
                 forward_only=False, optimizer="adam", decode="single"):
        self.voc_size = voc_size
        self.size = size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.learning_decay = learning_rate_decay
        self.max_grad_norm = max_gradient_norm
        self.foward_only = forward_only
        self.optimizer = optimizer
        self.decode_method=decode
        self.build_model()

    def _add_place_holders(self):
        self.keep_prob = tf.placeholder(tf.float32)
        self.src_toks = tf.placeholder(tf.int32, shape=[None, None])
        self.tgt_toks = tf.placeholder(tf.int32, shape=[None, None])
        self.src_mask = tf.placeholder(tf.int32, shape=[None, None])
        self.tgt_mask = tf.placeholder(tf.int32, shape=[None, None])
        self.beam_size = tf.placeholder(tf.int32)
        self.batch_size = tf.shape(self.src_mask)[1]
        self.len_inp = tf.shape(self.src_mask)[0]
        self.src_len = tf.cast(tf.reduce_sum(self.src_mask, axis=0), tf.int64)
        self.tgt_len = tf.cast(tf.reduce_sum(self.tgt_mask, axis=0), tf.int64)

    def setup_train(self):
        self.lr = tf.Variable(float(self.learning_rate), trainable=False)
        self.lr_decay_op = self.lr.assign(
            self.lr * self.learning_decay)
        self.global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()
        opt = get_optimizer(self.optimizer)(self.lr)
        gradients = tf.gradients(self.losses, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients,
                                                      self.max_grad_norm)
        self.gradient_norm = tf.global_norm(gradients)
        self.param_norm = tf.global_norm(params)
        self.updates = opt.apply_gradients(zip(clipped_gradients, params),
                                           global_step=self.global_step)

    def setup_embeddings(self):
        with vs.variable_scope("embeddings"):
            zeros = tf.zeros([1, self.size])
            enc = tf.get_variable("L_enc", [self.voc_size - 1, self.size])
            self.L_enc = tf.concat([zeros, enc], axis=0)
            dec = tf.get_variable("L_dec", [self.voc_size - 1, self.size])
            self.L_dec = tf.concat([zeros, dec], axis=0)
            self.encoder_inputs = embedding_ops.embedding_lookup(self.L_enc, self.src_toks)
            self.decoder_inputs = embedding_ops.embedding_lookup(self.L_dec, self.tgt_toks)

    def setup_encoder(self):
        with vs.variable_scope("Encoder"):
            inp = tf.nn.dropout(self.encoder_inputs, self.keep_prob)
            fw_cell = rnn_cell.GRUCell(self.size)
            fw_cell = rnn_cell.DropoutWrapper(
                fw_cell, output_keep_prob=self.keep_prob)
            self.encoder_fw_cell = rnn_cell.MultiRNNCell(
                [fw_cell] * self.num_layers, state_is_tuple=True)
            bw_cell = rnn_cell.GRUCell(self.size)
            bw_cell = rnn_cell.DropoutWrapper(
                bw_cell, output_keep_prob=self.keep_prob)
            self.encoder_bw_cell = rnn_cell.MultiRNNCell(
                [bw_cell] * self.num_layers, state_is_tuple=True)
            out, _ = rnn.bidirectional_dynamic_rnn(self.encoder_fw_cell,
                                                   self.encoder_bw_cell,
                                                   inp, self.src_len,
                                                   dtype=tf.float32,
                                                   time_major=True,
                                                   initial_state_fw=self.encoder_fw_cell.zero_state(
                                                       self.batch_size, dtype=tf.float32),
                                                   initial_state_bw=self.encoder_bw_cell.zero_state(
                                                       self.batch_size, dtype=tf.float32))
            out = tf.concat([out[0], out[1]], axis=2)
            self.encoder_output = out

    def setup_decoder(self):
        with vs.variable_scope("Decoder"):
            inp =  tf.nn.dropout(self.decoder_inputs, self.keep_prob)
            if self.num_layers > 1:
                with vs.variable_scope("RNN"):
                    decoder_cell = rnn_cell.GRUCell(self.size)
                    decoder_cell = rnn_cell.DropoutWrapper(decoder_cell,
                                                           output_keep_prob=self.keep_prob)
                    self.decoder_cell = rnn_cell.MultiRNNCell(
                        [decoder_cell] * (self.num_layers - 1), state_is_tuple=True)
                    inp, _ = rnn.dynamic_rnn(self.decoder_cell, inp, self.tgt_len,
                                             dtype=tf.float32, time_major=True,
                                             initial_state=self.decoder_cell.zero_state(
                                                 self.batch_size, dtype=tf.float32))

            with vs.variable_scope("Attn"):
                self.attn_cell = GRUCellAttn(self.size, self.len_inp,
                                             self.encoder_output, self.src_mask, self.decode_method)
                self.decoder_output, _ = rnn.dynamic_rnn(self.attn_cell, inp, self.tgt_len,
                                                         dtype=tf.float32, time_major=True,
                                                         initial_state=self.attn_cell.zero_state(
                                                             self.batch_size, dtype=tf.float32,
                                                         ))

    def setup_loss(self):
        with vs.variable_scope("Loss"):
            len_out = tf.shape(self.decoder_output)[0]
            logits2d = _linear(tf.reshape(self.decoder_output,
                                                   [-1, self.size]),
                                        self.voc_size, True, 1.0)
            self.outputs2d = tf.nn.log_softmax(logits2d)
            targets_no_GO = tf.slice(self.tgt_toks, [1, 0], [-1, -1])
            masks_no_GO = tf.slice(self.tgt_mask, [1, 0], [-1, -1])
            # easier to pad target/mask than to split decoder input since tensorflow does not support negative indexing
            labels1d = tf.reshape(tf.pad(targets_no_GO, [[0, 1], [0, 0]]), [-1])
            if self.foward_only or self.keep_prob==1.:
                labels1d = tf.one_hot(labels1d, depth=self.voc_size)
            else:
                labels1d = label_smooth(labels1d, self.voc_size)
            mask1d = tf.reshape(tf.pad(masks_no_GO, [[0, 1], [0, 0]]), [-1])
            losses1d = tf.nn.softmax_cross_entropy_with_logits(logits=logits2d, labels=labels1d) * tf.to_float(mask1d)
            losses2d = tf.reshape(losses1d, [len_out, self.batch_size])
            self.losses = tf.reduce_sum(losses2d) / tf.to_float(self.batch_size)

    def build_model(self):
        self._add_place_holders()
        with tf.variable_scope("Model", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_encoder()
            self.setup_decoder()
            self.setup_loss()
            if self.foward_only:
                self.setup_beam()
        if not self.foward_only:
            self.setup_train()
        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)

    def decode_step(self, inputs, state_inputs):
        beam_size = tf.shape(inputs)[0]
        with vs.variable_scope("Decoder", reuse=True):
            with vs.variable_scope("RNN", reuse=True):
                with vs.variable_scope("RNN", reuse=True):
                    rnn_out, rnn_outputs = self.decoder_cell(inputs, state_inputs[:self.num_layers-1])
            with vs.variable_scope("Attn", reuse=True):
                with vs.variable_scope("RNN", reuse=True):
                    if self.decode_method == 'average':
                        out, attn_outputs = self.attn_cell.beam_average(rnn_out, state_inputs[-1], beam_size)
                    elif self.decode_method == 'weight':
                        out, attn_outputs = self.attn_cell.beam_weighted(rnn_out, state_inputs[-1], beam_size)
                    elif self.decode_method == 'flat':
                        out, attn_outputs = self.attn_cell.beam_flat(rnn_out, state_inputs[-1], beam_size)
                    else:
                        out, attn_outputs = self.attn_cell.beam_single(rnn_out, state_inputs[-1], beam_size)
        state_outputs = rnn_outputs + (attn_outputs, )
        return out, state_outputs

    def setup_beam(self):
        time_0 = tf.constant(0)
        beam_seqs_0 = tf.constant([[util.SOS_ID]])
        beam_probs_0 = tf.constant([0.])
        cand_seqs_0 = tf.constant([[util.EOS_ID]])
        cand_probs_0 = tf.constant([-3e38])

        state_0 = tf.zeros([1, self.size])
        states_0 = [state_0] * self.num_layers

        def beam_cond(cand_probs, cand_seqs, time, beam_probs, beam_seqs, *states):
            return tf.logical_and(tf.reduce_max(beam_probs) >= tf.reduce_min(cand_probs),
                                  time < tf.reshape(self.len_inp, ()) + 10)

        def beam_step(cand_probs, cand_seqs, time, beam_probs, beam_seqs, *states):
            batch_size = tf.shape(beam_probs)[0]
            inputs = tf.reshape(tf.slice(beam_seqs, [0, time], [batch_size, 1]), [batch_size])
            decoder_input = embedding_ops.embedding_lookup(self.L_dec, inputs)
            decoder_output, state_output = self.decode_step(decoder_input, states)

            with vs.variable_scope("Loss", reuse=True):
                do2d = tf.reshape(decoder_output, [-1, self.size])
                logits2d = _linear(do2d, self.voc_size, True, 1.0)
                logprobs2d = tf.nn.log_softmax(logits2d)

            total_probs = logprobs2d + tf.reshape(beam_probs, [-1, 1])
            total_probs_noEOS = tf.concat([tf.slice(total_probs, [0, 0], [batch_size, util.EOS_ID]),
                                              tf.tile([[-3e38]], [batch_size, 1]),
                                              tf.slice(total_probs, [0, util.EOS_ID + 1],
                                                       [batch_size, self.voc_size - util.EOS_ID - 1])],
                                          axis=1)
            flat_total_probs = tf.reshape(total_probs_noEOS, [-1])

            beam_k = tf.minimum(tf.size(flat_total_probs), self.beam_size)
            next_beam_probs, top_indices = tf.nn.top_k(flat_total_probs, k=beam_k)

            next_bases = tf.floordiv(top_indices, self.voc_size)
            next_mods = tf.mod(top_indices, self.voc_size)

            next_states = [tf.gather(state, next_bases) for state in state_output]
            next_beam_seqs = tf.concat([tf.gather(beam_seqs, next_bases),
                                           tf.reshape(next_mods, [-1, 1])], axis=1)

            cand_seqs_pad = tf.pad(cand_seqs, [[0, 0], [0, 1]])
            beam_seqs_EOS = tf.pad(beam_seqs, [[0, 0], [0, 1]])
            new_cand_seqs = tf.concat([cand_seqs_pad, beam_seqs_EOS], axis=0)
            EOS_probs = tf.slice(total_probs, [0, util.EOS_ID], [batch_size, 1])

            new_cand_probs = tf.concat([cand_probs, tf.reshape(EOS_probs, [-1])], axis=0)
            cand_k = tf.minimum(tf.size(new_cand_probs), self.beam_size)
            next_cand_probs, next_cand_indices = tf.nn.top_k(new_cand_probs, k=cand_k)
            next_cand_seqs = tf.gather(new_cand_seqs, next_cand_indices)


            return [next_cand_probs, next_cand_seqs, time + 1, next_beam_probs, next_beam_seqs] + next_states

        var_shape = []
        var_shape.append((cand_probs_0, tf.TensorShape([None, ])))
        var_shape.append((cand_seqs_0, tf.TensorShape([None, None])))
        var_shape.append((time_0, time_0.get_shape()))
        var_shape.append((beam_probs_0, tf.TensorShape([None, ])))
        var_shape.append((beam_seqs_0, tf.TensorShape([None, None])))
        var_shape.extend([(state_0, tf.TensorShape([None, self.size])) for state_0 in states_0])
        loop_vars, loop_var_shapes = zip(*var_shape)
        self.loop_vars = loop_vars
        self.loop_var_shapes = loop_var_shapes
        ret_vars = tf.while_loop(cond=beam_cond, body=beam_step, loop_vars=loop_vars, shape_invariants=loop_var_shapes, back_prop=False)
        self.vars = ret_vars
        self.beam_output = ret_vars[1]
        self.beam_scores = ret_vars[0]

    def decode_beam(self, session, encoder_output, src_mask, len_inp, beam_size=128):
        input_feed = {}
        input_feed[self.encoder_output] = encoder_output
        input_feed[self.src_mask] = src_mask
        input_feed[self.len_inp] = len_inp
        input_feed[self.keep_prob] = 1.
        input_feed[self.beam_size] = beam_size
        output_feed = [self.beam_output, self.beam_scores]
        outputs = session.run(output_feed, input_feed)
        return outputs[0], outputs[1]

    def encode(self, session, src_toks, src_mask):
        input_feed = {}
        input_feed[self.src_toks] = src_toks
        input_feed[self.src_mask] = src_mask
        input_feed[self.keep_prob] = 1.
        output_feed = [self.encoder_output]
        outputs = session.run(output_feed, input_feed)
        return outputs[0]

    def train(self, session, src_toks, src_mask, tgt_toks, tgt_mask, dropout):
        input_feed = {}
        input_feed[self.src_toks] = src_toks
        input_feed[self.tgt_toks] = tgt_toks
        input_feed[self.src_mask] = src_mask
        input_feed[self.tgt_mask] = tgt_mask
        input_feed[self.keep_prob] = 1 - dropout
        output_feed = [self.updates, self.gradient_norm, self.losses, self.param_norm]
        outputs = session.run(output_feed, input_feed)
        return outputs[1], outputs[2], outputs[3]

    def test(self, session, src_toks, src_mask, tgt_toks, tgt_mask):
        input_feed = {}
        input_feed[self.src_toks] = src_toks
        input_feed[self.tgt_toks] = tgt_toks
        input_feed[self.src_mask] = src_mask
        input_feed[self.tgt_mask] = tgt_mask
        input_feed[self.keep_prob] = 1.
        output_feed = [self.losses]
        outputs = session.run(output_feed, input_feed)
        return outputs[0]
