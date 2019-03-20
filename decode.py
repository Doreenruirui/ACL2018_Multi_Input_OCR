from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
from six.moves import xrange
import tensorflow as tf
from multiprocessing import Pool
from os.path import join as pjoin
import model as ocr_model
from util import read_vocab, padded
import util

from flag import FLAGS
import re

reverse_vocab, vocab, data = None, None, None


def create_model(session, vocab_size, forward_only):
    model = ocr_model.Model(FLAGS.size, vocab_size,
                            FLAGS.num_layers, FLAGS.max_gradient_norm,
                            FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
                            forward_only=forward_only, decode=FLAGS.decode)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def tokenize_multi(sents, vocab):
    token_ids = []
    for sent in sents:
        token_ids.append(util.sentenc_to_token_ids(sent, vocab))
    token_ids = padded(token_ids)
    source = np.array(token_ids).T
    source_mask = (source != 0).astype(np.int32)
    return source, source_mask


def tokenize_single(sent, vocab):
    token_ids = util.sentenc_to_token_ids(sent, vocab)
    ones = [1] * len(token_ids)
    source = np.array(token_ids).reshape([-1, 1])
    mask = np.array(ones).reshape([-1, 1])
    return source, mask


def detokenize(sents, reverse_vocab):
    def detok_sent(sent):
        outsent = ''
        for t in sent:
            if t >= len(util._START_VOCAB):
                outsent += reverse_vocab[t]
        return outsent
    return [detok_sent(s) for s in sents]


def fix_sent(model, sess, sents):
    if FLAGS.decode == 'single':
        input_toks, mask = tokenize_single(sents[0], vocab)
        # len_inp * batch_size * num_units
        encoder_output = model.encode(sess, input_toks, mask)
        s1 = encoder_output.shape[0]
    else:
        input_toks, mask = tokenize_multi(sents, vocab)
        # len_inp * num_wit * num_units
        encoder_output = model.encode(sess, input_toks, mask)
        # len_inp * num_wit * (2 * size)
        s1, s2, s3 = encoder_output.shape
        # num_wit * len_inp * 1
        mask = np.transpose(mask, (1, 0))
        # num_wit * len_inp * (2 * size)
        encoder_output = np.transpose(encoder_output, (1, 0, 2))
    beam_toks, probs = model.decode_beam(sess, encoder_output, mask, s1, FLAGS.beam_size)
    beam_toks = beam_toks.tolist()
    probs = probs.tolist()
    # De-tokenize
    beam_strs = detokenize(beam_toks, reverse_vocab)
    return beam_strs, probs


def decode():
    global reverse_vocab, vocab
    folder_out = FLAGS.out_dir
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    print("Preparing NLC data in %s" % FLAGS.data_dir)
    vocab_path = pjoin(FLAGS.voc_dir, "vocab.dat")
    vocab, reverse_vocab = read_vocab(vocab_path)
    vocab_size = len(vocab)
    print("Vocabulary size: %d" % vocab_size)
    sess = tf.Session()
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, len(vocab), True)
    f_o = open(pjoin(folder_out,  'test.' + FLAGS.decode + '.o.txt'), 'w', encoding='utf-8')
    f_p = open(pjoin(folder_out, 'test.' + FLAGS.decode + '.p.txt'), 'w')
    line_id = 0
    with open(pjoin(FLAGS.data_dir,  'test.x.txt'), encoding='utf-8') as f_:
        for line in f_:
            sents = [ele for ele in line.strip('\n').split('\t')][:50]
            sents = [ele for ele in sents if len(ele.strip()) > 0]
            output_sents, output_probs = fix_sent(model, sess, sents)
            for out_sent in output_sents:
                f_o.write(out_sent + '\n')
            f_p.write('\n'.join(list(map(str, output_probs))) + '\n')
            line_id += 1
    f_o.close()
    f_p.close()


def main(_):
    decode()


if __name__ == "__main__":
    tf.app.run()
