from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import time
from os.path import join as pjoin
import numpy as np
from six.moves import xrange
import tensorflow as tf
import model as ocr_model
from flag import FLAGS
from util import pair_iter, read_vocab
import logging

logging.basicConfig(level=logging.INFO)

def create_model(session, vocab_size, forward_only):
    model = ocr_model.Model(FLAGS.size, vocab_size, FLAGS.num_layers,
                            FLAGS.max_gradient_norm, FLAGS.learning_rate,
                            FLAGS.learning_rate_decay_factor,
                            forward_only=forward_only,
                            optimizer=FLAGS.optimizer,
                            decode=FLAGS.decode)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    num_epoch = 0
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        num_epoch = int(ckpt.model_checkpoint_path.split('-')[1])
        print (num_epoch)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements()
                                            for v in tf.trainable_variables()))
    return model, num_epoch


def validate(model, sess, x_dev, y_dev):
    valid_costs, valid_lengths = [], []
    for source_tokens, source_mask, target_tokens, target_mask in pair_iter(x_dev,
                                                                            y_dev,
                                                                            FLAGS.batch_size,
                                                                            FLAGS.num_layers,
                                                                            max_seq_len=FLAGS.max_seq_len,
                                                                            sort_and_shuffle=False):
        cost = model.test(sess, source_tokens, source_mask, target_tokens, target_mask)
        valid_costs.append(cost * target_mask.shape[1])
        valid_lengths.append(np.sum(target_mask[1:, :]))
    valid_cost = sum(valid_costs) / float(sum(valid_lengths))
    return valid_cost


def train():
    """Train a translation model using NLC data."""
    # Prepare NLC data.
    logging.info("Get NLC data in %s" % FLAGS.data_dir)
    x_train = pjoin(FLAGS.data_dir, 'train.ids.x')
    y_train = pjoin(FLAGS.data_dir, 'train.ids.y')
    x_dev = pjoin(FLAGS.data_dir, FLAGS.dev + '.ids.x')
    y_dev = pjoin(FLAGS.data_dir, FLAGS.dev + '.ids.y')
    vocab_path = pjoin(FLAGS.voc_dir, "vocab.dat")
    vocab, _ = read_vocab(vocab_path)
    vocab_size = len(vocab)
    logging.info("Vocabulary size: %d" % vocab_size)
    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    file_handler = logging.FileHandler("{0}/log.txt".format(FLAGS.train_dir))
    logging.getLogger().addHandler(file_handler)

    # with open(os.path.join(FLAGS.train_dir, "flags.json"), 'w') as fout:
    #     json.dump(FLAGS.__flags, fout)
    with tf.Session() as sess:
        logging.info("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model, epoch = create_model(sess, vocab_size, False)

        # logging.info('Initial validation cost: %f' % validate(model, sess, x_dev, y_dev))

        if False:
            tic = time.time()
            params = tf.trainable_variables()
            num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
            toc = time.time()
            print ("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        best_epoch = 0
        previous_losses = []
        exp_cost = None
        exp_length = None
        exp_norm = None
        total_iters = 0
        start_time = time.time()
        while (FLAGS.epochs == 0 or epoch < FLAGS.epochs):
            epoch += 1
            print(epoch)
            current_step = 0

            ## Train
            epoch_tic = time.time()
            for source_tokens, source_mask, target_tokens, target_mask in pair_iter(x_train,
                                                                                    y_train,
                                                                                    FLAGS.batch_size,
                                                                                    FLAGS.num_layers,
                                                                                    max_seq_len=FLAGS.max_seq_len):
                # Get a batch and make a step.
                tic = time.time()
                grad_norm, cost, param_norm = model.train(sess,
                                                          source_tokens,
                                                          source_mask,
                                                          target_tokens,
                                                          target_mask,
                                                          dropout=FLAGS.dropout)
                toc = time.time()
                iter_time = toc - tic
                total_iters += np.sum(target_mask)
                tps = total_iters / (time.time() - start_time)
                current_step += 1
                lengths = np.sum(target_mask, axis=0)
                mean_length = np.mean(lengths)
                std_length = np.std(lengths)

                if not exp_cost:
                    exp_cost = cost
                    exp_length = mean_length
                    exp_norm = grad_norm
                else:
                    exp_cost = 0.99*exp_cost + 0.01*cost
                    exp_length = 0.99*exp_length + 0.01*mean_length
                    exp_norm = 0.99*exp_norm + 0.01*grad_norm

                cost = cost / mean_length

                print(current_step, cost)
                if current_step % FLAGS.print_every == 0:
                    logging.info('epoch %d, iter %d, cost %f, exp_cost %f, grad norm %f, param norm %f, tps %f, length mean/std %f/%f' %
                                 (epoch, current_step, cost, exp_cost / exp_length, grad_norm, param_norm, tps, mean_length, std_length))
            epoch_toc = time.time()

            ## Checkpoint
            checkpoint_path = os.path.join(FLAGS.train_dir, "best.ckpt")

            ## Validate
            valid_cost = validate(model, sess, x_dev, y_dev)

            logging.info("Epoch %d Validation cost: %f time: %f" % (epoch, valid_cost, epoch_toc - epoch_tic))

            if len(previous_losses) > 2 and valid_cost > previous_losses[-1]:
                logging.info("Annealing learning rate by %f" % FLAGS.learning_rate_decay_factor)
                sess.run(model.lr_decay_op)
                model.saver.restore(sess, checkpoint_path + ("-%d" % best_epoch))
            else:
                previous_losses.append(valid_cost)
                best_epoch = epoch
                model.saver.save(sess, checkpoint_path, global_step=epoch)
            sys.stdout.flush()

def main(_):
    train()

if __name__ == "__main__":
    tf.app.run()
