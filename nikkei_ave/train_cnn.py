"""classifier model
    |conv| → |relu| → |pool| → |affine| →  |relu| →  |dropout|
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.curdir)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from common import *
from nikkei import NikkeiData

def train():
  # Import data
  nikkei = NikkeiData()

  sess = tf.InteractiveSession()
  # Create the model
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, nikkei.in_size], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, nikkei.out_size], name='y-input')
    x_stock = tf.reshape(x, [-1, nikkei.in_days, nikkei.elem_size, 1])

  hidden1 = cnn2d_layer(x_stock, 1, 32, 'hidden1')
  hidden1_flat = tf.reshape(hidden1, [-1, 15*3*32])
  hidden2 = nn_layer(hidden1_flat, 15*3*32, 1024, 'hidden2')

  # Do not apply softmax activation yet, see below.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden2, keep_prob)
  y = nn_layer(dropped, 1024, 1, 'dropout', act=tf.identity)

  with tf.name_scope('loss'):
    diff = tf.nn.l2_loss(y_ - y)
    with tf.name_scope('total'):
      l2 = tf.reduce_mean(diff)
    with tf.name_scope('weight_decay'):
      tf.add_to_collection('losses', l2)
      loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
  tf.summary.scalar('loss', loss)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        loss)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(y, y_)
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/tb/cnn/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/tb/cnn/test')
  tf.global_variables_initializer().run()

  # Train
  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train:
        xs, ys = nikkei.fetch_train()
        k = FLAGS.dropout
    else:
        xs, ys = nikkei.fetch_test()
        k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  def save_hist(step):
    """ save histgram svg """
    xs, ys = nikkei.fetch_test()
    output = np.array(sess.run([y], {x: xs, keep_prob: 1.0}))
    output = output.reshape(-1)
    ys = ys.reshape(-1)
    loss = abs(output-ys)
    plt.figure()
    plt.hist(loss, bins=16)
    plt.savefig(FLAGS.log_dir + '/svg/cnn_' + str(step) + '.svg')

  for i in range(FLAGS.max_steps+1):
    if i % 10 == 0:  # Record summaries and test-set accuracy
      summary, acc = sess.run([merged, accuracy],
                        feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:  # Record train set summaries, and train
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
    if i % FLAGS.hist_interval == 0:
      save_hist(i)
  train_writer.close()
  test_writer.close()

  # save train model
  #saver = tf.train.Saver()
  #save_path = saver.save(sess, FLAGS.log_dir +
  #                  '/model/cnn_' + str(FLAGS.max_steps) + '_model.ckpt')
  #print("Model saved in file: %s" % save_path)


def main(_):
  # init tensor_borad_logdir
  tensor_borad_logdir = FLAGS.log_dir + '/tb/cnn'
  if tf.gfile.Exists(tensor_borad_logdir):
    tf.gfile.DeleteRecursively(tensor_borad_logdir)
  tf.gfile.MakeDirs(tensor_borad_logdir)
  # init histgram_logdir
  histgram_logdir = FLAGS.log_dir + '/svg'
  if not tf.gfile.Exists(histgram_logdir):
    tf.gfile.MakeDirs(histgram_logdir)
  train()

if __name__ == '__main__':
  parser = initArgParser()
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
