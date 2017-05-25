"""classifier model
    |affine|→|softmax|→|cross-entropy|
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
sys.path.append(os.curdir)  # カレントディレクトリのファイルをインポートするための設定

import tensorflow as tf
from common import *
from nikkei import NikkeiData

def train():
  # Import data
  nikkei = NikkeiData()

  sess = tf.InteractiveSession()
  # Create the model
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 210], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 1], name='y-input')
    x_stock = tf.reshape(x, [-1, 30, 7, 1])

  hidden1 = cnn2d_layer(x_stock, 1, 32, 'hidden1')
  hidden2 = cnn2d_layer(hidden1, 32, 64, 'hidden2')
  hidden2_flat = tf.reshape(hidden2, [-1, 7*7*64])
  hidden3 = nn_layer(hidden2_flat, 7*7*64, 1024, 'hidden3')

  # Do not apply softmax activation yet, see below.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden3, keep_prob)
  y = nn_layer(dropped, 1024, 1, 'dropout', act=tf.identity)

  with tf.name_scope('loss'):
    diff = tf.nn.l2_loss(y_ - y)
    with tf.name_scope('total'):
      loss = tf.reduce_mean(diff)
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
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
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

  for i in range(FLAGS.max_steps):
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
  train_writer.close()
  test_writer.close()

def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()

if __name__ == '__main__':
  parser = initArgParser()
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
