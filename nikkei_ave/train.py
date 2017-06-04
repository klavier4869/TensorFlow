"""classifier model
    (|affine| → |relu|)*5 → |dropout|
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
    x = tf.placeholder(tf.float32, [None, 150], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 1], name='y-input')

  hidden1 = nn_layer(x, 150, 1000, 'hidden1')
  hidden2 = nn_layer(hidden1, 1000, 750, 'hidden2')
  hidden3 = nn_layer(hidden2, 750, 500, 'hidden3')
  hidden4 = nn_layer(hidden3, 500, 250, 'hidden4')
  hidden5 = nn_layer(hidden4, 250, 125, 'hidden5')

  # Do not apply softmax activation yet, see below.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden5, keep_prob)
  y = nn_layer(dropped, 125, 1, 'dropout', act=tf.identity)

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

  saver = tf.train.Saver()
  save_path = saver.save(sess, "model/dnn/model.ckpt")
  print("Model saved in file: %s" % save_path)

def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()

if __name__ == '__main__':
  parser = initArgParser()
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
