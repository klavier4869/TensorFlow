from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
sys.path.append(os.curdir)  # カレントディレクトリのファイルをインポートするための設定

import numpy as np

import tensorflow as tf
from common import *
from nikkei import NikkeiData

import matplotlib.pyplot as plt

sess = tf.InteractiveSession()
# Create the model
x = tf.placeholder(tf.float32, [None, 210], name='x-input')
y_ = tf.placeholder(tf.float32, [None, 1], name='y-input')
x_stock = tf.reshape(x, [-1, 30, 7, 1])

hidden1 = cnn2d_layer(x_stock, 1, 32, 'hidden1')
hidden2 = cnn2d_layer(hidden1, 32, 64, 'hidden2')
hidden2_flat = tf.reshape(hidden2, [-1, 8*2*64])
hidden3 = nn_layer(hidden2_flat, 8*2*64, 1024, 'hidden3')

keep_prob = tf.placeholder(tf.float32)
dropped = tf.nn.dropout(hidden3, keep_prob)
y = nn_layer(dropped, 1024, 1, 'dropout', act=tf.identity)

tf.global_variables_initializer().run()
saver = tf.train.Saver()
saver.restore(sess, 'model/cnn/model.ckpt')

nikkei = NikkeiData()
xs, ys = nikkei.fetch_test()
output = np.array(sess.run([y], {x: xs, keep_prob: 1.0}))
output = output.reshape(-1)
ys = ys.reshape(-1)
loss = abs(output-ys)
print(loss)

plt.hist(loss, bins=16)
plt.savefig('cnn.svg')
