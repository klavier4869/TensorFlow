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

hidden1 = nn_layer(x, 210, 1000, 'hidden1')
hidden2 = nn_layer(hidden1, 1000, 750, 'hidden2')
hidden3 = nn_layer(hidden2, 750, 500, 'hidden3')
hidden4 = nn_layer(hidden3, 500, 250, 'hidden4')
hidden5 = nn_layer(hidden4, 250, 125, 'hidden5')

# Do not apply softmax activation yet, see below.
keep_prob = tf.placeholder(tf.float32)
dropped = tf.nn.dropout(hidden5, keep_prob)
y = nn_layer(dropped, 125, 1, 'dropout', act=tf.identity)

tf.global_variables_initializer().run()
saver = tf.train.Saver()
saver.restore(sess, 'model/dnn/model.ckpt')

nikkei = NikkeiData()
xs, ys = nikkei.fetch_test()
output = np.array(sess.run([y], {x: xs, keep_prob: 1.0}))
output = output.reshape(-1)
ys = ys.reshape(-1)
loss = abs(output-ys)
print(loss)

plt.hist(loss, bins=16)
plt.savefig('dnn.svg')
