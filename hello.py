# -*- coding: utf-8 -*-

#import numpy as np
#a = np.array([[1, 2], [3, 4]])
#w = np.array([[1, 2], [10, 20]])
#
#print(np.dot(a, w))
## [ 2.6  3. ] # plain nice old matix multiplication n x (n, m) -> m
#print(np.sum(np.expand_dims(a, -1) * w , axis=0))
## equivalent result [2.6, 3]

import tensorflow as tf
import numpy as np


a = np.array([1, 2, 1])
w = np.array([[.5, .6], [.7, .8], [.7, .8]])

a = tf.constant(a, dtype=tf.float64)
w = tf.constant(w)

with tf.Session() as sess:
  # they all produce the same result as numpy above
  print(tf.matmul(tf.expand_dims(a,0), w).eval())
  print((tf.reduce_sum(tf.multiply(tf.expand_dims(a,-1), w), axis=0)).eval())
  print((tf.reduce_sum(tf.multiply(a, tf.transpose(w)), axis=1)).eval())

  # Note tf.multiply is equivalent to "*"
  print((tf.reduce_sum(tf.expand_dims(a,-1) * w, axis=0)).eval())
  print((tf.reduce_sum(a * tf.transpose(w), axis=1)).eval())