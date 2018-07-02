# -*- coding: utf-8 -*-
# pip install -r requirements.txt
#import numpy as np
#a = np.array([[1, 2], [3, 4]])
#w = np.array([[1, 2], [10, 20]])
#
#print(np.dot(a, w))
## [ 2.6  3. ] # plain nice old matix multiplication n x (n, m) -> m
#print(np.sum(np.expand_dims(a, -1) * w , axis=0))
## equivalent result [2.6, 3]

import tensorflow as tf
import os
from skimage import data as skimage_data
import numpy as np
import matplotlib.pyplot as plt


"""
There are six categories of traffic signs in Belgium:
warning signs,
priority signs,
prohibitory signs,
mandatory signs,
signs related to parking and standing still on the road and,
lastly, designatory signs.
"""

a = [
  [1,2],
  [3,4]
]
b = [
  [10, 20],
  [100, 200]
]


x1 = tf.constant(a)
x2 = tf.constant(b)
result = tf.multiply(x1, x2)
my_variable = tf.get_variable("my_variable", [1, 2, 3])


with tf.Session() as sess:
  output = sess.run(result)
  print(output)


def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage_data.imread(f))
            labels.append(int(d))
    return images, labels


ROOT_PATH = "./"
train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training")
test_data_driectory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")

images, labels = load_data(train_data_directory)


# convert data sets to np arrays
np_images = np.array(images)
np_labels = np.array(labels)


print('np_images[0]', np_images[0])
print('np_images.size', np_images.size)
print('np_labels.size', np_labels.size)

# print('np_images.flags', np_images.flags)
# print('np_images.itemsize', np_images.itemsize)
# print('np_images.nbytes', np_images.nbytes)


plt.hist(labels, 62)
plt.show()

"""
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
"""
