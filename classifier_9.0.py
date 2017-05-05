#!/usr/bin/python3

# Copyright 2017. All Rights Reserved.
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
# =============================================================================


#######################################################################
# Description:                                                        # 
# This classifier took only one datasource, 
# Three conv layers with norm and pooling and three FC layers
#
# The purpose of this classifier is to make experiments with only one 
# data source, the data source for this blurred source
#######################################################################

import tensorflow as tf
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import initializer
import os.path
import os
import sys
import cv2
from matplotlib import pyplot as plt
import numpy as np
import util
import math
import csv
import pandas as pd

# Training set
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = BASE_DIR + '/dataset/' + 'Video Data-03-08-2015/extended_dataset.csv'
dataset = util.read_dataset(dataset_path, 244, split_teain_test = True, split_ratio = 0.7)
log_file = BASE_DIR + '/logs/classifier_9.1_blurry_test.csv'
train_set = dataset['train_set']
test_set = dataset['test_set']
blurry_set = dataset['blury_set']
cropped_set = dataset['cropped_set']
number_of_samples = len(train_set['X'])


f = open(log_file, 'wt')
writer = csv.writer(f)
writer.writerow( ('Epoch', 'Iteration', 'Phase', 'Accuracy', 'Loss') )
# For test purpose
# print train_set['X'][0].shape
# plt.imshow(train_set['X'][0], interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()

IMAGE_HEIGHT = 244
IMAGE_WIDTH = 244
NUM_CHANNELS = 3
NUMBER_OF_CLASSES = 10
# Layers depth
stage1_L1 = 8
stage1_L2 = 16
stage1_L3 = 32
stage1_L4 = 10000
stage1_L5 = 1000

# Layers depth
stage2_L1 = 8
stage2_L2 = 16
stage2_L3 = 32
stage2_L4 = 10000
stage2_L5 = 1000

# Layers depth
stage3_L1 = 8
stage3_L2 = 16
stage3_L3 = 32
stage3_L4 = 10000
stage3_L5 = 1000

L6 = 1000
L7 = 1000
L8 = 500
L9 = 200
L10 = 100
L11 = NUMBER_OF_CLASSES

# Network variables
x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
x_blury = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
x_cropped = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])

initializer = tf.contrib.layers.xavier_initializer()

w11 = tf.Variable(initializer(shape=[7, 7, NUM_CHANNELS, stage1_L1]))
b11 = tf.Variable(tf.ones([stage1_L1])/10) # L1 is the number of output channels

w12 = tf.Variable(initializer(shape=[5, 5, stage1_L1, stage1_L2]))
b12 = tf.Variable(tf.ones([stage1_L2])/10) # L2 is the number of output channels

w13 = tf.Variable(initializer(shape=[5, 5, stage1_L2, stage1_L3]))
b13 = tf.Variable(tf.ones([stage1_L3])/10) # L3 is the number of output channels

# Second stage
w21 = tf.Variable(initializer(shape=[7, 7, NUM_CHANNELS, stage2_L1]))
b21 = tf.Variable(tf.ones([stage2_L1])/10) # L1 is the number of output channels

w22 = tf.Variable(initializer(shape=[5, 5, stage2_L1, stage2_L2]))
b22 = tf.Variable(tf.ones([stage2_L2])/10) # L2 is the number of output channels

w23 = tf.Variable(initializer(shape=[5, 5, stage2_L2, stage2_L3]))
b23 = tf.Variable(tf.ones([stage2_L3])/10) # L3 is the number of output channels

# Third stage
w31 = tf.Variable(initializer(shape=[7, 7, NUM_CHANNELS, stage3_L1]))
b31 = tf.Variable(tf.ones([stage3_L1])/10) # L1 is the number of output channels

w32 = tf.Variable(initializer(shape=[5, 5, stage2_L1, stage3_L2]))
b32 = tf.Variable(tf.ones([stage3_L2])/10) # L2 is the number of output channels

w33 = tf.Variable(initializer(shape=[5, 5, stage2_L2, stage3_L3]))
b33 = tf.Variable(tf.ones([stage3_L3])/10) # L3 is the number of output channels

# Fully connected layers
w14 = tf.Variable(initializer(shape=[16*16*stage1_L3, stage1_L4]))
b14 = tf.Variable(tf.ones([stage1_L4])/10) # L4 is the number of output channels

# Fully connected layers
w15 = tf.Variable(initializer(shape=[stage1_L4, stage1_L5]))
b15 = tf.Variable(tf.ones([stage1_L5])/10) # L4 is the number of output channels

# Fully connected layers
# w24 = tf.Variable(initializer(shape=[16*16*stage2_L3, stage2_L4]))
# b24 = tf.Variable(tf.ones([stage2_L4])/10) # L4 is the number of output channels

# w25 = tf.Variable(initializer(shape=[stage2_L4, stage2_L5]))
# b25 = tf.Variable(tf.ones([stage2_L5])/10) # L4 is the number of output channels

# # Fully connected layers
# w34 = tf.Variable(initializer(shape=[16*16*stage3_L3, stage3_L4]))
# b34 = tf.Variable(tf.ones([stage3_L4])/10) # L4 is the number of output channels

# w35 = tf.Variable(initializer(shape=[stage3_L4, stage3_L5]))
# b35 = tf.Variable(tf.ones([stage3_L5])/10) # L4 is the number of output channels

w6 = tf.Variable(initializer(shape=[L6, L11]))
b6 = tf.Variable(tf.ones([L11])/10) # 13 is the number of output channels

# w7 = tf.Variable(initializer(shape=[L7, L8]))
# b7 = tf.Variable(tf.ones([L8])/10) # 13 is the number of output channels

# w8 = tf.Variable(initializer(shape=[L8, L9]))
# b8 = tf.Variable(tf.ones([L9])/10) # 13 is the number of output channels

# w9 = tf.Variable(initializer(shape=[L9, L10]))
# b9 = tf.Variable(tf.ones([L10])/10) # 13 is the number of output channels

# w10 = tf.Variable(initializer(shape=[L10, L11]))
b10 = tf.Variable(tf.ones([L11])/10) # 13 is the number of output channels
# To calculate the features size 
# More http://cs231n.github.io/convolutional-networks/#conv


# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float32)


################## Stage 1 #######################

# Convulational layer1
stride = 1  # output is still 112x112x8
Ycnv = tf.nn.conv2d(x_blury, w11, strides=[1, stride, stride, 1], padding='SAME')
Y1bn = tf.contrib.layers.batch_norm(Ycnv, center=True, scale=True,  is_training=True, scope='bn11')
y11rl = tf.nn.relu(Y1bn + b11)
y11 = tf.nn.max_pool(y11rl, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print y11.shape

# Convulational layer 2
stride = 2  # output is  28x28x16
Ycnv = tf.nn.conv2d(y11, w12, strides=[1, stride, stride, 1], padding='SAME')
Y1bn = tf.contrib.layers.batch_norm(Ycnv, center=True, scale=True,  is_training=True, scope='bn12')
y12rl = tf.nn.relu(Y1bn + b12)
y12 = tf.nn.max_pool(y12rl, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print y12.shape

# Convulational layer3
stride = 2  # output is  16x16x32
Ycnv = tf.nn.conv2d(y12, w13, strides=[1, stride, stride, 1], padding='SAME')
Y1bn = tf.contrib.layers.batch_norm(Ycnv, center=True, scale=True,  is_training=True, scope='bn13')
y13 = tf.nn.relu(Y1bn + b13)
#y3 = tf.nn.max_pool(y3rl, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print y13.shape

# reshape the output from the third convolution for the fully connected layer
yy = tf.reshape(y13, shape=[-1, 16 * 16 * stage1_L3])

y14 = tf.nn.relu(tf.matmul(yy, w14) + b14)
y14d = tf.nn.dropout(y14, pkeep)

print y14.shape

y15 = tf.nn.relu(tf.matmul(y14d, w15) + b15)
y15d = tf.nn.dropout(y15, pkeep)
print y15.shape

# concatination of stage1 and stage 2 
y5 = y15d
print y5.shape

# Fully connected layer 5
# y6 = tf.nn.relu(tf.matmul(y5, w6) + b6)
# print y6.shape

# Fully connected layer 10
Ylogits = tf.matmul(y5, w6) + b6
y = tf.nn.softmax(Ylogits)
print y.shape
y_decoded = tf.argmax(y, axis=1)

# labels
label = tf.placeholder(tf.float32, [None, NUMBER_OF_CLASSES])
label_decoded = tf.argmax(label, axis=1)

# loss function
#cross_entropy = -tf.reduce_sum(label * tf.log(y))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels = label)
cross_entropy = tf.reduce_mean(cross_entropy)*100
# % of correct answeres by comparing the maximum index of the predicted values and the truth ground
is_correct = tf.equal(tf.argmax(y,1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
accuracy_test = tf.reduce_mean(tf.cast(is_correct, tf.float32))
#tf.summary.scalar("cost_function", cross_entropy)
#tf.summary.scalar("accuracy", accuracy)

# learning rate decay
max_learning_rate = 0.003
min_learning_rate = 0.0001
decay_speed = 2000.0 # 0.003-0.0001-2000=>0.9826 done in 5000 iterations
i = 0
learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
#learning_rate = 0.0003
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# Initialize all variables 
init = tf.global_variables_initializer()

# Merge all summaries into a single operator
merged_summary_op = tf.summary.merge_all()

sess = tf.Session()
sess.run(init)

# Set the logs writer to the folder /tmp/tensorflow_logs
#summary_writer = tf.summary.FileWriter(BASE_DIR + '/logs',graph_def=sess.graph_def)

batch_size = 100
number_of_batches = number_of_samples / batch_size
for epoch in range(10):
	for i in range(number_of_batches):
		batch_x, batch_y = util.get_batch(train_set, batch_size, i)
		batch_x_blury, _ = util.get_batch(blurry_set, batch_size, i)
		batch_x_cropped, _ = util.get_batch(cropped_set, batch_size, i)
		train_data = {x: batch_x, label: batch_y, x_blury: batch_x_blury, x_cropped: batch_x_cropped}
		sess.run(train_step, feed_dict={x: batch_x, label: batch_y, x_blury: batch_x_blury, x_cropped: batch_x_cropped, pkeep: 0.75})			
		# Write logs for each iteration
		
		#summary, _ = sess.run([merged_summary_op, train_step], feed_dict=train_data)
		#summary_writer.add_summary(summary, epoch*number_of_batches + i)			

		# Write the accuracy for every iteration
		a,c = sess.run([accuracy, cross_entropy], feed_dict = {x: batch_x, label: batch_y, x_blury: batch_x_blury, x_cropped: batch_x_cropped, pkeep: 1.0})
		print 'Epoch ' + str(epoch) + ' iteration#:' + str(i) + ' Training accuracy:' + str(a)		
		writer.writerow( (epoch, i, 'train', a, c))

		

		if (i % 10) == 0:		
			test_data={x: test_set['X'], label: test_set['Y']}
			a,c = sess.run([accuracy_test, cross_entropy], feed_dict = {x_blury: test_set['X'], label: test_set['Y'], pkeep: 1.0})
			#tf.summary.scalar("Testing accuracy", a)
			print 'Epoch ' + str(epoch) + ' iteration#:' + str(i) + ' Testing accuracy:' + str(a)
			writer.writerow( (epoch, i, 'test', a, c))


# Find the confusion matrix		
y_d, l_d = sess.run([y_decoded, label_decoded], feed_dict={x_blury: test_set['X'], label: test_set['Y'], pkeep: 1.0})	
writer.writerow( (y_d, 0, 'test', 0, 0))
writer.writerow( (l_d, 0, 'test', 0, 0))
y_actu = pd.Series(l_d, name='Actual')
y_pred = pd.Series(y_d, name='Predicted')

df_confusion = pd.crosstab(y_actu, y_pred)			
util.plot_confusion_matrix(df_confusion)


saver_a = tf.train.Saver()
saver_a.save(sess, BASE_DIR + "/models/classifier_9_1_blurry_test")	
f.close()