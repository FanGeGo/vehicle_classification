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
# This file contains helper functions that can be used for general    #
# puposes						                                      #
#######################################################################

import pandas as pd
import os.path
import os
from PIL import Image
import sys
from os.path import basename
import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
import json
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_HEIGHT = 1280
IMAGE_WIDTH = 720
NUM_CHANNELS = 3

def get_batch(dataset, batch_size, batch_number):
	"""
	This function split the dataset into batches and return a specific batch number
	Args:
	dataset: 
	batch_size: specify the size of each batch
	batch_number: specify which batch to return
	Return
	Return a specific batch of the dataset as X and label 
	"""
	X = dataset['X']
	Y = dataset['Y']

	number_of_chunks = len(X)/batch_size
	batches_X = np.array_split(X,number_of_chunks);
	batches_Y = np.array_split(Y,number_of_chunks);
	if batch_number >= number_of_chunks:
		return [], []
	return batches_X[batch_number], batches_Y[batch_number]

def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
	"""
	This function has been taken from: 
	https://github.com/martin-gorner/tensorflow-mnist-tutorial/blob/master/mnist_4.2_batchnorm_convolutional.py
	"""    
	exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
	bnepsilon = 1e-5
	if convolutional:
		mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
	else:
		mean, variance = tf.nn.moments(Ylogits, [0])
	update_moving_everages = exp_moving_avg.apply([mean, variance])
	m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
	v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
	Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
	return Ybn, update_moving_everages


def read_dataset(dataset_path, resize = 0, split_teain_test = True, split_ratio = 0.7):
	"""Reads a .txt file containing pathes and labeles
	Args:
	   image_list_file: a .csv file -- the name of the csv file that contains labels with images names
	   resize: to resize images into a specific square size, default value means no resizing, 
	   split_teain_test: a boolean indicator, to split the dataset into two sets {train, test}
	   split_ratio: by default it is 70% for training and 30% for testing
	Returns:
	   dictionary of samples with labels
	"""          
	dataset = pd.DataFrame.from_csv(dataset_path, index_col = None, encoding='utf-8')  
	# Shuffle the dataset
	dataset = dataset.sample(frac=1).reset_index(drop = True)
	filenames = []
	images = []
	blury_images = []
	cropped_images = []
	labels = []
	#total_number = dataset.shape[0]
	total_number = 3000 # This is because of memory limit :(
	classes = dataset.Class.tolist()
	
	for i in range(dataset.shape[0]): 
		if i == 3000:
			break         	
		video_name = dataset['Video File Name'][i]
		video_name = os.path.splitext(video_name)[0]
		file_path = BASE_DIR + '/dataset/Video Data-03-08-2015/' + video_name + '/images/'
		image_name = dataset['Link'][i].split('\\')[1]
		filename = file_path + image_name

		# JSON file path that contains boxes locations of the detected objects
		json_path = file_path + '/out/' + image_name.split('.')[0] + '.json'		
		filenames.append(filename)                
		img = cv2.imread(filename,3)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)				
		blury_image, cropped_image = blur_image_around_object(img, json_path)
		
		# For test	
		#if (i > 2143):
			#cv2.imshow('Cropped Image',cropped_image)
			#cv2.waitKey(0)
		
		if (resize != 0):
		    img = cv2.resize(img, (resize, resize)) 
		    blury_image = cv2.resize(blury_image, (resize, resize)) 
		    cropped_image = cv2.resize(cropped_image, (resize, resize)) 

		images.append(img)        
		blury_images.append(blury_image)
		cropped_images.append(cropped_image)

		# Convert labels into one hot
		label = np.zeros(10)

		np.put(label,int(classes[i])-1,1)
		labels.append(label)

	if split_teain_test == True:
		
		max_train_index = int(math.floor(total_number*split_ratio))    	
		train_images = images[0:max_train_index]
		train_labels = labels[0:max_train_index]
		test_images = blury_images[max_train_index:total_number+1]
		test_labels = labels[max_train_index:total_number+1]
		blury_images_set = blury_images[0:max_train_index]
		cropped_images_set = cropped_images[0:max_train_index]

	train_set = {"X":train_images, "Y": train_labels}  
	test_set = {"X":test_images, "Y": test_labels} 
	blury_set = {"X":blury_images_set, "Y": train_labels} 
	cropped_set = {"X":cropped_images_set, "Y": train_labels} 
	dataset = {"train_set":train_set, "test_set": test_set, 'blury_set':blury_set, 'cropped_set': cropped_set}  
	return dataset

def blur_image_around_object(cv_image, json_path):		
	try:
		json_data = open(json_path)
		boxes = json.load(json_data)
	except:
		return cv_image, cv_image

	# if there is only one box then it must be the one
	if (len(boxes) == 1):
		x1 = boxes[0]['topleft']['x']
		y1 = boxes[0]['topleft']['y']
		x2 = boxes[0]['bottomright']['x']
		y2 = boxes[0]['bottomright']['y']
	else:
		# if there is more than one box, well it is a problem :( 
		# we need to check which box is the right one,
		# we have three rules for that, 
		# 1. Remove very small 
		# 2. Remove boxes that have less confidence
		# 3. Remove boxes are too far from the center
		# TODO: remove all unwanted boxes 
		areas = []
		horizental_locations = []
		confidences = []
		horizental_threshold_1 = 800
		horizental_threshold_2 = 200
		for i in range(len(boxes)):			
			x1 = boxes[i]['topleft']['x']
			y1 = boxes[i]['topleft']['y']
			x2 = boxes[i]['bottomright']['x']
			y2 = boxes[i]['bottomright']['y']			
			area = (x2 - x1)*(y2 - y1)
			horizental_location = x1
			if (x1 > horizental_threshold_1) or (x1 < horizental_threshold_2):
				confidence = 0
			else:
				confidence = boxes[i]['confidence']
			
			label = boxes[i]['label']
			areas.append(area)
			horizental_locations.append(horizental_location)
			confidences.append(confidence)

		np_confidences = np.array(confidences)
		max_confidence_index = np.where(np_confidences==max(confidences))[0]
		
		if (max_confidence_index.shape[0] == 1):
			best_box = boxes[max_confidence_index[0]]
		else:
			# Take the bigger one
			propsed_boxes = []
			proposed_areas = []
			for i in range(max_confidence_index.shape[0]):				
				proposed_areas.append(areas[max_confidence_index[i]])			
				propsed_boxes.append(boxes[max_confidence_index[i]])

			best_area_with_confidence = proposed_areas.index(max(proposed_areas))
			best_box = propsed_boxes[best_area_with_confidence]

		x1 = best_box['topleft']['x']
		y1 = best_box['topleft']['y']
		x2 = best_box['bottomright']['x']
		y2 = best_box['bottomright']['y']

	blur_image = cv2.blur(cv_image,(25,25)) #You can change the kernel size as you want
	object_image = cv_image[y1:y2, x1:x2]
	blur_image[y1:y2, x1:x2] = object_image
	cropped_image = object_image
	# For test	
	cv2.imshow('Averaging',cv_image)
	cv2.waitKey(0)
	cv2.imshow('Averaging',blur_image)
	cv2.waitKey(0)
	cv2.imshow('Averaging',cropped_image)
	cv2.waitKey(0)
	
	result = [blur_image, cropped_image]
	return result

def init_weights(shape, init_method='xavier', xavier_params = (None, None)):
	'''
	This function has been taken from here: 
	https://github.com/delip/blog-stuff/blob/master/tensorflow_ufp.ipynb
	'''
	if init_method == 'zeros':
		return tf.Variable(tf.zeros(shape, dtype=tf.float32))
	elif init_method == 'uniform':
		return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float32))
	else: #xavier
		(fan_in, fan_out) = xavier_params
		low = -4*np.sqrt(6.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1} 
		high = 4*np.sqrt(6.0/(fan_in + fan_out))
		return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))

# Reads pfathes of images together with their labels
# dataset_path = 'Video Data-03-08-2015/dataset.csv'
# training_set = read_labeled_image_list(dataset_path)

# samples = training_set['X']
# print len(training_set['X'])
# # Here is how to read image using opencv (Only for test)
# for i in range(len(samples)):
#     img = cv2.imread(samples[i],3)
#     #print img
#     #plt.imshow(img, interpolation = 'bicubic')
#     #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#     #plt.show()


def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
	'''
	This function is taken from the following link:
	http://stackoverflow.com/questions/2148543/how-to-write-a-confusion-matrix-in-python
	'''
	
	plt.matshow(df_confusion, cmap=cmap) # imshow
	#plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(df_confusion.columns))
	plt.xticks(tick_marks, df_confusion.columns, rotation=45)
	plt.yticks(tick_marks, df_confusion.index)
	#plt.tight_layout()
	plt.ylabel(df_confusion.index.name)
	plt.xlabel(df_confusion.columns.name)
	plt.show()

# print 'Finished!'

def one_hot_decode(data):
	out = []
	for i in range(len(data)):
		decode = tf.argmax(data[i], axis=0)
		out.append(decode)
	return out