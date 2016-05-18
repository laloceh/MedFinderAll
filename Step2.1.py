#!/usr/local/bin/python2.7
# https://github.com/bikz05/bag-of-words
# python getClass.py -t dataset/test --visualize

# Sinle Image
# python getClass.py -a svm -i dataset/test/aeroplane/test_1.jpg --visualize

import argparse as ap
import collections
import cv2
import imutils 
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.externals import joblib
from scipy.cluster.vq import *

def allClassifiers(image_paths):
	pkl_list = ['bofNN.pkl', 'bofRF.pkl', 'bofSVM.pkl']
	pred = []
	for pkl in pkl_list:
		clf, classes_names, stdSlr, k, voc = joblib.load(pkl)
	
		# Create feature extraction and keypoint detector objects
		fea_det = cv2.FeatureDetector_create("SIFT")
		des_ext = cv2.DescriptorExtractor_create("SIFT")

		# List where all the descriptors are stored
		des_list = []

		for image_path in image_paths:
			im = cv2.imread(image_path)
			if im == None:
				print "No such file {}\nCheck if the file exists".format(image_path)
				exit()
			kpts = fea_det.detect(im)
			kpts, des = des_ext.compute(im, kpts)
			des_list.append((image_path, des))   
			
		# Stack all the descriptors vertically in a numpy array
		descriptors = des_list[0][1]
		for image_path, descriptor in des_list[0:]:
			descriptors = np.vstack((descriptors, descriptor)) 

		# 
		test_features = np.zeros((len(image_paths), k), "float32")
		for i in xrange(len(image_paths)):
			words, distance = vq(des_list[i][1],voc)
			for w in words:
				test_features[i][w] += 1

		# Perform Tf-Idf vectorization
		nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
		idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

		# Scale the features
		test_features = stdSlr.transform(test_features)

		# Perform the predictions
		predictions =  [classes_names[i] for i in clf.predict(test_features)]
		
		pred.append(predictions[0])
	return pred


# Get the path of the testing set
parser = ap.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-t", "--testingSet", help="Path to testing Set")
group.add_argument("-i", "--image", help="Path to image")
parser.add_argument('-v',"--visualize", action='store_true')
parser.add_argument('-c', '--classifier', help='Classifier to use (svm, nn, rf, all)',
					required="True")
args = vars(parser.parse_args())

# Get the path of the testing image(s) and store them in a list
image_paths = []
image_paths = [args["image"]]

# Get the algorithm to use
classifier = args['classifier']
if classifier == 'nn' :
	print 'Using perceptron'
	# Load the classifier, class names, scaler, number of clusters and vocabulary 
	clf, classes_names, stdSlr, k, voc = joblib.load("bofNN.pkl")
elif classifier == 'rf':
	print 'Using random forest'
	# Load the classifier, class names, scaler, number of clusters and vocabulary 
	clf, classes_names, stdSlr, k, voc = joblib.load("bofRF.pkl")
elif classifier == 'svm':
	print 'Using svm'
	# Load the classifier, class names, scaler, number of clusters and vocabulary 
	clf, classes_names, stdSlr, k, voc = joblib.load("bofSVM.pkl")
elif classifier == 'all':
	print 'Using all classifiers, concensus'
	
if classifier == 'all':
	pred = allClassifiers(image_paths)
	print pred
	counter = collections.Counter(pred)
	print counter.most_common(1)
	if counter.most_common(1)[0][1] == 1:
		predictions = ['Try again, we could not find a good match']
	else:
		predictions = counter.most_common(1)[0]
	
else: 

	# Create feature extraction and keypoint detector objects
	fea_det = cv2.FeatureDetector_create("SIFT")
	des_ext = cv2.DescriptorExtractor_create("SIFT")

	# List where all the descriptors are stored
	des_list = []

	for image_path in image_paths:
		im = cv2.imread(image_path)
		if im == None:
			print "No such file {}\nCheck if the file exists".format(image_path)
			exit()
		kpts = fea_det.detect(im)
		kpts, des = des_ext.compute(im, kpts)
		des_list.append((image_path, des))   
		
	# Stack all the descriptors vertically in a numpy array
	descriptors = des_list[0][1]
	for image_path, descriptor in des_list[0:]:
		descriptors = np.vstack((descriptors, descriptor)) 

	# 
	test_features = np.zeros((len(image_paths), k), "float32")
	for i in xrange(len(image_paths)):
		words, distance = vq(des_list[i][1],voc)
		for w in words:
			test_features[i][w] += 1

	# Perform Tf-Idf vectorization
	nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
	idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

	# Scale the features
	test_features = stdSlr.transform(test_features)

	# Perform the predictions
	predictions =  [classes_names[i] for i in clf.predict(test_features)]
	if classifier == 'rf':
		print predictions
		print 1-max(clf.predict_proba(test_features)[0])

	elif classifier == 'nn':
		print predictions
		print clf.decision_function(test_features)
	else:
		print predictions
		print 1-max(clf.predict_proba(test_features)[0])

print 'Result ==>',predictions[0]
# Visualize the results, if "visualize" flag set to true by the user
if args["visualize"]:
    for image_path, prediction in zip(image_paths, predictions):
        image = cv2.imread(image_path)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        pt = (0, 3 * image.shape[0] // 4)
        cv2.putText(image, prediction, pt ,cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 0, 255], 2)
        cv2.imshow("Image", image)
        cv2.waitKey(3000)
