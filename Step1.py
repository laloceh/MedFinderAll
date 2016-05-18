#!/usr/local/bin/python2.7
# https://www.youtube.com/watch?v=iGZpJZhqEME

# https://github.com/bikz05/bag-of-words
# python findFeatures.py -t dataset/train/
import trainSVM
import trainRF
import trainNN
import trainGridSVM
import trainKGridSVM

import argparse as ap
import cv2
import imutils 
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
args = vars(parser.parse_args())

# Get the training classes names and store them in a list
train_path = args["trainingSet"]
training_names = os.listdir(train_path)

# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
training_classes = []

for training_name in training_names:

	if '.png' in training_name:
		className = training_name[:training_name.rfind("-")]
		image_paths.append(train_path + training_name) 
		
		if className not in training_classes:
			training_classes.append(className)
	#image_paths+=class_path
print 'Training classes', training_classes

class_id = 0
image_classes = []
for training_class in training_classes:
	count_id = 0
	for training_name in training_names:
		if '.png' in training_name:
			if training_class in training_name:
				count_id+=1
	image_classes+=[class_id]*count_id
	class_id+=1
	
print 'Image classes', image_classes
print 'Image paths', len(image_paths)

# Create feature extraction and keypoint detector objects
fea_det = cv2.FeatureDetector_create("SIFT")
des_ext = cv2.DescriptorExtractor_create("SIFT")

# List where all the descriptors are stored
des_list = []

for image_class in training_classes:
	x = 0
	print image_class
	for image_path in image_paths:
		if image_class in image_path:
			im = cv2.imread(image_path)
			kpts = fea_det.detect(im)
			kpts, des = des_ext.compute(im, kpts)
			#print len(des)
			#print len(kpts)
			print '	',x+1,image_path, len(des)
			des_list.append((image_path, des))
			#print des_list
			x+=1
print 'Des List', len(des_list)

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
print 'Descriptors',len(descriptors)

for image_path, descriptor in des_list[1:]:
	#print image_path, len(descriptor)
	descriptors = np.vstack((descriptors, descriptor))
    
print 'Desc Stack',len(descriptors)

# Perform k-means clustering
k = 100
voc, variance = kmeans(descriptors, k, 1) 
#print voc

# Calculate the histogram of features
#print 'Image paths',len(image_paths)
im_features = np.zeros((len(image_paths), k), "float32")
for i in xrange(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    #print words
    for w in words:
        im_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)
#print 'Im Features',len(im_features)

print 'Training....'
#
# Train the Linear SVM
#clf = LinearSVC()
#clf.fit(im_features, np.array(image_classes))
#print 'SVM Score:',clf.score(im_features, np.array(image_classes))

#clf = trainSVM.train(im_features, image_classes)
clf = trainKGridSVM.train(im_features, image_classes)

# Save the SVM
joblib.dump((clf, training_classes, stdSlr, k, voc), "bofSVM.pkl", compress=3)    
 

# Train the Random Forest
#clf = RandomForestClassifier(n_estimators=100)
#clf.fit(im_features, np.array(image_classes))
#print 'RF Score:',clf.score(im_features, np.array(image_classes))
clf = trainRF.train(im_features, image_classes)

# Save the RF
joblib.dump((clf, training_classes, stdSlr, k, voc), "bofRF.pkl", compress=3)    
   

# Train the Perceptron
clf = trainNN.train(im_features, image_classes)

#clf = Perceptron(n_iter=100, eta0=0.1)
#clf.fit(im_features, np.array(image_classes))
#print 'Perceptron Score:',clf.score(im_features, np.array(image_classes))

# Save the Perceptron
joblib.dump((clf, training_classes, stdSlr, k, voc), "bofNN.pkl", compress=3)    


