#!/usr/local/bin/python2.7
#!/usr/bin/python

import transform_image

import imutils 
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
import cv2

# ----------------MAIN --------------------	
print '***WAIT***'

print 'Transforming images'
class_dictionary = {}
#path = 'images/'
path = str(raw_input('Images\' folder to convert: '))
path = path + '/'
imsize = 200 #pixels
total = transform_image.imageNames(path, class_dictionary, imsize)
#print class_dictionary
#print ''
#print '***READ AND RESIZE IMAGES FINISHED***'
print '==== %d images processed ====' % total
#print ''
#print '//A dictionary with the classes was created\\'
imageClasses = class_dictionary.keys()
#print imageClasses
#print 'Total classes:', len(imageClasses)
