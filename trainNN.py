#!/usr/local/bin/python2.7
#!/usr/bin/python

from sklearn.linear_model import Perceptron
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from scipy.stats import sem
import numpy as np

def mean_score(score):
	return ("KFold Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(score), sem(score))
	
def evaluate_cross_validation(clf, X, y, K):
	# create a 10 k-fold cross validation iterator
	cv = KFold(len(y), K, shuffle=True, random_state=42)
	
	scores = cross_val_score(clf, X, y, cv=cv)
	print scores
	print 'NN ==>' + mean_score(scores)
	print ''

def train(im_features, image_classes):
	
	# Train the Perceptron
	clf = Perceptron(n_iter=100, eta0=0.1)
	
	clf.fit(im_features, np.array(image_classes))
	n_folds = 10
	kFoldScore = evaluate_cross_validation(clf, im_features, np.array(image_classes), n_folds)
	
	#print 'SVM Score:',clf.score(im_features, np.array(image_classes))
	#print 'SVM Score:', kFoldScore
	return clf
