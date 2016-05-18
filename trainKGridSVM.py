#!/usr/local/bin/python2.7
#!/usr/bin/python

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
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
	print 'SVM ==>' + mean_score(scores)
	print ''
	
def testGridSearch(clf, parameters, im_features, image_classes):
	grid = GridSearchCV(clf, parameters)
	grid.fit(im_features, image_classes)
	
	return grid.best_estimator_
	

def train(im_features, image_classes):
	
	# Train the Linear SVM
	#clf = LinearSVC()
	
	parameters = {'kernel': ['rbf', 'linear'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}
                    
	clf = SVC(probability=True)
	#print clf
	#print '=='
	
	clf = testGridSearch(clf, parameters, im_features, image_classes)
	#print clf
	n_folds = 10
	evaluate_cross_validation(clf, im_features, image_classes, n_folds)
	
	return clf
