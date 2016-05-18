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

def testGridSearch(svr, parameters, im_features, image_classes):
	clf = GridSearchCV(svr, parameters)
	clf.fit(im_features, image_classes)
	
	

def train(im_features, image_classes):
	
	# Train the Linear SVM
	#clf = LinearSVC()
	
	tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
                    
	#svr = SVC(probability=True)
	
	#best_params = testGridSearch(svr, parameters, im_features, image_classes)
	
	scores = ['precision', 'recall']

	for score in scores:
		print "# Tuning hyper-parameters for %s" % score
		print ''

		clf = GridSearchCV(SVC(probability=True), tuned_parameters, cv=10, scoring='%s_weighted' % score)
		clf.fit(im_features, image_classes)

		print "Best parameters set found on development set:"
		print ''
		print clf.best_params_
		print ''
		print("Grid scores on development set:")
		print
		for params, mean_score, scores in clf.grid_scores_:
			print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params)
		print ''
	
	print clf	
	return clf
