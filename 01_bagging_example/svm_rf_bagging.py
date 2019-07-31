import time
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

iris = datasets.load_iris()
seq_x, seq_y = iris.data, iris.target
#150:4
#150
seq_x_test = seq_x[0:40,:]
seq_y_test = seq_y[0:40]
seq_x = seq_x[41:150,:]
seq_y = seq_y[41:150]

seq_x = np.repeat(seq_x, 100, axis=0)  
seq_y = np.repeat(seq_y, 100, axis=0)  

#======= 01/ Single SVC
clf = OneVsRestClassifier(SVC(kernel='linear', probability=True))

start = time.time()
clf.fit(seq_x, seq_y)
end = time.time()
print ("\n")
print ("Single SVC:", end - start, clf.score(seq_x_test, seq_y_test))

#======= 02/ Ensemble SVC
n_estimators = 10
clf = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True), max_samples=1.0 / n_estimators, n_estimators=n_estimators))

start = time.time()
clf.fit(seq_x, seq_y)
end = time.time()
print ("\n")
print ("Bagging SVC:", end - start, clf.score(seq_x_test,seq_y_test))

#======= 03/ Random Forest
clf = RandomForestClassifier(min_samples_leaf=20)

start = time.time()
clf.fit(seq_x, seq_y)
end = time.time()
print ("\n")
print ("Random Forest:", end - start, clf.score(seq_x_test,seq_y_test))
