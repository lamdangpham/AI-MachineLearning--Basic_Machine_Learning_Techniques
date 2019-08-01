from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import datasets
import numpy as np

#=========================== 01/ IMPORT DATA
iris = datasets.load_iris()
seq_x, seq_y = iris.data, iris.target
#150:4
#150
#Suffle data
from random import shuffle
N = len(seq_y)
ind_list = [i for i in range(int(N))]
shuffle(ind_list)
seq_x_new  = seq_x[ind_list, :]
seq_y_new  = seq_y[ind_list]

seq_x_test = seq_x_new[0:20,:]
seq_y_test = seq_y_new[0:20]
seq_x = seq_x_new[21:150,:]
seq_y = seq_y_new[21:150]

seq_x = np.repeat(seq_x, 100, axis=0)  
seq_y = np.repeat(seq_y, 100, axis=0) 


#========================== 02/ SETUP PIPELINE PROCESS
from sklearn.preprocessing import StandardScaler

preprocessor = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                               ('scaler', StandardScaler())
                              ]
                       )

classifiers = [KNeighborsClassifier(3),
               SVC(kernel="rbf", C=0.025, probability=True),
               DecisionTreeClassifier(),
               RandomForestClassifier(),
               AdaBoostClassifier(),
               GradientBoostingClassifier()
              ] 

index = 0
for classifier in classifiers:
    print("\n")
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', classifier)
                          ]
                   )

    pipe.fit(seq_x, seq_y)   
    print("================ CLASSIFIER {} =========================== \n".format(index))
    print("{} \n".format(classifier))
    print("Model Score: %.3f" % pipe.score(seq_x_test, seq_y_test))
    index = index + 1
