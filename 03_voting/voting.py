import numpy as np
import pickle

#=========================================== 01/ Import Input Data
print("\n ==================================================================== IMPORT DATA...")
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

seq_x = mnist.train.images      #55000:784
seq_y = mnist.train.labels      #55000:10
seq_x_test = mnist.test.images  #10000:784
seq_y_test = mnist.test.labels  #10000:10

#normalize input images
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
seq_x      = scaler.fit_transform(seq_x)
seq_x_test = scaler.fit_transform(seq_x_test)

#tranfer input expected from one-hot format (2-D matrix) into one column vector
seq_y      = np.argmax(seq_y, axis=1)
seq_y_test = np.argmax(seq_y_test, axis=1)

#seq_x = seq_x[0:1000,:]
#seq_y = seq_y[0:1000]

print("\n ==================================================================== TRAINING...")

print("\n ==================================================================== 01: RANDOM FOREST")
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=50, n_jobs=10,)
rf_model.fit(seq_x,seq_y)
acc_rf =rf_model.score(seq_x_test, seq_y_test)
print('RF Score: ',acc_rf)

print("\n ==================================================================== 02: BAGGING with Decision Tree")
from sklearn.ensemble import BaggingClassifier 
bag_model = BaggingClassifier(n_estimators=50, max_samples=0.1, max_features = 0.7)
bag_model.fit(seq_x,seq_y)
acc_bag =bag_model.score(seq_x_test, seq_y_test)
print('Bagging Score: ',acc_bag)

print("\n ==================================================================== 03: AdaBoost with Decision Tree")
from sklearn.ensemble import AdaBoostClassifier 
ada_model = AdaBoostClassifier(n_estimators=100)
ada_model.fit(seq_x,seq_y)
acc_ada =ada_model.score(seq_x_test, seq_y_test)
print('Ada Boost Score: ',acc_ada)

print("\n ==================================================================== 04: Gradient boost")
from sklearn.ensemble import GradientBoostingClassifier
gra_model = GradientBoostingClassifier(n_estimators=100)
gra_model.fit(seq_x,seq_y)
acc_gra =gra_model.score(seq_x_test, seq_y_test)
print('Grad Boost: ',acc_gra)

print("\n ==================================================================== 05: Voting")
from sklearn.ensemble import VotingClassifier

voting_model = VotingClassifier(estimators=[('m01',gra_model),('m02',ada_model),('m03',rf_model),('m04',bag_model)], voting='soft' )
voting_model.fit(seq_x,seq_y)

acc_voting = voting_model.score(seq_x_test, seq_y_test)
print('Voting : ',acc_voting)

