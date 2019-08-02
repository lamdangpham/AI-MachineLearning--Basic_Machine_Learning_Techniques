import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

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

from sklearn.decomposition import TruncatedSVD
red_model = TruncatedSVD(n_components=10)
seq_x_reduced = red_model.fit_transform(seq_x)
[nS,nD] = np.shape(seq_x_reduced)

ind = 0
for i in range(int(nS)):
    if((seq_y[i] == 0) or (seq_y[i] == 1)):
        if(ind == 0):
            fig_seq_x = seq_x_reduced[i,:]
            fig_seq_y = seq_y[i]
            fig_seq_y = np.reshape(fig_seq_y, [1,1])
            fig_seq_x = np.reshape(fig_seq_x, [1,10])
        else:
            fig_seq_x = np.concatenate((fig_seq_x, np.reshape(seq_x_reduced[i,:], [1,10])), axis=0) 
            fig_seq_y = np.concatenate((fig_seq_y, np.reshape(seq_y[i], [1,1])), axis=0) 
        ind = ind + 1
[nS, nD] = np.shape(fig_seq_x)
fig_seq_y = np.reshape(fig_seq_y,[-1])

import seaborn as sns
import pandas as pd
# X is a matrix resulting from a dimensionality reduction method such as PCA
# Y is a list of labels for each instance
# c1 and c2 are column indices corresponding to the components that we wish to plot
# N is the number of instances
def nice_scatterplot(X, Y, c1, c2, N):
    lbl1 = 'Component {c1}'
    lbl2 = 'Component {c2}'    
    df = pd.DataFrame({lbl1:X[:N,c1], lbl2:X[:N,c2], 'label':Y[:N]})
    sns.lmplot(data=df, x=lbl1, y=lbl2, fit_reg=False, hue='label', scatter_kws={'alpha':0.5}) 

#all
nice_scatterplot(seq_x_reduced, seq_y, 0, 1, int(nS))

#0 and 1
#nice_scatterplot(fig_seq_x, fig_seq_y, 0, 1, int(nS))
plt.show()
