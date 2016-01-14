import numpy as np
import pandas as pd
from time import clock

# Read training data
train_data = pd.read_csv('train.csv')
target = train_data['label'].values
train = train_data.iloc[:,1:].values

# Read test data
test = pd.read_csv('test.csv').values

# Select classifier
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=100)

# MLPClassifier requires 0.18dev+ and is not available in 0.17
# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

from sklearn import svm
from sklearn import linear_model

# clf = linear_model.SGDClassifier()
# clf = svm.LinearSVC(tol=0.01, C=1)

clf = svm.SVC(gamma=0.001)

# algo = 'linear'
# tol = 0.1
# C = 1
# gamma = 0.01
# clf = svm.SVC(kernel=algo, tol=tol, C=C, gamma=gamma, shrinking=True)

# algo = 'rbf'
# tol = 0.001
# gamma =  0.00728932024638
# C = 2.82842712475
# clf = svm.SVC(kernel=algo, tol=tol, C=C, gamma=gamma, shrinking=True)

# Fit training data
start = clock()
clf.fit(train, target)
print(clock() - start)

# Predict and save results
predict = clf.predict(test)
predict_table = np.c_[range(1,len(test)+1), predict]
np.savetxt('predict.csv', predict_table, header='ImageId,Label', comments='', delimiter=',', fmt='%d')

# Visualize
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# train_square = dataset.iloc[:,1:].values.reshape(-1,28,28)
# plt.imshow(train_square[5000], cmap=cm.binary)
# plt.show()
