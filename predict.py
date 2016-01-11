import numpy as np
import pandas as pd
from time import clock

### Load data...

# Read training data
train_data = pd.read_csv('train.csv')
target = train_data['label'].values
train = train_data.iloc[:,1:].values

# Take fewer entries
target = target[0::10]
train = train[0::10]

# Read test data
test = pd.read_csv('test.csv').values

### Transform data

# Normalize data
# train_max = train.max()
# train = 2*train - train_max
# test = 2*test - train_max

from sklearn.decomposition import PCA

print('Reduction...')
start = clock()
COMPONENT_NUM = 35
pca = PCA(n_components=COMPONENT_NUM, whiten=True)
pca.fit(train)
train_data = pca.transform(train)
test_data = pca.transform(test)
print(clock() - start)

### Select Classifier

# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=100)

# MLPClassifier requires 0.18dev+ and is not available in 0.17
# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

# from sklearn import linear_model
# clf = linear_model.SGDClassifier()

from sklearn import svm

# clf = svm.LinearSVC( tol=0.01, C=1 )
# clf = svm.SVC(gamma=0.001)
clf = svm.SVC()

# algo = 'linear'
# tol = 0.01
# tol = 0.1
# C = 1
# gamma = 0.001
# gamma = 0.01

# algo = 'rbf'
# tol = 0.001
# gamma =  0.00728932024638
# C = 2.82842712475

# clf = svm.SVC( kernel=algo, tol=tol, C=C, gamma=gamma, shrinking=True, verbose=True)

# import logging
# logging.basicConfig()

# from sknn.mlp import Classifier, Layer
# clf = Classifier(
#     layers=[
#         Layer("Rectifier", units=10),
#         Layer("Linear")],
#     learning_rate=0.02,
#     n_iter=10)

### Fit training data

print('Fitting training data...')
start = clock()
clf.fit(train, target)
print(clock() - start)

### Predict and save results

print('Predicting...')
start = clock()
predict = clf.predict(test)
print(clock() - start)

predict_table = np.c_[range(1,len(test)+1), predict]
np.savetxt('predict.csv', predict_table, header='ImageId,Label', comments='', delimiter=',', fmt='%d')

### Visualize

# import matplotlib.pyplot as plt
# import matplotlib.cm as cm

# train_square = dataset.iloc[:,1:].values.reshape(-1,28,28)
# plt.imshow(train_square[5000], cmap=cm.binary)
# plt.show()
