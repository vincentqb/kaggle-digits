import numpy as np
import pandas as pd
from time import clock

### Load data

# Read training data
start = clock()
train_frame = pd.read_csv('train.csv')
label = train_frame['label'].values
train = train_frame.iloc[:,1:].values
print('Loaded {:d} train entries in {:.0f} seconds.'.format(len(train), clock() - start))

# Train on fewer entries
# label = label[0::10]
# train = train[0::10]

# Read test data 
start = clock()
test = pd.read_csv('test.csv').values
print('Loaded {:d} test entries in {:.0f} seconds.'.format(len(test), clock() - start))

### Visualize

# from random import randint
# 
# i = randint(0,len(train)-1)
# print("Displayed train entry {:d} labelled {:d}.".format(i, label[i]))
# 
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# 
# train_square = train.reshape(-1,28,28)
# plt.imshow(train_square[i], cmap=cm.binary)
# plt.show()

### Transform data

# Normalize data
# train_max = train.max()
# train = 2*train - train_max
# test = 2*test - train_max

from sklearn.decomposition import PCA

n_comp = 35
pca = PCA(n_components=n_comp, whiten=True)

start = clock()
pca.fit(train)
train = pca.transform(train)
test = pca.transform(test)
variance = sum(pca.explained_variance_ratio_)
print("Transformed data in {:.0f} seconds using {:d} components explaining {:.0%} of the variance.".format(clock() - start, n_comp, variance))

### Select Classifier

# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=100)

# MLPClassifier requires 0.18dev+ and is not available in 0.17
# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

# from sklearn import linear_model
# clf = linear_model.SGDClassifier()

# from sklearn.svm import LinearSVC
# clf = LinearSVC(tol=0.01, C=1)

from sklearn.svm import SVC
clf = SVC()

# algo = 'linear'
# tol = 0.01
# C = 1
# gamma = 0.01

# algo = 'rbf'
# tol = 0.001
# C = 2.82842712475
# gamma =  0.00728932024638

# from sklearn.svm import SVC
# clf = SVC(kernel=algo, tol=tol, C=C, gamma=gamma, shrinking=True)

# from lasagne import layers
# from lasagne.updates import nesterov_momentum
# from lasagne.nonlinearities import softmax
# from nolearn.lasagne import NeuralNet
# 
# clf = NeuralNet(
#         layers = [('input', layers.InputLayer),
#                   ('hidden', layers.DenseLayer),
#                   ('output', layers.DenseLayer),
#                   ],
# 
#         # Layer parameters
#         input_shape = (None,1,28,28),
#         hidden_num_units = 10,          # Number of units in hidden layer (10, 1000, ...)
#         output_nonlinearity = softmax,
#         output_num_units = 10,          # 10 target values for the digits 0, 1, 2, ..., 9
# 
#         # Optimization method
#         update = nesterov_momentum,
#         update_learning_rate = 0.001,   # 0.001, 0.0001, ...
#         update_momentum = 0.9,
#         max_epochs = 15,
# 
#         verbose = 1,
#         )

### Optimize classifer's parameters

from sklearn.grid_search import GridSearchCV

# Search on fewer entries
label_few = label[0::10]
train_few = train[0::10]

# Provide parameter spaces
params = [{'C': np.logspace(-1, 3), 'gamma': np.logspace(-4, -1)}]

# Run exhaustive grid search
start = clock()
gs = GridSearchCV(estimator = clf, param_grid = params, n_jobs = 2)
gs.fit(train_few, label_few)
print("Parameter optimized {} yielding {:.4f} in {:.0f} seconds.".format(gs.best_params_, gs.best_score_, clock() - start))

### Cross validation

from sklearn.cross_validation import cross_val_score

start = clock()
scores = cross_val_score(clf, train, label)
print("Performed cross validation in {:.0f} seconds: {:d} folds with accuracy {:0.4f} +/- {:0.4f}.".format(clock() - start, len(scores), scores.mean(), scores.std()))

### Fit training data

start = clock()
clf.fit(train, label)
print("Fitted training data in {:.0f} seconds.".format(clock() - start))

### Extrapolate to test data

start = clock()
predict = clf.predict(test)
print("Extrapolated to test data in {:.0f} seconds.".format(clock() - start))

### Save results

predict_table = np.c_[range(1,len(test)+1), predict]
np.savetxt('predict.csv', predict_table, header='ImageId,Label', comments='', delimiter=',', fmt='%d')
