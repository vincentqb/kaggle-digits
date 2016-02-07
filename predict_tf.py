import numpy as np
import pandas as pd
from time import clock

### Load data

# Read training data
start = clock()
train_frame = pd.read_csv('data/train.csv')
label = train_frame['label'].values
train = train_frame.iloc[:,1:].values
# train = train.reshape(-1,28,28)
print('Loaded {:d} train entries in {:.0f} seconds.'.format(len(train), clock() - start))

# Train on fewer entries
# label = label[0::10]
# train = train[0::10]

# Read test data 
start = clock()
test = pd.read_csv('data/test.csv').values
# test = test.reshape(-1,28,28)
print('Loaded {:d} test entries in {:.0f} seconds.'.format(len(test), clock() - start))

### Transform data

# from sklearn.preprocessing import StandardScaler
# train = StandardScaler().fit_transform(train)

### Select Classifier

# Tensor Flow is strict on the format of floats and ints
# train = train.astype(np.float32)
# label = label.astype(np.int32)
# test = test.astype(np.float32)

# Classifiers from Scikit Flow

# from skflow import TensorFlowLinearClassifier
# clf = TensorFlowLinearClassifier(n_classes = 10, batch_size = 256, steps = 1400, learning_rate = 0.01, optimizer = 'Adagrad')

from skflow import TensorFlowLinearRegressor
clf = TensorFlowLinearRegressor()

# from skflow import TensorFlowDNNClassifier
# Optimizer = SGD, Adam, Adagrad
# clf = TensorFlowDNNClassifier(hidden_units = [10, 20, 10], 
# 				n_classes = 10, batch_size = 256, steps = 1000, learning_rate = 0.01, optimizer = 'Adagrad')

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
