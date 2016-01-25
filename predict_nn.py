import numpy as np
import pandas as pd
from time import clock

### Load data

# Read training data
start = clock()
train_frame = pd.read_csv('train.csv')
label = train_frame['label'].values
train = train_frame.iloc[:,1:].values
train = train.reshape(-1,1,28,28)
print('Loaded {:d} train entries in {:.0f} seconds.'.format(len(train), clock() - start))

# Train on fewer entries
# label = label[0::10]
# train = train[0::10]

# Read test data 
start = clock()
test = pd.read_csv('test.csv').values
test = test.reshape(-1,1,28,28)
print('Loaded {:d} test entries in {:.0f} seconds.'.format(len(test), clock() - start))

### Select Classifier

from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import softmax, rectify
from nolearn.lasagne import NeuralNet

clf = NeuralNet(
    layers = [
    ('input', layers.InputLayer),
    ('conv1', layers.Conv2DLayer),      # First convolutional layer
    ('pool1', layers.MaxPool2DLayer),   # Like downsampling, for execution speed
    ('conv2', layers.Conv2DLayer),      # Second convolutional layer
    ('hidden3', layers.DenseLayer),     # Fully connected hidden layer
    ('output', layers.DenseLayer),
    ],

    input_shape = (None, 1, 28, 28),
    
    conv1_num_filters = 7, 
    conv1_filter_size = (3, 3), 
    conv1_nonlinearity = rectify,
        
    pool1_pool_size = (2, 2),
        
    conv2_num_filters = 12, 
    conv2_filter_size = (2, 2),    
    conv2_nonlinearity = rectify,
        
    hidden3_num_units = 100,

    output_num_units = 10, 
    output_nonlinearity = softmax,
    
    # Optimization method
    update_learning_rate = 0.001,
    update_momentum = 0.9,
    max_epochs = 15,

    verbose = 1,
    )

# Theano is strict on the format of floats and ints
train = train.astype(np.float32)
label = label.astype(np.int32)
test = test.astype(np.float32)

### Optimize classifer's parameters

def grid_search(train, label, params):
    from sklearn.grid_search import GridSearchCV
    
    # Search on fewer entries
    label_few = label[0::10]
    train_few = train[0::10]
    
    # Run exhaustive grid search
    start = clock()
    gs = GridSearchCV(estimator = clf, param_grid = params, n_jobs = 2)
    gs.fit(train_few, label_few)
    print("Parameter optimized {} yielding {:.4f} in {:.0f} seconds.".format(
        gs.best_params_, gs.best_score_, clock() - start))
    
# Parameter space to search
# params = [{'C': np.logspace(-1, 3), 'gamma': np.logspace(-4, -1)}]
# grid_search(train, label, params)

### Cross validation

from sklearn.cross_validation import cross_val_score

start = clock()
scores = cross_val_score(clf, train, label)
print("Performed {:d}-fold cross validation in {:.0f} seconds with accuracy {:0.4f} +/- {:0.4f}.".format(
    len(scores), clock() - start, scores.mean(), scores.std()))

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
