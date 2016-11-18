# Kaggle Digit Recognition 
import pandas as pd
import xgboost as xgb
from time import clock

# Read training data
start = clock()

train_frame = pd.read_csv('data/train.csv')
label = train_frame['label'].values
train = train_frame.iloc[:,1:].values

print('Loaded {:d} train entries in {:.0f} seconds.'.format(len(train), clock() - start))

# Train on fewer entries
# label = label[0::10]
# train = train[0::10]

# Read test data 
start = clock()

test_frame = pd.read_csv('data/test.csv')
test = test_frame.values

print('Loaded {:d} test entries in {:.0f} seconds.'.format(len(test), clock() - start))

# Convert to xgboost format

train_xgb = xgb.DMatrix(train, label)
test_xgb = xgb.DMatrix(test)

# Setup parameters for xgboost
param = {}

param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 10

num_round = 5

# Use softmax multi-class classification
param['objective'] = 'multi:softmax'

# Sklearn-like interface

# clf = xgb.XGBClassifier(param)
# clf.fit(train, label)
# clf.predict(test)

# Cross-validation

start = clock()
cv = xgb.cv(param, train_xgb, num_round, nfold = 10)
print("Cross-validated in {:.0f} seconds.".format(clock() - start))
print(cv)

# Train and predict

start = clock()
clf = xgb.train(param, train_xgb, num_round)
predict = clf.predict(test_xgb)
print("Trained and Extrapolated in {:.0f} seconds.".format(clock() - start))

# Save results

test_frame['ImageId'] = range(1,len(test)+1)
test_frame['Label'] = map(int, predict)
test_frame.to_csv('predict.csv', columns = ('ImageId', 'Label'), index = None)
