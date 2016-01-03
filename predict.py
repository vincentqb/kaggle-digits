from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

# File names
train_file_name = 'train.csv'
test_file_name = 'test.csv'
predict_file_name = 'predict.csv'

# Read training data
train_data = pd.read_csv(train_file_name)
target = train_data['label'].values
train = train_data.iloc[:,1:].values

# Read test data
test = pd.read_csv(test_file_name).values

# Use Classifier to create model and make prediction
rf = RandomForestClassifier()
rf.fit(train, target)
predict = rf.predict(test)

# Save result
predict_table = np.c_[range(1,len(test)+1), predict]
np.savetxt(predict_file_name, predict_table, header="ImageId,Label", comments="", delimiter=",", fmt="%d")
