from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

# Read training data
train_data = pd.read_csv('train.csv')
target = train_data['label'].values
train = train_data.iloc[:,1:].values

# Read test data
test = pd.read_csv('test.csv').values

# Use classifier to create model and make prediction
rf = RandomForestClassifier(n_estimators=100)
rf.fit(train, target)
predict = rf.predict(test)

# Save results
predict_table = np.c_[range(1,len(test)+1), predict]
np.savetxt('predict.csv', predict_table, header='ImageId,Label', comments='', delimiter=',', fmt='%d')

# Visualize
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# train_square = dataset.iloc[:,1:].values.reshape(-1,28,28)
# plt.imshow(train_square[5000], cmap=cm.binary)
# plt.show()
