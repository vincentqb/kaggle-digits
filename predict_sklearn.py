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
test_frame = pd.read_csv('data/test.csv')
test = test_frame.values
# test = test.reshape(-1,28,28)
print('Loaded {:d} test entries in {:.0f} seconds.'.format(len(test), clock() - start))

### Transform data

# from sklearn.preprocessing import StandardScaler
# train = StandardScaler().fit_transform(train)

def PCA(train, test):
    from sklearn.decomposition import PCA
    
    n_comp = 35
    pca = PCA(n_components=n_comp, whiten=True)
    
    start = clock()
    pca.fit(train)
    train = pca.transform(train)
    test = pca.transform(test)
    print("Transformed data in {:.0f} seconds using {:d} components explaining {:.0%} of the variance.".format(
        clock() - start, n_comp, sum(pca.explained_variance_ratio_)))
    return (train, test)

# (train, test) = PCA(train, test)

### Select Classifier

# MLPClassifier requires 0.18dev+ and is not available in 0.17
# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(algorithm = 'l-bfgs', alpha = 1e-5, hidden_layer_sizes = (5, 2), random_state = 1)

# from sklearn.linear_model import SGDClassifier
# clf = SGDClassifier()

# from sklearn.svm import LinearSVC
# clf = LinearSVC(tol = 0.01, C = 1)

def SVC():
    from sklearn.svm import SVC

    algo = 'linear'
    tol = 0.01
    C = 1
    gamma = 0.01
    
    algo = 'rbf'
    tol = 0.001
    C = 2.82842712475
    gamma =  0.00728932024638
    
    # clf = SVC(kernel=algo, tol=tol, C=C, gamma=gamma, shrinking=True)
    clf = SVC()

    return clf

# clf = SVC()

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

# from sklearn.cross_validation import cross_val_score
# 
# start = clock()
# scores = cross_val_score(clf, train, label)
# print("Performed {:d}-fold cross validation in {:.0f} seconds with accuracy {:0.4f} +/- {:0.4f}.".format(
#     len(scores), clock() - start, scores.mean(), scores.std()))

### Fit training data

start = clock()
clf.fit(train, label)
print("Fitted training data in {:.0f} seconds.".format(clock() - start))

### Extrapolate to test data

start = clock()
predict = clf.predict(test)
print("Extrapolated to test data in {:.0f} seconds.".format(clock() - start))

### Save results

test_frame['ImageId'] = range(1,len(test)+1)
test_frame['Label'] = predict
test_frame.to_csv("predict.csv", cols = ('ImageId', 'Label'), index = None)
