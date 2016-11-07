# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 13:10:45 2016

@author: Adam
"""

##crossvalidation and forward feature selection on regression for reported crime

# from exercise 6.2.1
from pylab import *
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
import pandas as pd
import neurolab as nl
from toolbox_02450 import feature_selector_lr, bmplot
from scipy import stats

# Load data from matlab file
#data = loadmat('../Data/body.mat')
#X = np.matrix(mat_data['X'])
#y = np.matrix(mat_data['y'])
#attributeNames = [name[0] for name in mat_data['attributeNames'][0]]


data = pd.read_csv('cleanSet.csv')
#print mat_data
#X = np.matrix(mat_data['X'])
#y = np.matrix(mat_data['y'], dtype=int)
#X = np.matrix(mat_data.drop('Cases',axis=1).dropna())
#y = np.matrix(mat_data['Cases'].apply(f)).T
#data_normalized = (data - data.mean()) / (data.std())
#X = np.matrix(data_normalized.drop(['reportedCrime','reportedCrimeVandalism'],axis=1).dropna())
def f(x):
    if x > 0:
        return 1
    else:
        return 0


data_normalized =  (data - data.mean()) / (data.max() - data.min())

X = np.matrix(data_normalized.drop('Cases',axis=1).dropna())
y = np.matrix(data['Cases'].apply(f)).T





attributeNames = data_normalized.drop(['Cases'],axis=1).dropna().columns.values
N, M = X.shape

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
#attributeNames = [u'Offset']+attributeNames
M = M + 1

#Neural network parameters
list_input_range = [[-1,1] for i in range(M)]
# Parameters for neural network classifier
n_hidden_units = 40      # number of hidden units
n_train = 2             # number of networks trained in each k-fold

# These parameters are usually adjusted to: (1) data specifics, (2) computational constraints
learning_goal = 200     # stop criterion 1 (train mse to be reached)
max_epochs = 1000       # stop criterion 2 (max epochs in training)


## Crossvalidation
# Create crossvalidation partition for evaluation
K = 2
CV = cross_validation.KFold(N,K,shuffle=True)

# Initialize variables
Features = np.zeros((M,K))
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_fs = np.empty((K,1))
Error_test_fs = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
Error_train_decision_tree = np.empty((K,1))
Error_test_decision_tree = np.empty((K,1))
Error_train_nn = np.empty((K,1))
Error_test_nn = np.empty((K,1))
Error_list_logistic_regression_fs = list()
Error_list_logistic_regression_all = list()
Error_list_decision_tree = list()
Error_list_nn = list()

bestnet = list()
error_hist = np.zeros((max_epochs,K))
errors = np.zeros(K)

k=0
for train_index, test_index in CV:
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10
    
    # Compute squared error without using the input data at all
    Error_train_nofeatures[k] = float(abs(0-y_train).sum())/y_train.shape[0]
    Error_test_nofeatures[k] = float(abs(0-y_train).sum())/y_train.shape[0]

    # Compute squared error with all features selected (no feature selection)
    m_all_attributes = lm.LogisticRegression().fit(X_train, y_train)
    Error_train[k] = float(abs(y_train.T-m_all_attributes.predict(X_train)).sum())/y_train.shape[0]
    Error_test[k] = float(abs(y_test.T-m_all_attributes.predict(X_test)).sum())/y_test.shape[0]
    Error_list_logistic_regression_all.append(abs(y_test.T-m_all_attributes.predict(X_test)))
    
    # Compute squared error with feature subset selection
    selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation)
    Features[selected_features,k]=1
    # .. alternatively you could use module sklearn.feature_selection
    m = lm.LogisticRegression().fit(X_train[:,selected_features], y_train)
    Error_train_fs[k] = float(abs(y_train.T-m.predict(X_train[:,selected_features])).sum())/y_train.shape[0]
    Error_test_fs[k] = float(abs(y_test.T-m.predict(X_test[:,selected_features])).sum())/y_test.shape[0]
    Error_list_logistic_regression_fs.append(abs(y_test.T-m.predict(X_test[:,selected_features])))
    
    #Compute neural network
    #ann = nl.net.newff([[0, 1], [0, 1]], [n_hidden_units, 1], [nl.trans.TanSig(),nl.trans.PureLin()])
    
    # train network
    best_train_error = 1e100
    for i in range(n_train):
        # Create randomly initialized network with 2 layers
        ann = nl.net.newff(list_input_range, [n_hidden_units, 1], [nl.trans.TanSig(),nl.trans.PureLin()])
        # train network
        train_error = ann.train(X_train, y_train, goal=learning_goal, epochs=max_epochs, show=round(max_epochs/8))
        # stores the best network        
        if train_error[-1]<best_train_error:
            bestnet.append(ann)
            best_train_error = train_error[-1]
            error_hist[range(len(train_error)),k] = train_error
    y_est = bestnet[k].sim(X_test)

    y_est = (y_est>.5).astype(int)
    errors[k] = (y_est!=y_test).sum().astype(float)/y_test.shape[0]
    print('Error rate: {0}%'.format(100*mean(errors)))
    #train_error             = ann.train(X_train, y_train, goal=learning_goal, epochs=max_epochs, show=round(max_epochs/8))
    list_of_training_errors = ann.train(X_train, y_train, epochs=max_epochs, show=round(max_epochs/8))    
    Error_train_nn[k] = list_of_training_errors[-1]  
    Error_test_nn[k] = np.square(y_test-ann.sim(X_test)).sum()/y_test.shape[0]
    
    Error_for_nn = bestnet[k].sim(X_test)
    Error_for_nn = (Error_for_nn>.5).astype(int)
    Error_list_nn.append((y_est!=y_test).astype(int))
    
    #Decision tree
    d_tree = DecisionTreeClassifier(min_samples_split=80).fit(X_train,y_train)
    Error_train_decision_tree[k] = float(abs(y_train.T-d_tree.predict(X_train)).sum())/y_train.shape[0]
    Error_test_decision_tree[k] = float(abs(y_test.T-d_tree.predict(X_test)).sum())/y_test.shape[0]
    Error_list_decision_tree.append(abs(y_test.T-d_tree.predict(X_test)))
    
    figure(k)
    subplot(1,2,1)
    plot(range(1,len(loss_record)), loss_record[1:])
    xlabel('Iteration')
    ylabel('Squared error (crossvalidation)')    
    
    subplot(1,3,3)
    bmplot(attributeNames, range(1,features_record.shape[1]), -features_record[:,1:])
    clim(-1.5,0)
    xlabel('Iteration')

    print('Cross validation fold {0}/{1}'.format(k+1,K))
    print('Train indices: {0}'.format(train_index))
    print('Test indices: {0}'.format(test_index))
    print('Features no: {0}\n'.format(selected_features.size))

    k+=1


# Display results
print("These are the errors in procent ")
print('Always predicting no fires:')
print('- Training error: {0}'.format((Error_train_nofeatures).mean())) 
print('- Test error:     {0}'.format((Error_test_nofeatures).mean()))
print('\n')
print('Logistic regression without feature selection:')
print('- Training error: {0}'.format((Error_train).mean() )) 
print('- Test error:     {0}'.format((Error_test).mean()))
print('\n')
print('Logistic regression with feature selection:')
print('- Training error: {0}'.format((Error_train_fs).mean()))
print('- Test error:     {0}'.format((Error_test_fs).mean()))
print('\n')
print('Decision tree classifier:')
print('- Training error: {0}'.format((Error_train_decision_tree).mean()))
print('- Test error:     {0}'.format((Error_test_decision_tree).mean()))
print('Neural network classifier')
print('- Test error:     {0}'.format((errors).mean()))





# Use T-test to check if classifiers are significantly different




figure(k)
subplot(1,3,2)
bmplot(attributeNames, range(1,Features.shape[1]+1), -Features)
clim(-1.5,0)
xlabel('Crossvalidation fold')
ylabel('Attribute')


# Inspect selected feature coefficients effect on the entire dataset and
# plot the fitted model residual error as function of each attribute to
# inspect for systematic structure in the residual
#==============================================================================
# f=2 # cross-validation fold to inspect
# ff=Features[:,f-1].nonzero()[0]
# m = lm.LogisticRegression().fit(X[:,ff], y)
# 
# y_est= m.predict(X[:,ff])
# residual=y-y_est
# 
# figure(k+1)
# title('Residual error vs. Attributes for features selected in cross-validation fold {0}'.format(f))
# for i in range(0,len(ff)):
#    subplot(2,ceil(len(ff)/2.0),i+1)
#    plot(X[:,ff[i]].A,residual.A,'.')
#    xlabel(attributeNames[ff[i]])
#    ylabel('residual error')
# 
# show() 
#==============================================================================

t_test_fs = Error_list_logistic_regression_fs[-1].T
t_test_decision_tree = Error_list_decision_tree[-1].T
t_test_log_all = Error_list_logistic_regression_all[-1].T
   
#==============================================================================
t_test_average = abs(y_test)
# 
# 
#==============================================================================
[tstatistic, pvalue] = stats.ttest_ind(t_test_fs,t_test_log_all)
print("\nLogistic regresion all vs logistic regression FS")
if pvalue<=0.05:
     print('Classifiers are significantly different. (p={0})'.format(pvalue[0]))
else:
    print('Classifiers are not significantly different (p={0})'.format(pvalue[0]))        


[tstatistic, pvalue] = stats.ttest_ind(t_test_fs,t_test_decision_tree)
print("\nLogistic regression fs vs decision tree")
if pvalue<=0.05:
     print('Classifiers are significantly different. (p={0})'.format(pvalue[0]))
else:
    print('Classifiers are not significantly different (p={0})'.format(pvalue[0]))        

[tstatistic, pvalue] = stats.ttest_ind(t_test_decision_tree,t_test_average)
print("\nDecision tree vs guessing 0 all times")
if pvalue<=0.05:
    print('Classifiers are significantly different. (p={0})'.format(pvalue[0]))
else:
    print('Classifiers are not significantly different (p={0})'.format(pvalue[0]))        

[tstatistic, pvalue] = stats.ttest_ind(t_test_log_all,t_test_average)
print("\nLogistic regression all vs guessing 0 all times")
if pvalue<=0.05:
    print('Classifiers are significantly different. (p={0})'.format(pvalue[0]))
else:
    print('Classifiers are not significantly different (p={0})'.format(pvalue[0]))        
#    

# Boxplot to compare classifier error distributions
figure()
boxplot(np.bmat('t_test_fs,t_test_decision_tree,t_test_average'))
xlabel('Linear Regression FS  vs.   Neural net')
ylabel('Cross-validation error [%]')

show()
#==============================================================================
#    
#==============================================================================
   
m_entire_dataset_fs = lm.LogisticRegression().fit(X[:,selected_features], y)
m_entire_dataset_all = lm.LogisticRegression().fit(X[:,:], y)

