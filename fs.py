# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 13:10:45 2016

@author: Adam Palmer, Thor Dahsltrøm

Based on the framework provided by -Det er vigtigt at skive___ 
"""

##crossvalidation and forward feature selection on regression for reported crime

# from exercise 6.2.1
from pylab import *
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import cross_validation
from sklearn import feature_selection as skfs
import pandas as pd
import neurolab as nl
from scipy import stats
from toolbox_02450 import feature_selector_lr, bmplot

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
data['youngVsOld'] = (data['populationShare65plus'] / data['youngUnskilled'] )
data_normalized =  (data - data.mean()) / (data.max() - data.min())
X = np.matrix(data_normalized.drop(['reportedCrime','reportedCrimeVandalism','urbanDegree'],axis=1).dropna())
# y = np.matrix((data['Cases'] / data['population']) *1e3)
y = np.matrix(data['reportedCrime']).T


attributeNames = data_normalized.drop(['reportedCrime','reportedCrimeVandalism'],axis=1).dropna().columns.values
N, M = X.shape

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
#attributeNames = [u'Offset']+attributeNames
M = M+1

#Neural network parameters
list_input_range = [[-1,1] for i in range(M)]
# Parameters for neural network classifier
n_hidden_units = 40      # number of hidden units
n_train = 2             # number of networks trained in each k-fold

# These parameters are usually adjusted to: (1) data specifics, (2) computational constraints
learning_goal = 180 # stop criterion 1 (train mse to be reached)
max_epochs = 1000       # stop criterion 2 (max epochs in training)


## Crossvalidation
# Create crossvalidation partition for evaluation
K = 4
CV = cross_validation.KFold(N,K,shuffle=True)

# Initialize variables
Features = np.zeros((M,K))
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_fs = np.empty((K,1))
Error_test_fs = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
Error_train_nn = np.empty((K,1))
Error_test_nn = np.empty((K,1))
Error_list_fs = list()
Error_list_nn = list()
k=0
for train_index, test_index in CV:
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10
    
    # Compute squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum()/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum()/y_test.shape[0]

    # Compute squared error with all features selected (no feature selection)
    m = lm.LinearRegression().fit(X_train, y_train)
    Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Compute squared error with feature subset selection
    #print skfs.f_regression(X_train, y_train, True)
    selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation)
    #print selected_features
    #Sprint features_record
    Features[selected_features,k]=1
    # .. alternatively you could use module sklearn.feature_selection
    
    
    m = lm.LinearRegression().fit(X_train[:,selected_features], y_train)
    Error_train_fs[k] = np.square(y_train-m.predict(X_train[:,selected_features])).sum()/y_train.shape[0]
    Error_test_fs[k] = np.square(y_test-m.predict(X_test[:,selected_features])).sum()/y_test.shape[0]
    Error_list_fs.append(abs(y_test-m.predict(X_test[:,selected_features])))
    #Compute neural network
    ann = nl.net.newff(list_input_range, [n_hidden_units, 1], [nl.trans.TanSig(),nl.trans.PureLin()])
    # train network 
    list_of_training_errors = ann.train(X_train, y_train, goal=learning_goal, epochs=max_epochs, show=round(100))    
    Error_train_nn[k] = list_of_training_errors[-1]  
    Error_test_nn[k] = np.square(y_test-ann.sim(X_test)).sum()/y_test.shape[0]
    Error_list_nn.append(abs(y_test - ann.sim(X_test)))
    
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
print('\n')
print('Average:\n')
print('- Training error: {0}'.format(np.sqrt(Error_train_nofeatures).mean())) 
print('- Test error:     {0}'.format(np.sqrt(Error_test_nofeatures).mean()))
print('Linear regression without feature selection:\n')
print('- Training error: {0}'.format(np.sqrt(Error_train).mean() )) 
print('- Test error:     {0}'.format(np.sqrt(Error_test).mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('\n\nLinear regression with feature selection:\n')
print('- Training error: {0}'.format(np.sqrt(Error_train_fs).mean()))
print('- Test error:     {0}'.format(np.sqrt(Error_test_fs).mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_fs.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test_fs.sum())/Error_test_nofeatures.sum()))
print('\n\nArtificial Neural Network:\n')
print('- Training error: {0}'.format(np.sqrt(Error_train_nn).mean()))
print('- Test error:     {0}'.format(np.sqrt(Error_test_nn).mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_nn.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test_nn.sum())/Error_test_nofeatures.sum()))

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
f=2 # cross-validation fold to inspect
ff=Features[:,f-1].nonzero()[0]
m = lm.LinearRegression().fit(X[:,ff], y)

y_est= m.predict(X[:,ff])
residual=y-y_est

figure(k+1)
title('Residual error vs. Attributes for features selected in cross-validation fold {0}'.format(f))
for i in range(0,len(ff)):
   subplot(2,ceil(len(ff)/2.0),i+1)
   plot(X[:,ff[i]].A,residual.A,'.')
   xlabel(attributeNames[ff[i]])
   ylabel('residual error')

show() 

t_test_fs = Error_list_fs[-1]
t_test_nn = Error_list_nn[-1]
   
t_test_average = abs(y_test - np.mean(y_train))


[tstatistic, pvalue] = stats.ttest_ind(t_test_fs,t_test_nn)
if pvalue<=0.05:
    print('Classifiers are significantly different. (p={0})'.format(pvalue[0]))
else:
    print('Classifiers are not significantly different (p={0})'.format(pvalue[0]))        

[tstatistic, pvalue] = stats.ttest_ind(t_test_fs,t_test_average)
if pvalue<=0.05:
    print('Classifiers are significantly different. (p={0})'.format(pvalue[0]))
else:
    print('Classifiers are not significantly different (p={0})'.format(pvalue[0]))        
    
# Boxplot to compare classifier error distributions
figure()
boxplot(np.bmat('t_test_fs, t_test_nn,t_test_average'))
xlabel('Linear Regression FS vs. Neural net vs. mean Error')
ylabel('Cross-validation error [%]')

show()
   
   
