####################################################################
# Machine Trading Analysis with Python                             #
# Algorithm Learning Multi-Layer Perceptron Methods                #
# (c) Diego Fernandez Garcia 2015-2017                             #
# www.exfinsis.com                                                 #
####################################################################

# 1. Packages Importing
import numpy as np
import pandas as pd
# import pandas_datareader.data as web
import sklearn.decomposition as fe
import sklearn.model_selection as cv
import sklearn.neural_network as ml
import sklearn.metrics as fa
import matplotlib.pyplot as plt

#########

# 2. Data Downloading or Reading

# Yahoo Finance
# data = web.DataReader('SPY', data_source='yahoo', start='2007-01-01', end='2017-01-01')
# spy = data['Adj Close']
# spy.columns = ['SPY.Adjusted']

# Data Reading
spy = pd.read_csv('Data//Machine-Trading-Analysis-Data.txt', index_col='Date', parse_dates=True)

#########

# 3. Feature Creation

# 3.1. Target Feature
rspy = np.log(spy/spy.shift(1))
rspy.columns = ['rspy']

# 3.2. Predictor Features
rspy1 = rspy.shift(1)
rspy1.columns = ['rspy1']
rspy2 = rspy.shift(2)
rspy2.columns = ['rspy2']
rspy3 = rspy.shift(3)
rspy3.columns = ['rspy3']
rspy4 = rspy.shift(4)
rspy4.columns = ['rspy4']
rspy5 = rspy.shift(5)
rspy5.columns = ['rspy5']
rspy6 = rspy.shift(6)
rspy6.columns = ['rspy6']
rspy7 = rspy.shift(7)
rspy7.columns = ['rspy7']
rspy8 = rspy.shift(8)
rspy8.columns = ['rspy8']
rspy9 = rspy.shift(9)
rspy9.columns = ['rspy9']

# 3.3. All Features
rspyall = rspy
rspyall = rspyall.join(rspy1)
rspyall = rspyall.join(rspy2)
rspyall = rspyall.join(rspy3)
rspyall = rspyall.join(rspy4)
rspyall = rspyall.join(rspy5)
rspyall = rspyall.join(rspy6)
rspyall = rspyall.join(rspy7)
rspyall = rspyall.join(rspy8)
rspyall = rspyall.join(rspy9)
rspyall = rspyall.dropna()

#########

# 4. Range Delimiting

# 4.1. Training Range
rspyt = rspyall['2007-01-01':'2014-01-01']

# 4.2. Testing Range
rspyf = rspyall['2014-01-01':'2016-01-01']

#########

# 5. Multi-Layer Perceptron Method Algorithm Learning

# 5.1. Multi-Layer Perceptron Method Algorithm Features

# 5.1.1. Features Selection
pfa = ['rspy1', 'rspy2', 'rspy5']

# 5.1.2. Features Extraction (Principal Component Analysis)
pfb = ['rspy1', 'rspy2', 'rspy3', 'rspy4', 'rspy5', 'rspy6', 'rspy7', 'rspy8', 'rspy9']
pcat = fe.PCA().fit_transform(rspyt[pfb], rspyt['rspy'])
pcaf = fe.PCA().fit_transform(rspyf[pfb], rspyf['rspy'])

# 5.2. Multi-Layer Perceptron Method Algorithm Training Optimal Parameter Selection

# 5.2.1. Time Series Cross-Validation
# Exhaustive Grid Search Time Series Cross-Validation with Parameter Array Specification
# TimeSeriesSplit = anchored time series cross-validation with
# initial training subset = validating subset ~ n_samples / (n_splits + 1) in size
# alpha = L2 regularization
cvannta = cv.GridSearchCV(ml.MLPRegressor(), cv=cv.TimeSeriesSplit(n_splits=10),
                          param_grid={"alpha": [0.0001, 0.001, 0.010, 0.100]}).fit(rspyt[pfa], rspyt['rspy'])
cvanntb = cv.GridSearchCV(ml.MLPRegressor(), cv=cv.TimeSeriesSplit(n_splits=10),
                          param_grid={"alpha": [0.0001, 0.001, 0.010, 0.100]}).fit(pcat, rspyt['rspy'])

# 5.2.2. Time Series Cross-Validation Optimal Parameter Selection
cvannpara = cvannta.best_estimator_.alpha
print("")
print("== Multi-Layer Perceptron Method Algorithm Training Optimal Parameter Selection ==")
print("")
print("Artificial Neural Network Regression A Optimal L2 Regularization: ", cvannpara)
cvannparb = cvanntb.best_estimator_.alpha
print("Artificial Neural Network Regression B Optimal L2 Regularization: ", cvannparb)
print("")

# 5.3. Multi-Layer Perceptron Method Algorithm Training
annta = ml.MLPRegressor(alpha=cvannpara).fit(rspyt[pfa], rspyt['rspy'])
anntb = ml.MLPRegressor(alpha=cvannparb).fit(pcat, rspyt['rspy'])

# 5.4. Multi-Layer Perceptron Method Algorithm Testing
annfa = annta.predict(rspyf[pfa])
annfb = anntb.predict(pcaf)

# 5.4.1. Algorithm Testing Charts

# Artificial Neural Network Regression A Testing Chart
annfadf = pd.DataFrame(annfa, index=rspyf.index)
fig1, ax = plt.subplots()
ax.plot(rspyf['rspy'])
ax.plot(annfadf, label='annfa')
plt.legend(loc='upper left')
plt.title('Artificial Neural Network Regression A Testing Chart')
plt.ylabel('Log Returns')
plt.xlabel('Date')
plt.show()

# Artificial Neural Network Regression B Testing Chart
annfbdf = pd.DataFrame(annfb, index=rspyf.index)
fig2, ax = plt.subplots()
ax.plot(rspyf['rspy'])
ax.plot(annfbdf, label='annfb')
plt.legend(loc='upper left')
plt.title('Artificial Neural Network Regression B Testing Chart')
plt.ylabel('Log Returns')
plt.xlabel('Date')
plt.show()

# 5.5. Multi-Layer Perceptron Method Algorithm Testing Forecasting Accuracy
annmaea = fa.mean_absolute_error(rspyf['rspy'], annfa)
annmaeb = fa.mean_absolute_error(rspyf['rspy'], annfb)
annmsea = fa.mean_squared_error(rspyf['rspy'], annfa)
annmseb = fa.mean_squared_error(rspyf['rspy'], annfb)
annrmsea = np.sqrt(annmsea)
annrmseb = np.sqrt(annmseb)
print("== Multi-Layer Perceptron Method Algorithm Testing Forecasting Accuracy ==")
print("")
print("Mean Absolute Error ", "A:", round(annmaea, 6), "B:", round(annmaeb, 6))
print("Mean Squared Error ", "A:", round(annmsea, 6), "B:", round(annmseb, 6))
print("Root Mean Squared Error ", "A:", round(annrmsea, 6), "B:", round(annrmseb, 6))