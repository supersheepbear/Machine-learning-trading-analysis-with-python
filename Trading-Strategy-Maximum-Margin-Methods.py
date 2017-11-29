####################################################################
# Machine Trading Analysis with Python                             #
# Trading Strategy Maximum Margin Methods                          #
# (c) Diego Fernandez Garcia 2015-2017                             #
# www.exfinsis.com                                                 #
####################################################################

# 1. Packages Importing
import numpy as np
import pandas as pd
# import pandas_datareader.data as web
import sklearn.model_selection as cv
import sklearn.svm as ml
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

# 4.3. Trading Range
rspys = rspyall['2016-01-01':'2017-01-01']

#########

# 5. Maximum Margin Method Algorithm Learning

# 5.1. Maximum Margin Method Algorithm Features

# 5.1.1. Features Selection
pfa = ['rspy1', 'rspy2', 'rspy5']

# 5.2. Maximum Margin Method Algorithm Training Optimal Parameter Selection

# 5.2.1. Time Series Cross-Validation
cvsvmta = cv.GridSearchCV(ml.SVR(kernel='rbf'), cv=cv.TimeSeriesSplit(n_splits=10),
                          param_grid={"C": [0.25, 0.50, 1.00, 1.25]}).fit(rspyt[pfa], rspyt['rspy'])

# 5.2.2. Time Series Cross-Validation Optimal Parameter Selection
cvsvmpara = cvsvmta.best_estimator_.C

# 5.3. Maximum Margin Method Algorithm Training
svmta = ml.SVR(kernel='rbf', C=cvsvmpara).fit(rspyt[pfa], rspyt['rspy'])

# 5.4. Maximum Margin Method Algorithm Forecasting
# Forecasting for Trading Subset
svmpa = svmta.predict(rspys[pfa])
svms = pd.DataFrame(svmpa, index=rspys.index)
svms.columns = ['svmpa']

#########

# 6. Maximum Margin Method Trading Strategy

# 6.1. Maximum Margin Method Trading Strategy Signals
# Previous Periods Data (avoid back-testing bias)
# Generate Trading Strategy Signals (buy stock=1 , sell stock=-1, do nothing=0)
svms['svmpa(-1)'] = svms['svmpa'].shift(1)
svms['svmpa(-2)'] = svms['svmpa'].shift(2)
svms['svmsig'] = 0
svmsig = 0
for i, r in enumerate(svms.iterrows()):
    if r[1]['svmpa(-2)'] - np.mean(svms['svmpa(-2)'][0:i]) < 0 and r[1]['svmpa(-1)'] - np.mean(svms['svmpa(-1)'][0:i]) > 0:
        svmsig = 1
    elif r[1]['svmpa(-2)'] - np.mean(svms['svmpa(-2)'][0:i]) > 0 and r[1]['svmpa(-1)'] - np.mean(svms['svmpa(-1)'][0:i]) < 0:
        svmsig = -1
    else:
        svmsig = 0
    svms.iloc[i, 3] = svmsig

# 6.1.1. Maximum Margin Method Trading Strategy Signals Chart
svms['svmpac'] = svms['svmpa'] - np.mean(svms['svmpa'])
svms['svmpacl'] = 0
svmsigc = svms['svmsig']
svmbuy = svmsigc[svmsigc == 1]
svmsell = svmsigc[svmsigc == -1]
fig1, ax = plt.subplots(2, sharex=True)
ax[0].plot(svms['svmpac'])
ax[0].plot(svms['svmpacl'], linestyle="--")
ax[0].legend(loc='upper right')
ax[1].plot(svmbuy, marker='^', linestyle='', color='g', label='svmbuy')
ax[1].plot(svmsell, marker='v', linestyle='', color='r', label='svmsell')
ax[1].legend(loc='upper right')
plt.suptitle('Maximum Margin Method Trading Strategy Signals Chart')
plt.show()

# 6.2. Maximum Margin Method Trading Strategy Positions
# Generate Trading Strategy Positions (own stock=1 , not own stock=0, short-selling not available)
svms['svmpos'] = 0
svmpos = 0
for i, r in enumerate(svms.iterrows()):
    if r[1]['svmsig'] == 1:
        svmpos = 1
    elif r[1]['svmsig'] == -1:
        svmpos = 0
    else:
        svmpos = svms['svmpos'][i-1]
    svms.iloc[i, 6] = svmpos

# 6.2.1. Maximum Margin Method Trading Strategy Positions Chart
svmposc = svms['svmpos']
svmopen = svmposc[svmposc == 1]
svmclosed = svmposc[svmposc == 0]
fig2, ax = plt.subplots(2, sharex=True)
ax[0].plot(svms['svmpac'])
ax[0].plot(svms['svmpacl'], linestyle="--")
ax[0].legend(loc='upper right')
ax[1].plot(svmopen, marker='.', linestyle='', color='g', label='svmopen')
ax[1].plot(svmclosed, marker='.', linestyle='', color='r', label='svmclosed')
ax[1].legend(loc='upper right')
plt.suptitle('Maximum Margin Method Trading Strategy Positions Chart')
plt.show()

##########

# 7. Maximum Margin Method Trading Strategy Performance Comparison

# 7.1. Maximum Margin Method Trading Strategy Daily Returns

# 7.1.1. Maximum Margin Method Strategy Daily Returns Without Trading Commissions
svms['svmdrt'] = rspys['rspy'] * svms['svmpos']

# 7.1.2. Maximum Margin Method Strategy Daily Returns With Trading Commissions (0.10% Per Trade)
svms['svmpos(-1)'] = svms['svmpos'].shift(1)
svms['svmtc'] = 0
svmtc = 0
for i, r in enumerate(svms.iterrows()):
    if (r[1]['svmsig'] == 1 or r[1]['svmsig'] == -1) and r[1]['svmpos'] != r[1]['svmpos(-1)']:
        svmtc = 0.001
    else:
        svmtc = 0.000
    svms.iloc[i, 9] = svmtc
svms['svmdrtc'] = (rspys['rspy'] * svms['svmpos']) - svms['svmtc']

# 7.2. Maximum Margin Method Trading Strategy Cumulative Returns

# 7.2.1. Maximum Margin Method Trading Strategy Cumulative Returns Calculation
svms['svmcrt'] = np.cumprod(svms['svmdrt'] + 1) - 1
svms['svmcrtc'] = np.cumprod(svms['svmdrtc'] + 1) - 1
svms['spycrt'] = np.cumprod(rspys['rspy'] + 1) - 1

# 7.2.2. Maximum Margin Method Trading Strategy Cumulative Returns Chart
fig3, ax = plt.subplots()
ax.plot(svms['svmcrt'])
ax.plot(svms['svmcrtc'])
ax.plot(svms['spycrt'])
ax.legend(loc='upper left')
plt.suptitle('Maximum Margin Method Trading Strategy Cumulative Returns Chart')
plt.show()

# 7.3. Maximum Margin Method Trading Strategy Performance Metrics

# 7.3.1. Maximum Margin Method Trading Strategy Annualized Returns
svmyrt = svms.iloc[251, 11]
svmyrtc = svms.iloc[251, 12]
spyyrt = svms.iloc[251, 13]

# 7.3.2. Maximum Margin Method Trading Strategy Annualized Standard Deviation
svmstd = np.std(svms['svmdrt']) * np.sqrt(252)
svmstdc = np.std(svms['svmdrtc']) * np.sqrt(252)
spystd = np.std(rspys['rspy']) * np.sqrt(252)

# 7.3.3. Maximum Margin Method Trading Strategy Annualized Sharpe Ratio
svmsr = svmyrt / svmstd
svmsrc = svmyrtc / svmstdc
spysr = spyyrt / spystd

# 7.3.3. Maximum Margin Method Trading Strategy Summary Results Data Table
svmdata = [{'0': '', '1': 'SVM', '2': 'SVMC', '3': 'SPY'},
        {'0': 'Annualized Return', '1': svmyrt, '2': svmyrtc, '3': spyyrt},
        {'0': 'Annualized Standard Deviation', '1': svmstd, '2': svmstdc, '3': spystd},
        {'0': 'Annualized Sharpe Ratio (Rf=0%)', '1': svmsr, '2': svmsrc, '3': spysr}]
svmtable = pd.DataFrame(svmdata)
print(svmtable)