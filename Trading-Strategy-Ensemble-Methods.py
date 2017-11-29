####################################################################
# Machine Trading Analysis with Python                             #
# Trading Strategy Ensemble Methods                                #
# (c) Diego Fernandez Garcia 2015-2017                             #
# www.exfinsis.com                                                 #
####################################################################

# 1. Packages Importing
import numpy as np
import pandas as pd
# import pandas_datareader.data as web
import sklearn.decomposition as fe
import sklearn.model_selection as cv
import sklearn.ensemble as ml
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

# 5. Ensemble Method Algorithm Learning

# 5.1. Ensemble Method Algorithm Features

# 5.1.1. Features Extraction (Principal Component Analysis)
# PCA for Training and Trading Subsets
pfb = ['rspy1', 'rspy2', 'rspy3', 'rspy4', 'rspy5', 'rspy6', 'rspy7', 'rspy8', 'rspy9']
pcat = fe.PCA().fit_transform(rspyt[pfb], rspyt['rspy'])
pcas = fe.PCA().fit_transform(rspys[pfb], rspys['rspy'])

# 5.2. Ensemble Method Algorithm Training Optimal Parameter Selection

# 5.2.1. Time Series Cross-Validation
cvgbmtb = cv.GridSearchCV(ml.GradientBoostingRegressor(), cv=cv.TimeSeriesSplit(n_splits=10),
                          param_grid={"max_depth": [1, 2, 3, 4, 5]}).fit(pcat, rspyt['rspy'])

# 5.2.2. Time Series Cross-Validation Optimal Parameter Selection
cvgbmparb = cvgbmtb.best_estimator_.max_depth

# 5.3. Ensemble Method Algorithm Training
gbmtb = ml.GradientBoostingRegressor(max_depth=cvgbmparb).fit(pcat, rspyt['rspy'])

# 5.4. Ensemble Method Algorithm Forecasting
# Forecasting for Trading Subset
gbmpb = gbmtb.predict(pcas)
gbms = pd.DataFrame(gbmpb, index=rspys.index)
gbms.columns = ['gbmpb']

#########

# 6. Ensemble Method Trading Strategy

# 6.1. Ensemble Method Trading Strategy Signals
# Previous Periods Data (avoid back-testing bias)
# Generate Trading Strategy Signals (buy stock=1 , sell stock=-1, do nothing=0)
gbms['gbmpb(-1)'] = gbms['gbmpb'].shift(1)
gbms['gbmpb(-2)'] = gbms['gbmpb'].shift(2)
gbms['gbmsig'] = 0
gbmsig = 0
for i, r in enumerate(gbms.iterrows()):
    if r[1]['gbmpb(-2)'] - np.mean(gbms['gbmpb(-2)'][0:i]) < 0 and r[1]['gbmpb(-1)'] - np.mean(gbms['gbmpb(-1)'][0:i]) > 0:
        gbmsig = 1
    elif r[1]['gbmpb(-2)'] - np.mean(gbms['gbmpb(-2)'][0:i]) > 0 and r[1]['gbmpb(-1)'] - np.mean(gbms['gbmpb(-1)'][0:i]) < 0:
        gbmsig = -1
    else:
        gbmsig = 0
    gbms.iloc[i, 3] = gbmsig

# 6.1.1. Ensemble Method Trading Strategy Signals Chart
gbms['gbmpbc'] = gbms['gbmpb'] - np.mean(gbms['gbmpb'])
gbms['gbmpbcl'] = 0
gbmsigc = gbms['gbmsig']
gbmbuy = gbmsigc[gbmsigc == 1]
gbmsell = gbmsigc[gbmsigc == -1]
fig1, ax = plt.subplots(2, sharex=True)
ax[0].plot(gbms['gbmpbc'])
ax[0].plot(gbms['gbmpbcl'], linestyle='--')
ax[0].legend(loc='upper right')
ax[1].plot(gbmbuy, marker='^', linestyle='', color='g', label='gbmbuy')
ax[1].plot(gbmsell, marker='v', linestyle='', color='r', label='gbmsell')
ax[1].legend(loc='upper right')
plt.suptitle('Ensemble Method Trading Strategy Signals Chart')
plt.show()

# 6.2. Ensemble Method Trading Strategy Positions
# Generate Trading Strategy Positions (own stock=1 , not own stock=0, short-selling not available)
gbms['gbmpos'] = 0
gbmpos = 0
for i, r in enumerate(gbms.iterrows()):
    if r[1]['gbmsig'] == 1:
        gbmpos = 1
    elif r[1]['gbmsig'] == -1:
        gbmpos = 0
    else:
        gbmpos = gbms['gbmpos'][i-1]
    gbms.iloc[i, 6] = gbmpos

# 6.2.1. Ensemble Method Trading Strategy Positions Chart
gbmposc = gbms['gbmpos']
gbmopen = gbmposc[gbmposc == 1]
gbmclosed = gbmposc[gbmposc == 0]
fig2, ax = plt.subplots(2, sharex=True)
ax[0].plot(gbms['gbmpbc'])
ax[0].plot(gbms['gbmpbcl'], linestyle='--')
ax[0].legend(loc='upper right')
ax[1].plot(gbmopen, marker='.', linestyle='', color='g', label='gbmopen')
ax[1].plot(gbmclosed, marker='.', linestyle='', color='r', label='gbmclosed')
ax[1].legend(loc='upper right')
plt.suptitle('Ensemble Method Trading Strategy Positions Chart')
plt.show()

##########

# 7. Ensemble Method Trading Strategy Performance Comparison

# 7.1. Ensemble Method Trading Strategy Daily Returns

# 7.1.1. Ensemble Method Strategy Daily Returns Without Trading Commissions
gbms['gbmdrt'] = rspys['rspy'] * gbms['gbmpos']

# 7.1.2. Ensemble Method Strategy Daily Returns With Trading Commissions (0.10% Per Trade)
gbms['gbmpos(-1)'] = gbms['gbmpos'].shift(1)
gbms['gbmtc'] = 0
gbmtc = 0
for i, r in enumerate(gbms.iterrows()):
    if (r[1]['gbmsig'] == 1 or r[1]['gbmsig'] == -1) and r[1]['gbmpos'] != r[1]['gbmpos(-1)']:
        gbmtc = 0.001
    else:
        gbmtc = 0.000
    gbms.iloc[i, 9] = gbmtc
gbms['gbmdrtc'] = (rspys['rspy'] * gbms['gbmpos']) - gbms['gbmtc']

# 7.2. Ensemble Method Trading Strategy Cumulative Returns

# 7.2.1. Ensemble Method Trading Strategy Cumulative Returns Calculation
gbms['gbmcrt'] = np.cumprod(gbms['gbmdrt'] + 1) - 1
gbms['gbmcrtc'] = np.cumprod(gbms['gbmdrtc'] + 1) - 1
gbms['spycrt'] = np.cumprod(rspys['rspy'] + 1) - 1

# 7.2.2. Ensemble Method Trading Strategy Cumulative Returns Chart
fig3, ax = plt.subplots()
ax.plot(gbms['gbmcrt'])
ax.plot(gbms['gbmcrtc'])
ax.plot(gbms['spycrt'])
ax.legend(loc='upper left')
plt.suptitle('Ensemble Method Trading Strategy Cumulative Returns Chart')
plt.show()

# 7.3. Ensemble Method Trading Strategy Performance Metrics

# 7.3.1. Ensemble Method Trading Strategy Annualized Returns
gbmyrt = gbms.iloc[251, 11]
gbmyrtc = gbms.iloc[251, 12]
spyyrt = gbms.iloc[251, 13]

# 7.3.2. Ensemble Method Trading Strategy Annualized Standard Deviation
gbmstd = np.std(gbms['gbmdrt']) * np.sqrt(252)
gbmstdc = np.std(gbms['gbmdrtc']) * np.sqrt(252)
spystd = np.std(rspys['rspy']) * np.sqrt(252)

# 7.3.3. Ensemble Method Trading Strategy Annualized Sharpe Ratio
gbmsr = gbmyrt / gbmstd
gbmsrc = gbmyrtc / gbmstdc
spysr = spyyrt / spystd

# 7.3.3. Ensemble Method Trading Strategy Summary Results Data Table
gbmdata = [{'0': '', '1': 'GBM', '2': 'GBMC', '3': 'SPY'},
        {'0': 'Annualized Return', '1': gbmyrt, '2': gbmyrtc, '3': spyyrt},
        {'0': 'Annualized Standard Deviation', '1': gbmstd, '2': gbmstdc, '3': spystd},
        {'0': 'Annualized Sharpe Ratio (Rf=0%)', '1': gbmsr, '2': gbmsrc, '3': spysr}]
gbmtable = pd.DataFrame(gbmdata)
print(gbmtable)