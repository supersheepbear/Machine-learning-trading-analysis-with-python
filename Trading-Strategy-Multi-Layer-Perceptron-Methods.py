####################################################################
# Machine Trading Analysis with Python                             #
# Trading Strategy Multi-Layer Perceptron Methods                  #
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

# 5. Multi-Layer Perceptron Method Algorithm Learning

# 5.1. Multi-Layer Perceptron Method Algorithm Features

# 5.1.1. Features Extraction (Principal Component Analysis)
# PCA for Training and Trading Subsets
pfb = ['rspy1', 'rspy2', 'rspy3', 'rspy4', 'rspy5', 'rspy6', 'rspy7', 'rspy8', 'rspy9']
pcat = fe.PCA().fit_transform(rspyt[pfb], rspyt['rspy'])
pcas = fe.PCA().fit_transform(rspys[pfb], rspys['rspy'])

# 5.2. Multi-Layer Perceptron Method Algorithm Training Optimal Parameter Selection

# 5.2.1. Time Series Cross-Validation
cvanntb = cv.GridSearchCV(ml.MLPRegressor(), cv=cv.TimeSeriesSplit(n_splits=10),
                          param_grid={"alpha": [0.0001, 0.001, 0.010, 0.100]}).fit(pcat, rspyt['rspy'])

# 5.2.2. Time Series Cross-Validation Optimal Parameter Selection
cvannparb = cvanntb.best_estimator_.alpha

# 5.3. Multi-Layer Perceptron Method Algorithm Training
anntb = ml.MLPRegressor(alpha=cvannparb).fit(pcat, rspyt['rspy'])

# 5.4. Multi-Layer Perceptron Method Forecasting
# Forecasting for Trading Subset
annpb = anntb.predict(pcas)
anns = pd.DataFrame(annpb, index=rspys.index)
anns.columns = ['annpb']

#########

# 6. Multi-Layer Perceptron Method Trading Strategy

# 6.1. Multi-Layer Perceptron Method Trading Strategy Signals
# Previous Periods Data (avoid back-testing bias)
# Generate Trading Strategy Signals (buy stock=1 , sell stock=-1, do nothing=0)
anns['annpb(-1)'] = anns['annpb'].shift(1)
anns['annpb(-2)'] = anns['annpb'].shift(2)
anns['annsig'] = 0
annsig = 0
for i, r in enumerate(anns.iterrows()):
    if r[1]['annpb(-2)'] - np.mean(anns['annpb(-2)'][0:i]) < 0 and r[1]['annpb(-1)'] - np.mean(anns['annpb(-1)'][0:i]) > 0:
        annsig = 1
    elif r[1]['annpb(-2)'] - np.mean(anns['annpb(-2)'][0:i]) > 0 and r[1]['annpb(-1)'] - np.mean(anns['annpb(-1)'][0:i]) < 0:
        annsig = -1
    else:
        annsig = 0
    anns.iloc[i, 3] = annsig

# 6.1.1. Multi-Layer Perceptron Method Trading Strategy Signals Chart
anns['annpbc'] = anns['annpb'] - np.mean(anns['annpb'])
anns['annpbcl'] = 0
annsigc = anns['annsig']
annbuy = annsigc[annsigc == 1]
annsell = annsigc[annsigc == -1]
fig1, ax = plt.subplots(2, sharex=True)
ax[0].plot(anns['annpbc'])
ax[0].plot(anns['annpbcl'], linestyle="--")
ax[0].legend(loc='upper right')
ax[1].plot(annbuy, marker='^', linestyle='', color='g', label='annbuy')
ax[1].plot(annsell, marker='v', linestyle='', color='r', label='annsell')
ax[1].legend(loc='upper right')
plt.suptitle('Multi-Layer Perceptron Method Trading Strategy Signals Chart')
plt.show()

# 6.2. Multi-Layer Perceptron Method Trading Strategy Positions
# Generate Trading Strategy Positions (own stock=1 , not own stock=0, short-selling not available)
anns['annpos'] = 0
annpos = 0
for i, r in enumerate(anns.iterrows()):
    if r[1]['annsig'] == 1:
        annpos = 1
    elif r[1]['annsig'] == -1:
        annpos = 0
    else:
        annpos = anns['annpos'][i-1]
    anns.iloc[i, 6] = annpos

# 6.2.1. Multi-Layer Perceptron Method Trading Strategy Positions Chart
annposc = anns['annpos']
annopen = annposc[annposc == 1]
annclosed = annposc[annposc == 0]
fig2, ax = plt.subplots(2, sharex=True)
ax[0].plot(anns['annpbc'])
ax[0].plot(anns['annpbcl'], linestyle="--")
ax[0].legend(loc='upper right')
ax[1].plot(annopen, marker='.', linestyle='', color='g', label='annopen')
ax[1].plot(annclosed, marker='.', linestyle='', color='r', label='annclosed')
ax[1].legend(loc='upper right')
plt.suptitle('Multi-Layer Perceptron Method Trading Strategy Positions Chart')
plt.show()

##########

# 7. Multi-Layer Perceptron Method Trading Strategy Performance Comparison

# 7.1. Multi-Layer Perceptron Method Trading Strategy Daily Returns

# 7.1.1. Multi-Layer Perceptron Method Strategy Daily Returns Without Trading Commissions
anns['anndrt'] = rspys['rspy'] * anns['annpos']

# 7.1.2. Multi-Layer Perceptron Method Strategy Daily Returns With Trading Commissions (0.10% Per Trade)
anns['annpos(-1)'] = anns['annpos'].shift(1)
anns['anntc'] = 0
anntc = 0
for i, r in enumerate(anns.iterrows()):
    if (r[1]['annsig'] == 1 or r[1]['annsig'] == -1) and r[1]['annpos'] != r[1]['annpos(-1)']:
        anntc = 0.001
    else:
        anntc = 0.000
    anns.iloc[i, 9] = anntc
anns['anndrtc'] = (rspys['rspy'] * anns['annpos']) - anns['anntc']

# 7.2. Multi-Layer Perceptron Method Trading Strategy Cumulative Returns

# 7.2.1. Multi-Layer Perceptron Method Trading Strategy Cumulative Returns Calculation
anns['anncrt'] = np.cumprod(anns['anndrt'] + 1) - 1
anns['anncrtc'] = np.cumprod(anns['anndrtc'] + 1) - 1
anns['spycrt'] = np.cumprod(rspys['rspy'] + 1) - 1

# 7.2.2. Multi-Layer Perceptron Method Trading Strategy Cumulative Returns Chart
fig3, ax = plt.subplots()
ax.plot(anns['anncrt'])
ax.plot(anns['anncrtc'])
ax.plot(anns['spycrt'])
ax.legend(loc='upper left')
plt.suptitle('Multi-Layer Perceptron Method Trading Strategy Cumulative Returns Chart')
plt.show()

# 7.3. Multi-Layer Perceptron Method Trading Strategy Performance Metrics

# 7.3.1. Multi-Layer Perceptron Method Trading Strategy Annualized Returns
annyrt = anns.iloc[251, 11]
annyrtc = anns.iloc[251, 12]
spyyrt = anns.iloc[251, 13]

# 7.3.2. Multi-Layer Perceptron Method Trading Strategy Annualized Standard Deviation
annstd = np.std(anns['anndrt']) * np.sqrt(252)
annstdc = np.std(anns['anndrtc']) * np.sqrt(252)
spystd = np.std(rspys['rspy']) * np.sqrt(252)

# 7.3.3. Multi-Layer Perceptron Method Trading Strategy Annualized Sharpe Ratio
annsr = annyrt / annstd
annsrc = annyrtc / annstdc
spysr = spyyrt / spystd

# 7.3.3. Multi-Layer Perceptron Method Trading Strategy Summary Results Data Table
anndata = [{'0': '', '1': 'ANN', '2': 'ANNC', '3': 'SPY'},
        {'0': 'Annualized Return', '1': annyrt, '2': annyrtc, '3': spyyrt},
        {'0': 'Annualized Standard Deviation', '1': annstd, '2': annstdc, '3': spystd},
        {'0': 'Annualized Sharpe Ratio (Rf=0%)', '1': annsr, '2': annsrc, '3': spysr}]
anntable = pd.DataFrame(anndata)
print(anntable)