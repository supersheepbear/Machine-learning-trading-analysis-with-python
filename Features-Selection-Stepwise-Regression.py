####################################################################
# Machine Trading Analysis with Python                             #
# Features Selection Stepwise Regression                           #
# (c) Diego Fernandez Garcia 2015-2017                             #
# www.exfinsis.com                                                 #
####################################################################

# 1. Packages Importing
import numpy as np
import pandas as pd
# import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.regression.linear_model as rg
import statsmodels.tools.tools as ct

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

#########

# 5. Predictor Features Selection

# 5.1. Predictor Features Stepwise Linear Regression

# 5.1.1. Linear Regression Predictor Features Selection (Step 1)
rspyt.loc[:, 'int'] = ct.add_constant(rspyt)
lmtapf = ['int', 'rspy1', 'rspy2', 'rspy3', 'rspy4', 'rspy5', 'rspy6', 'rspy7', 'rspy8', 'rspy9']
lmta = rg.OLS(rspyt['rspy'], rspyt[lmtapf], hasconst=bool).fit()
print("")
print("== Linear Regression Predictor Features Selection (Step 1) ==")
print("")
print(lmta.summary())
print("")

# 5.1.2. Linear Regression Predictor Features Selection (Step 2)
lmtbpf = ['int', 'rspy1', 'rspy2', 'rspy5']
lmtb = rg.OLS(rspyt['rspy'], rspyt[lmtbpf], hasconst=bool).fit()
print("")
print("== Linear Regression Predictor Features Selection (Step 2) ==")
print("")
print(lmtb.summary())
print("")

# 5.1.2. Linear Regression Predictor Features Selection Scatter Charts

# Linear Regression Predictor Features Selection Scatter Charts (Previous Day Returns)
fig1, ax = plt.subplots()
lmtb1 = np.polyfit(rspyt['rspy1'], rspyt['rspy'], deg=1)
ax.scatter(rspyt['rspy1'], rspyt['rspy'])
ax.plot(rspyt['rspy1'], lmtb1[0] * rspyt['rspy1'] + lmtb1[1], color='red')
ax.set_title('rspyt vs rspy1t')
ax.set_ylabel('rspyt')
ax.set_xlabel('rspy1t')
plt.show()

# Linear Regression Predictor Features Selection Scatter Charts (Second Previous Day Returns)
fig2, ax = plt.subplots()
lmtb2 = np.polyfit(rspyt['rspy2'], rspyt['rspy'], deg=1)
ax.scatter(rspyt['rspy2'], rspyt['rspy'])
ax.plot(rspyt['rspy2'], lmtb2[0] * rspyt['rspy2'] + lmtb2[1], color='red')
ax.set_title('rspyt vs rspy2t')
ax.set_ylabel('rspyt')
ax.set_xlabel('rspy2t')
plt.show()

# Linear Regression Predictor Features Selection Scatter Charts (Previous Week Returns)
fig3, ax = plt.subplots()
lmtb3 = np.polyfit(rspyt['rspy5'], rspyt['rspy'], deg=1)
ax.scatter(rspyt['rspy5'], rspyt['rspy'])
ax.plot(rspyt['rspy5'], lmtb3[0] * rspyt['rspy5'] + lmtb3[1], color='red')
ax.set_title('rspyt vs rspy5t')
ax.set_ylabel('rspyt')
ax.set_xlabel('rspy5t')
plt.show()

#########

# 6. Predictor Features Correlation

# 6.1. Predictor Features Correlation Matrix
crspypf = ['rspy1', 'rspy2', 'rspy3', 'rspy4', 'rspy5', 'rspy6', 'rspy7', 'rspy8', 'rspy9']
crspyt = rspyt[crspypf]
fig4, ax = plt.subplots()
cax = ax.matshow(crspyt.corr(), cmap="Blues")
fig4.colorbar(cax)
ax.set_xticklabels(['']+crspypf)
ax.set_yticklabels(['']+crspypf)
ax.set_title('Predictor Features Correlation Matrix')
plt.show()