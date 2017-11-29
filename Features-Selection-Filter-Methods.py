####################################################################
# Machine Trading Analysis with Python                             #
# Features Selection Filter Methods                                #
# (c) Diego Fernandez Garcia 2015-2017                             #
# www.exfinsis.com                                                 #
####################################################################

# 1. Packages Importing
import numpy as np
import pandas as pd
# import pandas_datareader.data as web
import sklearn.feature_selection as fs

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

# 5. Predictor Features Selection (Filter Methods)

# 5.1. Predictor Features
pfst = ['rspy1', 'rspy2', 'rspy3', 'rspy4', 'rspy5', 'rspy6', 'rspy7', 'rspy8', 'rspy9']
pfft = fs.f_regression(rspyt[pfst], rspyt['rspy'])
print("== Predictor Features ==")
print("")
print("Predictor Features Regression F Scores:")
print("['rspy1', 'rspy2', 'rspy3', 'rspy4', 'rspy5', 'rspy6', 'rspy7', 'rspy8', 'rspy9']")
np.set_printoptions(precision=2)
print(pfft[0])
print("")

# 5.2. False Discovery Rate
# Benjamini-Hochberg Procedure
fdrt = fs.SelectFdr(score_func=fs.f_regression, alpha=0.05).fit(rspyt[pfst], rspyt['rspy'])
print("== False Discovery Rate ==")
print("")
print("False Discovery Rate Features Selection:")
print(fdrt.get_support())
print("")

# 5.3. Family-Wise Error Rate
# Bonferroni Procedure
fwet = fs.SelectFwe(score_func=fs.f_regression, alpha=0.05).fit(rspyt[pfst], rspyt['rspy'])
print("== Family-Wise Error Rate ==")
print("")
print("Family-Wise Error Rate Features Selection:")
print(fwet.get_support())
print("")