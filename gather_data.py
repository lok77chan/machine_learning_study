import pandas as pd
import numpy as np
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
import quandl, math
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import os
import json
import pprint
import csv


df = pd.read_csv('WIKI-GOOGL.csv', parse_dates=True)
datetimeRow = pd.to_datetime(df['Date'])
df = df.set_index(pd.DatetimeIndex(datetimeRow))
df = df.sort_index()
print(df.head())

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]


forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.05 * len(df)))


df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=0)

# clf = svm.SVR()
# clf.fit(X_train, y_train)
# confidence = clf.score(X_test, y_test)
# print('svr score: {}'.format(confidence))

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print('LinearRegression: {}'.format(confidence))

forecast_set = clf.predict(X_lately)

print(forecast_set, confidence, forecast_out)

# for k in ['linear', 'poly', 'rbf', 'sigmoid']:
#     clf = svm.SVR(kernel=k)
#     clf.fit(X_train, y_train)
#     confidence = clf.score(X_test, y_test)
#     print('{}: {}'.format(k, confidence))

# plotting the result
style.use('ggplot')
df['Forecast'] = np.nan

last_data = df.iloc[-1].name
last_unix = last_data.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_data = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_data] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


