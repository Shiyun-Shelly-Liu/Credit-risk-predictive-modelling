# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 12:06:21 2024

@author: Shiyun Liu
"""
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import pandas as pd
from math import log

# Read in loan data from a CSV file
df = pd.read_csv('Loan_data.csv')

# Define the variable features
features = ['credit_lines_outstanding', 'debt_to_income',
            'payment_to_income', 'years_employed', 'fico_score']

# Calculate the payment_to_income ratio
df['payment_to_income'] = df['loan_amt_outstanding'] / df['income']

# Calculate the debt_to_income ratio
df['debt_to_income'] = df['total_debt_outstanding'] / df['income']

clf = LogisticRegression(random_state=0, solver='liblinear',
                         tol=1e-5, max_iter=10000).fit(df[features], df['default'])
print(clf.coef_, clf.intercept_)

# Use the following code to check yourself
y_pred = clf.predict(df[features])

fpr, tpr, thresholds = metrics.roc_curve(df['default'], y_pred)
print((1.0*(abs(df['default']-y_pred)).sum()) / len(df))
print(metrics.auc(fpr, tpr))


# %% maximum likelihood

x = df['default'].to_list()
y = df['fico_score'].to_list()
n = len(x)
print(len(x), len(y))

default = [0 for i in range(851)]
total = [0 for i in range(851)]

for i in range(n):
    y[i] = int(y[i])
    default[y[i]-300] += x[i]
    total[y[i]-300] += 1

for i in range(0, 551):
    default[i] += default[i-1]
    total[i] += total[i-1]


def log_likelihood(n, k):
    p = k/n
    if (p == 0 or p == 1):
        return 0
    return k*np.log(p) + (n-k)*np.log(1-p)


r = 10
dp = [[[-10**18, 0] for i in range(551)] for j in range(r+1)]

for i in range(r+1):
    for j in range(551):
        if (i == 0):
            dp[i][j][0] = 0
        else:
            for k in range(j):
                if (total[j] == total[k]):
                    continue
                if (i == 1):
                    dp[i][j][0] = log_likelihood(total[j], default[j])
                else:
                    if (dp[i][j][0] < (dp[i-1][k][0] + log_likelihood(total[j]-total[k], default[j] - default[k]))):
                        dp[i][j][0] = log_likelihood(
                            total[j]-total[k], default[j]-default[k]) + dp[i-1][k][0]
                        dp[i][j][1] = k

print(round(dp[r][550][0], 4))

k = 550
l = []
while r >= 0:
    l.append(k+300)
    k = dp[r][k][1]
    r -= 1

print(l)
