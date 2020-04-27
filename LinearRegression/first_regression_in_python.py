import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import seaborn as sns

sns.set()

data = pd.read_csv('1.01. Simple linear regression.csv')

x1 = data['SAT']
y = data['GPA']

# this is the first methode which i have created by using linear algebra

A = np.array([[1, x] for x in x1])
AtA_inv = np.linalg.inv(np.dot(A.T, A))
A_inv = np.dot(AtA_inv, A.T)
[b0, b1] = np.dot(A_inv, y)

plt.scatter(x1, y)
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)

f = [(b0 + b1 * x) for x in x1] # bias + weights*variables
plt.plot(x1, f, label = 'least squere method', lw = 1)
# plt.show()

# the other methode using statsmodels lib

x = sm.add_constant(x1)
results = sm.OLS(y, x).fit()
results.summary()

g = [(0.2750 + 0.0017 * x) for x in x1]
plt.plot(x1, g, label = 'OLS(statsmodels)', lw = 1)

plt.legend()
plt.show()


