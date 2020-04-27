import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import *
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
sns.set()
fig = plt.figure()

ax = fig.add_subplot(131, projection = '3d')
ax1 = fig.add_subplot(132)
ax2 = fig.add_subplot(133)

# importing the dataset
data = pd.read_csv('real_estate_price_size_year.csv')

# Setting the equation's sides
A = [[1, x1, 1] for x1 in data['size']]
sizes = data['size']
years = data['year']
for i in range(len(sizes)):
    A[i][2] = years[i]
A = np.array(A)
y = data['price']

# Calculation
AtA_inv = np.linalg.inv(np.dot(A.T, A))
A_inv = np.dot(AtA_inv, A.T)
[b0, b1, b2] = np.dot(A_inv, y)

f1 = [(b0 + b1 * x) for x in sizes ]
f2 = [(b2 * x) for x in years ]

f = []
for i in range(len(sizes)):
    f.append(f1[i] + f2[i])

# plot the sample data 
ax.scatter(sizes, years, y, c = 'b', marker = 'o', label = 'given data')
ax1.scatter(sizes, y)
ax2.scatter(years, y)
# plot the prodected values
ax.plot(sizes, years, f, c = 'r', label = 'prodected model')
ax1.plot(sizes, f1)
for i in range(len(years)):
    f2[i] += b0
ax2.plot(years, f2)

ax.set_xlabel('SIZES')
ax.set_ylabel('YEARS')
ax.set_zlabel('PRISES')

ax.legend()
plt.show()

# using statsmodels library

B = data[['size', 'year']]
B = sm.add_constant(B)
res = sm.OLS(y, B).fit()
res.summary()

# then repeat the process from line 33 


                         

