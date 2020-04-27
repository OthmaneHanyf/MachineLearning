import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import standerdization
import statsmodels.api as sm

# Make a 3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

# importing data
data = pd.read_csv('real_estate_price_size_year_feature_scaling.csv')

# standardize(normalize) the data
std_year = pd.DataFrame( [[x] for x in standerdization.standardize(data['year'])], columns=['std_year'])
std_size = pd.DataFrame( [[x] for x in standerdization.standardize(data['size'])], columns=['std_size'])

# Setting the sides of the equation x = (At*A)^(-1)*At*y
A = [[1, x1] for x1 in std_size['std_size']]
std_years = std_year['std_year']
for i in range(len(std_year['std_year'])):
    A[i].append(std_years[i])  
# A = sm.add_constant(A)
y = data['price']
A = np.array(A)

AtA_inv = np.linalg.inv(np.dot(A.T, A))
print(A.shape, A.T.shape)
A_inv = np.dot(AtA_inv, A.T)
x = np.dot(A_inv, y)
[b0, b1, b2] = x

# Define the two regression equations (one with a dummy = 1, the other with dummy = 0)
y_hat = b0 + b1 * std_size['std_size'] + b2 * std_year['std_year']

# Create one scatter plot which contains all observations
# Use the series 'Attendance' as color, and choose a colour map of your choice
# The colour map we've chosen is completely arbitrary
ax.scatter(std_size['std_size'], std_year['std_year'], y)
ax.plot(std_size['std_size'], std_year['std_year'], y_hat)
# Set the labels
ax.set_xlabel('SIZES', fontsize = 10)
ax.set_ylabel('YEARS', fontsize = 10)
ax.set_zlabel('PRISES', fontsize = 10)

ax.legend()
plt.show()



