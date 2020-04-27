
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression 
import pandas as pd 

data = pd.read_csv('1.01. Simple linear regression.csv')
a = data['SAT']
y = data['GPA']
A = a.values.reshape(-1, 1)

def adjusted_r_squared(regression_class_instance):
    # R^{2}_{adj.} = 1 - (1-R^{2}) \frac{n-1}{n-p-1}
    # n = the number of observations
    # p = the number of predictors
    n = A.shape[0]
    p = A.shape[1]
    r2 = r_squared(regression_class_instance)
    return 1 - (1 - r2)*(n - 1)/(n - p - 1)

def p_values(regression_class_instance, SD):
    return f_regression(A, y)[1].round(SD)

def r_squared(regression_class_instance):
    return regression_class_instance.score(A, y)

def coefficients(regression_class_instance):
    return regression_class_instance.coef_

def intercept(regression_class_instance): # Or the Bias
    return regression_class_instance.intercept_
    
def predict(newdata, regression_class_instance):
    return regression_class_instance.predict(newdata)

def main(sd):
    reg = LinearRegression()
    reg.fit(A, y) # (input, output)
    print(p_values(reg, sd))
