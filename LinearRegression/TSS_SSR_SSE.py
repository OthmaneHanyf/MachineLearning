from first_regression_in_python import x1, y, f

# Calculating the mean
mean = 0
for i in range(len(x1)):
    mean += y[i]
mean = mean/len(x1)

# Calculating tss, ssr and sse

tss = 0
ssr = 0
sse = 0
for i in range(len(x1)):
    tss += (y[i] - mean)*(y[i] - mean)
    ssr += (f[i] - mean)*(f[i] - mean)
    sse += (f[i] - y[i])*(f[i] - y[i])

r_squered = ssr/tss

    
