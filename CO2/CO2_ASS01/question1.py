import numpy as np
x = np.array([1000,1500,800,1200,2000], dtype=float)
y = np.array([50,75,40,60,90], dtype=float)
xbar= x.mean()
ybar=y.mean()
b1 = np.sum((x-xbar)*(y-ybar)) / np.sum((x-xbar)**2)
b0 = ybar - b1*xbar

print('\n')
print("Model form: Price = b0 + b1 * Area")
print('\n')
print("Regression coefficients: b0 =", b0, " b1 =", b1)
print("     Price =", b0, "+", b1, "* Area")
print('\n')

print("      b1 means each extra sq.ft increases price by", b1, "Lakhs")
print('\n')