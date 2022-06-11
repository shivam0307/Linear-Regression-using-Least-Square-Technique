# GATE-Score-vs-Academic-Performance
GATE Score vs Academic Performance using Mathematical Techniques
# The Problem is solved using Linear Regression with Least Square Techniques
# The following is Source Code in Python for above Model

import numpy as np

import matplotlib.pyplot as plt

x = np.array([95,85,80,70,60]) # Gate Scores
y = np.array([85,95,70,65,70]) # Academic Precentages

n = np.size(x)
m_x , m_y = np.mean(x) , np.mean(y)     #mean of x and y

print("X_mean: ",m_x)
print("Y_mean: ",m_y)

ss_xy = np.sum(y*x)-n*(m_x)*(m_y)    #numerator
ss_xx = np.sum(x*x)-n*(m_x)*(m_x)    #denominator

b0_1 = ss_xy / ss_xx                 #intercept
b0_0 = m_y - b0_1*m_x                #slope

print("intercept: ",b0_1)
print("slope: ",b0_0)

y_pred = b0_0 + b0_1*x

print(y_pred)

plt.scatter(x,y)
plt.plot(x,y_pred, color='k', marker='o')

import math

sigma_x = math.sqrt(n*np.sum(x*x)-np.sum(x)*np.sum(x))
sigma_y = math.sqrt(n*np.sum(y*y)-np.sum(y)*np.sum(y))

print("Standard Deviations: ", sigma_x, ",", sigma_y)

cov_xy = n*np.sum(x*y)-(np.sum(x)*np.sum(y))

print("Covariance: ",cov_xy)

r = cov_xy/(sigma_x*sigma_y)         #using statistical formula for correlation coefficient

print("Correlation Coefficient: ", r)

plt.scatter(x,y)
plt.plot(x,y_pred)

#Finding Coefficient of Determination by Statistical Formula for Coefficient of Determination
sse = np.sum((y-y_pred)*(y-y_pred))
sst = np.sum((y-m_y)*(y-m_y))
R2 = 1-(sse/sst)
print("Coefficient of Determination: ", R2)
