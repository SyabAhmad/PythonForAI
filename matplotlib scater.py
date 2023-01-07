from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x, y = load_diabetes(return_X_y=True)
diabetes = LinearRegression()
diabetes.fit(x,y)
pred = diabetes.predict(x)
plt.scatter(pred, y)
print(plt.scatter(pred, y))
plt.show()

# from sklearn import preprocessing
# from sklearn.linear_model import LinearRegression
# from sklearn.datasets import load_iris
# import numpy as np
#
# x,y = load_iris(return_X_y=True)
# iris = LinearRegression()
# iris.fit(x,y)
# print(iris.)
#
# #print(iris.predict(x))


