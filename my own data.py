from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import numpy as np

reg = linear_model.LinearRegression()
reg.fit([[1,2],[4,5],[7,8]],[9,2,6])
print(reg.coef_)


