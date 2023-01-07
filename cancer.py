from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

x1,y1 = load_breast_cancer(return_X_y=True)
x1 = x1[:,np.newaxis,2]

x1train = x1[:-20]
x1test = x1[-20:]

y1train = y1[:-20]
y1test = y1[-20:]

cancerobject = LinearRegression()

cancerobject.fit(x1train, y1train)

x1predection =  cancerobject.predict(x1test)
#y1predection = cancerobject.predict(y1test)

print(x1predection)
plt.plot(x1predection)
plt.show()
#print(y1predection)
#cancerobject.fit(trimedcancer, trim)

