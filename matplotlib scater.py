from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import matplotlib.pylab as plt

x, y = load_diabetes(return_X_y=True)
diabetes = LinearRegression()
diabetes.fit(x,y)
pred = diabetes.predict(x)
plt.scatter(pred, y)
print(plt.scatter(pred, y))

