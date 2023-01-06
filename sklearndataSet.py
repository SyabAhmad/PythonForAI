from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
X,y = load_breast_cancer(return_X_y=True)
cancer = KNeighborsRegressor()
cancer.fit(X, y)
print(cancer.predict(X))
cancer1 = LinearRegression()
cancer1.fit(X,y)
print(cancer.predict(X))
