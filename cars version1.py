import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
data = pd.read_csv("cars.csv", header=None)

cars1T = data[:-20]
cars2T = data[-20:]

cars3T = data[:-20]
cars4T = data[-20:]

linear = LinearRegression()
linear.fit(cars1T, cars2T)
cars4predection = linear.predict(cars4T)
plt.plot(cars4predection)
plt.show()

# print(linear.predict(cars4T))