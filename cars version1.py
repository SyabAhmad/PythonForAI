import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
data1= pd.read_csv("cars.csv")

print(data1.describe())




# trimedData = data[:20]
# print(trimedData)

# cars1T = data[:-20]
# cars2T = data[-20:]
#
# cars3T = data[:-20]
# cars4T = data[-20:]
#
# linear = LinearRegression()
# linear.fit(cars1T, cars2T)
# cars4predection = linear.predict(cars4T)
# plt.plot(cars4predection)
# plt.show()

# print(linear.predict(cars4T))