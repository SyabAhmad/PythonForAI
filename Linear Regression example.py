# Load the diabetes dataset  // done
# Use only one feature /// done
# Split the data into training/testing sets // done
# Split the targets into training/testing sets // done
# Create linear regression object
# Train the model using the training sets
# Make predictions using the testing set

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
diabetesX, diabetesY = load_diabetes(return_X_y=True)

diabetesX = diabetesX[:, np.newaxis, 2]

diabetesXTrain = diabetesX[:-20]
diabetesXTest = diabetesX[-20:]

diabetesYTrain = diabetesY[:-20]
diabetesYTest = diabetesY[-20:]

linearObject = LinearRegression()
linearObject.fit(diabetesXTrain,diabetesYTrain)

predectionY = linearObject.predict(diabetesXTest)

print(predectionY)

print(mean_squared_error(diabetesYTest, predectionY))
print(r2_score(diabetesYTest, predectionY))
plt.plot(predectionY)
plt.show()


