import numpy
from sklearn import preprocessing

inputLabels = ['red', 'green', 'blue', 'orange', 'yellow', 'black']
encoder = preprocessing.LabelEncoder()
encoder.fit(inputLabels)

testLabels = ["red", "black", "green"]
encodedValues = encoder.transform(testLabels)
print(encodedValues)
