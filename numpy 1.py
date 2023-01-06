import numpy as np
import sklearn.preprocessing
from sklearn import preprocessing

values = np.array([[2.1, -1.9, 5.5],[-1.5, 2.4, 3.5],[0.5, -7.9, 5.6],[5.9, 2.3, -5.8]])

binaryValues = preprocessing.Binarizer(threshold = 0.5).transform(values)

print(np.mean(binaryValues))
print(np.std(binaryValues))
print(np.mean(values))

dataScale = preprocessing.scale(values)
print(dataScale.mean(axis=0))
print(dataScale.std(axis=0))

noramlizaiton = preprocessing.normalize(binaryValues, norm='l1')
print(noramlizaiton)
print(noramlizaiton.mean(axis=0))

#for val in values:
#    print(binaryValues)
