import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

# musicData = pd.read_csv("music.csv")
# X = musicData.drop(columns=['genre'])
# Y = musicData[['genre']]
#
# Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)
# model = DecisionTreeClassifier()
# mod = model.fit(Xtrain, Ytrain)
# joblib.dump(mod,"musicfile.joblib")
model = joblib.load("musicfile.joblib")
predection = model.predict([[21,1],[34,0]])
print(predection)


# predictions = model.predict(Xtest)
# score = accuracy_score(Ytest, predictions)
# print(score)

