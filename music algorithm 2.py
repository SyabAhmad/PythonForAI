import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
musicData = pd.read_csv("music.csv")

X = musicData.drop(columns=['genre'])
Y = musicData[['genre']]

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(Xtrain, Ytrain)
predictions = model.predict(Xtest)
score = accuracy_score(Ytest, predictions)
print(score)

