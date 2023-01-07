import pandas as pd
from sklearn.tree import DecisionTreeClassifier
musicData = pd.read_csv("music.csv")

X = musicData.drop(columns=['genre'])
Y = musicData[['genre']]
print(X)
print(Y)
model = DecisionTreeClassifier()
model.fit(X, Y)
predictions = model.predict([[21,1],[57,0]])
print(predictions)

