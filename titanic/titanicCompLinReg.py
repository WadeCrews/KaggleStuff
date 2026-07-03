from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


titanicData = pd.read_csv("titanicData/train.csv").dropna()

#convert male to 0 and female to 1
titanicData["Sex"] = titanicData["Sex"].map({"male":0,"female":1})

titanicData["Age"] = titanicData["Age"].fillna(titanicData["Age"].mean())

X = titanicData[["Sex","Age"]]
y = titanicData[["Survived"]]
y_flat = np.ravel(y)

linearModel = LinearRegression()

linearModel.fit(X,y_flat)

testData = pd.read_csv("titanicData/test.csv")

#convert male to 0 and female to 1
testData["Sex"] = testData["Sex"].map({"male": 0, "female": 1})

#fill missing ages with mean age
testData["Age"] = testData["Age"].fillna(titanicData["Age"].mean())

X_test = testData[["Sex", "Age"]]

predictions = linearModel.predict(X_test)

print(predictions)