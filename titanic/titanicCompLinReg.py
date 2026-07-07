from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


titanicData = pd.read_csv("titanicData/train.csv")

#convert male to 0 and female to 1
titanicData["Sex"] = titanicData["Sex"].map({"male":0,"female":1})

#fill na ages with the mean of the data
titanicData["Age"] = titanicData["Age"].fillna(titanicData["Age"].mean())


#fill na fares with mean of the data
titanicData["Fare"] = titanicData["Fare"].fillna(

    titanicData["Fare"].median()

)

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

X = titanicData[features]
y = titanicData["Survived"]

#create a training and validation split, take 20% for validation, 80% for training.
X_train, X_validation, y_train, y_validation = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

linearModel = KNeighborsClassifier(n_neighbors=5)

linearModel.fit(X_train,y_train)

predictions = linearModel.predict(X_validation)

accuracy = accuracy_score(y_validation, predictions)

print("Validation accuracy:", accuracy)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)

scaledLinearModel = KNeighborsClassifier(n_neighbors=5)

scaledLinearModel.fit(X_train_scaled, y_train)

predictions = scaledLinearModel.predict(X_validation_scaled)

accuracy = accuracy_score(y_validation, predictions)

print("Scaled accuracy:", accuracy)