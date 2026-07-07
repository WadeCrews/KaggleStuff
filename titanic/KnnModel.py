from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

# -----------------------------
# Load training data
# -----------------------------
trainData = pd.read_csv("titanicData/train.csv")

# Convert Sex to numbers
trainData["Sex"] = trainData["Sex"].map({
    "male": 0,
    "female": 1
})

# Save these values so test data uses the same values
age_median = trainData["Age"].median()
fare_median = trainData["Fare"].median()

# Fill missing training values
trainData["Age"] = trainData["Age"].fillna(age_median)
trainData["Fare"] = trainData["Fare"].fillna(fare_median)

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

X = trainData[features]

# Use one pair of brackets so y is one-dimensional
y = trainData["Survived"]

# -----------------------------
# Scale all training data
# -----------------------------
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Use whichever settings worked best on validation data
finalModel = KNeighborsClassifier(
    n_neighbors=5
)

# Train using all of train.csv
finalModel.fit(X_scaled, y)

# -----------------------------
# Load Kaggle test data
# -----------------------------
testData = pd.read_csv("titanicData/test.csv")

testData["Sex"] = testData["Sex"].map({
    "male": 0,
    "female": 1
})

# Use values calculated from train.csv
testData["Age"] = testData["Age"].fillna(age_median)
testData["Fare"] = testData["Fare"].fillna(fare_median)

X_test = testData[features]

# Important: transform only, do not fit again
X_test_scaled = scaler.transform(X_test)

# Make predictions
test_predictions = finalModel.predict(X_test_scaled)

# -----------------------------
# Create Kaggle submission
# -----------------------------
submission = pd.DataFrame({
    "PassengerId": testData["PassengerId"],
    "Survived": test_predictions
})

submission.to_csv(
    "titanic_submission.csv",
    index=False
)

print(submission.head())
print("Submission file created.")