"""
Basic Classification Model
----------------------------
Demonstrates the core supervised learning workflow:
1. Load and understand a dataset
2. Split data into training and testing sets
3. Apply a simple classification algorithm
4. Evaluate the model

Dataset: Iris flower dataset (150 samples, 4 features, 3 classes)
Algorithm: Decision Tree Classifier
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ------------------------- 1. Load and understand the dataset ------------------------- #

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("=" * 60)
print("STEP 1: Dataset Overview")
print("=" * 60)
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nClass distribution:\n{df['species'].value_counts()}")
print(f"\nSummary statistics:\n{df.describe()}")

# Features (X) and target labels (y)
X = df[iris.feature_names]
y = df["species"]


# ------------------------- 2. Split into training and testing sets ------------------------- #

print("\n" + "=" * 60)
print("STEP 2: Train/Test Split")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")


# ------------------------- 3. Apply a simple classification algorithm ------------------------- #

print("\n" + "=" * 60)
print("STEP 3: Train the Model (Decision Tree)")
print("=" * 60)

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)
print("Model trained successfully.")

# Show which features mattered most
importances = pd.Series(model.feature_importances_, index=iris.feature_names)
print(f"\nFeature importances:\n{importances.sort_values(ascending=False)}")


# ------------------------- 4. Evaluate the model ------------------------- #

print("\n" + "=" * 60)
print("STEP 4: Evaluate on Test Set")
print("=" * 60)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2%}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")


# ------------------------- 5. Try a prediction on a new sample ------------------------- #

print("=" * 60)
print("STEP 5: Predict a New Sample")
print("=" * 60)

sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=iris.feature_names)  # looks like a Setosa
prediction = model.predict(sample)
print(f"Sample measurements: {sample.values[0]}")
print(f"Predicted species: {prediction[0]}")
