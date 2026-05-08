import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================
# Load Dataset
# =========================
df = pd.read_csv("data.csv")

print("Dataset Loaded Successfully\n")
print(df.head())

# =========================
# Data Cleaning
# =========================

# Remove customerID
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors='coerce'
)

# Fill missing values
df["TotalCharges"].fillna(df["TotalCharges"].mean(), inplace=True)

# =========================
# Encode Categorical Data
# =========================
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# =========================
# Split Dataset
# =========================
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Models
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

accuracies = {}

# =========================
# Training & Evaluation
# =========================
for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    accuracies[name] = accuracy

    print(f"\n===== {name} =====")

    print("Accuracy:", accuracy)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# =========================
# Accuracy Comparison Graph
# =========================
plt.figure(figsize=(8, 5))

sns.barplot(
    x=list(accuracies.keys()),
    y=list(accuracies.values())
)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1.0)

plt.show()

# =========================
# Churn Distribution
# =========================
plt.figure(figsize=(6, 4))

sns.countplot(x='Churn', data=df)

plt.title("Customer Churn Distribution")

plt.show()

# =========================
# Feature Importance
# =========================
rf_model = RandomForestClassifier()

rf_model.fit(X_train, y_train)

importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance
)

plt.title("Feature Importance")

plt.show()