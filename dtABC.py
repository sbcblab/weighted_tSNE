# Bruno Iochins Grisci
# February 2nd, 2025

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

data_file = "DATA/synthetic_dif_2ABC.csv"

seed = 20
depth = None
samples_split = 2


# Load the dataset
df = pd.read_csv(data_file)

# Separate features and target
X = df.drop(columns=["target"])  # Features
y = df["target"]  # Target labels

# Split the dataset into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier(class_weight='balanced', random_state=seed, max_depth=depth, min_samples_split=samples_split)  # Use RandomForestClassifier
clf.fit(X_train, y_train)

# Predict on the training set
y_train_pred = clf.predict(X_train)

# Predict on the test set
y_test_pred = clf.predict(X_test)

# Calculate the F1 score for the training set
f1_train = f1_score(y_train, y_train_pred, average="weighted")  # Use 'weighted' for multi-class F1 score

# Calculate the F1 score for the test set
f1_test = f1_score(y_test, y_test_pred, average="weighted")  # Use 'weighted' for multi-class F1 score

# Print the F1 scores
print(f"F1 Score (Training Set): {f1_train:.4f}")
print(f"F1 Score (Test Set): {f1_test:.4f}")

# Get feature importance values
feature_importance = clf.feature_importances_

# Create a DataFrame to display feature importance
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "weights": feature_importance
})

# Sort by importance (descending order)
sorted_importance_df = importance_df.sort_values(by="weights", ascending=False)

# Print the top 10 most important features
print("\nTop 10 Most Important Features:")
print(sorted_importance_df.head(10))

importance_df.to_csv(data_file.replace(".csv", f"_dt_{seed}_{depth}_{samples_split}.csv"), index=False)