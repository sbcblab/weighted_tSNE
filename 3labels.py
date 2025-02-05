# Bruno Iochins Grisci
# February 2nd, 2025

from sklearn.datasets import make_classification
import pandas as pd

# Set the parameters for the dataset
n_samples = 1000  # Number of samples
n_features = 100  # Total number of features
n_informative = 10  # Number of informative (relevant) features
n_classes = 3  # Number of distinct labels
n_clusters_per_class = 1  # Number of clusters per class

# Generate the synthetic dataset
X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative,
    n_redundant=0,  # No redundant features
    n_repeated=0,  # No repeated features
    n_classes=n_classes,
    n_clusters_per_class=n_clusters_per_class,
    random_state=42  # Set a random seed for reproducibility
)

# Create feature names
informative_features = [f"rel{i+1}" for i in range(n_informative)]  # Names for informative features
irrelevant_features = [f"irrel{i+1}" for i in range(n_features - n_informative)]  # Names for irrelevant features
feature_names = informative_features + irrelevant_features  # Combine the two lists

# Convert the dataset into a pandas DataFrame
df = pd.DataFrame(X, columns=feature_names)

# Map numeric targets to string labels
target_map = {0: "A", 1: "B", 2: "C"}
df["target"] = pd.Series(y).map(target_map)

# Save the original dataset to a .csv file
output_file_original = "DATA/synthetic_3ABC.csv"
df.to_csv(output_file_original, index=False)
print(f"Original dataset saved to {output_file_original}")

# Create a copy of the dataset for modification
df_modified = df.copy()

# Replace "A" and "B" with "AB" in the target column
df_modified["target"] = df_modified["target"].replace({"A": "AB", "B": "AB"})

# Save the modified dataset to a .csv file
output_file_modified = "DATA/synthetic_2ABC.csv"
df_modified.to_csv(output_file_modified, index=False)
print(f"Modified dataset saved to {output_file_modified}")