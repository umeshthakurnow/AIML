import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the iris dataset
iris = pd.read_csv(r"C:\Users\uasin\Downloads\iris_dataset.csv")

# Display basic information about the dataset
print("Dataset shape:", iris.shape)
print("\nFirst few rows:")
print(iris.head())
print("\nDataset Info:")
print(iris.info())
print("\nDataset Statistics:")
print(iris.describe())

# Separate features (X) and target (y)
# Features: all columns except the last one (sepal-length, sepal-width, petal-length, petal-width)
# Target: the last column (class - the iris species)
feature_columns = iris.columns[:-1]  # All columns except the last
target_column = iris.columns[-1]      # The last column (target)

X = iris[feature_columns]  # Select feature columns
y = iris[target_column]    # Select target column

print("\nFeature columns:", X.columns.tolist())
print("Target column:", target_column)
print("\nTarget classes:", y.unique())

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Standardize the features (important for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

# Evaluate the model
print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Logistic Regression')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
print("Confusion matrix visualization saved as 'confusion_matrix.png'")

# Model coefficients
print("\n" + "="*50)
print("MODEL COEFFICIENTS")
print("="*50)
for i, col in enumerate(X.columns):
    print(f"{col}: {model.coef_[0][i]:.4f}")

print(f"Intercept: {model.intercept_[0]:.4f}")

# Summary statistics
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Total samples: {len(iris)}")
print(f"Features used: {X.shape[1]}")
print(f"Number of classes: {len(y.unique())}")
print(f"Train-test split: 80-20")
print(f"Model Accuracy: {accuracy:.4f}")