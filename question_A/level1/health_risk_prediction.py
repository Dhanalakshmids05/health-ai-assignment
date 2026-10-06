import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


# ============================================================
# 1. PERSONAL SEED
# ============================================================

SEED = 3014


# ============================================================
# 2. LOAD DATASET
# ============================================================

heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features.copy()
y = heart_disease.data.targets.copy()


# ============================================================
# 3. CONVERT TARGET TO BINARY
# ============================================================

# 0 = no disease
# 1, 2, 3, 4 = disease/risk

y = (y.iloc[:, 0] > 0).astype(int)


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

# Use the most frequent value for missing values.
# The imputer is fitted ONLY on training data.

imputer = SimpleImputer(strategy="most_frequent")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


# ============================================================
# 6. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ============================================================
# 7. DISPLAY INFORMATION
# ============================================================

print("========== DATA CLEANING COMPLETE ==========")

print("\nSeed:", SEED)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

print("\nNumber of features:", X_train.shape[1])

print("\nTarget distribution:")
print(y.value_counts())

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
# ============================================================
# 8. LOGISTIC REGRESSION
# ============================================================

logistic_model = LogisticRegression(
    random_state=SEED,
    max_iter=1000
)

logistic_model.fit(X_train, y_train)

logistic_pred = logistic_model.predict(X_test)
# ============================================================
# 9. RANDOM FOREST
# ============================================================

random_forest_model = RandomForestClassifier(
    random_state=SEED
)

random_forest_model.fit(X_train, y_train)

random_forest_pred = random_forest_model.predict(X_test)
# ============================================================
# 10. MODEL EVALUATION
# ============================================================

print("\n========== MODEL RESULTS ==========")

# Logistic Regression
logistic_accuracy = accuracy_score(y_test, logistic_pred)
logistic_precision = precision_score(y_test, logistic_pred)
logistic_recall = recall_score(y_test, logistic_pred)

print("\nLogistic Regression:")
print("Accuracy :", logistic_accuracy)
print("Precision:", logistic_precision)
print("Recall   :", logistic_recall)


# Random Forest
random_forest_accuracy = accuracy_score(y_test, random_forest_pred)
random_forest_precision = precision_score(y_test, random_forest_pred)
random_forest_recall = recall_score(y_test, random_forest_pred)

print("\nRandom Forest:")
print("Accuracy :", random_forest_accuracy)
print("Precision:", random_forest_precision)
print("Recall   :", random_forest_recall)