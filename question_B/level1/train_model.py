import joblib

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


SEED = 3014


# ============================================================
# 1. LOAD DATASET
# ============================================================

heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features.copy()
y = heart_disease.data.targets.copy()


# ============================================================
# 2. CONVERT TARGET TO BINARY
# ============================================================

# 0 = no disease
# 1,2,3,4 = disease/risk

y = (y.iloc[:, 0] > 0).astype(int)


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)


# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================

imputer = SimpleImputer(
    strategy="most_frequent"
)

X_train = imputer.fit_transform(X_train)


# ============================================================
# 5. SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)


# ============================================================
# 6. TRAIN LOGISTIC REGRESSION
# ============================================================

model = LogisticRegression(
    random_state=SEED,
    max_iter=1000
)

model.fit(X_train, y_train)


# ============================================================
# 7. SAVE MODEL + PREPROCESSING
# ============================================================

model_data = {
    "model": model,
    "imputer": imputer,
    "scaler": scaler,
    "feature_names": X.columns.tolist(),
    "seed": SEED
}

joblib.dump(
    model_data,
    "question_B/level1/health_risk_model.joblib"
)

print("Model saved successfully.")
print("Features:", X.columns.tolist())
print("Seed:", SEED)