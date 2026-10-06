import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

SEED = 3014

print("Logistic Regression from Scratch")
print("Seed:", SEED)
# ============================================================
# LOAD DATASET
# ============================================================

heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features.copy()
y = heart_disease.data.targets.copy()

# Convert target to binary
# 0 = no disease
# 1, 2, 3, 4 = disease/risk
y = (y.iloc[:, 0] > 0).astype(int)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

imputer = SimpleImputer(strategy="most_frequent")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


# ============================================================
# SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Convert targets to NumPy arrays
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
print("\nSigmoid test:")
print(sigmoid(np.array([-2, 0, 2])))

def binary_cross_entropy(y_true, y_prob):
    epsilon = 1e-15

    y_prob = np.clip(y_prob, epsilon, 1 - epsilon)

    loss = -np.mean(
        y_true * np.log(y_prob)
        + (1 - y_true) * np.log(1 - y_prob)
    )

    return loss

def calculate_gradients(X, y, probabilities):
    n = len(y)

    error = probabilities - y

    dw = (X.T @ error) / n
    db = np.mean(error)

    return dw, db

def train_logistic_regression(
    X,
    y,
    learning_rate=0.01,
    epochs=2000,
    seed=3014
):
    rng = np.random.default_rng(seed)

    # Small random initial weights
    weights = rng.normal(
        loc=0.0,
        scale=0.01,
        size=X.shape[1]
    )

    bias = 0.0

    loss_history = []

    for epoch in range(epochs):

        # Forward pass
        z = X @ weights + bias

        probabilities = sigmoid(z)

        # Calculate loss
        loss = binary_cross_entropy(
            y,
            probabilities
        )

        # Calculate gradients
        dw, db = calculate_gradients(
            X,
            y,
            probabilities
        )

        # Update weights
        weights -= learning_rate * dw
        bias -= learning_rate * db

        loss_history.append(loss)

    return weights, bias, loss_history

def predict(X, weights, bias, threshold=0.5):

    probabilities = sigmoid(
        X @ weights + bias
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return predictions, probabilities

# ============================================================
# TRAIN OUR LOGISTIC REGRESSION
# ============================================================

weights, bias, loss_history = train_logistic_regression(
    X_train,
    y_train,
    learning_rate=0.01,
    epochs=2000,
    seed=SEED
)

print("\nTraining completed.")

print("Final loss:", loss_history[-1])
print("Bias:", bias)

print("\nWeights:")
print(weights)

# ============================================================
# OWN CONFUSION MATRIX
# ============================================================

def confusion_matrix_manual(y_true, y_pred):

    true_negative = 0
    false_positive = 0
    false_negative = 0
    true_positive = 0

    for actual, predicted in zip(y_true, y_pred):

        if actual == 0 and predicted == 0:
            true_negative += 1

        elif actual == 0 and predicted == 1:
            false_positive += 1

        elif actual == 1 and predicted == 0:
            false_negative += 1

        elif actual == 1 and predicted == 1:
            true_positive += 1

    return (
        true_negative,
        false_positive,
        false_negative,
        true_positive
    )
    
# ============================================================
# MAKE TEST PREDICTIONS
# ============================================================

test_predictions, test_probabilities = predict(
    X_test,
    weights,
    bias,
    threshold=0.5
)

print("\nFirst 10 predicted probabilities:")
print(test_probabilities[:10])

print("\nFirst 10 predictions:")
print(test_predictions[:10])

print("\nFirst 10 actual values:")
print(y_test[:10])

# ============================================================
# CALCULATE METRICS FROM OUR CONFUSION MATRIX
# ============================================================

tn, fp, fn, tp = confusion_matrix_manual(
    y_test,
    test_predictions
)

accuracy = (tp + tn) / (tp + tn + fp + fn)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0

recall = tp / (tp + fn) if (tp + fn) > 0 else 0


print("\n========== SCRATCH MODEL RESULTS ==========")

print("True Negative :", tn)
print("False Positive:", fp)
print("False Negative:", fn)
print("True Positive :", tp)

print("\nAccuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)

# ============================================================
# TOP 3 FEATURE WEIGHTS
# ============================================================

feature_names = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]

weight_information = list(
    zip(feature_names, weights)
)

top_features = sorted(
    weight_information,
    key=lambda item: abs(item[1]),
    reverse=True
)[:3]

print("\n========== TOP 3 FEATURE WEIGHTS ==========")

for rank, (feature, weight) in enumerate(
    top_features,
    start=1
):
    print(
        f"{rank}. {feature}: {weight:.6f}"
    )
    