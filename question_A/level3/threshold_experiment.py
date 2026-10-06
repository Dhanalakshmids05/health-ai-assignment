import numpy as np

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


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
# 1, 2, 3, 4 = disease/risk

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
X_test = imputer.transform(X_test)


# ============================================================
# 5. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# ============================================================
# 6. SIGMOID
# ============================================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ============================================================
# 7. BINARY CROSS-ENTROPY
# ============================================================

def binary_cross_entropy(y_true, y_prob):

    epsilon = 1e-15

    y_prob = np.clip(
        y_prob,
        epsilon,
        1 - epsilon
    )

    loss = -np.mean(
        y_true * np.log(y_prob)
        + (1 - y_true) * np.log(1 - y_prob)
    )

    return loss


# ============================================================
# 8. CALCULATE GRADIENTS
# ============================================================

def calculate_gradients(
    X,
    y,
    probabilities
):

    n = len(y)

    error = probabilities - y

    dw = (X.T @ error) / n

    db = np.mean(error)

    return dw, db


# ============================================================
# 9. TRAIN LOGISTIC REGRESSION FROM SCRATCH
# ============================================================

def train_logistic_regression(
    X,
    y,
    learning_rate=0.01,
    epochs=2000,
    seed=3014
):

    rng = np.random.default_rng(seed)

    weights = rng.normal(
        loc=0.0,
        scale=0.01,
        size=X.shape[1]
    )

    bias = 0.0

    for epoch in range(epochs):

        z = X @ weights + bias

        probabilities = sigmoid(z)

        loss = binary_cross_entropy(
            y,
            probabilities
        )

        dw, db = calculate_gradients(
            X,
            y,
            probabilities
        )

        weights -= learning_rate * dw

        bias -= learning_rate * db

    return weights, bias


# ============================================================
# 10. TRAIN MODEL
# ============================================================

weights, bias = train_logistic_regression(
    X_train,
    y_train,
    learning_rate=0.01,
    epochs=2000,
    seed=SEED
)


# ============================================================
# 11. GET PROBABILITIES
# ============================================================

test_probabilities = sigmoid(
    X_test @ weights + bias
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

def confusion_matrix_manual(
    y_true,
    y_pred
):

    true_negative = 0
    false_positive = 0
    false_negative = 0
    true_positive = 0

    for actual, predicted in zip(
        y_true,
        y_pred
    ):

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
# 13. CALCULATE METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    tn, fp, fn, tp = confusion_matrix_manual(
        y_true,
        y_pred
    )

    accuracy = (
        (tp + tn)
        / (tp + tn + fp + fn)
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    return accuracy, precision, recall


# ============================================================
# 14. BASELINE AT THRESHOLD 0.50
# ============================================================

baseline_predictions = (
    test_probabilities >= 0.50
).astype(int)

baseline_accuracy, baseline_precision, baseline_recall = (
    calculate_metrics(
        y_test,
        baseline_predictions
    )
)


print("========== BASELINE ==========")

print("Threshold:", 0.50)
print("Accuracy:", baseline_accuracy)
print("Precision:", baseline_precision)
print("Recall:", baseline_recall)


# ============================================================
# 15. FIND THRESHOLD FOR RECALL >= 0.90
# ============================================================

best_threshold = None
best_precision = -1
best_accuracy = None
best_recall = None


thresholds = np.arange(
    0.50,
    0.00,
    -0.01
)


for threshold in thresholds:

    predictions = (
        test_probabilities >= threshold
    ).astype(int)

    accuracy, precision, recall = calculate_metrics(
        y_test,
        predictions
    )

    if recall >= 0.90:

        if precision > best_precision:

            best_threshold = threshold
            best_accuracy = accuracy
            best_precision = precision
            best_recall = recall


# ============================================================
# 16. DISPLAY LEVEL 3 RESULT
# ============================================================

print("\n========== LEVEL 3 RESULT ==========")

print(
    "Selected threshold:",
    round(best_threshold, 2)
)

print(
    "Accuracy:",
    best_accuracy
)

print(
    "Precision:",
    best_precision
)

print(
    "Recall:",
    best_recall
)


# ============================================================
# 17. COMPARE WITH BASELINE
# ============================================================

print("\n========== COMPARISON ==========")

print(
    "Precision change:",
    best_precision - baseline_precision
)

print(
    "Recall change:",
    best_recall - baseline_recall
)