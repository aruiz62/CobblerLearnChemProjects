
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# =====================================
# STEP 1: LOAD DATASET
# =====================================

folder = Path(__file__).resolve().parent
output = folder / "Random_Forest_Results"
output.mkdir(exist_ok=True)

df = pd.read_csv(
    folder / "AR_BLA_Final_ML_Dataset.csv",
    low_memory=False
)

print("ORIGINAL DATASET:", df.shape)

# =====================================
# STEP 2: SELECT MOLECULAR FEATURES
# =====================================

fingerprints = [
    f"FP_{i}" for i in range(2048)
]

descriptors = [
    "Molecular_Weight",
    "LogP",
    "TPSA",
    "H_Bond_Donors",
    "H_Bond_Acceptors",
    "Rotatable_Bonds",
    "Ring_Count",
    "Heavy_Atom_Count"
]

features = fingerprints + descriptors

missing = [c for c in features if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing feature columns: {missing[:20]}"
    )

X = df[features].apply(
    pd.to_numeric, errors="coerce"
)

# =====================================
# STEP 3: CREATE ACTIVE/INACTIVE LABELS
# =====================================

labels = (
    df["hit_call_antagonist"]
    .astype("string")
    .str.strip()
    .str.lower()
)

y = labels.map({
    "active": 1,
    "inactive": 0
})

valid = y.notna()

X = X.loc[valid].copy()
y = y.loc[valid].astype(int)

print("\nCLASS DISTRIBUTION:")
print(y.value_counts())

# =====================================
# STEP 4: CHECK MISSING VALUES
# =====================================

print("\nMISSING MOLECULAR VALUES:")
print(X.isna().sum().sum())

# Remove infinite values
X = X.replace([np.inf, -np.inf], np.nan)

# =====================================
# STEP 5: CHECK DUPLICATE STRUCTURES
# =====================================

# Identical molecular features should not
# appear in both training and testing.

feature_hash = pd.util.hash_pandas_object(
    X.fillna(-999999),
    index=False
)

groups = feature_hash.astype(str)

conflicts = pd.DataFrame({
    "group": groups.to_numpy(),
    "label": y.to_numpy()
}).groupby("group")["label"].nunique()

conflicting_groups = set(
    conflicts[conflicts > 1].index
)

print("\nUnique feature groups:", groups.nunique())
print("Conflicting label groups:", len(conflicting_groups))

# Exclude ambiguous structures for this baseline.
keep = ~groups.isin(conflicting_groups)

X = X.loc[keep].copy()
y = y.loc[keep].copy()
groups = groups.loc[keep].copy()

# =====================================
# STEP 6: TRAIN/TEST SPLIT
# =====================================

# Split unique structures first.
unique_groups = pd.DataFrame({
    "group": groups.to_numpy(),
    "label": y.to_numpy()
}).drop_duplicates("group")

train_groups, test_groups = train_test_split(
    unique_groups["group"],
    test_size=0.20,
    random_state=42,
    stratify=unique_groups["label"]
)

train_mask = groups.isin(set(train_groups))
test_mask = groups.isin(set(test_groups))

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

print("\nTRAINING CHEMICALS:", len(X_train))
print("TESTING CHEMICALS:", len(X_test))

# =====================================
# STEP 7: IMPUTE MISSING VALUES
# =====================================

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(
    strategy="median",
    keep_empty_features=True
)

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# =====================================
# STEP 8: TRAIN RANDOM FOREST
# =====================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =====================================
# STEP 9: MAKE PREDICTIONS
# =====================================

predictions = model.predict(X_test)

probabilities = model.predict_proba(
    X_test
)[:, 1]

# =====================================
# STEP 10: EVALUATE MODEL
# =====================================

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(
    y_test, predictions, zero_division=0
)
recall = recall_score(
    y_test, predictions, zero_division=0
)
f1 = f1_score(
    y_test, predictions, zero_division=0
)
auc = roc_auc_score(
    y_test, probabilities
)

print("\nRANDOM FOREST RESULTS")
print("----------------------------")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC-AUC:", round(auc, 4))

print("\nCLASSIFICATION REPORT:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Inactive", "Active"],
    zero_division=0
))

# =====================================
# STEP 11: SAVE MODEL METRICS
# =====================================

metrics = pd.DataFrame([{
    "Model": "Random Forest",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1,
    "ROC_AUC": auc
}])

metrics.to_csv(
    output / "Random_Forest_Metrics.csv",
    index=False
)

# =====================================
# STEP 12: CONFUSION MATRIX
# =====================================

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    display_labels=["Inactive", "Active"],
    cmap="Blues"
)

plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()
plt.savefig(
    output / "Confusion_Matrix.png",
    dpi=300
)
plt.close()

# =====================================
# STEP 13: ROC CURVE
# =====================================

RocCurveDisplay.from_predictions(
    y_test,
    probabilities
)

plt.title("Random Forest - ROC Curve")
plt.tight_layout()
plt.savefig(
    output / "ROC_Curve.png",
    dpi=300
)
plt.close()

# =====================================
# STEP 14: FEATURE IMPORTANCE
# =====================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance", ascending=False
)

importance.to_csv(
    output / "Feature_Importance.csv",
    index=False
)

top = importance.head(20)

plt.figure(figsize=(10, 7))
plt.barh(
    top["Feature"][::-1],
    top["Importance"][::-1]
)

plt.xlabel("Feature Importance")
plt.title("Top 20 Molecular Features")
plt.tight_layout()
plt.savefig(
    output / "Feature_Importance.png",
    dpi=300
)
plt.close()

# =====================================
# STEP 15: SAVE PREDICTIONS
# =====================================

results = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": predictions,
    "Probability_Active": probabilities
})

results.to_csv(
    output / "Random_Forest_Predictions.csv",
    index=False
)

print("\nALL RESULTS SAVED!")
print("Folder:", output)
# ==========================================
# STEP 16: SAVE ROC CURVE
# ==========================================

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, probabilities)

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test, probabilities)

# Create ROC graph
plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    color="blue",
    linewidth=2,
    label=f"Random Forest (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    color="red",
    linestyle="--",
    label="Random Guessing"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Random Forest ROC Curve")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

# Save graph
plt.savefig(
    output / "ROC_Curve.png",
    dpi=300,
    bbox_inches="tight"
)

# Close without pausing
plt.close()

print("ROC curve saved successfully!")