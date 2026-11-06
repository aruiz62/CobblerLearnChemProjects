
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Step 1: Load our dataset
folder = Path(__file__).resolve().parent
df = pd.read_csv(folder / "Androgen_ML_Features.csv")

print("Total chemicals:", len(df))

# Step 2: Select molecular features
# Exclude chemical IDs and assay activity labels
X = df.drop(columns=[
    "DTXSID",
    "BLA_Activity",
    "Viability_Activity"
])

# Convert BLA activity into binary labels
# Active = 1, Inactive = 0
y = df["BLA_Activity"].map({
    "Active": 1,
    "Inactive": 0
})

print("Molecular features:", X.shape[1])
print("Active chemicals:", (y == 1).sum())
print("Inactive chemicals:", (y == 0).sum())

# Step 3: Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining chemicals:", len(X_train))
print("Testing chemicals:", len(X_test))

# Step 4: Create Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

# Step 5: Train the model
print("\nTraining Random Forest...")
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Step 7: Evaluate the results
print("\nRandom Forest Results")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Inactive", "Active"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from pathlib import Path

# Create confusion matrix visualization
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Inactive", "Active"],
    cmap="Blues"
)

plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()

# Save inside the same folder as the Python script
folder = Path(__file__).resolve().parent
plt.savefig(folder / "Random_Forest_Confusion_Matrix.png", dpi=300)

plt.show()
