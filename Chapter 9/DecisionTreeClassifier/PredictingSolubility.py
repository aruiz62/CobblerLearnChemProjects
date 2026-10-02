import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree


# --------------------------------
# 1. Create the dataset
# --------------------------------

data = {
    "Molecule": [
        "Molecule 1", "Molecule 2", "Molecule 3", "Molecule 4",
        "Molecule 5", "Molecule 6", "Molecule 7", "Molecule 8",
        "Molecule 9", "Molecule 10", "Molecule 11", "Molecule 12"
    ],

    "Molecular Weight": [
        180, 250, 80, 300, 150, 400,
        90, 200, 130, 275, 135, 220
    ],

    "Hydrogen Bond Donors": [
        5, 2, 1, 1, 4, 3,
        0, 2, 3, 1, 1, 3
    ],

    "Hydrogen Bond Acceptors": [
        6, 3, 2, 2, 5, 4,
        1, 3, 4, 2, 3, 2
    ],

    "Water Solubility": [
        1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 0, 1
    ]
}

df = pd.DataFrame(data)

print("Chemical Dataset:")
print(df.to_string(index=False))


# --------------------------------
# 2. Separate features and target
# --------------------------------

features = [
    "Molecular Weight",
    "Hydrogen Bond Donors",
    "Hydrogen Bond Acceptors"
]

X = df[features]
y = df["Water Solubility"]


# --------------------------------
# 3. Create and train decision tree
# --------------------------------

model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X, y)

print("\nDecision tree successfully trained!")


# --------------------------------
# 4. Display the decision tree
# --------------------------------

plt.rcParams["font.family"] = "Times New Roman"

plt.figure(figsize=(14, 8))

tree_plot = plot_tree(
    model,
    feature_names=features,
    class_names=["Not Soluble", "Soluble"],
    filled=True,
    rounded=True,
    fontsize=11
)


# Change white boxes to yellow
for box in tree_plot:

    patch = box.get_bbox_patch()

    if patch is not None:

        red, green, blue, alpha = patch.get_facecolor()

        # If the box is white or almost white,
        # change it to yellow
        if red > 0.90 and green > 0.90 and blue > 0.90:
            patch.set_facecolor("yellow")


plt.title(
    "Decision Tree for Water Solubility",
    fontsize=16,
    fontname="Times New Roman"
)

plt.tight_layout()


# --------------------------------
# 5. Save the image
# --------------------------------

plt.savefig(
    "decision_tree_solubility.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nTree image saved as decision_tree_solubility.png")