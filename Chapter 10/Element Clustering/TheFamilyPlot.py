import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Create Group 1 data
data = {
    "Element": ["Lithium", "Sodium", "Potassium",
                "Rubidium", "Cesium", "Francium"],

    "Symbol": ["Li", "Na", "K", "Rb", "Cs", "Fr"],

    "Atomic Number": [3, 11, 19, 37, 55, 87],

    "Atomic Radius": [167, 190, 243, 265, 298, 348],

    "Ionization Energy": [520.2, 495.8, 418.8,
                          403.0, 375.7, 380.0]
}

df = pd.DataFrame(data)

# Save CSV
df.to_csv("group1_elements.csv", index=False)

# Preview data
print("Group 1 Element Data:")
print(df)

# Select atomic radius and ionization energy
X = df[["Atomic Radius", "Ionization Energy"]]

# Keep running until user chooses to stop
while True:

    k = int(input("\nEnter a k value: "))

    # K-Means clustering
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    df["Cluster"] = kmeans.fit_predict(X)

    centers = kmeans.cluster_centers_

    # Create plot
    plt.figure(figsize=(9, 6))

    plt.scatter(
        df["Atomic Radius"],
        df["Ionization Energy"],
        c=df["Cluster"],
        cmap="viridis",
        s=100
    )

    # Label elements
    for i in range(len(df)):
        plt.annotate(
            df["Symbol"].iloc[i],
            (df["Atomic Radius"].iloc[i],
             df["Ionization Energy"].iloc[i]),
            xytext=(5, 5),
            textcoords="offset points"
        )

    # Cluster centers as large black Xs
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        c="black",
        marker="X",
        s=300,
        label="Cluster Centers"
    )

    plt.xlabel("Atomic Radius (pm)")
    plt.ylabel("First Ionization Energy (kJ/mol)")
    plt.title(f"Group 1 K-Means Clustering (k = {k})")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig(f"group1_kmeans_k{k}.png", dpi=300)

    plt.show()

    # Show clusters
    print("\nCluster Assignments:")
    print(df[["Element", "Symbol", "Cluster"]])

    # Ask if user wants another k
    again = input("\nTry another k value? (yes/no): ")

    if again.lower() != "yes":
        print("Finished!")
        break
