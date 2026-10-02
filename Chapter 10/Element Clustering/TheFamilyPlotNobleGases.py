import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Create Noble Gas data
data = {
    "Element": [
        "Helium",
        "Neon",
        "Argon",
        "Krypton",
        "Xenon",
        "Radon"
    ],

    "Symbol": [
        "He",
        "Ne",
        "Ar",
        "Kr",
        "Xe",
        "Rn"
    ],

    "Atomic Number": [
        2, 10, 18, 36, 54, 86
    ],

    "Atomic Radius": [
        31, 38, 71, 88, 108, 120
    ],

    "Ionization Energy": [
        2372.3,
        2080.7,
        1520.6,
        1350.8,
        1170.4,
        1037.0
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save as CSV
df.to_csv("noble_gases.csv", index=False)

# Preview data
print("Noble Gas Data:")
print(df)

# Select features
X = df[["Atomic Radius", "Ionization Energy"]]

# Loop to test different k values
while True:

    k = int(input("\nEnter a k value: "))

    # K-Means
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    df["Cluster"] = kmeans.fit_predict(X)

    # Cluster centers
    centers = kmeans.cluster_centers_

    # Create scatter plot
    plt.figure(figsize=(9, 6))

    plt.scatter(
        df["Atomic Radius"],
        df["Ionization Energy"],
        c=df["Cluster"],
        cmap="viridis",
        s=100
    )

    # Label each noble gas
    for i in range(len(df)):
        plt.annotate(
            df["Symbol"].iloc[i],
            (
                df["Atomic Radius"].iloc[i],
                df["Ionization Energy"].iloc[i]
            ),
            xytext=(5, 5),
            textcoords="offset points"
        )

    # Add cluster centers as black Xs
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        c="black",
        marker="X",
        s=300,
        label="Cluster Centers"
    )

    # Labels
    plt.xlabel("Atomic Radius (pm)")
    plt.ylabel("First Ionization Energy (kJ/mol)")
    plt.title(f"Noble Gas K-Means Clustering (k = {k})")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save graph
    plt.savefig(f"noble_gases_k{k}.png", dpi=300)

    plt.show()

    # Show cluster assignments
    print("\nCluster Assignments:")
    print(df[["Element", "Symbol", "Cluster"]])

    # Ask whether to try another k
    again = input("\nTry another k value? (yes/no): ")

    if again.lower() != "yes":
        print("Finished!")
        break