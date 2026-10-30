
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

# STEP 1: Load the Iris dataset
iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

print("\nIRIS DATASET")
print(df.head())
print("\nTotal flowers:", len(df))
print("\nFeatures:", list(df.columns))

# STEP 2: Prepare data without species labels
X = df.copy()

# STEP 3: Train K-Means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)

print("\nCLUSTER COUNTS")
print(df["Cluster"].value_counts().sort_index())

# STEP 4: Scatter plot of predicted clusters
plt.figure(figsize=(9, 6))

for cluster in sorted(df["Cluster"].unique()):
    subset = df[df["Cluster"] == cluster]
    plt.scatter(
        subset["petal length (cm)"],
        subset["petal width (cm)"],
        label=f"Cluster {cluster}",
        alpha=0.7
    )

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("K-Means Clustering of Iris Flowers")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("iris_clusters.png", dpi=300)
plt.show()

# STEP 5: Compare with actual species
df["Species"] = [iris.target_names[i] for i in iris.target]

comparison = pd.crosstab(df["Cluster"], df["Species"])

print("\nCLUSTERS VS ACTUAL SPECIES")
print(comparison)

percentages = pd.crosstab(
    df["Cluster"],
    df["Species"],
    normalize="index"
) * 100

print("\nSPECIES PERCENTAGE IN EACH CLUSTER")
print(percentages.round(2))

score = adjusted_rand_score(iris.target, df["Cluster"])
print("\nAdjusted Rand Index:", round(score, 4))

# STEP 6: Plot actual species
plt.figure(figsize=(9, 6))

for species in iris.target_names:
    subset = df[df["Species"] == species]
    plt.scatter(
        subset["petal length (cm)"],
        subset["petal width (cm)"],
        label=species,
        alpha=0.7
    )

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Actual Iris Species")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("iris_actual_species.png", dpi=300)
plt.show()

# STEP 7: Enhancement - normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

normalized_model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Normalized Cluster"] = normalized_model.fit_predict(X_scaled)

normalized_score = adjusted_rand_score(
    iris.target,
    df["Normalized Cluster"]
)

print("\nNORMALIZED K-MEANS")
print("Adjusted Rand Index:", round(normalized_score, 4))

# STEP 8: Enhancement - compare different k values
k_values = range(2, 9)
inertias = []

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(9, 6))
plt.plot(list(k_values), inertias, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Iris Clustering")
plt.xticks(list(k_values))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("iris_elbow.png", dpi=300)
plt.show()

print("\nAnalysis completed!")
