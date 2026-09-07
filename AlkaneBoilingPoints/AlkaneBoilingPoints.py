import matplotlib.pyplot as plt

# Number of carbon atoms in the first 10 straight-chain alkanes
carbons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Corresponding boiling points in degrees Celsius
boiling_points = [-161.5, -88.6, -42.1, -0.5, 36.1,
                  68.7, 98.4, 125.6, 150.8, 174.1]

# Create scatterplot
plt.scatter(carbons, boiling_points)

# Add title and axis labels
plt.title("Boiling Point vs. Number of Carbon Atoms")
plt.xlabel("Number of Carbon Atoms")
plt.ylabel("Boiling Point (°C)")

# Display the graph
plt.savefig("AlkaneBoilingPoints.png")
plt.show()