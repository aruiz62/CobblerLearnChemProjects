
import numpy as np
from scipy.signal import convolve2d

# Step 2: Create a 5x5 image of a plus sign
image = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0]
])

print("Original Image:")
print(image)
print("Image Shape:", image.shape)

# Step 3: Create a horizontal edge filter
filter_kernel = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
])

print("\nFilter:")
print(filter_kernel)

# Step 4: Apply the filter
feature_map = convolve2d(image, filter_kernel, mode="valid")

print("\nFeature Map:")
print(feature_map)

# Step 5: Analyze the results
print("\nHighest Value:", feature_map.max())
print("Lowest Value:", feature_map.min())

print("Highest Value Locations:",
      np.argwhere(feature_map == feature_map.max()))

print("Lowest Value Locations:",
      np.argwhere(feature_map == feature_map.min()))


# Step 6: Visualize the image and feature map

import matplotlib.pyplot as plt
from pathlib import Path

output_folder = Path(__file__).resolve().parent

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(feature_map, cmap="coolwarm")
plt.title("Feature Map")
plt.colorbar()

plt.tight_layout()
plt.savefig(output_folder / "convolution_comparison.png", dpi=300)
plt.show()

print("Graph saved successfully!")
