import pandas as pd

# Load the dataset
df = pd.read_csv("alkane_dataset.csv")

# Count how many rows we started with
original_rows = len(df)

# Remove any row that contains a missing value
clean_df = df.dropna()

# Count how many rows remain
remaining_rows = len(clean_df)

# Calculate how many rows were removed
rows_removed = original_rows - remaining_rows

# Calculate the percentage of the dataset that was lost
percent_lost = (rows_removed / original_rows) * 100


# Show the results
print("Original number of rows:", original_rows)
print("Rows remaining:", remaining_rows)
print("Rows removed:", rows_removed)
print("Percentage of dataset lost:", round(percent_lost, 2), "%")