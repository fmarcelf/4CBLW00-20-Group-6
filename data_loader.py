"""
Date: 30 April 2025
Author: Beloslava Malakova
Description: Data Loader + initial EDA
"""

import os
import pandas as pd
import time

def load_all_data(base_path="data-london", log_every=200):
    all_dfs = []
    file_count = 0
    start_time = time.time()

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path)
                    df['source_file'] = file
                    df['month_folder'] = os.path.basename(root)
                    all_dfs.append(df)
                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")
                file_count += 1
                if file_count % log_every == 0:
                    print(f"Processed {file_count} files...")

    total_time = time.time() - start_time
    print(f"Done! Total files: {file_count}, Total time: {total_time:.2f} seconds")

    return pd.concat(all_dfs, ignore_index=True)

# === Load and inspect the data ===
all_data = load_all_data()

# Drop fully-empty columns (like 'Context')
# all_data = all_data.dropna(axis=1, how='all')

# shape and columns
print("Shape of combined dataset:", all_data.shape)
print("Columns in the dataset:")
print(all_data.columns)

# missing values
print("Missing values per column:")
print(all_data.isnull().sum())

# basic statistics
print("Basic statistics:")
print(all_data.describe(include='all').transpose())

#  insights
print("Top 10 crime types:")
print(all_data['Crime type'].value_counts().head(10))

print("Top 5 files by number of records:")
print(all_data['source_file'].value_counts().head())

print("Time range in 'Month' column:")
print(all_data['Month'].min(), "→", all_data['Month'].max())

# Preview of the data
print("Preview of the data:")
print(all_data.head())
