import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Define sample data for Persons and Households
persons_data = {
    "sex": ["F", "F", "F", "F", "F", "F", "F", "F", "M"],
    "age": ["18_19", "25_29", "65_69", "35_39", "25_29", "35_39", "80_84", "45_49", "40_44"],
    "ethnicity": ["W", "W", "B", "W", "W", "A", "W", "W", "W"],
    "religion": ["C", "NS", "N", "N", "C", "M", "C", "C", "C"],
    "marital": ["Single", "Married", "Separated", "Divorced", "Single", "Single", "Married", "Married", "Married"],
    "qualification": ["level3", "level4+", "no", "other", "other", "level4+", "no", "level1", "level4+"],
    "Person_ID": list(range(1, 10))
}

households_data = {
    "religion": ["M", "M", "M", "O", "O", "A", "M", "M", "M"],
    "ethnicity": ["N", "H", "J", "B", "B", "B", "S", "H", "B"],
    "hh_comp": ["1FL-nC", "1FL-nC", "1FL-nC", "1FM-nA", "1FM-nA", "1H-nA", "1PA", "1FL-nC", "1PA"],
    "hh_size": [1, 1, 1, 1, 1, 4, 1, 1, 1],
    "hh_ID": list(range(1, 10))
}

# Create DataFrames
persons_df = pd.DataFrame(persons_data)
households_df = pd.DataFrame(households_data)

# Function to encode categorical data safely for both datasets
def safe_encode(column):
    # Combine the data from both datasets for this column
    combined = pd.concat([persons_df[column], households_df[column]], ignore_index=True)
    encoder = LabelEncoder().fit(combined)
    persons_df[column] = encoder.transform(persons_df[column])
    households_df[column] = encoder.transform(households_df[column])

# Apply encoding to overlapping categorical columns
for column in ['religion', 'ethnicity']:
    safe_encode(column)

# Encode other categorical data separately for each dataset
for column in ['sex', 'marital', 'qualification', 'hh_comp']:
    if column in persons_df:
        persons_df[column] = LabelEncoder().fit_transform(persons_df[column])
    if column in households_df:
        households_df[column] = LabelEncoder().fit_transform(households_df[column])

# Convert age to the midpoint of range
persons_df['age'] = persons_df['age'].apply(lambda x: (int(x.split('_')[0]) + int(x.split('_')[-1])) / 2)

# Perform hierarchical clustering
linked = linkage(persons_df[['sex', 'age', 'ethnicity', 'religion', 'marital', 'qualification']], method='ward')

# Plotting the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top',
           labels=persons_df['Person_ID'].values,
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram of Persons')
plt.xlabel('Person ID')
plt.ylabel('Distance')
plt.show()
