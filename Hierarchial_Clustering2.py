import pandas as pd
import numpy as np

# Set a seed for reproducibility
np.random.seed(0)

# Create random data for persons
age_groups = ['18_19', '25_29', '35_39', '45_49', '65_69', '80_84', '40_44']
marital_status = ['Single', 'Married', 'Cohabiting', 'Divorced', 'Separated']
children_status = [0, 1, 2, 3]  # Number of children

persons = pd.DataFrame({
    'Person_ID': range(1, 101),
    'Age': np.random.choice(age_groups, 100, p=[0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1]),
    'Marital': np.random.choice(marital_status, 100),
    'Children': np.random.choice(children_status, 100, p=[0.5, 0.2, 0.2, 0.1])
})

# Create random data for households
composition_types = [
    '1PE', '1PA', '1FM-0C', '1FC-0C', '1FE', '1FM-nC', '1FC-nC', '1FL-nC',
    '1FM-nA', '1FC-nA', '1FL-nA', '1H-nC', '1H-nS', '1H-nE', '1H-nA'
]

households = pd.DataFrame({
    'Household_ID': range(1, 31),
    'Composition': np.random.choice(composition_types, 30),
    'Slots': np.random.choice([1, 2, 3, 4], 30)  # Assuming households can have 1-4 slots
})

# Convert ages to midpoint for numerical processing
age_to_midpoint = {
    '18_19': 18.5, '25_29': 27, '35_39': 37, '45_49': 47,
    '65_69': 67, '80_84': 82, '40_44': 42
}
persons['Age'] = persons['Age'].map(age_to_midpoint)


from sklearn.preprocessing import LabelEncoder

# Encode categorical variables
le_marital = LabelEncoder()
persons['Marital'] = le_marital.fit_transform(persons['Marital'])

from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Perform clustering based on age and marital status
linked = linkage(persons[['Age', 'Marital', 'Children']], method='ward')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked, labels=persons['Person_ID'].values)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Person ID')
plt.ylabel('Distance')
plt.show()

from scipy.cluster.hierarchy import fcluster

# Define the cut-off distance to create distinct clusters
cut_off_distance = 10  # This value might need adjustments based on the dendrogram
clusters = fcluster(linked, t=cut_off_distance, criterion='distance')

# Add cluster labels to the persons DataFrame
persons['Cluster'] = clusters

# Group persons by their cluster to see how they fit into potential households
clustered_persons = persons.groupby('Cluster').agg({
    'Age': 'mean',  # Average age might help in further analysis
    'Marital': lambda x: x.mode()[0],  # Most common marital status in the cluster
    'Children': 'sum',  # Total number of children in the cluster
    'Person_ID': list  # List of persons in each cluster
})

print(clustered_persons)


# Define a function to match clusters to households based on household requirements
def match_clusters_to_households(clustered_persons, households):
    matches = {}
    for idx, household in households.iterrows():
        # Extract household requirements
        comp = household['Composition']
        slots = household['Slots']

        # Identify a suitable cluster for this household
        for cluster_id, row in clustered_persons.iterrows():
            # Define matching conditions based on the composition of the household
            if comp.startswith('1PE') and len(row['Person_ID']) == 1 and row['Age'] > 65:
                # Single pensioner
                condition = True
            elif comp.startswith('1PA') and len(row['Person_ID']) == 1 and row['Age'] <= 65:
                # Single adult (non-pensioner)
                condition = True
            elif comp.startswith('1FM-0C') and len(row['Person_ID']) == 2 and row['Children'] == 0:
                # Married couple with no children
                condition = True
            # Add more conditions here as per the composition rules
            else:
                condition = False

            # If a suitable cluster is found, assign it to the household
            if condition and len(row['Person_ID']) <= slots:
                matches[household['Household_ID']] = cluster_id
                # Once matched, remove this cluster from consideration
                clustered_persons = clustered_persons.drop(cluster_id)
                break

    return matches


# Perform matching
matches = match_clusters_to_households(clustered_persons, households)
print("Household Matches:", matches)
