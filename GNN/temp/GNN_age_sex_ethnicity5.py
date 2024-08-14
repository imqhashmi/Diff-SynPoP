import torch
from torch_geometric.data import Data

# Define node features for 10 person nodes with unique IDs as features
person_nodes = torch.arange(10).view(10, 1).float()  # Person nodes with unique IDs as features
age_nodes = torch.tensor([
    [0],  # Child
    [1],  # Adult
    [2]   # Elder
], dtype=torch.float)

sex_nodes = torch.tensor([
    [0],  # Male
    [1]   # Female
], dtype=torch.float)

ethnicity_nodes = torch.tensor([
    [0],  # White
    [1],  # Black
    [2]   # Asian
], dtype=torch.float)

# Combine all nodes
node_features = torch.cat([person_nodes, age_nodes, sex_nodes, ethnicity_nodes], dim=0)

# Define edges based on the cross table
# Person nodes: 0-9
# Age nodes: 10-12 (Child, Adult, Elder)
# Sex nodes: 13-14 (Male, Female)
# Ethnicity nodes: 15-17 (White, Black, Asian)
edge_index = torch.tensor([
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # Person IDs
    [11, 11, 11, 10, 10, 10, 10, 10, 12, 12, 14, 14, 14, 14, 14, 14, 13, 13, 14, 14, 16, 15, 15, 15, 15, 15, 17, 15, 15, 15]  # Characteristic IDs
], dtype=torch.long)

# Define the data object for PyTorch Geometric
data = Data(x=node_features, edge_index=edge_index)

# Aggregate function to count the observed data
def aggregate_counts(data):
    observed_counts = {
        'Child-Male-White': 0,
        'Child-Male-Black': 0,
        'Child-Male-Asian': 0,
        'Child-Female-White': 2,
        'Child-Female-Black': 0,
        'Child-Female-Asian': 1,

        'Adult-Male-White': 1,
        'Adult-Male-Black': 2,
        'Adult-Male-Asian': 1,
        'Adult-Female-White': 0,
        'Adult-Female-Black': 0,
        'Adult-Female-Asian': 0,

        'Elder-Male-White': 0,
        'Elder-Male-Black': 0,
        'Elder-Male-Asian': 1,
        'Elder-Female-White': 1,
        'Elder-Female-Black': 1,
        'Elder-Female-Asian': 0
    }

    age_mapping = {10: 'Child', 11: 'Child', 12: 'Elder'}
    sex_mapping = {13: 'Male', 14: 'Female'}
    ethnicity_mapping = {15: 'White', 16: 'Black', 17: 'Asian'}

    for i in range(data.edge_index.shape[1]):
        person_node = data.edge_index[0, i]
        characteristic_node = data.edge_index[1, i]

        if 10 <= characteristic_node <= 12:
            age = age_mapping[characteristic_node]
        elif 13 <= characteristic_node <= 14:
            sex = sex_mapping[characteristic_node]
        elif 15 <= characteristic_node <= 17:
            ethnicity = ethnicity_mapping[characteristic_node]

        if age and sex and ethnicity:
            key = f"{age}-{sex}-{ethnicity}"
            if key in observed_counts:
                observed_counts[key] += 1

    return observed_counts

# Calculate and print the aggregate counts
aggregate_results = aggregate_counts(data)
print(aggregate_results)
