import networkx as nx
import numpy as np

# Distributions
age_dist = {'Child': 0.25, 'Adult': 0.6, 'Elder': 0.15}
sex_dist = {'Male': 0.5, 'Female': 0.5}
ethnicity_dist = {'White': 0.6, 'Black': 0.2, 'Asian': 0.2}
religion_dist = {'Christian': 0.5, 'Jew': 0.1, 'Muslim': 0.4}

n = 100

# Create a directed graph
G = nx.DiGraph()


# Helper function to sample based on distribution
def sample_distribution(distribution, size):
    return np.random.choice(list(distribution.keys()), size=size, p=list(distribution.values()))


# Generate nodes for each attribute
ages = {f'A{i + 1}': age for i, age in enumerate(sample_distribution(age_dist, n))}
sexs = {f'S{i + 1}': sex for i, sex in enumerate(sample_distribution(sex_dist, n))}
ethnicities = {f'E{i + 1}': eth for i, eth in enumerate(sample_distribution(ethnicity_dist, n))}
religions = {f'R{i + 1}': rel for i, rel in enumerate(sample_distribution(religion_dist, n))}

# Add attribute nodes to the graph
for key, value in ages.items():
    G.add_node(key, type='age', label=value)

for key, value in sexs.items():
    G.add_node(key, type='sex', label=value)

for key, value in ethnicities.items():
    G.add_node(key, type='ethnicity', label=value)

for key, value in religions.items():
    G.add_node(key, type='religion', label=value)

for i in range(n):
    G.add_node(f'P{i}', type='person', label=f'P{i}')

# Add edges between person and attribute nodes
for i in range(n):
    G.add_edge(f'P{i}', f'A{i + 1}')
    G.add_edge(f'P{i}', f'S{i + 1}')
    G.add_edge(f'P{i}', f'E{i + 1}')
    G.add_edge(f'P{i}', f'R{i + 1}')

import pyvis
from pyvis.network import Network
nt = Network(height='750px', width='100%')
nt.from_nx(G)
nt.show('nx.html', notebook=False)