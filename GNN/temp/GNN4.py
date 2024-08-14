import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Distributions
age_dist = {'Child': 0.25, 'Adult': 0.6, 'Elder': 0.15}
sex_dist = {'Male': 0.5, 'Female': 0.5}
religion_dist = {'Christian': 0.5, 'Jew': 0.1, 'Muslim': 0.4}

crosstable1 = {'Adult-Female-Christian': 10, 'Adult-Female-Jew': 6, 'Adult-Female-Muslim': 5,
               'Adult-Male-Christian': 19, 'Adult-Male-Jew': 4, 'Adult-Male-Muslim': 13,
               'Child-Female-Christian': 6, 'Child-Female-Jew': 2, 'Child-Female-Muslim': 5,
               'Child-Male-Christian': 6, 'Child-Male-Jew': 1, 'Child-Male-Muslim': 6,
               'Elder-Female-Christian': 5, 'Elder-Female-Jew': 1, 'Elder-Female-Muslim': 5,
               'Elder-Male-Christian': 3, 'Elder-Male-Jew': 0, 'Elder-Male-Muslim': 3}
n = 100

# Create a directed graph
G = nx.DiGraph()

# Helper function to sample based on distribution
def sample_distribution(distribution, size):
    return np.random.choice(list(distribution.keys()), size=size, p=list(distribution.values()))

# Generate nodes for each attribute
ages= {f'A{i+1}': age for i, age in enumerate(sample_distribution(age_dist, n))}
sexs = {f'S{i+1}': sex for i, sex in enumerate(sample_distribution(sex_dist, n))}
religions = {f'R{i+1}': rel for i, rel in enumerate(sample_distribution(religion_dist, n))}


# Add attribute nodes to the graph
for key, value in ages.items():
    G.add_node(key, type='age', label=value)

for key, value in sexs.items():
    G.add_node(key, type='sex', label=value)

for key, value in religions.items():
    G.add_node(key, type='religion', label=value)

for i in range(n):
    G.add_node(f'P{i}', type='person', label=f'P{i}')


# Add edges between person and attribute nodes
for i in range(n):
    G.add_edge(f'P{i}', f'A{i+1}')
    G.add_edge(f'P{i}', f'S{i+1}')
    G.add_edge(f'P{i}', f'R{i+1}')

#draw the graph using pyViz
# import pyvis
# from pyvis.network import Network
# nt = Network(height='750px', width='100%')
# nt.from_nx(G)
# nt.show('nx.html', notebook=False)

import dgl

def build_dgl_graph(nx_graph):
    g = dgl.from_networkx(nx_graph, node_attrs=['feature'])
    return g

dgl_graph = build_dgl_graph(G)
