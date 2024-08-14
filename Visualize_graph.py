import os
import pandas as pd
import networkx as nx
from pyvis.network import Network

# Assuming the path setup and CSV loading are correct and unchanged
path = os.path.join(os.path.dirname(os.getcwd()), 'Diff-SynPoP')
df = pd.read_csv(os.path.join(path, 'synthetic_households.csv'))

# Sample a fraction of the dataframe for processing
df = df.sample(frac=0.1)

# Create a new graph
G = nx.Graph()

# Track person assignments to ensure uniqueness
assigned_persons = set()

# Add household nodes and check for person uniqueness
for idx, row in df.iterrows():
    persons = row['assigned_persons'].strip('[]').split(', ')
    # Convert to int and filter empty strings
    persons = [int(person) for person in persons if person != '']
    for person in persons:
        if person not in assigned_persons:
            G.add_node(idx, label=str('H' + str(idx)), color='blue')  # Ensure labels are strings for consistency
            G.add_node(person, label=str(person), color='red')
            G.add_edge(idx, person, color='black')
            assigned_persons.add(person)
        else:
            print(f"Duplicate assignment detected: Person {person} is already assigned to a household.")

# Visualize the graph
nt = Network(height='1000px', width='100%')
nt.from_nx(G)
nt.show('nx.html', notebook=False)
