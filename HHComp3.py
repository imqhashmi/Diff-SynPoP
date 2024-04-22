import pandas as pd
import numpy as np
import plotly as py
import plotly.graph_objects as go

# Define the composition_size_mapping and HH_composition
composition_size_mapping = {
    '1PE-0C': 1, '1PA-0C': 1, '1FE-0C': 1,
    '1FM-0C': 2, '1FS-0C': 1, '1FC-0C': 2,
    '1FM-1C': 3, '1FS-1C': 3, '1FC-1C': 3, '1FL-1C': 3,
    '1FM-nC': '4+', '1FM-nA': '4+', '1FS-nC': '4+', '1FS-nA': '4+',
    '1FC-nC': '4+', '1FC-nA': '4+', '1FL-nC': '4+', '1FL-nA': '4+',
    '1H-1C': '4+', '1H-nC': '4+', '1H-nA': '4+', '1H-nE': '4+', '1H-X': '4+'
}

HH_composition = {
    '1PE-0C': 515, '1PA-0C': 1333, '1FE-0C': 178, '1FM-0C': 417,
    '1FM-1C': 241, '1FM-nC': 379, '1FM-nA': 139, '1FS-0C': 4,
    '1FS-1C': 0, '1FS-nC': 0, '1FS-nA': 0, '1FC-0C': 469,
    '1FC-1C': 129, '1FC-nC': 93, '1FC-nA': 14, '1FL-1C': 210,
    '1FL-nC': 151, '1FL-nA': 99, '1H-1C': 78, '1H-nC': 68,
    '1H-nA': 3, '1H-nE': 6, '1H-X': 326
}
HH_size = {'1': 1848, '2': 1472, '3': 723, '4': 512, '5': 173, '6': 85, '7': 19, '8+': 20}

# Create a list to hold the data
data = []

# Fill the data list with the compositions
for composition, count in HH_composition.items():
    for _ in range(count):
        data.append({
            "HHID": None,  # Will be filled in next
            "Composition": composition,
            "Size": composition_size_mapping[composition]
        })

# Create the DataFrame
df = pd.DataFrame(data)

# Assign unique HHID values
df['HHID'] = range(1, len(df) + 1)

# obtain HHdf composition aggregats as a dictionary
HHdf_comp = df.groupby('Composition').size().to_dict()
HHdf_size = df.groupby('Size').size().to_dict()

fig = go.Figure()
fig.add_trace(go.Bar(x=list(HH_composition.keys()), y=list(HH_composition.values()), name='HH Composition'))
fig.add_trace(go.Bar(x=list(HHdf_comp.keys()), y=list(HHdf_comp.values()), name='HH Composition'))
fig.show()

fig = go.Figure()
fig.add_trace(go.Bar(x=list(HH_size.keys()), y=list(HH_size.values()), name='HH Size'))
fig.add_trace(go.Bar(x=list(HHdf_size.keys()), y=list(HHdf_size.values()), name='HH Size'))
fig.show()