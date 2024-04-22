import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import random
import time
import numpy as np
import plotly as py
import pandas as pd
import plotly.graph_objects as go

import InputData as ID
import InputCrossTables as ICT

file_path = os.path.join(os.path.dirname(os.getcwd()), 'Diff-SynPoP', 'synthetic_population.csv')
persons_df = pd.read_csv(file_path)

area = 'E02005924'
HH_composition = ID.getHHcomdictionary(ID.HHcomdf, area)
HH_size = ID.getdictionary(ID.HHsizedf, area)
Total = sum(HH_size.values())

print(Total)
print(HH_composition)
print(HH_size)


composition_size_mapping = {
    # 1 person
    '1PE-0C': 1, '1PA-0C': 1, '1FE-0C': 1,
    # 2 people
    '1FM-0C': 2, '1FS-0C': 1, '1FC-0C': 2,
    # 3 people
    '1FM-1C': 3, '1FS-1C': 3, '1FC-1C': 3, '1FL-1C': 3,
    # more people
    '1FM-nC': '4+', '1FM-nA': '4+', '1FS-nC': '4+', '1FS-nA': '4+',
    '1FC-nC': '4+', '1FC-nA': '4+', '1FL-nC': '4+', '1FL-nA': '4+',
    '1H-1C': '4+', '1H-nC': '4+', '1H-nA': '4+', '1H-nE': '4+', '1H-X': '4+'
}
# Generate household entries, treating '8+' sizes as 8
households = []
for comp_code, count in HH_composition.items():
    size = composition_size_mapping.get(comp_code, '4+')  # Fallback for missing mappings
    if isinstance(size, str):  # Handle '4+' category
        total_count_for_range = sum(HH_size[str(i)] for i in range(4, 8)) + HH_size['8+']
        for i in range(4, 9):
            size_key = str(i) if i < 8 else '8+'
            proportion = HH_size[size_key] / total_count_for_range
            households_for_size = round(count * proportion)
            for _ in range(households_for_size):
                households.append((comp_code, i, []))
    else:
        for _ in range(count):
            households.append((comp_code, size, []))
# Create DataFrame with Household Size as integers
HHdf = pd.DataFrame(households, columns=['Composition', 'Size', 'Persons'])
HHdf['Size'] = HHdf['Size'].apply(lambda x: 8 if x == '8+' else x).astype(int)

#obtain HHdf composition aggregats as a dictionary
HHdf_composition = HHdf['Composition'].value_counts().to_dict()
HHdf_size = HHdf['Size'].value_counts().to_dict()

# fig = go.Figure()
# fig.add_trace(go.Bar(x=list(HH_composition.keys()), y=list(HH_composition.values()), name='HH Composition'))
# fig.add_trace(go.Bar(x=list(HHdf_composition.keys()), y=list(HHdf_composition.values()), name='HH Composition'))
# fig.show()
#
# fig = go.Figure()
# fig.add_trace(go.Bar(x=list(HH_size.keys()), y=list(HH_size.values()), name='HH Size'))
# fig.add_trace(go.Bar(x=list(HHdf_size.keys()), y=list(HHdf_size.values()), name='HH Size'))
# fig.show()