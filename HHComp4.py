# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
#
# # Define the composition-size mapping and initial household composition
# composition_size_mapping = {
#     '1PE-0C': 1, '1PA-0C': 1, '1FE-0C': 1,
#     '1FM-0C': 2, '1FS-0C': 1, '1FC-0C': 2,
#     '1FM-1C': 3, '1FS-1C': 3, '1FC-1C': 3, '1FL-1C': 3,
#     '1FM-nC': '4+', '1FM-nA': '4+', '1FS-nC': '4+', '1FS-nA': '4+',
#     '1FC-nC': '4+', '1FC-nA': '4+', '1FL-nC': '4+', '1FL-nA': '4+',
#     '1H-1C': '4+', '1H-nC': '4+', '1H-nA': '4+', '1H-nE': '4+', '1H-X': '4+'
# }
#
# HH_composition = {
#     '1PE-0C': 515, '1PA-0C': 1333, '1FE-0C': 178, '1FM-0C': 417,
#     '1FM-1C': 241, '1FM-nC': 379, '1FM-nA': 139, '1FS-0C': 4,
#     '1FS-1C': 0, '1FS-nC': 0, '1FS-nA': 0, '1FC-0C': 469,
#     '1FC-1C': 129, '1FC-nC': 93, '1FC-nA': 14, '1FL-1C': 210,
#     '1FL-nC': 151, '1FL-nA': 99, '1H-1C': 78, '1H-nC': 68,
#     '1H-nA': 3, '1H-nE': 6, '1H-X': 326
# }
#
# HH_size = {'1': 1848, '2': 1472, '3': 723, '4': 512, '5': 173, '6': 85, '7': 19, '8+': 20}
#
# # Create the initial DataFrame
# data = []
# for composition, count in HH_composition.items():
#     for _ in range(count):
#         data.append({
#             "HHID": None,
#             "Composition": composition,
#             "Size": composition_size_mapping[composition]
#         })
#
# df = pd.DataFrame(data)
# df['HHID'] = range(1, len(df) + 1)
# df['Size'] = df['Size'].astype(str)
#
#
# # Custom IPF algorithm to adjust '4+' sizes
# def HH_IPF_Size(df, target_distribution, max_iterations=20, tolerance=1):
#     for _ in range(max_iterations):
#         current_distribution = df['Size'].value_counts().to_dict()
#         diff_distribution = {k: target_distribution.get(k, 0) - current_distribution.get(k, 0)
#                              for k in set(target_distribution) | set(current_distribution)}
#
#         if all(abs(val) <= tolerance for val in diff_distribution.values()):
#             return df
#
#         for size, diff in diff_distribution.items():
#             if diff > 0 and size != '4+':
#                 potential_updates = df[(df['Size'] == '4+') | (df['Size'].astype(str) > size)]
#                 if not potential_updates.empty:
#                     update_indices = potential_updates.sample(min(len(potential_updates), diff)).index
#                     df.loc[update_indices, 'Size'] = size
#             elif diff < 0:
#                 potential_downgrades = df[df['Size'] == size]
#                 if not potential_downgrades.empty:
#                     downgrade_indices = potential_downgrades.sample(min(len(potential_downgrades), -diff)).index
#                     df.loc[downgrade_indices, 'Size'] = '4+'
#     return df
#
#
# # Convert the target size distribution to string keys for consistency
# target_distribution_str_keys = {str(k): v for k, v in HH_size.items()}
# # Adjust the DataFrame to match the target size distribution
# HHdf = HH_IPF_Size(df, target_distribution_str_keys)
#
# HHdf_composition = HHdf['Composition'].value_counts().to_dict()
# HHdf_size = HHdf['Size'].value_counts().to_dict()
#
#
# fig = go.Figure()
# fig.add_trace(go.Bar(x=list(HH_size.keys()), y=list(HH_size.values()), name='HH Size'))
# fig.add_trace(go.Bar(x=list(HHdf_size.keys()), y=list(HHdf_size.values()), name='HH Size'))
# fig.show()
#

import pandas as pd
import numpy as np

# Composition-size mapping and initial household composition
composition_size_mapping = {
    '1PE-0C': 1, '1PA-0C': 1, '1FE-0C': 1,
    '1FM-0C': 2, '1FS-0C': 1, '1FC-0C': 2,
    '1FM-1C': 3, '1FS-1C': 3, '1FC-1C': 3, '1FL-1C': 3
}

HH_composition = {
    '1PE-0C': 515, '1PA-0C': 1333, '1FE-0C': 178, '1FM-0C': 417,
    '1FM-1C': 241, '1FM-nC': 379, '1FM-nA': 139, '1FS-0C': 4,
    '1FS-1C': 0, '1FS-nC': 0, '1FS-nA': 0, '1FC-0C': 469,
    '1FC-1C': 129, '1FC-nC': 93, '1FC-nA': 14, '1FL-1C': 210,
    '1FL-nC': 151, '1FL-nA': 99, '1H-1C': 78, '1H-nC': 68,
    '1H-nA': 3, '1H-nE': 6, '1H-X': 326
}

HH_size_distribution = {'1': 1848, '2': 1472, '3': 723, '4': 512, '5': 173, '6': 85, '7': 19, '8+': 20}

# Create the initial DataFrame
data = [{'HHID': i+1, 'Composition': comp, 'Size': size}
        for comp, count in HH_composition.items()
        for size in [composition_size_mapping[comp]]*count
        for i in range(count)]
df = pd.DataFrame(data)

# Function to perform IPF adjustments
def perform_ipf(df, target_distribution):
    # Expand '4+' into individual counts for sizes 4 through '8+'
    expanded_sizes = {**{str(i): 0 for i in range(1, 8)}, '8+': 0}
    for size, count in target_distribution.items():
        if size in expanded_sizes:
            expanded_sizes[size] = count

    # Initial adjustment: Assign '4+' based on remaining distribution needs
    for _ in range(len(df)):
        current_dist = df['Size'].value_counts().to_dict()
        for size, req_count in expanded_sizes.items():
            current_count = current_dist.get(size, 0)
            if req_count > current_count and '4+' in df['Size'].values:
                idx = df[df['Size'] == '4+'].sample(1).index
                df.at[idx, 'Size'] = size
                break

    return df

# Adjust DataFrame to match target size distribution
df_adjusted = perform_ipf(df, HH_size_distribution)

# Verify the final size distribution
final_distribution = df_adjusted['Size'].value_counts().sort_index()
final_distribution
