import os
import pandas as pd
import random
import time
import torch

import InputData as ID
import InputCrossTables as ICT

# create local path from current working directory
persons_df = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), 'Diff-SynPoP', 'synthetic_population.csv')) # saving the loaded csv file to a pandas dataframe

area = 'E02005924'
HH_composition = ID.getHHcomdictionary(ID.HHcomdf, area)
HH_Total = sum(HH_composition.values())
HH_size = ID.getdictionary(ID.HHsizedf, area)
# Provided data
composition_size_mapping = {
    '1PE-0C': 1, '1PA-0C': 1, '1FE-0C': 1,
    '1FM-0C': 2, '1FS-0C': 1, '1FC-0C': 2,
    '1FM-1C': 3, '1FS-1C': 3, '1FC-1C': 3, '1FL-1C': 3
}

# Initialize DataFrame
df = pd.DataFrame(index=range(HH_Total), columns=['HHID', 'Composition', 'Size', 'Persons'])
# loop through the HH_Composition dictionary and create a dataframe

for key, value in HH_composition.items():
    for i in range(value):
        df.loc[i, 'HHID'] = i
        df.loc[i, 'Composition'] = key
        df.loc[i, 'Size'] = composition_size_mapping[key]
