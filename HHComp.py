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
Total = ID.get_HH_com_total(ID.HHcomdf, area)
HH_composition = ID.getHHcomdictionary(ID.HHcomdf, area)
print(HH_composition)
HH_size = ID.getdictionary(ID.HHsizedf, area)

df = pd.DataFrame(index=range(Total), columns=['HHID', 'Composition', 'Size', 'Persons'])
df['HHID'] = range(1, Total+1)

