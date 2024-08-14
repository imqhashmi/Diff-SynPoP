### Importing Libraries

import sys

import os
import ast
import time
import random
import itertools
import numpy as np
import plotly as py
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# importing InputData and InputCrossTables for processing UK census data files
import InputData as ID
import InputCrossTables as ICT

from torch.nn import init
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ExponentialLR

torch.set_printoptions(threshold=10000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

### Feed Forward Network

class FFNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(FFNetwork, self).__init__()
        layers = []
        in_dims = [input_dim] + hidden_dims[:-1] # these are the input dimensions for each layer
        for in_dim, out_dim in zip(in_dims, hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim)) # a linear layer
            layers.append(nn.BatchNorm1d(out_dim)) # a batch normalization layer (helps with quicker covnergence)
            layers.append(nn.ReLU()) # non-linearity / activation function (ReLU performed the best among others)
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        x = self.layers(x)
        return self.output_layer(x)
    
def initialize_input_tensor(total, category_lengths, dicts):
    # ensuring the input_tensor is a float tensor since probabilities will be floats
    input_tensor = torch.zeros(total, sum(category_lengths.values()), device=device, dtype=torch.float32)
    start = 0
    for dict, length in zip(dicts, category_lengths.values()):
        probabilities = np.array(list(dict.values()), dtype=np.float32)  # explicitly making it float32 to handle division correctly
        probabilities /= probabilities.sum()  # Normalize to create a distribution

        # using probabilities to sample categories for each individual
        choices = np.random.choice(range(length), total, p=probabilities)
        for i, choice in enumerate(choices):
            input_tensor[i, start + choice] = 1  # assigning selected category as '1'
        start += length
    return input_tensor

### Dictionaries, Cross Tables, & Input Tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu' # checking to see if a cuda device is available
path = os.path.join(os.path.dirname(os.getcwd()), 'Diff-SynPoP')

# selected MSOA
area = 'E02005924' # geography code for one of the oxford output areas (selected for this work)

total = ID.get_total(ID.age5ydf, area) # getting the total number of individuals in our MSOA
# num_households = ID.get_HH_com_total(ID.HHcomdf, area) # getting the total number of households in our MSOA
num_households = 4852

print("Total number of individuals: ", total)
print("Total number of households: ", num_households)
print()

# getting the distributions of individual attributes in the selected MSOA
# saving the extracted distributions into respective dictionaries
sex_dict = ID.getdictionary(ID.sexdf, area) # sex
age_dict = ID.getdictionary(ID.age5ydf, area) # age
ethnic_dict = ID.getdictionary(ID.ethnicdf, area) # ethnicity
religion_dict = ID.getdictionary(ID.religiondf, area) # religion
marital_dict = ID.getdictionary(ID.maritaldf, area) # marital status
qual_dict = ID.getdictionary(ID.qualdf, area) # highest qualification level

seg_dict = ID.getFinDictionary(ID.seg_df, area) # socio-economic grade
occupation_dict = ID.getFinDictionary(ID.occupation_df, area) # occupation
economic_act_dict = ID.getFinDictionary(ID.economic_act_df, area) #economic activity
approx_social_grade_dict = ID.getFinDictionary(ID.approx_social_grade_df, area) # approximated social grade
general_health_dict = ID.getFinDictionary(ID.general_health_df, area) # general health
industry_dict = ID.getFinDictionary(ID.industry_df, area) # industry of occupation

hh_comp_dict = ID.getHHcomdictionary(ID.HHcomdf, area) # household composition
hh_id = range(1, num_households + 1)
hh_size_dict = ID.getdictionary(ID.HHsizedf, area)  # household size

hh_comp_dict_mod = {index: value for index, (_, value) in enumerate(hh_comp_dict.items())}
ethnic_dict_hh = ID.getdictionary(ID.ethnicdf, area) # ethnicity of reference person of a household
religion_dict_hh = ID.getdictionary(ID.religiondf, area) # religion of reference person of a household
car_ownership_dict = ID.getHHDictionary(ID.car_ownership_df, area) # car or van ownership / availability

# getting the length (number of classes) for each attribute
category_lengths = {
    'sex': len(sex_dict),
    'age': len(age_dict),
    'ethnicity': len(ethnic_dict),
    'religion': len(religion_dict),
    'marital': len(marital_dict),
    'qual': len(qual_dict),
    'seg': len(seg_dict),
    'occupation': len(occupation_dict),
    'economic_act': len(economic_act_dict),
    'approx_social_grade': len(approx_social_grade_dict),
    'general_health': len(general_health_dict),
    'industry': len(industry_dict)
}

category_lengths_hh = {
    'composition': len(hh_comp_dict),
    'ref_ethnicity': len(ethnic_dict_hh),
    'ref_religion': len(religion_dict_hh),
    'car_ownership': len(car_ownership_dict)
}

cross_table1 = ICT.getdictionary(ICT.ethnic_by_sex_by_age, area)
cross_table2 = ICT.getdictionary(ICT.religion_by_sex_by_age, area)
cross_table3 = ICT.getdictionary(ICT.marital_by_sex_by_age, area)
cross_table4 = ICT.getdictionary(ICT.qualification_by_sex_by_age, area)
cross_table5 = ICT.getdictionary(ICT.HH_composition_by_sex_by_age, area)

cross_table6 = ICT.getFinDictionary(ICT.seg, area)
cross_table7 = ICT.getFinDictionary(ICT.occupation, area)
cross_table8 = ICT.getFinDictionary(ICT.economic_act, area)
cross_table9 = ICT.getFinDictionary(ICT.approx_social_grade, area)
cross_table10 = ICT.getFinDictionary(ICT.general_health, area)
cross_table11 = ICT.getFinDictionary(ICT.industry, area)

cross_table1_hh = ICT.getdictionary(ICT.HH_composition_by_Ethnicity, area)
cross_table2_hh = ICT.getdictionary(ICT.HH_composition_by_Religion, area)
cross_table3_hh = ICT.getHHDictionary(ICT.car_ownership, area)


print(cross_table1)