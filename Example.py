import sys
import os
import time
import random
import numpy as np
import pandas
import plotly as py
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import torch
import torch.nn as nn
import torch.optim as optim
import torch
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
    # Ensuring the input_tensor is a float tensor since probabilities will be floats
    input_tensor = torch.zeros(total, sum(category_lengths.values()), device=device, dtype=torch.float32)
    start = 0
    for dict, length in zip(dicts, category_lengths.values()):
        probabilities = np.array(list(dict.values()), dtype=np.float32)  # Explicitly making it float32 to handle division correctly
        probabilities /= probabilities.sum()  # Normalize to create a distribution

        # Using probabilities to sample categories for each individual
        choices = np.random.choice(range(length), total, p=probabilities)
        for i, choice in enumerate(choices):
            input_tensor[i, start + choice] = 1  # Assigning selected category as '1'
        start += length
    return input_tensor



def agemap(age):
    if age in ['0_4', '5_7', '8_9', '10_14', '15']:
        return "0-15"
    elif age in ['16_17', '18_19', '20_24']:
        return "16-24"
    elif age in ['25_29', '30_34']:
        return "25-34"
    elif age in ['35_39', '40_44', '45_49']:
        return "35-49"
    elif age in ['50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85+']:
        return "50+"

def reverse_agemap(age):
    if age=="0-15":
        return ['0_4', '5_7', '8_9', '10_14', '15']
    elif age=="16-24":
        return ['16_17', '18_19', '20_24']
    elif age=="25-34":
        return ['25_29', '30_34']
    elif age=="35-49":
        return ['35_39', '40_44', '45_49']
    elif age=="50+":
        return ['50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85+']


def key_replace(key):
    temp = key.split(' ')
    return temp[0] + ' ' + temp[1].replace('0', '')

device = 'cuda' if torch.cuda.is_available() else 'cpu' # checking to see if a cuda device is available
path = os.path.join(os.path.dirname(os.getcwd()), 'Diff-SynPoP')

#MSOA
area = 'E02005924' # geography code for one of the oxford areas (selected for this work)
total = ID.get_total(ID.age5ydf, area) # getting the total number of individuals in our MSOA
hh_total = ID.get_total(ID.HHcomdf, area)

# getting the distributions of individual attributes in the selected MSOA
# saving the extracted distributions into respective dictionaries
sex_dict = ID.getdictionary(ID.sexdf, area) # sex
age_dict = ID.getdictionary(ID.age5ydf, area) # age
ethnic_dict = ID.getdictionary(ID.ethnicdf, area) # ethnicity
religion_dict = ID.getdictionary(ID.religiondf, area) # religion
marital_dict = ID.getdictionary(ID.maritaldf, area) # marital status
qual_dict = ID.getdictionary(ID.qualdf, area) # highest qualification level

# print(len(sex_dict), len(age_dict), len(ethnic_dict), len(religion_dict), len(marital_dict), len(qual_dict))
# print(len(sex_dict) +  len(age_dict) + len(ethnic_dict) + len(religion_dict) + len(marital_dict) + len(qual_dict))


hh_comp_dict = ID.getdictionary(ID.HHcomdf, area)  # household composition
# ct5 = ['1PE', '1PA', '1FE', '1FM-0C', '1FM-nC', '1FM-nA', '1FC-0C', '1FC-nC', '1FC-nA', '1FL-nC', '1FL-nA', '1H-nC', '1H-nA', '1H-nE']
# #iterate hh_comp_dict and remove keys that are not in ct5
# hh_comp_dict = {k: v for k, v in hh_comp_dict.items() if k in ct5}
hh_id = range(1, hh_total + 1)
hh_size_dict = ID.getdictionary(ID.HHsizedf, area)  # household size
hh_comp_by_size = ICT.get_hh_comp_by_size_crosstable(area)

hh_comp_dict_mod = {index: value for index, (_, value) in enumerate(hh_comp_dict.items())}
ethnic_dict_hh = ID.getdictionary(ID.ethnicdf, area) # ethnicity of reference person of a household
religion_dict_hh = ID.getdictionary(ID.religiondf, area) # religion of reference person of a household

# getting the length (number of classes) for each attribute
category_lengths = {
    'sex': len(sex_dict),
    'age': len(age_dict),
    'ethnicity': len(ethnic_dict),
    'religion': len(religion_dict),
    'marital': len(marital_dict),
    'qual': len(qual_dict)
}

# getting the length (number of classes) for each attribute
category_dicts = {
    'ethnicity': ethnic_dict_hh,
    'religion': religion_dict_hh,
    'hh_comp': hh_comp_dict,
    'hh_size': hh_size_dict
}

# instantiating networks for each characteristic
input_dim = sum(len(d.keys()) for d in [sex_dict, age_dict, ethnic_dict, religion_dict, marital_dict, qual_dict])
print("Input dimension: ", input_dim)
hidden_dims = [64, 32]

input_dim_hh = sum(len(d.keys()) for d in [ethnic_dict_hh, religion_dict_hh, hh_comp_dict, hh_size_dict])
print("Input HH dimension: ", input_dim_hh)
hidden_dims_hh = [64, 32]

# With CUDA
sex_net = FFNetwork(input_dim, hidden_dims, len(sex_dict)).to(device).cuda()
age_net = FFNetwork(input_dim, hidden_dims, len(age_dict)).to(device).cuda()
ethnic_net = FFNetwork(input_dim, hidden_dims, len(ethnic_dict)).to(device).cuda()
religion_net = FFNetwork(input_dim, hidden_dims, len(religion_dict)).to(device).cuda()
marital_net = FFNetwork(input_dim, hidden_dims, len(marital_dict)).to(device).cuda()
qual_net = FFNetwork(input_dim, hidden_dims, len(qual_dict)).to(device).cuda()
# hh_comp_net = FFNetwork(input_dim, hidden_dims, len(hh_comp_dict)).to(device).cuda()
# age_hh_net = FFNetwork(input_dim, hidden_dims, len(age_hh_dict)).to(device).cuda()



# With CUDA
ethnic_net_hh = FFNetwork(input_dim_hh, hidden_dims_hh, len(ethnic_dict_hh)).to(device).cuda()
religion_net_hh = FFNetwork(input_dim_hh, hidden_dims_hh, len(religion_dict_hh)).to(device).cuda()
hh_comp_net = FFNetwork(input_dim_hh, hidden_dims, len(hh_comp_dict)).to(device).cuda()
hh_size_net = FFNetwork(input_dim_hh, hidden_dims, len(hh_size_dict)).to(device).cuda()

input_tensor = torch.empty(total, input_dim).to(device)
init.kaiming_normal_(input_tensor)

input_tensor_hh = torch.empty(hh_total, input_dim_hh).to(device)
init.kaiming_normal_(input_tensor_hh)

# ACCESSORY FUNCTIONS (INDIVIDUAL GENERATION)
# defining the Gumbel-Softmax function
def gumbel_softmax_sample(logits, temperature=0.5):
    gumbel_noise = -torch.log(-torch.log(torch.rand(logits.shape, device=device)))
    y = logits + gumbel_noise
    return torch.nn.functional.softmax(y / temperature, dim=-1)

def generate_population(input_tensor, temperature=0.5):
    sex_logits = sex_net(input_tensor)
    age_logits = age_net(input_tensor)
    ethnicity_logits = ethnic_net(input_tensor)
    religion_logits = religion_net(input_tensor)
    marital_logits = marital_net(input_tensor)
    qual_logits = qual_net(input_tensor)

    sex = gumbel_softmax_sample(sex_logits, temperature)
    age = gumbel_softmax_sample(age_logits, temperature)
    ethnicity = gumbel_softmax_sample(ethnicity_logits, temperature)
    religion = gumbel_softmax_sample(religion_logits, temperature)
    marital = gumbel_softmax_sample(marital_logits, temperature)
    qual = gumbel_softmax_sample(qual_logits, temperature)
    return torch.cat([sex, age, ethnicity, religion, marital, qual], dim=-1)


def generate_households(input_tensor, temperature=0.5):
    ethnicity_logits = ethnic_net_hh(input_tensor_hh)
    religion_logits = religion_net_hh(input_tensor_hh)
    hh_comp_logits = hh_comp_net(input_tensor_hh)
    hh_size_logits = hh_size_net(input_tensor_hh)

    ethnicity_hh = gumbel_softmax_sample(ethnicity_logits, temperature)
    religion_hh = gumbel_softmax_sample(religion_logits, temperature)
    hh_comp = gumbel_softmax_sample(hh_comp_logits, temperature)
    hh_size = gumbel_softmax_sample(hh_size_logits, temperature)

    return torch.cat([ethnicity_hh, religion_hh, hh_comp, hh_size], dim=-1)

def gumbel_softmax_to_one_hot(tensor, attributes):
    one_hot_encoded = []
    for key, item in attributes.items():
        start_idx = item[1]
        end_idx = item[2]
        attribute_tensor = tensor[:, start_idx:end_idx]
        max_indices = torch.argmax(attribute_tensor, dim=1)
        one_hot = F.one_hot(max_indices, num_classes=len(item[0]))
        one_hot_encoded.append(one_hot)

    return torch.cat(one_hot_encoded, dim=1)

def decode_tensor(row, attributes):
    row = row.cpu().detach().numpy()
    result = []
    for key, value in attributes.items():
        start_idx = value[1]
        end_idx = value[2]
        attribute_tensor = row[:, start_idx:end_idx]
        # Get the index of the maximum value
        max_index = np.argmax(attribute_tensor)
        # print(max_index, attribute_tensor)
        # Get the corresponding key
        decoded_key = value[0][max_index]
        result.append(decoded_key)
    return result

encoded_population = generate_population(input_tensor) # [:3]
encoded_household = generate_households(input_tensor_hh)

# Define the attribute sizes (number of categories for each attribute)
attributes= {
    'sex': list(sex_dict.keys()),
    'age_group': list(age_dict.keys()),
    'ethnicity': list(ethnic_dict.keys()),
    'religion': list(religion_dict.keys()),
    'marital_status': list(marital_dict.keys()),
    'qualification': list(qual_dict.keys())
}

attributes_hh = {
    'ethnicity': list(ethnic_dict_hh.keys()),
    'religion': list(religion_dict_hh.keys()),
    'hh_comp': list(hh_comp_dict.keys()),
    'hh_size': list(hh_size_dict.keys())
}

# get all age groups for children
children = ['0_4', '5_7', '8_9', '10_14', '15', '16_17']
adults = ['18_19', '20_24', '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64']
elders = ['65_69', '70_74', '75_79', '80_84', '85+']

def extract_random_sample(person_list, size):
    if (len(person_list)>0):
        size = int(size)
        random_samples = random.sample(person_list, size)
        # remove random samples from persons
        for sample in random_samples:
            del persons[sample]
        return random_samples

def handle_composition(composition, size, persons):
    if composition == '1PE' or composition == '1FE': # assuming 1PE will have size = 1
        # Handle One person: Pensioner (person who is above 65)
        # Handle One family: All pensioner (a family consisting of persons all above 65)
        persons_filtered = [k for k, v in persons.items() if v[1] in elders]
        return extract_random_sample(persons_filtered, size)

    elif composition == '1PA':
        # Handle One person: Other (a single person who is above 18 and below 65)
        persons_filtered = [k for k, v in persons.items() if v[1] in adults]
        return extract_random_sample(persons_filtered, size)

    elif composition == '1FM-0C':
        # Handle One family: Married Couple: No children
        persons_filtered_married_male = [k for k, v in persons.items() if v[4] == 'Married' and v[0]=='M']
        persons_filtered_married_female = [k for k, v in persons.items() if v[4] == 'Married' and v[0]=='F']
        # get random samples from both lists
        return extract_random_sample(persons_filtered_married_male, 1) + extract_random_sample(persons_filtered_married_female, 1)
    # elif composition == '1FM-1C':
    #     # Handle One family: Married Couple: One dependent child
    #     pass
    # elif composition == '1FM-nC':
    #     # Handle One family: Married Couple: Having dependent children
    #     pass
    # elif composition == '1FM-nA':
    #     # Handle One family: Married Couple: all children non-dependent
    #     pass
    # elif composition == '1FS-0C':
    #     # Handle One family: Separated Couple: No children
    #     pass
    # elif composition == '1FS-1C':
    #     # Handle One family: Separated Couple: One dependent child
    #     pass
    # elif composition == '1FS-nC':
    #     # Handle One family: Separated Couple: Having dependent children
    #     pass
    # elif composition == '1FS-nA':
    #     # Handle One family: Separated Couple: all children non-dependent
    #     pass
    # elif composition == '1FC-0C':
    #     # Handle One family: Cohabiting Couple: No children
    #     pass
    # elif composition == '1FC-1C':
    #     # Handle One family: Cohabiting Couple: One dependent child
    #     pass
    # elif composition == '1FC-nC':
    #     # Handle One family: Cohabiting Couple: Having dependent children
    #     pass
    # elif composition == '1FC-nA':
    #     # Handle One family: Cohabiting Couple: all children non-dependent
    #     pass
    # elif composition == '1FL-1C':
    #     # Handle One family: Lone Parent: One dependent child
    #     pass
    # elif composition == '1FL-nC':
    #     # Handle One family: Lone Parent: Having dependent children
    #     pass
    # elif composition == '1FL-nA':
    #     # Handle One family: Lone parent: all children non-dependent
    #     pass
    # elif composition == '1H-1C':
    #     # Handle Other households: One dependent child
    #     pass
    # elif composition == '1H-nC':
    #     # Handle Other households: Having dependent children
    #     pass
    # elif composition == '1H-nA':
    #     # Handle Other households: All adults
    #     pass
    # elif composition == '1H-nE':
    #     # Handle Other households: All pensioners
    #     pass
    # else:
    #     # Handle unknown composition
    #     pass

def set_starting_indices(attributes):
    # Calculate the starting indices for each attribute
    start_idx = 0
    for key, value in attributes.items():
        size = len(value)
        # add the start and end indices to the attributes dictionary
        attributes[key] = (attributes[key], start_idx, start_idx + size)
        start_idx += size

set_starting_indices(attributes)
set_starting_indices(attributes_hh)

persons = {}
for i in range(len(encoded_population)):
    persons[i] = decode_tensor(encoded_population[i].unsqueeze(0), attributes)

persons_original = persons.copy()

households = {}
for j in range(len(encoded_household)):
    households[j] = decode_tensor(encoded_household[j].unsqueeze(0), attributes_hh)


for i, household in households.items():
    eth = household[0]
    rel = household[1]
    comp = household[2]
    size = household[3]
    #get all rows from persons filtered by eth, rel
    persons_filtered = {key: value for key, value in persons.items() if value[2] == eth and value[3] == rel}
    selected_indices = handle_composition(comp, int(size), persons_filtered)
    print(selected_indices)

