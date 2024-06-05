### Importing Libraries

import sys

import os
import ast
import math
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

hh_comp_dict = ID.getHHcomdictionary(ID.HHcomdf, area) # household composition
hh_id = range(1, num_households + 1)
hh_size_dict = ID.getdictionary(ID.HHsizedf, area)  # household size

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

category_lengths_hh = {
    'composition': len(hh_comp_dict),
    'ref_ethnicity': len(ethnic_dict_hh),
    'ref_religion': len(religion_dict_hh)
}

cross_table1 = ICT.getdictionary(ICT.ethnic_by_sex_by_age, area)
cross_table2 = ICT.getdictionary(ICT.religion_by_sex_by_age, area)
cross_table3 = ICT.getdictionary(ICT.marital_by_sex_by_age, area)
cross_table4 = ICT.getdictionary(ICT.qualification_by_sex_by_age, area)
cross_table5 = ICT.getdictionary(ICT.HH_composition_by_sex_by_age, area)

cross_table1_hh = ICT.getdictionary(ICT.HH_composition_by_Ethnicity, area)
cross_table2_hh = ICT.getdictionary(ICT.HH_composition_by_Religion, area)

weights = ID.getweights(ID.age5ydf, area)

cross_table5a = {}
for sexkey, sexvalue in sex_dict.items():
    for agekey, agevalue in age_dict.items():
        for hhkey, hhvalue in hh_comp_dict.items():
            val = cross_table5[sexkey + ' ' + ID.agemap(agekey) + ' ' + hhkey]
            ages = ID.reverse_agemap(ID.agemap(agekey))
            # divide the val in cross_table5a ages using weights
            for age in ages:
                cross_table5a[sexkey + ' ' + age + ' ' + hhkey] = round(val * weights[age])

cross_table_tensor1 = torch.tensor(list(cross_table1.values()), dtype=torch.float32).to(device).cuda()
cross_table_tensor2 = torch.tensor(list(cross_table2.values()), dtype=torch.float32).to(device).cuda()
cross_table_tensor3 = torch.tensor(list(cross_table3.values()), dtype=torch.float32).to(device).cuda()
cross_table_tensor4 = torch.tensor(list(cross_table4.values()), dtype=torch.float32).to(device).cuda()
cross_table_tensor5 = torch.tensor(list(cross_table5a.values()), dtype=torch.float32).to(device).cuda()

cross_table_tensor1_hh = torch.tensor(list(cross_table1_hh.values()), dtype=torch.float32).to(device).cuda()
cross_table_tensor2_hh = torch.tensor(list(cross_table2_hh.values()), dtype=torch.float32).to(device).cuda()

input_dim = sum(len(d.keys()) for d in [sex_dict, age_dict, ethnic_dict, religion_dict, marital_dict, qual_dict])
hidden_dims = [64, 32]

input_dim_hh = sum(len(d.keys()) for d in [ethnic_dict_hh, religion_dict_hh])
hidden_dims_hh = [64, 32]

### Accessory Functions

# defining the Gumbel-Softmax function
def gumbel_softmax_sample(logits, temperature=0.5):
    gumbel_noise = -torch.log(-torch.log(torch.rand(logits.shape, device=device)))
    y = logits + gumbel_noise
    return torch.nn.functional.softmax(y / temperature, dim=-1)

sex_net = FFNetwork(input_dim, hidden_dims, len(sex_dict)).to(device).cuda()
age_net = FFNetwork(input_dim, hidden_dims, len(age_dict)).to(device).cuda()
ethnic_net = FFNetwork(input_dim, hidden_dims, len(ethnic_dict)).to(device).cuda()
religion_net = FFNetwork(input_dim, hidden_dims, len(religion_dict)).to(device).cuda()
marital_net = FFNetwork(input_dim, hidden_dims, len(marital_dict)).to(device).cuda()
qual_net = FFNetwork(input_dim, hidden_dims, len(qual_dict)).to(device).cuda()

input_tensor = torch.empty(total, input_dim).to(device)
init.kaiming_normal_(input_tensor)
    
ethnic_net_hh = FFNetwork(input_dim_hh, hidden_dims_hh, len(ethnic_dict_hh)).to(device).cuda()
religion_net_hh = FFNetwork(input_dim_hh, hidden_dims_hh, len(religion_dict_hh)).to(device).cuda()

input_tensor_hh = torch.empty(num_households, input_dim_hh).to(device)
init.kaiming_normal_(input_tensor_hh)
    
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
    
def generate_households(input_tensor_hh, temperature=0.5):    
    ethnicity_logits = ethnic_net_hh(input_tensor_hh)
    religion_logits = religion_net_hh(input_tensor_hh)
    
    ethnicity_hh = gumbel_softmax_sample(ethnicity_logits, temperature)
    religion_hh = gumbel_softmax_sample(religion_logits, temperature)
    # hh_comp = torch.zeros((sum(hh_comp_dict.values()), len(hh_comp_dict)), dtype=torch.float).to(device)
    hh_comp = torch.zeros((4852, len(hh_comp_dict)), dtype=torch.float).to(device)
    
    row_index = 0
    for key, value in sorted(hh_comp_dict_mod.items()):
        eye_matrix = torch.eye(len(hh_comp_dict))[key].to(device)
        hh_comp[row_index:row_index+value] = eye_matrix.repeat(value, 1)
        row_index += value
            
    return torch.cat([hh_comp, ethnicity_hh, religion_hh], dim=-1)

def aggregate(encoded_tensor, cross_table, category_dicts):
    # calculating split sizes based on category dictionaries
    split_sizes = [len(cat_dict) for cat_dict in category_dicts]
    # making sure that tensor dimension matches the total category count
    if encoded_tensor.size(1) != sum(split_sizes):
        raise ValueError("Size mismatch between encoded_tensor and category_dicts")

    # splitting the tensor into category-specific probabilities
    category_probs = torch.split(encoded_tensor, split_sizes, dim=1)

    # initializing the aggregated tensor
    aggregated_tensor = torch.zeros(len(cross_table.keys()), device=device)

    # aggregating the tensor based on the cross table
    for i, key in enumerate(cross_table.keys()):
        category_keys = key.split(' ')
        expected_count = torch.ones(encoded_tensor.size(0), device=device) # Initialize as a vector

        # multiplying probabilities across each category
        for cat_index, cat_key in enumerate(category_keys):
            category_index = list(category_dicts[cat_index].keys()).index(cat_key)
            expected_count *= category_probs[cat_index][:, category_index]

        # aggregating the expected counts
        aggregated_tensor[i] = torch.sum(expected_count)
    return aggregated_tensor

def decode_tensor(encoded_tensor, category_dicts):
    # calculating the split sizes from the category dictionaries
    split_sizes = [len(cat_dict) for cat_dict in category_dicts]

    # dynamic tensor splitting
    category_encoded_tensors = torch.split(encoded_tensor, split_sizes, dim=1)

    # decoding each category
    decoded_categories = []
    for cat_tensor, cat_dict in zip(category_encoded_tensors, category_dicts):
        cat_labels = list(cat_dict.keys())
        decoded_cat = [cat_labels[torch.argmax(ct).item()] for ct in cat_tensor]
        decoded_categories.append(decoded_cat)

    # combining the decoded categories
    return list(zip(*decoded_categories))

def keep_categories(encoded_tensor, category_lengths, categories_to_keep):
    # calculating the indices for the categories to be kept
    keep_indices = []
    current_index = 0
    for category, length in category_lengths.items():
        if category in categories_to_keep:
            indices = torch.arange(start=current_index, end=current_index + length, device=encoded_tensor.device)
            keep_indices.append(indices)
        current_index += length

    # concatenating all keep_indices and using them to index the tensor
    keep_indices = torch.cat(keep_indices, dim=0)
    kept_tensor = encoded_tensor[:, keep_indices]
    return kept_tensor

def plot(target, computed, cross_table, name):
    accuracy = rmse_accuracy(target.cpu(), computed.cpu())
    # creating bar plots for both dictionaries
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(cross_table.keys()),
        y=list(target.tolist()),
        name='Target',
        marker_color='#636EFA'
    ))
    fig.add_trace(go.Bar(
        x=list(cross_table.keys()),
        y=list(computed.tolist()),
        name='Computed',
        marker_color='#EF553B'
    ))

    fig.update_layout(
        title= name + ' [' + "RMSE:" + str(accuracy) + ']',
        xaxis_tickangle=-90,
        xaxis_title='Categories',
        yaxis_title='Counts',
        barmode='group',
        bargap=0.5,
        width=9000
    )

    fig.write_html(f"{name}.html")
    fig.show()
    
def plot_radar_triplets(target, computed, cross_table, name):
    accuracy = rmse_accuracy(target.cpu(), computed.cpu())
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r = list(computed.tolist()),
        theta=list(cross_table.keys()),
        name='Generated Population',
        line=dict(width=3)
    ))
    fig.add_trace(go.Scatterpolar(
        r = list(target.tolist()),
        theta=list(cross_table.keys()),
        name='Actual Population',
        line=dict(width=3)
    ))
    
    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          type='linear',
          gridcolor='black',
          linecolor='black'
        )),
      showlegend=True,
      width=1500,
      height=1500,
      margin=dict(l=500, r=500, t=500, b=500),
      font=dict(size=15)
    )
    
#     fig.write_html(f"{attribute}-radar-chart.html")
    fig.show()    

def rmse_accuracy(computed_tensor, target_tensor):
    mse = torch.mean((target_tensor - computed_tensor) ** 2)
    rmse = torch.sqrt(mse)
    max_possible_error = torch.sqrt(torch.sum(target_tensor ** 2))
    accuracy = 1 - (rmse / max_possible_error)
    return accuracy.item()

def rmse_loss(aggregated_tensor, target_tensor):
    return torch.sqrt(torch.mean((aggregated_tensor - target_tensor) ** 2))

def weights_init(m):
    if isinstance(m, nn.Linear):
        # nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)

def combined_rmse_loss(aggregated_tensor1, aggregated_tensor2, aggregated_tensor3, aggregated_tensor4, target_tensor1, target_tensor2, target_tensor3, target_tensor4):
    # concatenating the target and computed tensors along the characteristic dimension (dim=1)
    concatenated_tensor = torch.cat([target_tensor1, target_tensor2, target_tensor3, target_tensor4])
    aggregated_cat_tensor = torch.cat([aggregated_tensor1, aggregated_tensor2, aggregated_tensor3, aggregated_tensor4])
    # calculating RMSE loss on the concatenated tensor
    loss = torch.sqrt(torch.mean((aggregated_cat_tensor - concatenated_tensor) ** 2))
    return loss

def combined_rmse_loss_hh(aggregated_tensor1, aggregated_tensor2, target_tensor1, target_tensor2):
    # concatenating the target and computed tensors along the characteristic dimension (dim=1)
    concatenated_tensor = torch.cat([target_tensor1, target_tensor2])
    aggregated_cat_tensor = torch.cat([aggregated_tensor1, aggregated_tensor2])
    # calculating RMSE loss on the concatenated tensor
    loss = torch.sqrt(torch.mean((aggregated_cat_tensor - concatenated_tensor) ** 2))
    return loss

fixed_hh = {"1PE": 1, "1PA": 1, "1FM-0C": 2, "1FC-0C": 2}
three_or_more_hh = {'1FM-2C', '1FM-nA', '1FC-2C', '1FC-nA'}
two_or_more_hh = {'1FL-2C', '1FL-nA', '1H-2C'}

hh_size_dist_org = ID.getdictionary(ID.HHsizedf, area)
values_size_org, weights_size_org = zip(*hh_size_dist_org.items())
rk = ['1']
household_size_dist = {key: value for key, value in hh_size_dist_org.items() if key not in rk}
values_size, weights_size = zip(*household_size_dist.items())
rk = ['1', '2']
household_size_dist_na = {key: value for key, value in hh_size_dist_org.items() if key not in rk}
values_size_na, weights_size_na = zip(*household_size_dist_na.items())

def fit_household_size(composition):
    if composition in fixed_hh:
        return fixed_hh[composition]
    elif composition in three_or_more_hh:
        return int(random.choices(values_size_na, weights=weights_size_na)[0])
    elif composition in two_or_more_hh:
        return int(random.choices(values_size, weights=weights_size)[0])
    else:
        return int(random.choices(values_size_org, weights=weights_size_org)[0])
    

child_ages = ["0_4", "5_7", "8_9", "10_14", "15"]
adult_ages = ["16_17", "18_19", "20_24", "25_29", "30_34", "35_39", "40_44", "45_49", "50_54", "55_59", "60_64"]
elder_ages = ["65_69", "70_74", "75_79", "80_84", "85+"]

def extract_random_sample(persons, size):
    if size > len(persons):
        return []
    sampled_persons = persons['Person_ID'].sample(size).tolist()
    # sampled_persons_indices = sampled_persons.index
    # persons = persons.drop(sampled_persons_indices)
    return sampled_persons

def handle_composition(composition, size, persons):
    if composition == '1PE' or composition == '1FE':
        # Handle One person: Pensioner (person who is above 65)
        # Handle One family: All pensioner (a family consisting of persons all above 65)
        elders = persons[persons.apply(lambda row: row['age'] in elder_ages, axis=1)]
        return extract_random_sample(elders, size)

    elif composition == '1PA':
        # Handle One person: Other (a single person who is above 18 and below 65)
        adults = persons[persons.apply(lambda row: row['age'] in adult_ages, axis=1)]
        return extract_random_sample(adults, size)

    elif composition == '1FM-0C':
        # Handle One family: Married Couple: No children
        married_male = persons[persons.apply(lambda row: row['marital'] == 'Married' and row['sex'] == 'M', axis=1)]
        married_female = persons[persons.apply(lambda row: row['marital'] == 'Married' and row['sex'] == 'F', axis=1)]
        return extract_random_sample(married_male, 1) + extract_random_sample(married_female, 1)
    
    elif composition == '1FM-2C':
        # Handle One family: Married Couple: Having dependent children
        married_male = persons[persons.apply(lambda row: row['marital'] == 'Married' and row['sex'] == 'M', axis=1)]
        married_female = persons[persons.apply(lambda row: row['marital'] == 'Married' and row['sex'] == 'F', axis=1)]
        children = persons[persons.apply(lambda row: row['age'] in child_ages, axis=1)]
        return extract_random_sample(married_male, 1) + extract_random_sample(married_female, 1) + extract_random_sample(children, size-2)
    
    elif composition == '1FM-nA':
        # Handle One family: Married Couple: all children non-dependent
        married_male = persons[persons.apply(lambda row: row['marital'] == 'Married' and row['sex'] == 'M', axis=1)]
        married_female = persons[persons.apply(lambda row: row['marital'] == 'Married' and row['sex'] == 'F', axis=1)]
        children = persons[persons.apply(lambda row: row['age'] in adult_ages, axis=1)]
        return extract_random_sample(married_male, 1) + extract_random_sample(married_female, 1) + extract_random_sample(children, size-2)
    
    elif composition == '1FC-0C':
        # Handle One family: Cohabiting Couple: No children
        male = persons[persons.apply(lambda row: row['marital'] != 'Married' and row['sex'] == 'M', axis=1)]
        female = persons[persons.apply(lambda row: row['marital'] != 'Married' and row['sex'] == 'F', axis=1)]
        return extract_random_sample(male, 1) + extract_random_sample(female, 1)
    
    elif composition == '1FC-2C':
        # Handle One family: Cohabiting Couple: Having dependent children
        male = persons[persons.apply(lambda row: row['marital'] != 'Married' and row['sex'] == 'M', axis=1)]
        female = persons[persons.apply(lambda row: row['marital'] != 'Married' and row['sex'] == 'F', axis=1)]
        children = persons[persons.apply(lambda row: row['age'] in child_ages, axis=1)]
        return extract_random_sample(male, 1) + extract_random_sample(female, 1) + extract_random_sample(children, size-2)
    
    elif composition == '1FC-nA':
        # Handle One family: Cohabiting Couple: all children non-dependent
        male = persons[persons.apply(lambda row: row['marital'] != 'Married' and row['sex'] == 'M', axis=1)]
        female = persons[persons.apply(lambda row: row['marital'] != 'Married' and row['sex'] == 'F', axis=1)]
        children = persons[persons.apply(lambda row: row['age'] in adult_ages, axis=1)]
        return extract_random_sample(male, 1) + extract_random_sample(female, 1) + extract_random_sample(children, size-2)
    
    elif composition == '1FL-2C':
        # Handle One family: Lone Parent: Having dependent children
        parent = persons[persons.apply(lambda row: row['marital'] != 'Married', axis=1)]
        children = persons[persons.apply(lambda row: row['age'] in child_ages, axis=1)]
        return extract_random_sample(parent, 1) + extract_random_sample(children, size-1)
    
    elif composition == '1FL-nA':
        # Handle One family: Lone parent: all children non-dependent
        parent = persons[persons.apply(lambda row: row['marital'] != 'Married', axis=1)]
        children = persons[persons.apply(lambda row: row['age'] in adult_ages, axis=1)]
        return extract_random_sample(parent, 1) + extract_random_sample(children, size-1)
    
    elif composition == '1H-2C':
        # Handle Other households: Having dependent children
        adults = persons[persons.apply(lambda row: row['age'] in adult_ages, axis=1)]
        children = persons[persons.apply(lambda row: row['age'] in child_ages, axis=1)]
        return extract_random_sample(adults, 1) + extract_random_sample(children, size-1)
    
    elif composition == '1H-nA':
        # Handle Other households: All adults
        adults_and_children = persons[persons.apply(lambda row: row['age'] in adult_ages, axis=1)]
        return extract_random_sample(adults_and_children, size)
    
    elif composition == '1H-nE':
        # Handle Other households: All pensioners
        elders = persons[persons.apply(lambda row: row['age'] in elder_ages, axis=1)]
        return extract_random_sample(elders, size)
    
    elif composition == '1H-nS':
        # Handle Other households: All pensioners
        students = persons[persons.apply(lambda row: row['qualification'] != 'no', axis=1)]
        return extract_random_sample(students, size)
    
### Training / Plotting

loss_history = []
loss_history_hh = []

accuracy_history = []
accuracy_history_hh = []

# recording execution start time
start = time.time()

# training loop
optimizer = torch.optim.Adam([{'params': sex_net.parameters()},
                              {'params': age_net.parameters()},
                              {'params': ethnic_net.parameters()},
                              {'params': religion_net.parameters()},
                              {'params': marital_net.parameters()},
                              {'params': qual_net.parameters()}], lr=0.01)
scheduler = StepLR(optimizer, step_size=20, gamma=0.25)

optimizer_hh = torch.optim.Adam([{'params': ethnic_net_hh.parameters()},
                                 {'params': religion_net_hh.parameters()}], lr=0.0005)
scheduler_hh = StepLR(optimizer_hh, step_size=20, gamma=0.25)

sex_net.apply(weights_init)
age_net.apply(weights_init)
ethnic_net.apply(weights_init)
religion_net.apply(weights_init)
marital_net.apply(weights_init)
qual_net.apply(weights_init)

ethnic_net_hh.apply(weights_init)
religion_net_hh.apply(weights_init)

number_of_epochs = 300
for epoch in range(number_of_epochs+1):
    optimizer.zero_grad()
    optimizer_hh.zero_grad()

    # generating and aggregating encoded population for sex, age, ethnicity
    # generating and aggregating encoded households for reference religion, reference ethnicity, and composition
    encoded_population = generate_population(input_tensor)
    encoded_households = generate_households(input_tensor_hh)

    categories_to_keep = ['sex', 'age', 'ethnicity']
    kept_tensor = keep_categories(encoded_population, category_lengths, categories_to_keep)
    aggregated_population1 = aggregate(kept_tensor, cross_table1, [sex_dict, age_dict, ethnic_dict])

    categories_to_keep = ['sex', 'age', 'religion']
    kept_tensor = keep_categories(encoded_population, category_lengths, categories_to_keep)
    aggregated_population2 = aggregate(kept_tensor, cross_table2, [sex_dict, age_dict, religion_dict])

    categories_to_keep = ['sex', 'age', 'marital']
    kept_tensor = keep_categories(encoded_population, category_lengths, categories_to_keep)
    aggregated_population3 = aggregate(kept_tensor, cross_table3, [sex_dict, age_dict, marital_dict])

    categories_to_keep = ['sex', 'age', 'qual']
    kept_tensor = keep_categories(encoded_population, category_lengths, categories_to_keep)
    aggregated_population4 = aggregate(kept_tensor, cross_table4, [sex_dict, age_dict, qual_dict])

    categories_to_keep_hh = ['composition', 'ref_ethnicity']
    kept_tensor_hh = keep_categories(encoded_households, category_lengths_hh, categories_to_keep_hh)
    aggregated_households1 = aggregate(kept_tensor_hh, cross_table1_hh, [hh_comp_dict, ethnic_dict_hh])

    categories_to_keep_hh = ['composition', 'ref_religion']
    kept_tensor_hh = keep_categories(encoded_households, category_lengths_hh, categories_to_keep_hh)
    aggregated_households2 = aggregate(kept_tensor_hh, cross_table2_hh, [hh_comp_dict, religion_dict_hh])

    loss = combined_rmse_loss(aggregated_population1,
                              aggregated_population2,
                              aggregated_population3,
                              aggregated_population4,
                              cross_table_tensor1,
                              cross_table_tensor2,
                              cross_table_tensor3,
                              cross_table_tensor4)

    accuracy1 = rmse_accuracy(aggregated_population1, cross_table_tensor1)
    accuracy2 = rmse_accuracy(aggregated_population2, cross_table_tensor2)
    accuracy3 = rmse_accuracy(aggregated_population3, cross_table_tensor3)
    accuracy4 = rmse_accuracy(aggregated_population4, cross_table_tensor4)
    accuracy = (accuracy1 + accuracy2 + accuracy3 + accuracy4) / 4
    
    loss_hh = combined_rmse_loss_hh(aggregated_households1,
                                   aggregated_households2,
                                   cross_table_tensor1_hh,
                                   cross_table_tensor2_hh)

    accuracy1_hh = rmse_accuracy(aggregated_households1, cross_table_tensor1_hh)
    accuracy2_hh = rmse_accuracy(aggregated_households2, cross_table_tensor2_hh)
    accuracy_hh = (accuracy1_hh + accuracy2_hh) / 2

    loss_history.append(loss)
    accuracy_history.append(accuracy)
    
    loss_history_hh.append(loss_hh)
    accuracy_history_hh.append(accuracy_hh)

    loss.backward()
    optimizer.step()
    
    loss_hh.backward()
    optimizer_hh.step()
        
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Ind Loss: {loss}, Ind Accuracy: {accuracy}, HH Loss: {loss_hh}, HH Accuracy: {accuracy_hh}")


records = decode_tensor(encoded_population, [sex_dict, age_dict, ethnic_dict, religion_dict, marital_dict, qual_dict])
persons = pd.DataFrame(records, columns=['sex', 'age', 'ethnicity', 'religion', 'marital', 'qualification'])
persons['Person_ID'] = range(1, len(persons) + 1)

records_hh = decode_tensor(encoded_households, [hh_comp_dict, ethnic_dict_hh, religion_dict_hh])
households = pd.DataFrame(records_hh, columns=['composition', 'ref_ethnicity', 'ref_religion'])
households["size"] = households["composition"].apply(fit_household_size)
households['assigned_persons'] = [[] for _ in range(len(households))]

for i, household in households.iterrows():

    eth = household['ref_ethnicity']
    rel = household['ref_religion']
    comp = household['composition']
    size = household['size']
    
    persons_filtered = persons[(persons['ethnicity'] == eth) & (persons['religion'] == rel)]
    persons_to_assign = handle_composition(comp, int(size), persons_filtered)
    households.at[i, 'assigned_persons'] = persons_to_assign

aggregated_population1 = aggregated_population1.round().long().cuda()
aggregated_population2 = aggregated_population2.round().long().cuda()
aggregated_population3 = aggregated_population3.round().long().cuda()
aggregated_population4 = aggregated_population4.round().long().cuda()

aggregated_households1 = aggregated_households1.round().long().cuda()
aggregated_households2 = aggregated_households2.round().long().cuda()

plot(cross_table_tensor1, aggregated_population1, cross_table1, 'Age-Sex-Ethnicity')
plot(cross_table_tensor2, aggregated_population2, cross_table2, 'Age-Sex-Religion')
plot(cross_table_tensor3, aggregated_population3, cross_table3, 'Age-Sex-MaritalStatus')
plot(cross_table_tensor4, aggregated_population4, cross_table4, 'Age-Sex-Qualification')

plot_radar_triplets(cross_table_tensor1, aggregated_population1, cross_table1, 'Age-Sex-Ethnicity')
plot_radar_triplets(cross_table_tensor2, aggregated_population2, cross_table2, 'Age-Sex-Religion')
plot_radar_triplets(cross_table_tensor3, aggregated_population3, cross_table3, 'Age-Sex-MaritalStatus')
plot_radar_triplets(cross_table_tensor4, aggregated_population4, cross_table4, 'Age-Sex-Qualification')

plot(cross_table_tensor1_hh, aggregated_households1, cross_table1_hh, 'Household Composition By Ethnicity')
plot(cross_table_tensor2_hh, aggregated_households2, cross_table2_hh, 'Household Composition By Religion')

plot_radar_triplets(cross_table_tensor1_hh, aggregated_households1, cross_table1_hh, 'Household Composition By Ethnicity')
plot_radar_triplets(cross_table_tensor2_hh, aggregated_households2, cross_table2_hh, 'Household Composition By Religion')

loss_history = [tensor.to('cpu').item() for tensor in loss_history]
accuracy_history = [tensor for tensor in accuracy_history]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(len(loss_history)), loss_history, label='Training Loss')
plt.title('Training Loss Over Epochs For Individuals')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(len(accuracy_history)), accuracy_history, label='Training Accuracy')
plt.title('Training Accuracy Over Epochs For Individuals')
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('individual_generation_convergence.png')
plt.show()

loss_history_hh = [tensor.to('cpu').item() for tensor in loss_history_hh]
accuracy_history_hh = [tensor for tensor in accuracy_history_hh]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(len(loss_history_hh)), loss_history_hh, label='Training Loss')
plt.title('Training Loss Over Epochs For Households')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(len(accuracy_history_hh)), accuracy_history_hh, label='Training Accuracy')
plt.title('Training Accuracy Over Epochs For Households')
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('household_generation_convergence.png')
plt.show()

age_categories_to_single = ["0_4", "5_7", "8_9", "10_14", "15"]
persons.loc[persons['age'].isin(age_categories_to_single), 'marital'] = 'Single'
persons.loc[persons['age'].isin(age_categories_to_single), 'qualification'] = 'no'
persons.to_csv('synthetic_population.csv', index=False)

households.to_csv('synthetic_households.csv', index=False)

# recording execution end time
end = time.time()
duration = end - start

# converting the recording time to hours, minutes, and seconds
hours = int(duration // 3600)
minutes = int((duration % 3600) // 60)
seconds = duration % 60
print(f"Duration: {hours} hours, {minutes} minutes, {seconds:.2f} seconds")

persons_df_copy = persons.copy()

import plotly.graph_objects as go
import math

def plot_radar(attribute, attribute_dict, show):
    
    categories_gen = persons_df[attribute].unique().astype(str).tolist()    
    count_gen = [(persons_df[attribute] == str).sum() for str in categories_gen]
    categories, count_act = list(attribute_dict.keys()), list(attribute_dict.values())

    gen_combined = list(zip(categories_gen, count_gen))
    gen_combined = sorted(gen_combined, key=lambda x: categories.index(x[0]))
    categories_gen, count_gen = zip(*gen_combined)
    count_gen = list(count_gen)   
    range = (0, ((max(count_act) + max(count_gen)) / 2) * 1.1)
    
    count_act = [max(val, 10) for val in count_act]
    count_gen = [max(val, 10) for val in count_gen] 
    
    squared_errors = [(actual - predicted) ** 2 for actual, predicted in zip(count_act, count_gen)]
    mean_squared_error = sum(squared_errors) / len(count_act)
    rmse = math.sqrt(mean_squared_error)
    max_possible_error = math.sqrt(sum(x**2 for x in count_act))
    accuracy = 1 - (rmse / max_possible_error)
    
    categories.append(categories[0])
    count_act.append(count_act[0])
    count_gen.append(count_gen[0])

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r = count_gen,
        theta=categories,
        name='Generated Population',
        line=dict(width=10)
    ))
    fig.add_trace(go.Scatterpolar(
        r = count_act,
        theta=categories,
        name='Actual Population',
        line=dict(width=10)
    ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          type='log',
          tickvals=[10, 100, 1000],
          tickmode='array'
        )),
      showlegend=True,
      width=2500,
      height=2500,
      margin=dict(l=500, r=500, t=500, b=500),
      font=dict(size=65)
    )
    
    fig.write_html(f"{attribute}-radar-chart.html")
    fig.show() if show == 'yes' else None

# note: set show to 'yes' for displaying the plot
plot_radar('age', age_dict, show='yes')
plot_radar('religion', religion_dict, show='yes')
plot_radar('ethnicity', ethnic_dict, show='yes')
plot_radar('marital', marital_dict, show='yes')
plot_radar('qualification', qual_dict, show='yes')

households_df_plot = households.copy(deep=True)
households_df_plot.loc[(households_df_plot['composition'].str.contains('1P')), 'composition'] = '1P'
households_df_plot.loc[(households_df_plot['composition'].str.contains('1H')), 'composition'] = '1H'
households_df_plot.loc[(households_df_plot['composition'].str.contains('1F')), 'composition'] = '1F'

comp_list = households_df_plot['composition'].unique().astype(str).tolist()
ethnic_dict = ID.getdictionary(ID.ethnicdf, area)
ethnic_list = [str(k) for k in ethnic_dict.keys()]
combinations = pd.DataFrame(list(itertools.product(comp_list, ethnic_list)), columns=['composition', 'ref_ethnicity'])
counts_df = households_df_plot.groupby(['composition', 'ref_ethnicity']).size().reset_index(name='counts')
comp_ethnic_df_gen = combinations.merge(counts_df, on=['composition', 'ref_ethnicity'], how='left').fillna(0)
comp_ethnic_df_gen['category'] = comp_ethnic_df_gen.apply(lambda row: '{} {}'.format(row['composition'], row['ref_ethnicity']), axis=1)
comp_ethnic_df_gen.drop(['composition', 'ref_ethnicity'], axis=1, inplace=True)
comp_ethnic_df_gen['counts'] = comp_ethnic_df_gen['counts'].astype(int)

comp_list = households_df_plot['composition'].unique().astype(str).tolist()
religion_dict = ID.getdictionary(ID.religiondf, area)
rel_list = [str(k) for k in religion_dict.keys()]
combinations = pd.DataFrame(list(itertools.product(comp_list, rel_list)), columns=['composition', 'ref_religion'])
counts_df = households_df_plot.groupby(['composition', 'ref_religion']).size().reset_index(name='counts')
comp_rel_df_gen = combinations.merge(counts_df, on=['composition', 'ref_religion'], how='left').fillna(0)
comp_rel_df_gen['category'] = comp_rel_df_gen.apply(lambda row: '{} {}'.format(row['composition'], row['ref_religion']), axis=1)
comp_rel_df_gen.drop(['composition', 'ref_religion'], axis=1, inplace=True)
comp_rel_df_gen['counts'] = comp_rel_df_gen['counts'].astype(int)

cats = ["1P", "1F", "1H"]
cross_table1_hh = ICT.getdictionary(ICT.HH_composition_by_Ethnicity, area)
cross_table2_hh = ICT.getdictionary(ICT.HH_composition_by_Religion, area)

new_dict = {}
for key, value in cross_table1_hh.items():
    for cat in cats:
        if key.startswith(cat):
            key_mod = f"{cat} " + key.split(" ")[1]
            if key_mod in new_dict:
                new_dict[key_mod] += value
            else:
                new_dict[key_mod] = value 
comp_ethnic_df_act = pd.DataFrame({'category': new_dict.keys(), 'counts': new_dict.values()})

new_dict = {}
for key, value in cross_table2_hh.items():
    for cat in cats:
        if key.startswith(cat):
            key_mod = f"{cat} " + key.split(" ")[1]
            if key_mod in new_dict:
                new_dict[key_mod] += value
            else:
                new_dict[key_mod] = value    
comp_rel_df_act = pd.DataFrame({'category': new_dict.keys(), 'counts': new_dict.values()})

import plotly.graph_objects as go

def normalize_list(lst):
    min_val = min(lst)
    max_val = max(lst)
    normalized_lst = [(x - min_val) / (max_val - min_val) for x in lst]
    return normalized_lst

def plot_radar_hh(attribute, act_df, gen_df, show):
    
    categories_gen = gen_df["category"].astype(str).tolist()
    count_gen = gen_df["counts"].tolist()
    categories = act_df["category"].astype(str).tolist()
    count_act = act_df["counts"].tolist()

    gen_combined = list(zip(categories_gen, count_gen))
    gen_combined = sorted(gen_combined, key=lambda x: categories.index(x[0]))
    categories_gen, count_gen = zip(*gen_combined)
    count_gen = list(count_gen)   
    range = (0, ((max(count_act) + max(count_gen)) / 2) * 1.1)
    
    count_act = [max(val, 10) for val in count_act]
    count_gen = [max(val, 10) for val in count_gen]

    squared_errors = [(actual - predicted) ** 2 for actual, predicted in zip(count_act, count_gen)]
    mean_squared_error = sum(squared_errors) / len(count_act)
    rmse = math.sqrt(mean_squared_error)
    max_possible_error = math.sqrt(sum(x**2 for x in count_act))
    accuracy = 1 - (rmse / max_possible_error)
    
    categories.append(categories[0])
    count_act.append(count_act[0])
    count_gen.append(count_gen[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r = count_act,
        theta=categories,
        name='Actual Households',
        line=dict(width=10)
    ))
    
    fig.add_trace(go.Scatterpolar(
        r = count_gen,
        theta=categories,
        name='Generated Households',
        line=dict(width=10)
    ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
            visible=True,
            type='log',
          showline=False,
          showticklabels=False,
          gridcolor='black'
        )),
      showlegend=True,
      width=2500,
      height=2500,
      margin=dict(l=500, r=500, t=500, b=500),
      font=dict(size=65)
    )
    
    fig.write_html(f"comp-{attribute}-radar-chart.html")
    fig.show() if show == 'yes' else None

# note: set show to 'yes' for displaying the plot
plot_radar_hh('ethnicity', comp_ethnic_df_act, comp_ethnic_df_gen, show='yes')
plot_radar_hh('religion', comp_rel_df_act, comp_rel_df_gen, show='yes')