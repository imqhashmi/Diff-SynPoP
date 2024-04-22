# IMPORTING LIBRARIES

import sys

# appending the system path to run the file on kaggle
# not required if you are running it locally
# sys.path.insert(1, '/kaggle/input/diffspop/Diff-SynPoP')

import os
import time
import random
import numpy as np
import plotly as py
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import torch
import torch.nn as nn
import torch.optim as optim

# importing InputData and InputCrossTables for processing UK census data files
import InputData as ID
import InputCrossTables as ICT

from torch.nn import init
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ExponentialLR

# FEED FORWARD NEURAL NETWORK

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

# Staging 1: Individual Generation
# DICTIONARIES, CROSS TABLES, & INPUT TENSOR (INDIVIDUALS)

device = 'cuda' if torch.cuda.is_available() else 'cpu' # checking to see if a cuda device is available 
path = os.path.join(os.path.dirname(os.getcwd()), 'Diff-SynPoP')

#MSOA
area = 'E02005924' # geography code for one of the oxford areas (selected for this work)
total = ID.get_total(ID.age5ydf, area) # getting the total number of individuals in our MSOA

# getting the distributions of individual attributes in the selected MSOA
# saving the extracted distributions into respective dictionaries
sex_dict = ID.getdictionary(ID.sexdf, area) # sex
age_dict = ID.getdictionary(ID.age5ydf, area) # age
ethnic_dict = ID.getdictionary(ID.ethnicdf, area) # ethnicity
religion_dict = ID.getdictionary(ID.religiondf, area) # religion
marital_dict = ID.getdictionary(ID.maritaldf, area) # marital status
qual_dict = ID.getdictionary(ID.qualdf, area) # highest qualification level
# hh_comp_dict = ID.getHHcomdictionary(ID.HHcomdf, area) # household composition
# age_hh_dict = {'0_15': 2110, '16_24': 1360, '25_34': 2441, '35_49': 2388, '50+': 2581} # dictionary for mapped ages for households

# getting the length (number of classes) for each attribute
category_lengths = {
    'sex': len(sex_dict),
    'age': len(age_dict),
    'ethnicity': len(ethnic_dict),
    'religion': len(religion_dict),
    'marital': len(marital_dict),
    'qual': len(qual_dict)
#     'hh_comp': len(hh_comp_dict)
#     'age_hh': len(age_hh_dict)
}

# print(category_lengths)

cross_table1 = ICT.getdictionary(ICT.ethnic_by_sex_by_age, area)
cross_table2 = ICT.getdictionary(ICT.religion_by_sex_by_age, area)
cross_table3 = ICT.getdictionary(ICT.marital_by_sex_by_age, area)
cross_table4 = ICT.getdictionary(ICT.qualification_by_sex_by_age, area)
# cross_table5 = ICT.getdictionary(ICT.HH_composition_by_sex_by_age, area)

cross_table_tensor1 = torch.tensor(list(cross_table1.values()), dtype=torch.float32).to(device).cuda()
cross_table_tensor2 = torch.tensor(list(cross_table2.values()), dtype=torch.float32).to(device).cuda()
cross_table_tensor3 = torch.tensor(list(cross_table3.values()), dtype=torch.float32).to(device).cuda()
cross_table_tensor4 = torch.tensor(list(cross_table4.values()), dtype=torch.float32).to(device).cuda()
# cross_table_tensor5 = torch.tensor(list(cross_table5.values()), dtype=torch.float32).to(device).cuda()

# instantiating networks for each characteristic
input_dim = sum(len(d.keys()) for d in [sex_dict, age_dict, ethnic_dict, religion_dict, marital_dict, qual_dict])
print("Input dimension: ", input_dim)
hidden_dims = [64, 32]

sex_net = FFNetwork(input_dim, hidden_dims, len(sex_dict)).to(device).cuda()
age_net = FFNetwork(input_dim, hidden_dims, len(age_dict)).to(device).cuda()
ethnic_net = FFNetwork(input_dim, hidden_dims, len(ethnic_dict)).to(device).cuda()
religion_net = FFNetwork(input_dim, hidden_dims, len(religion_dict)).to(device).cuda()
marital_net = FFNetwork(input_dim, hidden_dims, len(marital_dict)).to(device).cuda()
qual_net = FFNetwork(input_dim, hidden_dims, len(qual_dict)).to(device).cuda()
# hh_comp_net = FFNetwork(input_dim, hidden_dims, len(hh_comp_dict)).to(device).cuda()
# age_hh_net = FFNetwork(input_dim, hidden_dims, len(age_hh_dict)).to(device).cuda()

# input for the networks
# input_tensor = torch.randn(total, input_dim).to(device)  # Random noise as input, adjust as necessary

input_tensor = torch.empty(total, input_dim).to(device)
init.kaiming_normal_(input_tensor)

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
#     hh_comp_logits = hh_comp_net(input_tensor)
#     age_hh_logits = age_hh_net(input_tensor)
    
    sex = gumbel_softmax_sample(sex_logits, temperature)
    age = gumbel_softmax_sample(age_logits, temperature)
    ethnicity = gumbel_softmax_sample(ethnicity_logits, temperature)
    religion = gumbel_softmax_sample(religion_logits, temperature)
    marital = gumbel_softmax_sample(marital_logits, temperature)
    qual = gumbel_softmax_sample(qual_logits, temperature)

    return torch.cat([sex, age, ethnicity, religion, marital, qual], dim=-1)
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

# encoded_population = generate_population(input_tensor).cuda()
# records = decode_tensor(encoded_population, [sex_dict, age_dict, ethnic_dict, religion_dict, marital_dict, qual_dict])
# print(records)
# categories_to_keep = ['sex', 'age', 'marital']  # Categories to keep
# kept_tensor = keep_categories(encoded_population, category_lengths, categories_to_keep)
# aggregated_tensor = aggregate(kept_tensor, cross_table3, [sex_dict, age_dict, marital_dict])

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

# generated_population = generate_population(input_tensor)
# records = decode_tensor(generated_population, [sex_dict, age_dict, ethnic_dict, religion_dict, marital_dict, qual_dict])
# df = pd.DataFrame(records, columns=['sex', 'age', 'ethnicity', 'religion', 'marital', 'qualification'])
# print(df)

# TRAINING / PLOTTING (INDIVIDUAL GENERATION)

loss_history = []
accuracy_history = []

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

sex_net.apply(weights_init)
age_net.apply(weights_init)
ethnic_net.apply(weights_init)
religion_net.apply(weights_init)
marital_net.apply(weights_init)
qual_net.apply(weights_init)
# hh_comp_net.apply(weights_init)

number_of_epochs = 200
for epoch in range(number_of_epochs+1):
    optimizer.zero_grad()

    # generating and aggregating encoded population for sex, age, ethnicity
    encoded_population = generate_population(input_tensor)
    
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
    
#     categories_to_keep = ['sex', 'age_hh', 'hh_comp']
#     kept_tensor = keep_categories(encoded_population, category_lengths, categories_to_keep)
#     aggregated_population5 = aggregate(kept_tensor, cross_table5, [sex_dict, age_hh_dict, hh_comp_dict])

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
    
    loss_history.append(loss)
    accuracy_history.append(accuracy)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}, Accuracy: {accuracy}")
        
aggregated_population1 = aggregated_population1.round().long().cuda()
aggregated_population2 = aggregated_population2.round().long().cuda()
aggregated_population3 = aggregated_population3.round().long().cuda()
aggregated_population4 = aggregated_population4.round().long().cuda()
# aggregated_population5 = aggregated_population5.round().long().cuda()

loss_history = [tensor.to('cpu').item() for tensor in loss_history]
accuracy_history = [tensor for tensor in accuracy_history]

records = decode_tensor(encoded_population, [sex_dict, age_dict, ethnic_dict, religion_dict, marital_dict, qual_dict])
persons_df = pd.DataFrame(records, columns=['sex', 'age', 'ethnicity', 'religion', 'marital', 'qualification'])
age_categories_to_single = ["0_4", "5_7", "8_9", "10_14", "15"]
persons_df.loc[persons_df['age'].isin(age_categories_to_single), 'marital'] = 'Single'
persons_df.loc[persons_df['age'].isin(age_categories_to_single), 'qualification'] = 'no'
persons_df['Person_ID'] = range(1, len(persons_df) + 1) # assigning a person ID to each row
persons_df.to_csv('synthetic_population.csv', index=False)

# recording execution end time
end = time.time()
duration = end - start

# converting the recordind time to hours, minutes, and seconds
hours = int(duration // 3600)
minutes = int((duration % 3600) // 60)
seconds = duration % 60
print(f"Duration: {hours} hours, {minutes} minutes, {seconds:.2f} seconds")

# Stage 2: Household Generation
# DICTIONARIES, CROSS TABLES, & INPUT TENSOR (HOUSEHOLDS)
# Generating empty households with household size,and reference person ethnicity and religion

device = 'cuda' if torch.cuda.is_available() else 'cpu' # checking to see if a cuda device is available

area = 'E02005924' # geography code for one of the oxford areas (selected for this work)
num_households = ID.get_HH_com_total(ID.HHcomdf, area)
print("Total number of households: ", num_households)
print()

hh_comp_dict = ID.getHHcomdictionary(ID.HHcomdf, area) # household composition
hh_comp_dict_mod = {index: value for index, (_, value) in enumerate(hh_comp_dict.items())}
ethnic_dict_hh = ID.getdictionary(ID.ethnicdf, area) # ethnicity of reference person of a household
religion_dict_hh = ID.getdictionary(ID.religiondf, area) # religion of reference person of a household
print("Household Composition Dictionary: ", hh_comp_dict)
print()

category_lengths_hh = {
    'composition': len(hh_comp_dict),
    'ref_ethnicity': len(ethnic_dict_hh),
    'ref_religion': len(religion_dict_hh)
}

cross_table1_hh = ICT.getdictionary(ICT.HH_composition_by_Ethnicity, area)
cross_table2_hh = ICT.getdictionary(ICT.HH_composition_by_Religion, area)

cross_table_tensor1_hh = torch.tensor(list(cross_table1_hh.values()), dtype=torch.float32).to(device).cuda()
cross_table_tensor2_hh = torch.tensor(list(cross_table2_hh.values()), dtype=torch.float32).to(device).cuda()

input_dim = sum(len(d.keys()) for d in [ethnic_dict_hh, religion_dict_hh])
# print("Input dimension: ", input_dim)
hidden_dims = [64, 32]

# hh_comp_net = FFNetwork(input_dim, hidden_dims, len(hh_comp_dict)).to(device).cuda()
ethnic_net_hh = FFNetwork(input_dim, hidden_dims, len(ethnic_dict_hh)).to(device).cuda()
religion_net_hh = FFNetwork(input_dim, hidden_dims, len(religion_dict_hh)).to(device).cuda()

input_tensor_hh = torch.empty(num_households, input_dim).to(device)
init.kaiming_normal_(input_tensor_hh)

def generate_households(input_tensor, temperature=0.5):
    ethnicity_logits = ethnic_net_hh(input_tensor)
    religion_logits = religion_net_hh(input_tensor)
    
    ethnicity_hh = gumbel_softmax_sample(ethnicity_logits, temperature)
    religion_hh = gumbel_softmax_sample(religion_logits, temperature)
    hh_comp = torch.zeros((sum(hh_comp_dict.values()), len(hh_comp_dict)), dtype=torch.float).to(device)
    
    row_index = 0
    for key, value in sorted(hh_comp_dict_mod.items()):
        eye_matrix = torch.eye(len(hh_comp_dict))[key].to(device)
        hh_comp[row_index:row_index+value] = eye_matrix.repeat(value, 1)
        row_index += value
            
    return torch.cat([hh_comp, ethnicity_hh, religion_hh], dim=-1)

def combined_rmse_loss_hh(aggregated_tensor1, aggregated_tensor2, target_tensor1, target_tensor2):
    # concatenating the target and computed tensors along the characteristic dimension (dim=1)
    concatenated_tensor = torch.cat([target_tensor1, target_tensor2])
    aggregated_cat_tensor = torch.cat([aggregated_tensor1, aggregated_tensor2])
    # calculating RMSE loss on the concatenated tensor
    loss = torch.sqrt(torch.mean((aggregated_cat_tensor - concatenated_tensor) ** 2))
    return loss

generated_households = generate_households(input_tensor_hh)
records_hh = decode_tensor(generated_households, [hh_comp_dict, ethnic_dict_hh, religion_dict_hh])
df = pd.DataFrame(records_hh, columns=['composition', 'ref_ethnicity', 'ref_religion'])
print(df)

loss_history = []
accuracy_history = []

# recording execution start time
start = time.time()

# training loop
optimizer = torch.optim.Adam([{'params': ethnic_net_hh.parameters()},
                              {'params': religion_net_hh.parameters()}], lr=0.001)
scheduler = StepLR(optimizer, step_size=20, gamma=0.25)

ethnic_net_hh.apply(weights_init)
religion_net_hh.apply(weights_init)

number_of_epochs = 500
for epoch in range(number_of_epochs+1):
    optimizer.zero_grad()

    # generating and aggregating encoded households for reference religion, reference ethnicity, and composition
    encoded_households = generate_households(input_tensor_hh)
    
    categories_to_keep_hh = ['composition', 'ref_ethnicity']
    kept_tensor_hh = keep_categories(encoded_households, category_lengths_hh, categories_to_keep_hh)
    aggregated_households1 = aggregate(kept_tensor_hh, cross_table1_hh, [hh_comp_dict, ethnic_dict_hh])

    categories_to_keep_hh = ['composition', 'ref_religion']
    kept_tensor_hh = keep_categories(encoded_households, category_lengths_hh, categories_to_keep_hh)
    aggregated_households2 = aggregate(kept_tensor_hh, cross_table2_hh, [hh_comp_dict, religion_dict_hh])

    loss = combined_rmse_loss_hh(aggregated_households1,
                                 aggregated_households2,
                                 cross_table_tensor1_hh,
                                 cross_table_tensor2_hh)

    accuracy1 = rmse_accuracy(aggregated_households1, cross_table_tensor1_hh)
    accuracy2 = rmse_accuracy(aggregated_households2, cross_table_tensor2_hh)
    accuracy = (accuracy1 + accuracy2) / 2
    
    loss_history.append(loss)
    accuracy_history.append(accuracy)

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss}, Accuracy: {accuracy}")
        
aggregated_households1 = aggregated_households1.round().long().cuda()
aggregated_households2 = aggregated_households2.round().long().cuda()

loss_history = [tensor.to('cpu').item() for tensor in loss_history]
accuracy_history = [tensor for tensor in accuracy_history]

records_hh = decode_tensor(encoded_households, [hh_comp_dict, ethnic_dict_hh, religion_dict_hh])
households_df = pd.DataFrame(records_hh, columns=['composition', 'ref_ethnicity', 'ref_religion'])
households_df['household_ID'] = range(1, len(households_df) + 1) # assigning a household ID to each row

# recording execution end time
end = time.time()
duration = end - start

# converting the recordind time to hours, minutes, and seconds
hours = int(duration // 3600)
minutes = int((duration % 3600) // 60)
seconds = duration % 60
print(f"Duration: {hours} hours, {minutes} minutes, {seconds:.2f} seconds")

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

households_df["size"] = households_df["composition"].apply(fit_household_size)
households_df['assigned_persons'] = [[] for _ in range(len(households_df))]

# individuals are in persons_df
# households are in households_df
# Stage 3: Assigning Individuals to Households
# Assigning individuals to households based on the generated synthetic population and households

child_ages = ["0_4", "5_7", "8_9", "10_14", "15"]
adult_ages = ["16_17", "18_19", "20_24", "25_29", "30_34", "35_39", "40_44", "45_49", "50_54", "55_59", "60_64"]
elder_ages = ["65_69", "70_74", "75_79", "80_84", "85+"]

inter_ethnic_ratio = 0.1
inter_rel_ratio = 0.015

# Composition # 1:'1PE' One person: Pensioner (person who is above 65)

# Composition # 2:'1PA' One person: Other (a single person who is above 18 and below 65)
        
# Composition # 3:'1FM-0C' One family: Married Couple: No children
                
# Composition # 4:'1FC-0C' One family: Cohabiting Couple: No children

# Composition # 5:'1FE' One family: All pensioner (a family consisting of persons all above 65)
            
# Composition # 6:'1FM-2C' One family: Married Couple: Having dependent children
                
# Composition # 7:'1FC-2C' One family: Cohabiting Couple: Having dependent children
                
# Composition # 8:'1FL-2C' One family: Lone Parent: Having dependent children
            
# Composition # 9:'1FM-nA' One family: Married Couple: all children non-dependent
                
# Composition # 10:'1FC-nA' One family: Cohabiting Couple: all children non-dependent
                
# Composition # 11:'1FL-nA' One family: Lone parent: all children non-dependent
            
# Composition # 12:'1H-2C' Other households: Having dependent children
            
# Composition # 13:'1H-nS' Other households: All student
        
# Composition # 14:'1H-nE' Other households: All pensioners

# Composition # 15:'1H-nA' Other households: All adults

