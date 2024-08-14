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
income_dict = ID.income_df # income

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
# weekly_income_dict = ID.getHHDictionary(ID.weekly_income_df, area) # total and net weekly income

# getting the length (number of classes) for each attribute
category_lengths = {
    'sex': len(sex_dict),
    'age': len(age_dict),
    'ethnicity': len(ethnic_dict),
    'religion': len(religion_dict),
    'marital': len(marital_dict),
    'qual': len(qual_dict),
    'income': len(income_dict),
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

cross_table_tensor6 = torch.tensor(list(cross_table6.values()), dtype=torch.float32).to(device).cuda()
cross_table_tensor7 = torch.tensor(list(cross_table7.values()), dtype=torch.float32).to(device).cuda()
cross_table_tensor8 = torch.tensor(list(cross_table8.values()), dtype=torch.float32).to(device).cuda()
cross_table_tensor9 = torch.tensor(list(cross_table9.values()), dtype=torch.float32).to(device).cuda()
cross_table_tensor10 = torch.tensor(list(cross_table10.values()), dtype=torch.float32).to(device).cuda()
cross_table_tensor11 = torch.tensor(list(cross_table11.values()), dtype=torch.float32).to(device).cuda()

cross_table_tensor1_hh = torch.tensor(list(cross_table1_hh.values()), dtype=torch.float32).to(device).cuda()
cross_table_tensor2_hh = torch.tensor(list(cross_table2_hh.values()), dtype=torch.float32).to(device).cuda()
cross_table_tensor3_hh = torch.tensor(list(cross_table3_hh.values()), dtype=torch.float32).to(device).cuda()

input_dim = sum(len(d.keys()) for d in [sex_dict, age_dict, ethnic_dict, religion_dict, marital_dict, qual_dict, seg_dict, occupation_dict, economic_act_dict, approx_social_grade_dict, general_health_dict, industry_dict])
hidden_dims = [64, 32]

input_dim_hh = sum(len(d.keys()) for d in [ethnic_dict_hh, religion_dict_hh, car_ownership_dict])
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
income_net = FFNetwork(input_dim, hidden_dims, len(income_dict)).to(device).cuda()
seg_net = FFNetwork(input_dim, hidden_dims, len(seg_dict)).to(device).cuda()
occupation_net = FFNetwork(input_dim, hidden_dims, len(occupation_dict)).to(device).cuda()
economic_act_net = FFNetwork(input_dim, hidden_dims, len(economic_act_dict)).to(device).cuda()
approx_social_grade_net = FFNetwork(input_dim, hidden_dims, len(approx_social_grade_dict)).to(device).cuda()
general_health_net = FFNetwork(input_dim, hidden_dims, len(general_health_dict)).to(device).cuda()
industry_net = FFNetwork(input_dim, hidden_dims, len(industry_dict)).to(device).cuda()

input_tensor = torch.empty(total, input_dim).to(device)
init.kaiming_normal_(input_tensor)
    
ethnic_net_hh = FFNetwork(input_dim_hh, hidden_dims_hh, len(ethnic_dict_hh)).to(device).cuda()
religion_net_hh = FFNetwork(input_dim_hh, hidden_dims_hh, len(religion_dict_hh)).to(device).cuda()
car_ownership_net_hh = FFNetwork(input_dim_hh, hidden_dims_hh, len(car_ownership_dict)).to(device).cuda()

input_tensor_hh = torch.empty(num_households, input_dim_hh).to(device)
init.kaiming_normal_(input_tensor_hh)

def generate_population(input_tensor, temperature=0.5):    
    sex_logits = sex_net(input_tensor)
    age_logits = age_net(input_tensor)
    ethnicity_logits = ethnic_net(input_tensor)
    religion_logits = religion_net(input_tensor)
    marital_logits = marital_net(input_tensor)
    qual_logits = qual_net(input_tensor)
    income_logits = income_net(input_tensor)
    seg_logits = seg_net(input_tensor)
    occupation_logits = occupation_net(input_tensor)
    economic_act_logits = economic_act_net(input_tensor)
    approx_social_grade_logits = approx_social_grade_net(input_tensor)    
    general_health_logits = general_health_net(input_tensor)
    industry_logits = industry_net(input_tensor)

    sex = gumbel_softmax_sample(sex_logits, temperature)
    age = gumbel_softmax_sample(age_logits, temperature)
    ethnicity = gumbel_softmax_sample(ethnicity_logits, temperature)
    religion = gumbel_softmax_sample(religion_logits, temperature)
    marital = gumbel_softmax_sample(marital_logits, temperature)
    qual = gumbel_softmax_sample(qual_logits, temperature)
    income = gumbel_softmax_sample(income_logits, temperature)
    seg = gumbel_softmax_sample(seg_logits, temperature)
    occupation = gumbel_softmax_sample(occupation_logits, temperature)
    economic_act = gumbel_softmax_sample(economic_act_logits, temperature)
    approx_social_grade = gumbel_softmax_sample(approx_social_grade_logits, temperature)    
    general_health = gumbel_softmax_sample(general_health_logits, temperature)
    industry = gumbel_softmax_sample(industry_logits, temperature)

    return torch.cat([sex, age, ethnicity, religion, marital, qual, income, seg, occupation, economic_act, approx_social_grade, general_health, industry], dim=-1)
    
def generate_households(input_tensor_hh, temperature=0.5):    
    ethnicity_logits = ethnic_net_hh(input_tensor_hh)
    religion_logits = religion_net_hh(input_tensor_hh)
    car_ownership_logits = car_ownership_net_hh(input_tensor_hh)
    
    ethnicity_hh = gumbel_softmax_sample(ethnicity_logits, temperature)
    religion_hh = gumbel_softmax_sample(religion_logits, temperature)
    car_ownership_hh = gumbel_softmax_sample(car_ownership_logits, temperature)
    
    # hh_comp = torch.zeros((sum(hh_comp_dict.values()), len(hh_comp_dict)), dtype=torch.float).to(device)
    hh_comp = torch.zeros((4852, len(hh_comp_dict)), dtype=torch.float).to(device)
    
    row_index = 0
    for key, value in sorted(hh_comp_dict_mod.items()):
        eye_matrix = torch.eye(len(hh_comp_dict))[key].to(device)
        hh_comp[row_index:row_index+value] = eye_matrix.repeat(value, 1)
        row_index += value
            
    return torch.cat([hh_comp, ethnicity_hh, religion_hh, car_ownership_hh], dim=-1)

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

def rmse_accuracy(computed_tensor, target_tensor):
    mse = torch.mean((target_tensor - computed_tensor) ** 2)
    rmse = torch.sqrt(mse)
    max_possible_error = torch.sqrt(torch.sum(target_tensor ** 2))
    accuracy = 1 - (rmse / max_possible_error)
    return accuracy.item()



training = False

P = generate_population(input_tensor)
print(decode_tensor(P, [sex_dict, age_dict, ethnic_dict, religion_dict, marital_dict, qual_dict, income_dict, seg_dict, occupation_dict, economic_act_dict, approx_social_grade_dict, general_health_dict, industry_dict]))

categories_to_keep = ['seg']
kept_tensor = keep_categories(P, category_lengths, categories_to_keep)
aggregated_population6 = aggregate(kept_tensor, cross_table6, [seg_dict])
print(aggregated_population6)

def rmse_loss(aggregated_tensor, target_tensor):
    return torch.sqrt(torch.mean((aggregated_tensor - target_tensor) ** 2))

def weights_init(m):
    if isinstance(m, nn.Linear):
        # nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)

def combined_rmse_loss(aggregated_tensor1,
                       aggregated_tensor2,
                       aggregated_tensor3,
                       aggregated_tensor4,
                       aggregated_tensor6,
                       aggregated_tensor7,
                       aggregated_tensor8,
                       aggregated_tensor9,
                       aggregated_tensor10,
                       aggregated_tensor11,
                       target_tensor1,
                       target_tensor2,
                       target_tensor3,
                       target_tensor4,
                       target_tensor6,
                       target_tensor7,
                       target_tensor8,
                       target_tensor9,
                       target_tensor10,
                       target_tensor11):
    # concatenating the target and computed tensors along the characteristic dimension (dim=1)
    concatenated_tensor = torch.cat([target_tensor1, target_tensor2, target_tensor3, target_tensor4, target_tensor6, target_tensor7, target_tensor8, target_tensor9, target_tensor10, target_tensor11])
    aggregated_cat_tensor = torch.cat([aggregated_tensor1, aggregated_tensor2, aggregated_tensor3, aggregated_tensor4, aggregated_tensor6, aggregated_tensor7, aggregated_tensor8, aggregated_tensor9, aggregated_tensor10, aggregated_tensor11])
    # calculating RMSE loss on the concatenated tensor
    loss = torch.sqrt(torch.mean((aggregated_cat_tensor - concatenated_tensor) ** 2))
    return loss

def combined_rmse_loss_hh(aggregated_tensor1, aggregated_tensor2, aggregated_tensor3, target_tensor1, target_tensor2, target_tensor3):
    # concatenating the target and computed tensors along the characteristic dimension (dim=1)
    concatenated_tensor = torch.cat([target_tensor1, target_tensor2, target_tensor3])
    aggregated_cat_tensor = torch.cat([aggregated_tensor1, aggregated_tensor2, aggregated_tensor3])
    # calculating RMSE loss on the concatenated tensor
    loss = torch.sqrt(torch.mean((aggregated_cat_tensor - concatenated_tensor) ** 2))
    return loss


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
        return extract_random_sample(married_male, 1) + extract_random_sample(married_female,
                                                                              1) + extract_random_sample(children,
                                                                                                         size - 2)

    elif composition == '1FM-nA':
        # Handle One family: Married Couple: all children non-dependent
        married_male = persons[persons.apply(lambda row: row['marital'] == 'Married' and row['sex'] == 'M', axis=1)]
        married_female = persons[persons.apply(lambda row: row['marital'] == 'Married' and row['sex'] == 'F', axis=1)]
        children = persons[persons.apply(lambda row: row['age'] in adult_ages, axis=1)]
        return extract_random_sample(married_male, 1) + extract_random_sample(married_female,
                                                                              1) + extract_random_sample(children,
                                                                                                         size - 2)

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
        return extract_random_sample(male, 1) + extract_random_sample(female, 1) + extract_random_sample(children,
                                                                                                         size - 2)

    elif composition == '1FC-nA':
        # Handle One family: Cohabiting Couple: all children non-dependent
        male = persons[persons.apply(lambda row: row['marital'] != 'Married' and row['sex'] == 'M', axis=1)]
        female = persons[persons.apply(lambda row: row['marital'] != 'Married' and row['sex'] == 'F', axis=1)]
        children = persons[persons.apply(lambda row: row['age'] in adult_ages, axis=1)]
        return extract_random_sample(male, 1) + extract_random_sample(female, 1) + extract_random_sample(children,
                                                                                                         size - 2)

    elif composition == '1FL-2C':
        # Handle One family: Lone Parent: Having dependent children
        parent = persons[persons.apply(lambda row: row['marital'] != 'Married', axis=1)]
        children = persons[persons.apply(lambda row: row['age'] in child_ages, axis=1)]
        return extract_random_sample(parent, 1) + extract_random_sample(children, size - 1)

    elif composition == '1FL-nA':
        # Handle One family: Lone parent: all children non-dependent
        parent = persons[persons.apply(lambda row: row['marital'] != 'Married', axis=1)]
        children = persons[persons.apply(lambda row: row['age'] in adult_ages, axis=1)]
        return extract_random_sample(parent, 1) + extract_random_sample(children, size - 1)

    elif composition == '1H-2C':
        # Handle Other households: Having dependent children
        adults = persons[persons.apply(lambda row: row['age'] in adult_ages, axis=1)]
        children = persons[persons.apply(lambda row: row['age'] in child_ages, axis=1)]
        return extract_random_sample(adults, 1) + extract_random_sample(children, size - 1)

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
                              {'params': qual_net.parameters()},
                              {'params': seg_net.parameters()},
                              {'params': occupation_net.parameters()},
                              {'params': economic_act_net.parameters()},
                              {'params': approx_social_grade_net.parameters()},
                              {'params': general_health_net.parameters()},
                              {'params': industry_net.parameters()}], lr=0.01)
scheduler = StepLR(optimizer, step_size=20, gamma=0.25)

optimizer_hh = torch.optim.Adam([{'params': ethnic_net_hh.parameters()},
                                 {'params': religion_net_hh.parameters()},
                                 {'params': car_ownership_net_hh.parameters()}], lr=0.0005)
scheduler_hh = StepLR(optimizer_hh, step_size=20, gamma=0.25)

sex_net.apply(weights_init)
age_net.apply(weights_init)
ethnic_net.apply(weights_init)
religion_net.apply(weights_init)
marital_net.apply(weights_init)
qual_net.apply(weights_init)
seg_net.apply(weights_init)
occupation_net.apply(weights_init)
economic_act_net.apply(weights_init)
approx_social_grade_net.apply(weights_init)
general_health_net.apply(weights_init)
industry_net.apply(weights_init)

ethnic_net_hh.apply(weights_init)
religion_net_hh.apply(weights_init)
car_ownership_net_hh.apply(weights_init)

if training:
    number_of_epochs = 100
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

        categories_to_keep = ['seg']
        kept_tensor = keep_categories(encoded_population, category_lengths, categories_to_keep)
        aggregated_population6 = aggregate(kept_tensor, cross_table6, [seg_dict])

        categories_to_keep = ['occupation']
        kept_tensor = keep_categories(encoded_population, category_lengths, categories_to_keep)
        aggregated_population7 = aggregate(kept_tensor, cross_table7, [occupation_dict])

        categories_to_keep = ['economic_act']
        kept_tensor = keep_categories(encoded_population, category_lengths, categories_to_keep)
        aggregated_population8 = aggregate(kept_tensor, cross_table8, [economic_act_dict])

        categories_to_keep = ['approx_social_grade']
        kept_tensor = keep_categories(encoded_population, category_lengths, categories_to_keep)
        aggregated_population9 = aggregate(kept_tensor, cross_table9, [approx_social_grade_dict])

        categories_to_keep = ['general_health']
        kept_tensor = keep_categories(encoded_population, category_lengths, categories_to_keep)
        aggregated_population10 = aggregate(kept_tensor, cross_table10, [general_health_dict])

        categories_to_keep = ['industry']
        kept_tensor = keep_categories(encoded_population, category_lengths, categories_to_keep)
        aggregated_population11 = aggregate(kept_tensor, cross_table11, [industry_dict])

        categories_to_keep_hh = ['composition', 'ref_ethnicity']
        kept_tensor_hh = keep_categories(encoded_households, category_lengths_hh, categories_to_keep_hh)
        aggregated_households1 = aggregate(kept_tensor_hh, cross_table1_hh, [hh_comp_dict, ethnic_dict_hh])

        categories_to_keep_hh = ['composition', 'ref_religion']
        kept_tensor_hh = keep_categories(encoded_households, category_lengths_hh, categories_to_keep_hh)
        aggregated_households2 = aggregate(kept_tensor_hh, cross_table2_hh, [hh_comp_dict, religion_dict_hh])

        categories_to_keep_hh = ['car_ownership']
        kept_tensor_hh = keep_categories(encoded_households, category_lengths_hh, categories_to_keep_hh)
        aggregated_households3 = aggregate(kept_tensor_hh, cross_table3_hh, [car_ownership_dict])

        loss = combined_rmse_loss(aggregated_population1,
                                  aggregated_population2,
                                  aggregated_population3,
                                  aggregated_population4,
                                  aggregated_population6,
                                  aggregated_population7,
                                  aggregated_population8,
                                  aggregated_population9,
                                  aggregated_population10,
                                  aggregated_population11,
                                  cross_table_tensor1,
                                  cross_table_tensor2,
                                  cross_table_tensor3,
                                  cross_table_tensor4,
                                  cross_table_tensor6,
                                  cross_table_tensor7,
                                  cross_table_tensor8,
                                  cross_table_tensor9,
                                  cross_table_tensor10,
                                  cross_table_tensor11)

        accuracy1 = rmse_accuracy(aggregated_population1, cross_table_tensor1)
        accuracy2 = rmse_accuracy(aggregated_population2, cross_table_tensor2)
        accuracy3 = rmse_accuracy(aggregated_population3, cross_table_tensor3)
        accuracy4 = rmse_accuracy(aggregated_population4, cross_table_tensor4)
        accuracy6 = rmse_accuracy(aggregated_population6, cross_table_tensor6)
        accuracy7 = rmse_accuracy(aggregated_population7, cross_table_tensor7)
        accuracy8 = rmse_accuracy(aggregated_population8, cross_table_tensor8)
        accuracy9 = rmse_accuracy(aggregated_population9, cross_table_tensor9)
        accuracy10 = rmse_accuracy(aggregated_population10, cross_table_tensor10)
        accuracy11 = rmse_accuracy(aggregated_population11, cross_table_tensor11)
        accuracy = (accuracy1 + accuracy2 + accuracy3 + accuracy4 + accuracy6 + accuracy7 + accuracy8 + accuracy9 + accuracy10 + accuracy11) / 10

        loss_hh = combined_rmse_loss_hh(aggregated_households1,
                                        aggregated_households2,
                                        aggregated_households3,
                                        cross_table_tensor1_hh,
                                        cross_table_tensor2_hh,
                                        cross_table_tensor3_hh)

        accuracy1_hh = rmse_accuracy(aggregated_households1, cross_table_tensor1_hh)
        accuracy2_hh = rmse_accuracy(aggregated_households2, cross_table_tensor2_hh)
        accuracy3_hh = rmse_accuracy(aggregated_households3, cross_table_tensor3_hh)
        accuracy_hh = (accuracy1_hh + accuracy2_hh + accuracy3_hh) / 3

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

    aggregated_population1 = aggregated_population1.round().long().cuda()
    aggregated_population2 = aggregated_population2.round().long().cuda()
    aggregated_population3 = aggregated_population3.round().long().cuda()
    aggregated_population4 = aggregated_population4.round().long().cuda()
    aggregated_population6 = aggregated_population6.round().long().cuda()
    aggregated_population7 = aggregated_population7.round().long().cuda()
    aggregated_population8 = aggregated_population8.round().long().cuda()
    aggregated_population9 = aggregated_population9.round().long().cuda()
    aggregated_population10 = aggregated_population10.round().long().cuda()
    aggregated_population11 = aggregated_population11.round().long().cuda()

    aggregated_households1 = aggregated_households1.round().long().cuda()
    aggregated_households2 = aggregated_households2.round().long().cuda()
    aggregated_households3 = aggregated_households3.round().long().cuda()

# recording execution end time
    end = time.time()
    duration = end - start

    # converting the recording time to hours, minutes, and seconds
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = duration % 60
    print(f"Duration: {hours} hours, {minutes} minutes, {seconds:.2f} seconds")

    child_ages = ["0_4", "5_7", "8_9", "10_14", "15"]
    adult_ages = ["16_17", "18_19", "20_24", "25_29", "30_34", "35_39", "40_44", "45_49", "50_54", "55_59", "60_64"]
    elder_ages = ["65_69", "70_74", "75_79", "80_84", "85+"]



    records = decode_tensor(encoded_population, [sex_dict, age_dict, ethnic_dict, religion_dict, marital_dict, qual_dict, seg_dict, occupation_dict, economic_act_dict, approx_social_grade_dict, general_health_dict, industry_dict])
    persons = pd.DataFrame(records, columns=['sex', 'age', 'ethnicity', 'religion', 'marital', 'qualification', 'seg', 'occupation', 'economic_act', 'approx_social_grade', 'general_health', 'industry'])
    persons['Person_ID'] = range(1, len(persons) + 1)

    records_hh = decode_tensor(encoded_households, [hh_comp_dict, ethnic_dict_hh, religion_dict_hh, car_ownership_dict])
    households = pd.DataFrame(records_hh, columns=['composition', 'ref_ethnicity', 'ref_religion', 'car_ownership'])
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

    age_categories_to_single = ["0_4", "5_7", "8_9", "10_14", "15"]
    persons.loc[persons['age'].isin(age_categories_to_single), 'marital'] = 'Single'
    persons.loc[persons['age'].isin(age_categories_to_single), 'qualification'] = 'no'
    persons.to_csv('synthetic_population.csv', index=False)
    households.to_csv('synthetic_households.csv', index=False)