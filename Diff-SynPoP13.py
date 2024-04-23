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

# importing InputData and InputCrossTables for processing UK census data files
import InputData as ID
import InputCrossTables as ICT

from torch.nn import init
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ExponentialLR


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

# #MSOA
area = 'E02005924' # geography code for one of the oxford areas (selected for this work)
total = ID.get_total(ID.age5ydf, area) # getting the total number of individuals in our MSOA
hh_total = ID.get_total(ID.HHcomdf, area)

sex_dict = ID.getdictionary(ID.sexdf, area) # sex
age_dict = ID.getdictionary(ID.age5ydf, area) # age
ethnic_dict = ID.getdictionary(ID.ethnicdf, area) # ethnicity
religion_dict = ID.getdictionary(ID.religiondf, area) # religion
hh_comp_dict = ID.getdictionary(ID.HHcomdf, area)  # household composition
ct5 = ['1PE', '1PA', '1FE', '1FM-0C', '1FM-nC', '1FM-nA', '1FC-0C', '1FC-nC', '1FC-nA', '1FL-nC', '1FL-nA', '1H-nC', '1H-nA', '1H-nE']
#iterate hh_comp_dict and remove keys that are not in ct5
hh_comp_dict = {k: v for k, v in hh_comp_dict.items() if k in ct5}
# hh_id = range(1, hh_total + 1)
hh_size_dict = ID.getdictionary(ID.HHsizedf, area)  # household size


# getting the length (number of classes) for each attribute
category_dicts = {
    'sex': sex_dict,
    'age': age_dict,
    'hh_comp': hh_comp_dict,
    'hh_size': hh_size_dict
}
# generate a dictionary of category lengths
category_lengths = {key: len(value) for key, value in category_dicts.items()}


cross_table5 = ICT.getdictionary(ICT.HH_composition_by_sex_by_age, area)
# iterate cross_table5 using compherension
# cross_table5 = {k.split(' ')[2]: v for k, v in cross_table5.items()}
weights = ID.getweights(ID.age5ydf, area)
# create a new cross table dictionary of sex, age and hh_comp
cross_table5a = {}
for sexkey, sexvalue in sex_dict.items():
    for agekey, agevalue in age_dict.items():
        for hhkey, hhvalue in hh_comp_dict.items():
            val = cross_table5[sexkey + ' ' + agemap(agekey) + ' ' + hhkey]
            ages = reverse_agemap(agemap(agekey))
            # divide the val in cross_table5a ages using weights
            for age in ages:
                cross_table5a[sexkey + ' ' + age + ' ' + hhkey] = round(val * weights[age])

cross_table6 = ICT.getdictionary(ICT.HH_composition_by_Ethnicity, area)
# iterate crosstable6 using compherension and remove keys that doesn't have 0
cross_table6 = {k: v for k, v in cross_table6.items() if '0' in k.split(' ')[1]}
#rename W0 to W, B0 to B, A0 to A, M0 to M
cross_table6 = {key_replace(k): v for k, v in cross_table6.items()}
cross_table7 = ICT.getdictionary(ICT.HH_composition_by_Religion, area)
cross_table8 = ICT.get_hh_comp_by_size_crosstable(area)
cross_table8 = {k: v for k, v in cross_table8.items() if k.split(' ')[0] in ct5}

cross_table_tensor5 = torch.tensor(list(cross_table5a.values()), dtype=torch.float32).to(device).cuda()
cross_table_tensor6 = torch.tensor(list(cross_table6.values()), dtype=torch.float32).to(device).cuda()
cross_table_tensor7 = torch.tensor(list(cross_table7.values()), dtype=torch.float32).to(device).cuda()
cross_table_tensor8 = torch.tensor(list(cross_table8.values()), dtype=torch.float32).to(device).cuda()


# instantiating networks for each characteristic
input_dim = sum(len(d.keys()) for d in category_dicts.values())
hidden_dims = [64, 32]

# With CUDA
sex_net = FFNetwork(input_dim, hidden_dims, len(sex_dict)).to(device).cuda()
age_net = FFNetwork(input_dim, hidden_dims, len(age_dict)).to(device).cuda()
ethnic_net = FFNetwork(input_dim, hidden_dims, len(ethnic_dict)).to(device).cuda()
religion_net = FFNetwork(input_dim, hidden_dims, len(religion_dict)).to(device).cuda()
hh_comp_net = FFNetwork(input_dim, hidden_dims, len(hh_comp_dict)).to(device).cuda()
hh_size_net = FFNetwork(input_dim, hidden_dims, len(hh_size_dict)).to(device).cuda()

# input for the networks
# input_tensor = torch.empty(total, input_dim).to(device)
input_tensor = initialize_input_tensor(hh_total, category_lengths, category_dicts.values()).to(device)
init.kaiming_normal_(input_tensor)

training = True
# training = False

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
    hh_comp_logits = hh_comp_net(input_tensor)
    hh_size_logits = hh_size_net(input_tensor)

    sex = gumbel_softmax_sample(sex_logits, temperature)
    age = gumbel_softmax_sample(age_logits, temperature)
    ethnicity = gumbel_softmax_sample(ethnicity_logits, temperature)
    religion = gumbel_softmax_sample(religion_logits, temperature)
    hh_comp = gumbel_softmax_sample(hh_comp_logits, temperature)
    hh_size = gumbel_softmax_sample(hh_size_logits, temperature)

    # return torch.cat([sex, age, ethnicity, religion, hh_comp, hh_size], dim=-1)
    return torch.cat([sex, age, hh_comp, hh_size], dim=-1)

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


# encoded_population = generate_population(input_tensor)
# records = decode_tensor(encoded_population, category_dicts.values())
# print(records)
# df = pd.DataFrame(records, columns=['sex', 'age', 'ethnicity', 'religion', 'marital', 'qualification', 'composition', 'size'])


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

def combined_rmse_loss(aggregated_tensor1, aggregated_tensor2,
                       target_tensor1, target_tensor2):
    # concatenating the target and computed tensors along the characteristic dimension (dim=1)
    concatenated_tensor = torch.cat([target_tensor1, target_tensor2])
    aggregated_cat_tensor = torch.cat([aggregated_tensor1, aggregated_tensor2])
    # calculating RMSE loss on the concatenated tensor
    loss = torch.sqrt(torch.mean((aggregated_cat_tensor - concatenated_tensor) ** 2))
    return loss

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
def plot_radar(attribute, attribute_dict):

    categories_gen = hh_df[attribute].unique().astype(str).tolist()
    count_gen = [(hh_df[attribute] == str).sum() for str in categories_gen]
    categories, count_act = list(attribute_dict.keys()), list(attribute_dict.values())

    gen_combined = list(zip(categories_gen, count_gen))
    gen_combined = sorted(gen_combined, key=lambda x: categories.index(x[0]))
    categories_gen, count_gen = zip(*gen_combined)
    count_gen = list(count_gen)
    range = (0, ((max(count_act) + max(count_gen)) / 2) * 1.1)

    count_act = [max(val, 10) for val in count_act]
    count_gen = [max(val, 10) for val in count_gen]

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
      showlegend=False,
      width=2500,
      height=2500,
      margin=dict(l=500, r=500, t=500, b=500),
      font=dict(size=65)
    )

    fig.write_html(f"{attribute}-radar-chart.html")
    fig.show()

if training:
    loss_history = []
    accuracy_history = []

    # recording execution start time
    start = time.time()

    # training loop
    optimizer = torch.optim.Adam([{'params': sex_net.parameters()},
                                  {'params': age_net.parameters()},
                                  # {'params': ethnic_net.parameters()},
                                  # {'params': religion_net.parameters()},
                                    {'params': hh_comp_net.parameters()},
                                    {'params': hh_size_net.parameters()}], lr=0.01)

    scheduler = StepLR(optimizer, step_size=20, gamma=0.25)

    sex_net.apply(weights_init)
    age_net.apply(weights_init)
    ethnic_net.apply(weights_init)
    religion_net.apply(weights_init)
    hh_comp_net.apply(weights_init)
    hh_size_net.apply(weights_init)

    number_of_epochs = 100
    for epoch in range(number_of_epochs+1):
        optimizer.zero_grad()

        # generating and aggregating encoded population for sex, age, ethnicity
        encoded_population = generate_population(input_tensor)

        categories_to_keep = ['sex', 'age', 'hh_comp']
        kept_tensor = keep_categories(encoded_population, category_lengths, categories_to_keep)
        aggregated_population5 = aggregate(kept_tensor, cross_table5a, [sex_dict, age_dict, hh_comp_dict])

        # categories_to_keep = ['hh_comp', 'ethnicity']
        # kept_tensor = keep_categories(encoded_population, category_lengths, categories_to_keep)
        # aggregated_population6 = aggregate(kept_tensor, cross_table6, [hh_comp_dict, ethnic_dict])
        #
        # categories_to_keep = ['hh_comp', 'religion']
        # kept_tensor = keep_categories(encoded_population, category_lengths, categories_to_keep)
        # aggregated_population7 = aggregate(kept_tensor, cross_table7, [hh_comp_dict, religion_dict])

        categories_to_keep = ['hh_comp', 'hh_size']
        kept_tensor = keep_categories(encoded_population, category_lengths, categories_to_keep)
        aggregated_population8 = aggregate(kept_tensor, cross_table8, [hh_comp_dict, hh_size_dict])


        loss = combined_rmse_loss(aggregated_population5,
                                  # aggregated_population6,
                                  # aggregated_population7,
                                  aggregated_population8,
                                  cross_table_tensor5,
                                  # cross_table_tensor6,
                                  # cross_table_tensor7,
                                  cross_table_tensor8)

        accuracy5 = rmse_accuracy(aggregated_population5, cross_table_tensor5)
        # accuracy6 = rmse_accuracy(aggregated_population6, cross_table_tensor6)
        # accuracy7 = rmse_accuracy(aggregated_population7, cross_table_tensor7)
        accuracy8 = rmse_accuracy(aggregated_population8, cross_table_tensor8)

        accuracy = (accuracy5 + accuracy8) / 2

        loss_history.append(loss)
        accuracy_history.append(accuracy)

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss}, Accuracy: {accuracy}")

    # With CUDA
    aggregated_population5 = aggregated_population5.round().long().cuda()
    # aggregated_population6 = aggregated_population6.round().long().cuda()
    # aggregated_population7 = aggregated_population7.round().long().cuda()
    aggregated_population8 = aggregated_population8.round().long().cuda()


    # plot(cross_table_tensor1, aggregated_population1, cross_table1, 'Age-Sex-Ethnicity')
    # plot(cross_table_tensor2, aggregated_population2, cross_table2, 'Age-Sex-Religion')
    # plot(cross_table_tensor3, aggregated_population3, cross_table3, 'Age-Sex-MaritalStatus')
    # plot(cross_table_tensor4, aggregated_population4, cross_table4, 'Age-Sex-Qualification')
    # plot(cross_table_tensor5, aggregated_population5, cross_table5, 'Age-Sex-HouseholdComposition')

    loss_history = [tensor.to('cpu').item() for tensor in loss_history]
    accuracy_history = [tensor for tensor in accuracy_history]

    records = decode_tensor(encoded_population, category_dicts.values())
    hh_df = pd.DataFrame(records, columns=['sex', 'age', 'hh_comp', 'hh_size'])
    # age_categories_to_single = ["0_4", "5_7", "8_9", "10_14", "15"]
    # persons_df.loc[persons_df['age'].isin(age_categories_to_single), 'marital'] = 'Single'
    # persons_df.loc[persons_df['age'].isin(age_categories_to_single), 'qualification'] = 'no'
    hh_df['hh_ID'] = range(1, len(hh_df) + 1) # assigning a person ID to each row
    hh_df.to_csv('households-20240422.csv', index=False)



    # note: set show to 'yes' for displaying the plot
    plot_radar('age', age_dict)
    # plot_radar('religion', religion_dict)
    # plot_radar('ethnicity', ethnic_dict)
    plot_radar('hh_comp', hh_comp_dict)
    plot_radar('hh_size', hh_size_dict)
