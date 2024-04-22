import sys

# appending the system path to run the file on kaggle
# not required if you are running it locally
sys.path.insert(1, '/kaggle/input/diffspop/Diff-SynPoP')

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

"""# **Dictionaries, Cross Tables, & Input Tensor (Individuals)**"""

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

# With CUDA
cross_table_tensor1 = torch.tensor(list(cross_table1.values()), dtype=torch.float32).to(device).cuda()
cross_table_tensor2 = torch.tensor(list(cross_table2.values()), dtype=torch.float32).to(device).cuda()
cross_table_tensor3 = torch.tensor(list(cross_table3.values()), dtype=torch.float32).to(device).cuda()
cross_table_tensor4 = torch.tensor(list(cross_table4.values()), dtype=torch.float32).to(device).cuda()
# cross_table_tensor5 = torch.tensor(list(cross_table5.values()), dtype=torch.float32).to(device).cuda()

# Without CUDA
# cross_table_tensor1 = torch.tensor(list(cross_table1.values()), dtype=torch.float32).to(device)
# cross_table_tensor2 = torch.tensor(list(cross_table2.values()), dtype=torch.float32).to(device)
# cross_table_tensor3 = torch.tensor(list(cross_table3.values()), dtype=torch.float32).to(device)
# cross_table_tensor4 = torch.tensor(list(cross_table4.values()), dtype=torch.float32).to(device)
# # cross_table_tensor5 = torch.tensor(list(cross_table5.values()), dtype=torch.float32).to(device)

# instantiating networks for each characteristic
input_dim = sum(len(d.keys()) for d in [sex_dict, age_dict, ethnic_dict, religion_dict, marital_dict, qual_dict])
print("Input dimension: ", input_dim)
hidden_dims = [64, 32]

# With CUDA
sex_net = FFNetwork(input_dim, hidden_dims, len(sex_dict)).to(device).cuda()
age_net = FFNetwork(input_dim, hidden_dims, len(age_dict)).to(device).cuda()
ethnic_net = FFNetwork(input_dim, hidden_dims, len(ethnic_dict)).to(device).cuda()
religion_net = FFNetwork(input_dim, hidden_dims, len(religion_dict)).to(device).cuda()
marital_net = FFNetwork(input_dim, hidden_dims, len(marital_dict)).to(device).cuda()
qual_net = FFNetwork(input_dim, hidden_dims, len(qual_dict)).to(device).cuda()
# hh_comp_net = FFNetwork(input_dim, hidden_dims, len(hh_comp_dict)).to(device).cuda()
# age_hh_net = FFNetwork(input_dim, hidden_dims, len(age_hh_dict)).to(device).cuda()

# Without CUDA
# sex_net = FFNetwork(input_dim, hidden_dims, len(sex_dict)).to(device)
# age_net = FFNetwork(input_dim, hidden_dims, len(age_dict)).to(device)
# ethnic_net = FFNetwork(input_dim, hidden_dims, len(ethnic_dict)).to(device)
# religion_net = FFNetwork(input_dim, hidden_dims, len(religion_dict)).to(device)
# marital_net = FFNetwork(input_dim, hidden_dims, len(marital_dict)).to(device)
# qual_net = FFNetwork(input_dim, hidden_dims, len(qual_dict)).to(device)
# # hh_comp_net = FFNetwork(input_dim, hidden_dims, len(hh_comp_dict)).to(device)
# # age_hh_net = FFNetwork(input_dim, hidden_dims, len(age_hh_dict)).to(device)

# input for the networks
# input_tensor = torch.randn(total, input_dim).to(device)  # Random noise as input, adjust as necessary

input_tensor = torch.empty(total, input_dim).to(device)
init.kaiming_normal_(input_tensor)

"""# **Accessory Functions (Individual Generation)**"""

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
#     hh_comp = gumbel_softmax_sample(hh_comp_logits, temperature)
#     age_hh = gumbel_softmax_sample(age_hh_logits, temperature)

#     age_hh = torch.zeros((age.shape[0], 5), dtype=torch.float).to(device)
#     for i, a in enumerate(age):
#         argmax_item = torch.argmax(a).item()
#         if argmax_item in range(0, 5):
#             age_hh[i] = torch.eye(5)[0]
#         elif argmax_item in range(5, 8):
#             age_hh[i] = torch.eye(5)[1]
#         elif argmax_item in range(8, 10):
#             age_hh[i] = torch.eye(5)[2]
#         elif argmax_item in range(10, 13):
#             age_hh[i] = torch.eye(5)[3]
#         elif argmax_item in range(13, 21):
#             age_hh[i] = torch.eye(5)[4]

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

generated_population = generate_population(input_tensor)
records = decode_tensor(generated_population, [sex_dict, age_dict, ethnic_dict, religion_dict, marital_dict, qual_dict])
df = pd.DataFrame(records, columns=['sex', 'age', 'ethnicity', 'religion', 'marital', 'qualification'])
print(df)

"""# **Training / Plotting (Individual Generation)**"""

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

# With CUDA
aggregated_population1 = aggregated_population1.round().long().cuda()
aggregated_population2 = aggregated_population2.round().long().cuda()
aggregated_population3 = aggregated_population3.round().long().cuda()
aggregated_population4 = aggregated_population4.round().long().cuda()
# aggregated_population5 = aggregated_population5.round().long().cuda()

# Without CUDA
# aggregated_population1 = aggregated_population1.round().long()
# aggregated_population2 = aggregated_population2.round().long()
# aggregated_population3 = aggregated_population3.round().long()
# aggregated_population4 = aggregated_population4.round().long()
# # aggregated_population5 = aggregated_population5.round().long()

plot(cross_table_tensor1, aggregated_population1, cross_table1, 'Age-Sex-Ethnicity')
plot(cross_table_tensor2, aggregated_population2, cross_table2, 'Age-Sex-Religion')
plot(cross_table_tensor3, aggregated_population3, cross_table3, 'Age-Sex-MaritalStatus')
plot(cross_table_tensor4, aggregated_population4, cross_table4, 'Age-Sex-Qualification')
# plot(cross_table_tensor5, aggregated_population5, cross_table5, 'Age-Sex-HouseholdComposition')

loss_history = [tensor.to('cpu').item() for tensor in loss_history]
accuracy_history = [tensor for tensor in accuracy_history]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(len(loss_history)), loss_history, label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(len(accuracy_history)), accuracy_history, label='Training Accuracy')
plt.title('Training Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

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

persons_df_copy = persons_df.copy()

"""# **Radar Charts For Individual Attributes**"""

import plotly.graph_objects as go

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
    fig.show() if show == 'yes' else None

# note: set show to 'yes' for displaying the plot
plot_radar('age', age_dict, show='no')
plot_radar('religion', religion_dict, show='no')
plot_radar('ethnicity', ethnic_dict, show='no')
plot_radar('marital', marital_dict, show='no')
plot_radar('qualification', qual_dict, show='no')

"""# **Dictionaries, Cross Tables, & Input Tensor (Households)**"""

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

# With CUDA
cross_table_tensor1_hh = torch.tensor(list(cross_table1_hh.values()), dtype=torch.float32).to(device).cuda()
cross_table_tensor2_hh = torch.tensor(list(cross_table2_hh.values()), dtype=torch.float32).to(device).cuda()

# Without CUDA
# cross_table_tensor1_hh = torch.tensor(list(cross_table1_hh.values()), dtype=torch.float32).to(device)
# cross_table_tensor2_hh = torch.tensor(list(cross_table2_hh.values()), dtype=torch.float32).to(device)

input_dim = sum(len(d.keys()) for d in [ethnic_dict_hh, religion_dict_hh])
# print("Input dimension: ", input_dim)
hidden_dims = [64, 32]

# With CUDA
# hh_comp_net = FFNetwork(input_dim, hidden_dims, len(hh_comp_dict)).to(device).cuda()
ethnic_net_hh = FFNetwork(input_dim, hidden_dims, len(ethnic_dict_hh)).to(device).cuda()
religion_net_hh = FFNetwork(input_dim, hidden_dims, len(religion_dict_hh)).to(device).cuda()

# Without CUDA
# # hh_comp_net = FFNetwork(input_dim, hidden_dims, len(hh_comp_dict)).to(device)
# ethnic_net_hh = FFNetwork(input_dim, hidden_dims, len(ethnic_dict_hh)).to(device)
# religion_net_hh = FFNetwork(input_dim, hidden_dims, len(religion_dict_hh)).to(device)

input_tensor_hh = torch.empty(num_households, input_dim).to(device)
init.kaiming_normal_(input_tensor_hh)

"""# **Accessory Functions (Household Fitting)**"""

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

"""# **Training / Plotting (Household Fitting)**"""

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

number_of_epochs = 1000
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

# With CUDA
aggregated_households1 = aggregated_households1.round().long().cuda()
aggregated_households2 = aggregated_households2.round().long().cuda()

# Without CUDA
# aggregated_households1 = aggregated_households1.round().long()
# aggregated_households2 = aggregated_households2.round().long()

plot(cross_table_tensor1_hh, aggregated_households1, cross_table1_hh, 'Household Composition By Ethnicity')
plot(cross_table_tensor2_hh, aggregated_households2, cross_table2_hh, 'Household Composition By Religion')

loss_history = [tensor.to('cpu').item() for tensor in loss_history]
accuracy_history = [tensor for tensor in accuracy_history]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(len(loss_history)), loss_history, label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(len(accuracy_history)), accuracy_history, label='Training Accuracy')
plt.title('Training Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

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

"""# **Household Size Allocation**"""

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

"""# **Age Categories & Household Ratios**"""

# individuals are in persons_df
# households are in households_df

child_ages = ["0_4", "5_7", "8_9", "10_14", "15"]
adult_ages = ["16_17", "18_19", "20_24", "25_29", "30_34", "35_39", "40_44", "45_49", "50_54", "55_59", "60_64"]
elder_ages = ["65_69", "70_74", "75_79", "80_84", "85+"]

inter_ethnic_ratio = 0.1
inter_rel_ratio = 0.015

"""# **Composition Classification**"""

def check_composition(households_df, persons_df):
    composition_classes = []

    for index, row in households_df.iterrows():
        assigned_persons = row['assigned_persons']
        composition_class = ""

        num_children = 0
        num_adults = 0
        num_elders = 0

        num_children = sum(persons_df.loc[persons_df['Person_ID'].isin(assigned_persons), 'age'].isin(child_ages))
        num_adults = sum(persons_df.loc[persons_df['Person_ID'].isin(assigned_persons), 'age'].isin(adult_ages))
        num_elders = sum(persons_df.loc[persons_df['Person_ID'].isin(assigned_persons), 'age'].isin(elder_ages))

        cc = "other"

        if len(assigned_persons) > 2:
            if sum((persons_df['Person_ID'].isin(assigned_persons)) & (persons_df['marital'].isin(["Separated", "Widowed", "Divorced"]))) == 1:
                cc = "1FL-nA"
            elif sum((persons_df['Person_ID'].isin(assigned_persons)) & (persons_df['marital'].isin(["Marital"]))) == 2:
                cc = "1FM-nA"
            else:
                cc = "1FC-nA"

        if len(assigned_persons) == num_elders:
            if row['composition'] == "1H-nE":
                cc = "1H-nE"
            if row['composition'] == "1FE":
                cc = "1FE"

        if len(assigned_persons) == num_adults:
            cc = "1H-nA"

        if not any(persons_df[(persons_df['Person_ID'].isin(assigned_persons)) & (persons_df['qualification'] == "no")]):
            cc = "1H-nS"

        if num_children > 0:
            if (num_adults + num_elders) == 1:
                if row['composition'] == "1FL-2C":
                    cc = "1FL-2C"
                elif row['composition'] == "1H-2C":
                    cc = "1H-2C"
            elif len(assigned_persons) == num_children:
                cc = "other"
            elif all(persons_df[(persons_df['Person_ID'].isin(assigned_persons)) & (persons_df['age'].isin(elder_ages + adult_ages))]['marital'] == 'Married'):
                cc = "1FM-2C"
            else:
                cc = "1FC-2C"

        if len(assigned_persons) == 2:
            if all(persons_df[(persons_df['Person_ID'].isin(assigned_persons)) & (persons_df['age'].isin(adult_ages))]['marital'] == 'Married'):
                cc = "1FM-0C"
            elif len(set(persons_df.loc[persons_df['Person_ID'].isin(assigned_persons), 'marital'])) == 1:
                cc = "1FC-0C"

        if len(assigned_persons) == 1:
            if num_elders == 1:
                cc = "1PE"
            elif num_adults == 1:
                cc = "1PA"

        if len(assigned_persons) == 0:
            cc = "other"

        composition_classes.append(cc)

    households_df['composition_class'] = composition_classes

    return households_df

"""# **Household Assignment Training Loop**"""

import random

def initial_assignment(persons_dataframe, households_df):

    persons_df = persons_dataframe.copy()

    shuffled_persons = persons_df.sample(frac=1).reset_index(drop=True)
    assigned_persons_dict = {household_id: [] for household_id in households_df['household_ID']}

    for idx, household_row in households_df.iterrows():
        household_id = household_row['household_ID']
        size = household_row['size']
        ref_religion = household_row['ref_religion']
        ref_ethnicity = household_row['ref_ethnicity']

        eligible_persons = shuffled_persons[
            (shuffled_persons['religion'] == ref_religion) &
            (shuffled_persons['ethnicity'] == ref_ethnicity)
        ]

        if len(eligible_persons) < size:
            continue

        sampled_persons = eligible_persons.sample(n=size)
        assigned_persons_dict[household_id] = list(sampled_persons['Person_ID'])
        shuffled_persons = shuffled_persons.drop(sampled_persons.index)

    households_df['assigned_persons'] = households_df['household_ID'].map(assigned_persons_dict)

    return households_df

import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def regular_assignment(persons_df, households_df):

    correctly_assigned_households = households_df[households_df['composition'] == households_df['composition_class']]
    households_df = households_df[~households_df.index.isin(correctly_assigned_households.index)]
    households_df['assigned_persons'] = [[] for _ in range(len(households_df))]



    label_encoders = {}
    encoded_data = persons_df.copy()
    for column in persons_df.columns:
        label_encoders[column] = LabelEncoder()
        encoded_data[column] = label_encoders[column].fit_transform(persons_df[column])

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(encoded_data)
    num_clusters = 1000  # we can choose the number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(scaled_data)

    persons_df['cluster'] = kmeans.labels_
    persons_df = persons_df.sort_values(by='cluster').reset_index(drop=True)

#     assigned_persons_ids = [person_id for sublist in correctly_assigned_households['assigned_persons'] for person_id in sublist]
#     persons_df = persons_df[~persons_df['Person_ID'].isin(assigned_persons_ids)].reset_index(drop=True)

    shuffled_persons = persons_df.sample(frac=1).reset_index(drop=True)
    assigned_persons_dict = {household_id: [] for household_id in households_df['household_ID']}


    for idx, household_row in households_df.iterrows():
        household_id = household_row['household_ID']
        size = household_row['size']
        ref_religion = household_row['ref_religion']
        ref_ethnicity = household_row['ref_ethnicity']
        composition = household_row['composition']

        filtered_rows = correctly_assigned_households[correctly_assigned_households['composition'] == composition]

        if not filtered_rows.empty:
            assigned_persons = filtered_rows.iloc[0]['assigned_persons']
        else:
#             print("No matching composition found for " + composition)
            eligible_persons = shuffled_persons[
                (shuffled_persons['religion'] == ref_religion) &
                (shuffled_persons['ethnicity'] == ref_ethnicity)
            ]

            if len(eligible_persons) < size:
                continue

            sampled_persons = eligible_persons.sample(n=size)
            assigned_persons_dict[household_id] = list(sampled_persons['Person_ID'])
            continue


#             filtered_rows = correctly_assigned_households[correctly_assigned_households['composition'] != composition]
#             assigned_persons = filtered_rows.iloc[0]['assigned_persons']

#         assigned_persons = correctly_assigned_households[correctly_assigned_households['composition'] == composition].iloc[0]['assigned_persons']
        persons_to_add = []

        for given_person_id in assigned_persons:
            given_person_cluster = persons_df.loc[persons_df['Person_ID'] == given_person_id, 'cluster'].iloc[0]
            same_cluster_df = persons_df.loc[persons_df['cluster'] == given_person_cluster]
            same_cluster_df = same_cluster_df[same_cluster_df['Person_ID'] != given_person_id]
            sampled_person_id = np.random.choice(same_cluster_df['Person_ID'])
            persons_to_add.append(sampled_person_id)

#         eligible_persons = shuffled_persons[
#             (shuffled_persons['religion'] == ref_religion) &
#             (shuffled_persons['ethnicity'] == ref_ethnicity)
#         ]

#         if len(eligible_persons) < size:
#             continue

#         sampled_persons = eligible_persons.sample(n=size)
#         assigned_persons_dict[household_id] = list(sampled_persons['Person_ID'])
        assigned_persons_dict[household_id] = persons_to_add
#         shuffled_persons = shuffled_persons.drop(sampled_persons.index)

    households_df['assigned_persons'] = households_df['household_ID'].map(assigned_persons_dict)

    households_df = pd.concat([households_df, correctly_assigned_households], ignore_index=True)

    return households_df

def calculate_hh_comp_loss(households_df):

    losses = (households_df['composition'] != households_df['composition_class']).astype(int)
    total_loss = losses.sum()
    normalized_loss = total_loss / len(households_df)

    return normalized_loss

updated_households_df = initial_assignment(persons_df, households_df)
# print(updated_households_df)

number_of_epochs = 10
persons_df_copy = persons_df.copy()
for epoch in range(number_of_epochs+1):

    # households_df_some = updated_households_df.head(50)
    labeled_households_df = check_composition(updated_households_df, persons_df)
    # print(labeled_households_df)

    loss = calculate_hh_comp_loss(labeled_households_df)
    print("Total loss:", loss)

    updated_households_df = regular_assignment(persons_df_copy, labeled_households_df)

updated_households_df = updated_households_df.sample(frac=1, random_state=42).reset_index(drop=True)
updated_households_df.to_csv('households.csv', index=False)

"""# **Radar Charts For Households**"""

households_df_plot = updated_households_df.copy(deep=True)
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

    categories.append(categories[0])
    count_act.append(count_act[0])
    count_gen.append(count_gen[0])

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r = count_act,
        theta=categories,
        name='Actual Population',
        line=dict(width=10)
    ))

    fig.add_trace(go.Scatterpolar(
        r = count_gen,
        theta=categories,
        name='Generated Population',
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

    fig.write_html(f"comp-{attribute}-radar-chart.html")
    fig.show() if show == 'yes' else None

# note: set show to 'yes' for displaying the plot
plot_radar_hh('ethnicity', comp_ethnic_df_act, comp_ethnic_df_gen, show='no')
plot_radar_hh('religion', comp_rel_df_act, comp_rel_df_gen, show='no')