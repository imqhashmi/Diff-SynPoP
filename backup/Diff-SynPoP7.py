import sys

# appending the system path to run the file on kaggle
# not required if you are running it locally
# sys.path.insert(1, '/kaggle/input/diffspop/Diff-SynPoP')

import os
import time
import numpy as np
import plotly as py
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import torch
import torch.nn as nn
import torch.optim as optim

# imporint InputData and InputCrossTables for processing UK census data files
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

# getting the length (number of classes) for each attribute 
category_lengths = {
    'sex': len(sex_dict),
    'age': len(age_dict),
    'ethnicity': len(ethnic_dict),
    'religion': len(religion_dict),
    'marital': len(marital_dict),
    'qual': len(qual_dict)
}

cross_table1 = ICT.getdictionary(ICT.ethnic_by_sex_by_age, area)
cross_table2 = ICT.getdictionary(ICT.religion_by_sex_by_age, area)
cross_table3 = ICT.getdictionary(ICT.marital_by_sex_by_age, area)
cross_table4 = ICT.getdictionary(ICT.qualification_by_sex_by_age, area)

cross_table_tensor1 = torch.tensor(list(cross_table1.values()), dtype=torch.float32).to(device).cuda()
cross_table_tensor2 = torch.tensor(list(cross_table2.values()), dtype=torch.float32).to(device).cuda()
cross_table_tensor3 = torch.tensor(list(cross_table3.values()), dtype=torch.float32).to(device).cuda()
cross_table_tensor4 = torch.tensor(list(cross_table4.values()), dtype=torch.float32).to(device).cuda()

# instantiating networks for each characteristic
input_dim = sum(len(d.keys()) for d in [sex_dict, age_dict, ethnic_dict, religion_dict, marital_dict, qual_dict])
hidden_dims = [64, 32]

sex_net = FFNetwork(input_dim, hidden_dims, len(sex_dict)).to(device).cuda()
age_net = FFNetwork(input_dim, hidden_dims, len(age_dict)).to(device).cuda()
ethnic_net = FFNetwork(input_dim, hidden_dims, len(ethnic_dict)).to(device).cuda()
relgion_net = FFNetwork(input_dim, hidden_dims, len(religion_dict)).to(device).cuda()
marital_net = FFNetwork(input_dim, hidden_dims, len(marital_dict)).to(device).cuda()
qual_net = FFNetwork(input_dim, hidden_dims, len(qual_dict)).to(device).cuda()

# input for the networks
# input_tensor = torch.randn(total, input_dim).to(device)  # Random noise as input, adjust as necessary

input_tensor = torch.empty(total, input_dim).to(device)
init.kaiming_normal_(input_tensor)

# defining the Gumbel-Softmax function
def gumbel_softmax_sample(logits, temperature=0.5):
    gumbel_noise = -torch.log(-torch.log(torch.rand(logits.shape, device=device)))
    y = logits + gumbel_noise
    return torch.nn.functional.softmax(y / temperature, dim=-1)

def generate_population(input_tensor, temperature=0.5):
    sex_logits = sex_net(input_tensor)
    age_logits = age_net(input_tensor)
    ethnicity_logits = ethnic_net(input_tensor)
    relgion_logits = relgion_net(input_tensor)
    marital_logits = marital_net(input_tensor)
    qual_logits = qual_net(input_tensor)
    
    sex = gumbel_softmax_sample(sex_logits, temperature)
    age = gumbel_softmax_sample(age_logits, temperature)
    ethnicity = gumbel_softmax_sample(ethnicity_logits, temperature)
    relgion = gumbel_softmax_sample(relgion_logits, temperature)
    marital = gumbel_softmax_sample(marital_logits, temperature)
    qual = gumbel_softmax_sample(qual_logits, temperature)

    return torch.cat([sex, age, ethnicity, relgion, marital, qual], dim=-1)

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
    # saving plot
    #py.offline.plot(fig, filename= path + '/plots/' + str(name) + '.html')
    # showing plot
    fig.show()

# encoded_population = generate_population(input_tensor).cuda()
# records = decode_tensor(encoded_population, [sex_dict, age_dict, ethnic_dict, religion_dict, marital_dict, qual_dict])
# print(records)
# categories_to_keep = ['sex', 'age', 'marital']  # Categories to keep
# kept_tensor = keep_categories(encoded_population, category_lengths, categories_to_keep)
# aggregated_tensor = aggregate(kept_tensor, cross_table3, [sex_dict, age_dict, marital_dict])
#
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

loss_history = []
accuracy_history = []

# recording execution start time
start = time.time()

# training loop
optimizer = torch.optim.Adam([{'params': sex_net.parameters()},
                              {'params': age_net.parameters()},
                              {'params': ethnic_net.parameters()},
                              {'params': relgion_net.parameters()},
                              {'params': marital_net.parameters()},
                              {'params': qual_net.parameters()}], lr=0.01)
scheduler = StepLR(optimizer, step_size=20, gamma=0.25)

sex_net.apply(weights_init)
age_net.apply(weights_init)
ethnic_net.apply(weights_init)
relgion_net.apply(weights_init)
marital_net.apply(weights_init)
qual_net.apply(weights_init)

number_of_epochs = 100
for epoch in range(number_of_epochs):
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

    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {loss}, Accuracy: {accuracy}")

aggregated_population1 = aggregated_population1.round().long().cuda()
aggregated_population2 = aggregated_population2.round().long().cuda()
aggregated_population3 = aggregated_population3.round().long().cuda()
aggregated_population4 = aggregated_population4.round().long().cuda()

plot(cross_table_tensor1, aggregated_population1, cross_table1, 'Age-Sex-Ethnicity')
plot(cross_table_tensor2, aggregated_population2, cross_table2, 'Age-Sex-Religion')
plot(cross_table_tensor3, aggregated_population3, cross_table3, 'Age-Sex-MaritalStatus')
plot(cross_table_tensor4, aggregated_population4, cross_table4, 'Age-Sex-Qualification')

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
df = pd.DataFrame(records, columns=['sex', 'age', 'ethnicity', 'religion', 'marital', 'qualification'])
age_categories_to_single = ["0_4", "5_7", "8_9", "10_14", "15"]
df.loc[df['age'].isin(age_categories_to_single), 'marital'] = 'Single'
df.loc[df['age'].isin(age_categories_to_single), 'qualification'] = 'no'


df.to_csv('synthetic_population.csv', index=False)

# recording execution end time
end = time.time()
duration = end - start

# converting the recordind time to hours, minutes, and seconds
hours = int(duration // 3600)
minutes = int((duration % 3600) // 60)
seconds = duration % 60
print(f"Duration: {hours} hours, {minutes} minutes, {seconds:.2f} seconds")
