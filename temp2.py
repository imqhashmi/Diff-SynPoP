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

def assign_size(composition, hh_size_dict):
    # calculate weights for each household size
    hh_size_weights = {key: value / sum(hh_size_dict.values()) for key, value in hh_size_dict.items()}
    household_sizes = {
        '1PE': '1', '1PA': '1', '1FE': '1',
        '1FM-0C': '2', '1FM-1C': '3', '1FM-2C': '4', '1FM-nA': '3+',
        '1FS-0C': '2', '1FS-1C': '3', '1FS-2C': '4', '1FS-nA': '3+',
        '1FC-0C': '2', '1FC-1C': '3', '1FC-2C': '4', '1FC-nA': '3+',
        '1FL-1C': '2', '1FL-2C': '3', '1FL-nA': '2+',
        '1H-1C': '3+', '1H-2C': '3+', '1H-nA': '3+', '1H-nE': '3+',
        '1H-nS': '2+'
    }
    expected_size = household_sizes[composition]
    # if expected size has +
    if '+' in expected_size:
        expected_size = int(expected_size.replace('+', ''))
        # get random choice from the expected size till 8
        expected_sizes = list(range(expected_size, 9))
        return random.choices(expected_sizes, weights=[hh_size_weights[str(size)] for size in expected_sizes])[0]

    else:
        return int(expected_size)

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

# DICTIONARIES, CROSS TABLES, & INPUT TENSOR (INDIVIDUALS)**
device = 'cuda' if torch.cuda.is_available() else 'cpu' # checking to see if a cuda device is available 
path = os.path.join(os.path.dirname(os.getcwd()), 'Diff-SynPoP')

area = 'E02005924' # geography code for one of the oxford areas (selected for this work)
num_households = ID.get_HH_com_total(ID.HHcomdf, area)
print("Total number of households: ", num_households)
persondf = pd.read_csv(os.path.join(path, 'persons.csv')) # reading the persons file

hh_comp_dict = ID.getHHcomdictionary(ID.HHcomdf, area) # household composition
hh_size_dict = ID.getdictionary(ID.HHsizedf, area)  # household size

composition = []
for key, value in hh_comp_dict.items():
    composition.extend([key] * value)

hh_df = pd.DataFrame(composition, columns=['composition'])
hh_df['size'] = hh_df['composition'].apply(lambda x: int(assign_size(x, hh_size_dict)))

category_lengths_hh = {
    'size': len(hh_size_dict), #because HH_size dic is not the same size as HH_comp_dict
    'composition': len(hh_comp_dict)
}

target_tensor1 = torch.tensor(list(hh_comp_dict.values()), dtype=torch.float32).to(device).cuda()
target_tensor2 = torch.tensor(list(hh_size_dict.values()), dtype=torch.float32).to(device).cuda()

input_dim = len(hh_comp_dict.keys()) + len(hh_size_dict.keys())
print("Input dimension: ", input_dim)
hidden_dims = [64, 32]

hh_comp_net = FFNetwork(input_dim, hidden_dims, len(hh_comp_dict)).to(device).cuda()
hh_size_net = FFNetwork(input_dim, hidden_dims, len(hh_size_dict)).to(device).cuda()

input_tensor_hh = torch.empty(num_households, input_dim).to(device)
init.kaiming_normal_(input_tensor_hh)

def gumbel_softmax_sample(logits, temperature=0.5):
    gumbel_noise = -torch.log(-torch.log(torch.rand(logits.shape, device=device)))
    y = logits + gumbel_noise
    return torch.nn.functional.softmax(y / temperature, dim=-1)
def generate_households(input_tensor, temperature=0.5):
    hh_comp_logits = hh_comp_net(input_tensor)
    hh_size_logits = hh_size_net(input_tensor)

    hh_comp = gumbel_softmax_sample(hh_comp_logits, temperature)
    hh_size = gumbel_softmax_sample(hh_size_logits, temperature)
    return torch.cat([hh_comp, hh_size], dim=-1)

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

def aggregate_decoded(encoded_tensor):
    # split the encoded tensor into the the two categories and display the first category
    hh_comp_tensor, hh_size_tensor = torch.split(encoded_tensor, [len(hh_comp_dict), len(hh_size_dict)], dim=1)
    hh_comp_labels = list(hh_comp_dict.keys())
    hh_size_labels = list(hh_size_dict.keys())
    hh_comp_decoded = [hh_comp_labels[torch.argmax(hct).item()] for hct in hh_comp_tensor]
    hh_size_decoded = [hh_size_labels[torch.argmax(hst).item()] for hst in hh_size_tensor]

    # aggregating the composition and size separately
    aggregated_hh_comp = {key: 0 for key in hh_comp_dict.keys()}
    aggregated_hh_size = {key: 0 for key in hh_size_dict.keys()}
    for hh_comp, hh_size in zip(hh_comp_decoded, hh_size_decoded):
        aggregated_hh_comp[hh_comp] += 1
        aggregated_hh_size[hh_size] += 1
    return aggregated_hh_comp, aggregated_hh_size

def aggregate_encoded(encoded_tensor):
    # Assume encoded_tensor is split into two parts: hh_comp_tensor and hh_size_tensor
    hh_comp_tensor, hh_size_tensor = torch.split(encoded_tensor, [len(hh_comp_dict), len(hh_size_dict)], dim=1)
    # Use torch.sum to aggregate along the batch dimension (assuming your data is batch-first)
    aggregated_hh_comp = torch.sum(hh_comp_tensor, dim=0)
    aggregated_hh_size = torch.sum(hh_size_tensor, dim=0)
    return aggregated_hh_comp, aggregated_hh_size

def tensor_to_dicts(tensor,  dict):
    labels = list(dict.keys())
    result = {label: count.item() for label, count in zip(labels, tensor)}
    return result

def rmse_accuracy(computed_tensor, target_tensor):
    mse = torch.mean((target_tensor - computed_tensor) ** 2)
    rmse = torch.sqrt(mse)
    max_possible_error = torch.sqrt(torch.sum(target_tensor ** 2))
    accuracy = 1 - (rmse / max_possible_error)
    return accuracy.item()

def combined_rmse_loss(aggregated_tensor1, aggregated_tensor2, target_tensor1, target_tensor2, encoded_tensor):
    # concatenating the target and computed tensors along the characteristic dimension (dim=1)
    concatenated_tensor = torch.cat([target_tensor1, target_tensor2])
    aggregated_cat_tensor = torch.cat([aggregated_tensor1, aggregated_tensor2])
    # calculating RMSE loss on the concatenated tensor
    loss = torch.sqrt(torch.mean((aggregated_cat_tensor - concatenated_tensor) ** 2))
    return loss


def weights_init(m):
    if isinstance(m, nn.Linear):
        # nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)

# households = generate_households(input_tensor_hh)
def evaluate_accuracy(encoded_tensor, category_dicts, household_sizes, doprint=False):
    decoded_results = decode_tensor(encoded_tensor, category_dicts)
    correct_predictions = 0
    total_predictions = len(decoded_results)
    for composition, size in decoded_results:
        size = int(size)  # Ensure that size is treated as an integer
        if evaluate_size(composition, size):
            correct_predictions += 1
    if doprint:
        print(f"Correct Predictions: {correct_predictions}, Total Predictions: {total_predictions}")
    return correct_predictions / total_predictions


# training loop
optimizer = torch.optim.Adam([{'params': hh_comp_net.parameters()},
                              {'params': hh_size_net.parameters()}], lr=0.001)
scheduler = StepLR(optimizer, step_size=20, gamma=0.25)

hh_comp_net.apply(weights_init)
hh_size_net.apply(weights_init)

number_of_epochs = 100
for epoch in range(number_of_epochs+1):
    optimizer.zero_grad()

    encoded_households = generate_households(input_tensor_hh)
    aggregated_hh_comp, aggregated_hh_size  = aggregate_encoded(encoded_households)

    compliance = evaluate_accuracy(encoded_households, [hh_comp_dict, hh_size_dict], household_sizes)

    loss = combined_rmse_loss(aggregated_hh_comp, aggregated_hh_size,
                                 target_tensor1, target_tensor2, encoded_households)

    # Evaluate accuracy based on RMSE and compliance with expected household sizes
    rmse_accuracy1 = rmse_accuracy(aggregated_hh_comp, target_tensor1)
    rmse_accuracy2 = rmse_accuracy(aggregated_hh_size, target_tensor2)
    average_rmse_accuracy = (rmse_accuracy1 + rmse_accuracy2) / 2


    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}, RMSE Accuracy: {average_rmse_accuracy}, Compliance: {(1 - compliance)}")