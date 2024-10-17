import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mpld3 as mpld3
import plotly.graph_objs as go
import plotly.io as pio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import commons as cm

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

device = 'cuda' if torch.cuda.is_available() else 'cpu' # checking to see if a cuda device is available
num_epochs = 1000

total_households = 2835
hh_comp = {'1PE': 485, '1PA': 590, '1FE': 192, '1FM-0C': 310, '1FM-nA': 92, '1FC-0C': 135, '1FC-nA': 5, '1FL-nA': 57, '1H-nS': 37, '1H-nE': 18, '1H-nA': 184, '1FM-2C': 489, '1FC-2C': 38, '1FL-2C': 144, '1H-2C': 44}
hh_size =  {'1': 1051, '2': 864, '3': 405, '4': 347, '5': 116, '6': 36, '7': 13, '8': 3}
hh_ethnic =  {'W': 6066, 'M': 272, 'A': 581, 'B': 135, 'O': 155}
hh_religion = {'C': 3628, 'B': 76, 'H': 80, 'J': 88, 'M': 293, 'S': 4, 'O': 41, 'N': 2356, 'NS': 643}

hh_comp_by_size = {'1PE 1': 467, '1PE 2': 0, '1PE 3': 0, '1PE 4': 0, '1PE 5': 0, '1PE 6': 0, '1PE 7': 0, '1PE 8': 0, '1PA 1': 607, '1PA 2': 0, '1PA 3': 0, '1PA 4': 0, '1PA 5': 0, '1PA 6': 0, '1PA 7': 0, '1PA 8': 0, '1FE 1': 205, '1FE 2': 0, '1FE 3': 0, '1FE 4': 0, '1FE 5': 0, '1FE 6': 0, '1FE 7': 0, '1FE 8': 0, '1FM-0C 1': 0, '1FM-0C 2': 332, '1FM-0C 3': 0, '1FM-0C 4': 0, '1FM-0C 5': 0, '1FM-0C 6': 0, '1FM-0C 7': 0, '1FM-0C 8': 0, '1FM-nA 1': 0, '1FM-nA 2': 0, '1FM-nA 3': 48, '1FM-nA 4': 31, '1FM-nA 5': 6, '1FM-nA 6': 0, '1FM-nA 7': 4, '1FM-nA 8': 1, '1FC-0C 1': 0, '1FC-0C 2': 132, '1FC-0C 3': 0, '1FC-0C 4': 0, '1FC-0C 5': 0, '1FC-0C 6': 0, '1FC-0C 7': 0, '1FC-0C 8': 0, '1FC-nA 1': 0, '1FC-nA 2': 0, '1FC-nA 3': 3, '1FC-nA 4': 5, '1FC-nA 5': 0, '1FC-nA 6': 0, '1FC-nA 7': 0, '1FC-nA 8': 0, '1FL-nA 1': 0, '1FL-nA 2': 0, '1FL-nA 3': 16, '1FL-nA 4': 19, '1FL-nA 5': 10, '1FL-nA 6': 0, '1FL-nA 7': 3, '1FL-nA 8': 1, '1H-nS 1': 0, '1H-nS 2': 0, '1H-nS 3': 13, '1H-nS 4': 13, '1H-nS 5': 3, '1H-nS 6': 0, '1H-nS 7': 2, '1H-nS 8': 0, '1H-nE 1': 0, '1H-nE 2': 0, '1H-nE 3': 6, '1H-nE 4': 6, '1H-nE 5': 0, '1H-nE 6': 0, '1H-nE 7': 0, '1H-nE 8': 0, '1H-nA 1': 0, '1H-nA 2': 0, '1H-nA 3': 87, '1H-nA 4': 70, '1H-nA 5': 26, '1H-nA 6': 0, '1H-nA 7': 3, '1H-nA 8': 1, '1FM-2C 1': 0, '1FM-2C 2': 0, '1FM-2C 3': 204, '1FM-2C 4': 171, '1FM-2C 5': 80, '1FM-2C 6': 2, '1FM-2C 7': 21, '1FM-2C 8': 9, '1FC-2C 1': 0, '1FC-2C 2': 0, '1FC-2C 3': 17, '1FC-2C 4': 16, '1FC-2C 5': 6, '1FC-2C 6': 0, '1FC-2C 7': 5, '1FC-2C 8': 2, '1FL-2C 1': 0, '1FL-2C 2': 0, '1FL-2C 3': 66, '1FL-2C 4': 59, '1FL-2C 5': 10, '1FL-2C 6': 2, '1FL-2C 7': 2, '1FL-2C 8': 1, '1H-2C 1': 0, '1H-2C 2': 0, '1H-2C 3': 16, '1H-2C 4': 16, '1H-2C 5': 7, '1H-2C 6': 0, '1H-2C 7': 3, '1H-2C 8': 0}
hh_comp_by_ethnic = {'1PE W': 436, '1PE M': 1, '1PE A': 6, '1PE B': 4, '1PE O': 0, '1PA W': 536, '1PA M': 21, '1PA A': 34, '1PA B': 8, '1PA O': 5, '1FE W': 162, '1FE M': 0, '1FE A': 7, '1FE B': 0, '1FE O': 3, '1FM-0C W': 263, '1FM-0C M': 6, '1FM-0C A': 19, '1FM-0C B': 1, '1FM-0C O': 4, '1FM-2C W': 388, '1FM-2C M': 3, '1FM-2C A': 52, '1FM-2C B': 9, '1FM-2C O': 21, '1FM-nA W': 75, '1FM-nA M': 0, '1FM-nA A': 8, '1FM-nA B': 2, '1FM-nA O': 6, '1FC-0C W': 129, '1FC-0C M': 5, '1FC-0C A': 12, '1FC-0C B': 1, '1FC-0C O': 3, '1FC-2C W': 55, '1FC-2C M': 0, '1FC-2C A': 2, '1FC-2C B': 0, '1FC-2C O': 2, '1FC-nA W': 5, '1FC-nA M': 0, '1FC-nA A': 0, '1FC-nA B': 0, '1FC-nA O': 1, '1FL-2C W': 130, '1FL-2C M': 10, '1FL-2C A': 13, '1FL-2C B': 10, '1FL-2C O': 0, '1FL-nA W': 62, '1FL-nA M': 0, '1FL-nA A': 3, '1FL-nA B': 0, '1FL-nA O': 3, '1H-2C W': 40, '1H-2C M': 4, '1H-2C A': 5, '1H-2C B': 5, '1H-2C O': 1, '1H-nS W': 8, '1H-nS M': 3, '1H-nS A': 4, '1H-nS B': 0, '1H-nS O': 0, '1H-nE W': 9, '1H-nE M': 1, '1H-nE A': 0, '1H-nE B': 0, '1H-nE O': 1, '1H-nA W': 194, '1H-nA M': 7, '1H-nA A': 23, '1H-nA B': 2, '1H-nA O': 2}
hh_comp_by_religion =  {'1PE C': 316, '1PE B': 1, '1PE H': 2, '1PE J': 7, '1PE M': 4, '1PE S': 0, '1PE O': 1, '1PE N': 77, '1PE NS': 39, '1PA C': 282, '1PA B': 4, '1PA H': 3, '1PA J': 4, '1PA M': 18, '1PA S': 0, '1PA O': 13, '1PA N': 222, '1PA NS': 58, '1FE C': 101, '1FE B': 0, '1FE H': 3, '1FE J': 4, '1FE M': 2, '1FE S': 0, '1FE O': 0, '1FE N': 48, '1FE NS': 14, '1FM-0C C': 143, '1FM-0C B': 4, '1FM-0C H': 4, '1FM-0C J': 6, '1FM-0C M': 6, '1FM-0C S': 2, '1FM-0C O': 3, '1FM-0C N': 97, '1FM-0C NS': 28, '1FM-2C C': 233, '1FM-2C B': 4, '1FM-2C H': 10, '1FM-2C J': 7, '1FM-2C M': 29, '1FM-2C S': 0, '1FM-2C O': 2, '1FM-2C N': 150, '1FM-2C NS': 38, '1FM-nA C': 44, '1FM-nA B': 4, '1FM-nA H': 1, '1FM-nA J': 0, '1FM-nA M': 4, '1FM-nA S': 0, '1FM-nA O': 2, '1FM-nA N': 30, '1FM-nA NS': 6, '1FC-0C C': 40, '1FC-0C B': 1, '1FC-0C H': 3, '1FC-0C J': 0, '1FC-0C M': 3, '1FC-0C S': 0, '1FC-0C O': 1, '1FC-0C N': 91, '1FC-0C NS': 11, '1FC-2C C': 21, '1FC-2C B': 1, '1FC-2C H': 0, '1FC-2C J': 1, '1FC-2C M': 3, '1FC-2C S': 0, '1FC-2C O': 0, '1FC-2C N': 27, '1FC-2C NS': 6, '1FC-nA C': 1, '1FC-nA B': 0, '1FC-nA H': 0, '1FC-nA J': 0, '1FC-nA M': 1, '1FC-nA S': 0, '1FC-nA O': 0, '1FC-nA N': 4, '1FC-nA NS': 0, '1FL-2C C': 78, '1FL-2C B': 2, '1FL-2C H': 0, '1FL-2C J': 0, '1FL-2C M': 13, '1FL-2C S': 0, '1FL-2C O': 1, '1FL-2C N': 60, '1FL-2C NS': 9, '1FL-nA C': 39, '1FL-nA B': 0, '1FL-nA H': 0, '1FL-nA J': 0, '1FL-nA M': 2, '1FL-nA S': 0, '1FL-nA O': 0, '1FL-nA N': 14, '1FL-nA NS': 13, '1H-2C C': 26, '1H-2C B': 0, '1H-2C H': 2, '1H-2C J': 0, '1H-2C M': 2, '1H-2C S': 0, '1H-2C O': 0, '1H-2C N': 23, '1H-2C NS': 2, '1H-nS C': 4, '1H-nS B': 2, '1H-nS H': 0, '1H-nS J': 0, '1H-nS M': 0, '1H-nS S': 0, '1H-nS O': 0, '1H-nS N': 7, '1H-nS NS': 2, '1H-nE C': 8, '1H-nE B': 0, '1H-nE H': 0, '1H-nE J': 0, '1H-nE M': 1, '1H-nE S': 0, '1H-nE O': 0, '1H-nE N': 1, '1H-nE NS': 1, '1H-nA C': 95, '1H-nA B': 5, '1H-nA H': 5, '1H-nA J': 2, '1H-nA M': 6, '1H-nA S': 0, '1H-nA O': 3, '1H-nA N': 88, '1H-nA NS': 24}
hh_comp_by_sex_by_age = {'M 0_15 1PE': 0, 'M 0_15 1PA': 1, 'M 0_15 1FE': 0, 'M 0_15 1FM-0C': 0, 'M 0_15 1FM-2C': 386, 'M 0_15 1FM-nA': 0, 'M 0_15 1FC-0C': 0, 'M 0_15 1FC-2C': 35, 'M 0_15 1FC-nA': 0, 'M 0_15 1FL-2C': 126, 'M 0_15 1FL-nA': 0, 'M 0_15 1H-2C': 44, 'M 0_15 1H-nS': 0, 'M 0_15 1H-nE': 0, 'M 0_15 1H-nA': 0, 'M 16_24 1PE': 0, 'M 16_24 1PA': 9, 'M 16_24 1FE': 0, 'M 16_24 1FM-0C': 3, 'M 16_24 1FM-2C': 68, 'M 16_24 1FM-nA': 40, 'M 16_24 1FC-0C': 11, 'M 16_24 1FC-2C': 7, 'M 16_24 1FC-nA': 2, 'M 16_24 1FL-2C': 39, 'M 16_24 1FL-nA': 20, 'M 16_24 1H-2C': 20, 'M 16_24 1H-nS': 11, 'M 16_24 1H-nE': 0, 'M 16_24 1H-nA': 55, 'M 25_34 1PE': 0, 'M 25_34 1PA': 94, 'M 25_34 1FE': 0, 'M 25_34 1FM-0C': 63, 'M 25_34 1FM-2C': 62, 'M 25_34 1FM-nA': 22, 'M 25_34 1FC-0C': 104, 'M 25_34 1FC-2C': 17, 'M 25_34 1FC-nA': 1, 'M 25_34 1FL-2C': 1, 'M 25_34 1FL-nA': 10, 'M 25_34 1H-2C': 16, 'M 25_34 1H-nS': 9, 'M 25_34 1H-nE': 0, 'M 25_34 1H-nA': 153, 'M 35_49 1PE': 0, 'M 35_49 1PA': 116, 'M 35_49 1FE': 0, 'M 35_49 1FM-0C': 47, 'M 35_49 1FM-2C': 290, 'M 35_49 1FM-nA': 16, 'M 35_49 1FC-0C': 29, 'M 35_49 1FC-2C': 31, 'M 35_49 1FC-nA': 2, 'M 35_49 1FL-2C': 7, 'M 35_49 1FL-nA': 9, 'M 35_49 1H-2C': 20, 'M 35_49 1H-nS': 1, 'M 35_49 1H-nE': 0, 'M 35_49 1H-nA': 62, 'M 50+ 1PE': 118, 'M 50+ 1PA': 104, 'M 50+ 1FE': 171, 'M 50+ 1FM-0C': 180, 'M 50+ 1FM-2C': 123, 'M 50+ 1FM-nA': 82, 'M 50+ 1FC-0C': 10, 'M 50+ 1FC-2C': 8, 'M 50+ 1FC-nA': 4, 'M 50+ 1FL-2C': 5, 'M 50+ 1FL-nA': 21, 'M 50+ 1H-2C': 22, 'M 50+ 1H-nS': 0, 'M 50+ 1H-nE': 6, 'M 50+ 1H-nA': 49, 'F 0_15 1PE': 0, 'F 0_15 1PA': 0, 'F 0_15 1FE': 0, 'F 0_15 1FM-0C': 0, 'F 0_15 1FM-2C': 410, 'F 0_15 1FM-nA': 0, 'F 0_15 1FC-0C': 0, 'F 0_15 1FC-2C': 60, 'F 0_15 1FC-nA': 0, 'F 0_15 1FL-2C': 106, 'F 0_15 1FL-nA': 0, 'F 0_15 1H-2C': 41, 'F 0_15 1H-nS': 0, 'F 0_15 1H-nE': 0, 'F 0_15 1H-nA': 0, 'F 16_24 1PE': 0, 'F 16_24 1PA': 16, 'F 16_24 1FE': 0, 'F 16_24 1FM-0C': 5, 'F 16_24 1FM-2C': 68, 'F 16_24 1FM-nA': 25, 'F 16_24 1FC-0C': 18, 'F 16_24 1FC-2C': 11, 'F 16_24 1FC-nA': 3, 'F 16_24 1FL-2C': 37, 'F 16_24 1FL-nA': 20, 'F 16_24 1H-2C': 25, 'F 16_24 1H-nS': 7, 'F 16_24 1H-nE': 0, 'F 16_24 1H-nA': 82, 'F 25_34 1PE': 0, 'F 25_34 1PA': 65, 'F 25_34 1FE': 0, 'F 25_34 1FM-0C': 74, 'F 25_34 1FM-2C': 75, 'F 25_34 1FM-nA': 13, 'F 25_34 1FC-0C': 96, 'F 25_34 1FC-2C': 22, 'F 25_34 1FC-nA': 0, 'F 25_34 1FL-2C': 34, 'F 25_34 1FL-nA': 8, 'F 25_34 1H-2C': 13, 'F 25_34 1H-nS': 7, 'F 25_34 1H-nE': 0, 'F 25_34 1H-nA': 143, 'F 35_49 1PE': 0, 'F 35_49 1PA': 76, 'F 35_49 1FE': 0, 'F 35_49 1FM-0C': 40, 'F 35_49 1FM-2C': 304, 'F 35_49 1FM-nA': 16, 'F 35_49 1FC-0C': 23, 'F 35_49 1FC-2C': 27, 'F 35_49 1FC-nA': 3, 'F 35_49 1FL-2C': 76, 'F 35_49 1FL-nA': 13, 'F 35_49 1H-2C': 37, 'F 35_49 1H-nS': 0, 'F 35_49 1H-nE': 0, 'F 35_49 1H-nA': 41, 'F 50+ 1PE': 329, 'F 50+ 1PA': 123, 'F 50+ 1FE': 173, 'F 50+ 1FM-0C': 174, 'F 50+ 1FM-2C': 95, 'F 50+ 1FM-nA': 80, 'F 50+ 1FC-0C': 9, 'F 50+ 1FC-2C': 6, 'F 50+ 1FC-nA': 3, 'F 50+ 1FL-2C': 30, 'F 50+ 1FL-nA': 48, 'F 50+ 1H-2C': 29, 'F 50+ 1H-nS': 0, 'F 50+ 1H-nE': 17, 'F 50+ 1H-nA': 78}

class HouseholdModel(nn.Module):
    def __init__(self, input_size, hidden_layers, output_sizes):
        super(HouseholdModel, self).__init__()
        layers = []
        in_dim = input_size
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.hidden_layers = nn.Sequential(*layers)
        self.size_output = nn.Linear(in_dim, output_sizes['size'])
        self.composition_output = nn.Linear(in_dim, output_sizes['composition'])
        self.ethnic_output = nn.Linear(in_dim, output_sizes['ethnic'])
        self.religion_output = nn.Linear(in_dim, output_sizes['religion'])

    def forward(self, x, temperature=0.75):
        x = self.hidden_layers(x)
        size_output = F.gumbel_softmax(self.size_output(x), tau=temperature, hard=False)
        composition_output = F.gumbel_softmax(self.composition_output(x), tau=temperature, hard=False)
        ethnic_output = F.gumbel_softmax(self.ethnic_output(x), tau=temperature, hard=False)
        religion_output = F.gumbel_softmax(self.religion_output(x), tau=temperature, hard=False)
        return size_output, composition_output, ethnic_output, religion_output

def aggregate(outputs, cross_table, category_dicts):
    """
    Aggregates soft counts based on the output tensors and cross table, filtering out zero values.

    Parameters:
    - outputs: List of tensors corresponding to each characteristic's probabilities.
    - cross_table: Dictionary representing the target cross table.
    - category_dicts: List of dictionaries for each category (e.g., [sex_dict, age_dict]).

    Returns:
    - Aggregated tensor of counts based on the cross table.
    """
    keys = [key for key, value in cross_table.items()]
    aggregated_tensor = torch.zeros(len(keys), device=device)

    for i, key in enumerate(keys):
        category_keys = key.split(' ')
        expected_count = torch.ones(outputs[0].size(0), device=device)
        for output, category_key, category_dict in zip(outputs, category_keys, category_dicts):
            category_index = list(category_dict.keys()).index(category_key)
            expected_count *= output[:, category_index]
        aggregated_tensor[i] = torch.sum(expected_count)
    return aggregated_tensor

def decode_outputs(size_output, composition_output, ethnic_output, religion_output):
    size_decoded = size_output.argmax(dim=1).tolist()
    composition_decoded = composition_output.argmax(dim=1).tolist()
    ethnic_decoded = ethnic_output.argmax(dim=1).tolist()
    religion_decoded = religion_output.argmax(dim=1).tolist()

    size_labels = [list(hh_size.keys())[idx] for idx in size_decoded]
    composition_labels = [list(hh_comp.keys())[idx] for idx in composition_decoded]
    ethnic_labels = [list(hh_ethnic.keys())[idx] for idx in ethnic_decoded]
    religion_labels = [list(hh_religion.keys())[idx] for idx in religion_decoded]

    decoded_df = pd.DataFrame({
        'Size': size_labels,
        'Composition': composition_labels,
        'Ethnic' : ethnic_labels,
        'Religion': religion_labels})

    return decoded_df

input_size = 10
hidden_layers = [128, 64]
output_sizes = {
    'size': len(hh_size),
    'composition': len(hh_comp),
    'ethnic': len(hh_ethnic),
    'religion': len(hh_religion)
}

model = HouseholdModel(input_size, hidden_layers, output_sizes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    optimizer.zero_grad()
    x = torch.randn(total_households, input_size, device=device, requires_grad=True)
    size_output, composition_output, ethnic_output, religion_output = model(x)

    size_aggregated = aggregate([composition_output, size_output], hh_comp_by_size, [hh_comp, hh_size])
    ethnic_aggregated = aggregate([composition_output, ethnic_output], hh_comp_by_ethnic, [hh_comp, hh_ethnic])
    religion_aggregated = aggregate([composition_output, religion_output], hh_comp_by_religion, [hh_comp, hh_religion])

    size_target = torch.tensor([value for value in hh_comp_by_size.values()], dtype=torch.float32, device=device)
    ethnic_target = torch.tensor([value for value in hh_comp_by_ethnic.values()], dtype=torch.float32, device=device)
    religion_target = torch.tensor([value for value in hh_comp_by_religion.values()], dtype=torch.float32, device=device)

    # Calculate loss and backpropagate
    size_loss = torch.sqrt(nn.functional.mse_loss(size_aggregated, size_target))
    ethnic_loss = torch.sqrt(nn.functional.mse_loss(ethnic_aggregated, ethnic_target))
    religion_loss = torch.sqrt(nn.functional.mse_loss(religion_aggregated, religion_target))
    total_loss = size_loss +  ethnic_loss + religion_loss
    total_loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(epoch, ": ", 'Total Loss:', total_loss.item(),
              'Size Loss:', size_loss.item(),
              'Ethnic Loss', ethnic_loss.item(),
              'Religion Loss', religion_loss.item())

df = decode_outputs(size_output, composition_output, ethnic_output, religion_output)