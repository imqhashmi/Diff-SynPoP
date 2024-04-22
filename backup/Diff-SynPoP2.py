import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
import plotly.graph_objects as go
import InputData as ID
import InputCrossTables as ICT
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#MSOA
area = 'E02005924'
total = ID.get_total(ID.age5ydf, area)

sex_dict = ID.getdictionary(ID.sexdf, area)
age_dict = ID.getdictionary(ID.age5ydf, area)
ethnic_dict = ID.getdictionary(ID.ethnicdf, area)
religion_dict = ID.getdictionary(ID.religiondf, area)
mstatusdf_dict = ID.getdictionary(ID.mstatusdf, area)
qualdf_dict = ID.getdictionary(ID.qualdf, area)

sex_age_ethnic = ICT.getdictionary(ICT.ethnic_by_sex_by_age, area)
sex_age_religion = ICT.getdictionary(ICT.religion_by_sex_by_age, area)
sex_age_mstatus = ICT.getdictionary(ICT.marital_by_sex_by_age, area)
sex_age_qual = ICT.getdictionary(ICT.qualification_by_sex_by_age, area)



cross_table_tensor = torch.tensor(list(cross_table.values()), dtype=torch.float32)

def calculate_proportions(dictionary):
    total_count = sum(dictionary.values())
    proportions = [count / total_count for count in dictionary.values()]
    # Convert to log probabilities; add a small value to avoid log(0)
    log_probs = torch.log(torch.tensor(proportions) + 1e-10)
    return log_probs

# Create leaf tensors for logits
sex_logits = torch.empty(total, len(sex_dict), requires_grad=True, device=device)
age_logits = torch.empty(total, len(age_dict), requires_grad=True, device=device)
ethnicity_logits = torch.empty(total, len(ethnic_dict), requires_grad=True, device=device)

# Set the values of the logits tensors
sex_logits.data = calculate_proportions(sex_dict).repeat(total, 1)
age_logits.data = calculate_proportions(age_dict).repeat(total, 1)
ethnicity_logits.data = calculate_proportions(ethnic_dict).repeat(total, 1)

# check if logits are leaf tensors
# print("Is sex_logits a leaf tensor?", sex_logits.is_leaf)
# print("Is age_logits a leaf tensor?", age_logits.is_leaf)
# print("Is ethnicity_logits a leaf tensor?", ethnicity_logits.is_leaf)


def gumbel_softmax_sample(logits, temperature):
    gumbel_noise = -torch.log(-torch.log(torch.rand(logits.shape, device=device)))
    y = logits + gumbel_noise
    return torch.nn.functional.softmax(y / temperature, dim=-1)

def generate_population_gumbel_softmax(temperature=0.5):
    sex = F.gumbel_softmax(sex_logits, tau=temperature, hard=False)
    age = F.gumbel_softmax(age_logits, tau=temperature, hard=False)
    ethnicity = F.gumbel_softmax(ethnicity_logits, tau=temperature, hard=False)
    return torch.cat([sex, age, ethnicity], dim=-1)

def aggregate_softmax_encoded_tensor(encoded_tensor):
    # Using a simpler implementation to ensure gradient flow
    aggregated_tensor = torch.zeros(len(cross_table.keys()), device=encoded_tensor.device)

    sex_categories = list(sex_dict.keys())
    age_categories = list(age_dict.keys())
    ethnic_categories = list(ethnic_dict.keys())

    # Make sure no in-place operation that disrupts gradient flow
    sex_probs, age_probs, ethnicity_probs = torch.split(encoded_tensor,
                [len(sex_dict), len(age_dict), len(ethnic_dict)], dim=1)

    for i, key in enumerate(cross_table.keys()):
        sex_key, age_key, ethnicity_key = key.split(' ')
        sex_index = sex_categories.index(sex_key)
        age_index = age_categories.index(age_key)
        ethnicity_index = ethnic_categories.index(ethnicity_key)
        # Use a simpler aggregation method to ensure gradients flow
        expected_count = torch.sum(sex_probs[:, sex_index] * age_probs[:, age_index] * ethnicity_probs[:, ethnicity_index])
        aggregated_tensor[i] = expected_count
    return aggregated_tensor

def decode_tensor(encoded_tensor):
    sex_categories = list(sex_dict.keys())
    age_categories = list(age_dict.keys())
    ethnic_categories = list(ethnic_dict.keys())

    # split encoded tensor into individual categories
    encoded_sex, encoded_age, encoded_ethnicity = torch.split(encoded_tensor,[len(sex_categories), len(age_categories),
                                                      len(ethnic_categories)], dim=1)

    decoded_sex = [sex_categories[torch.argmax(s).item()] for s in encoded_sex]
    decoded_age = [age_categories[torch.argmax(a).item()] for a in encoded_age]
    decoded_ethnicity = [ethnic_categories[torch.argmax(e).item()] for e in encoded_ethnicity]
    return list(zip(decoded_sex, decoded_age, decoded_ethnicity))

# test above functions:
# population = generate_population_gumbel_softmax()
# aggregated_population = aggregate_softmax_encoded_tensor(population)
# decoded_population = decode_tensor(population)

number_of_epochs = 100 # or any other suitable number of epochs
temperature = 0.5  # Initial temperature for Gumbel-Softmax


# optimizer = torch.optim.Adam([sex_logits, age_logits, ethnicity_logits], lr=0.01)
optimizer = torch.optim.SGD([sex_logits, age_logits, ethnicity_logits], lr=0.001, momentum=0.9)
# optimizer = torch.optim.RMSprop([sex_logits, age_logits, ethnicity_logits], lr=0.001, alpha=0.99)

# Loss function
def mse_loss(aggregated_tensor, target_tensor):
    return torch.mean((aggregated_tensor - target_tensor) ** 2)
def rmse_loss(aggregated_tensor, target_tensor):
    return torch.sqrt(torch.mean((aggregated_tensor - target_tensor) ** 2))

def kl_divergence_loss(generated_distribution, target_distribution):
    # Adding a small epsilon to avoid log(0)
    epsilon = 1e-8
    generated_distribution = generated_distribution + epsilon
    target_distribution = target_distribution + epsilon

    # Normalizing distributions to sum to 1
    generated_distribution = generated_distribution / generated_distribution.sum()
    target_distribution = target_distribution / target_distribution.sum()

    kl_div = (target_distribution * torch.log(target_distribution / generated_distribution)).sum()
    return kl_div

# Training loop
for epoch in range(number_of_epochs):
    optimizer.zero_grad()

    # Generate and aggregate encoded population
    encoded_population = generate_population_gumbel_softmax(temperature)
    aggregated_population = aggregate_softmax_encoded_tensor(encoded_population)

    # Compute and backpropagate loss
    loss = rmse_loss(aggregated_population, cross_table_tensor)

    # Check for NaN in loss
    if torch.isnan(loss):
        print(f"NaN detected in loss at epoch {epoch}")
        break

    loss.backward()
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_([sex_logits, age_logits, ethnicity_logits], max_norm=1.0)

    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")