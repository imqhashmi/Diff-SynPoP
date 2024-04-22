import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
import plotly.graph_objects as go

# Data
total = 500
sex_dict = {'M': 240, 'F': 260}
age_dict = {'Child': 300, 'Adult': 200}
ethnic_dict = {'White': 200, 'Black': 100, 'Asian': 200}

cross_table = {'M-Child-White': 66, 'M-Child-Black': 31, 'M-Child-Asian': 46,
               'M-Adult-White': 38, 'M-Adult-Black': 14, 'M-Adult-Asian': 35,
               'F-Child-White': 67, 'F-Child-Black': 26, 'F-Child-Asian': 71,
               'F-Adult-White': 44, 'F-Adult-Black': 17, 'F-Adult-Asian': 45}

cross_table_tensor = torch.tensor(list(cross_table.values()), dtype=torch.float32)
# Initialize logits as learnable parameters
sex_logits = torch.randn(total, len(sex_dict.keys()), requires_grad=True)
age_logits = torch.randn(total, len(age_dict.keys()), requires_grad=True)
ethnicity_logits = torch.randn(total, len(ethnic_dict.keys()), requires_grad=True)

# def generate_population():
#     population = []
#     sex = random.choices(list(sex_dict.keys()), weights=[p / total for p in list(sex_dict.values())], k=total)
#     age = random.choices(list(age_dict.keys()), weights=[p / total for p in list(age_dict.values())], k=total)
#     ethnicity = random.choices(list(ethnic_dict.keys()), weights=[p / total for p in list(ethnic_dict.values())], k=total)
#     # Combine the three lists into a list of tuples using zip
#     population = [(s, a, e) for s, a, e in zip(sex, age, ethnicity)]
#     return population

# def generate_population():
#     population = []
#     for _ in range(total):
#         sex = random.choice(list(sex_dict.keys()))
#         age = random.choice(list(age_dict.keys()))
#         ethnicity = random.choice(list(ethnic_dict.keys()))
#         population.append((sex, age, ethnicity))
#     return population
def gumbel_softmax_sample(logits, temperature):
    gumbel_noise = -torch.log(-torch.log(torch.rand(logits.shape, device=logits.device)))
    y = logits + gumbel_noise
    return torch.nn.functional.softmax(y / temperature, dim=-1)

def generate_population_gumbel_softmax(temperature=0.5):
    sex = gumbel_softmax_sample(sex_logits, temperature)
    age = gumbel_softmax_sample(age_logits, temperature)
    ethnicity = gumbel_softmax_sample(ethnicity_logits, temperature)
    return torch.cat([sex, age, ethnicity], dim=-1)

def aggregate_population(population):
    cross_table = {}
    # Nested loops to generate all combinations
    for s in list(sex_dict.keys()):
        for a in list(age_dict.keys()):
            for e in list(ethnic_dict.keys()):
                cross_table[s + "-" + a + "-" + e] = 0

    for person in population:
        key = person[0] + "-" + person[1] + "-" + person[2]
        cross_table[key] += 1
    return cross_table

def encode_tensor(population_data):
    sex_num = torch.tensor([list(sex_dict.keys()).index(individual[0]) for individual in population_data], dtype=torch.float32)
    age_num = torch.tensor([list(age_dict.keys()).index(individual[1]) for individual in population_data], dtype=torch.float32)
    ethnic_num = torch.tensor([list(ethnic_dict.keys()).index(individual[2]) for individual in population_data],
                              dtype=torch.float32)
    return torch.stack([sex_num, age_num, ethnic_num], dim=1)

def decode_tensor(encoded_data):
    sex_categories = list(sex_dict.keys())
    age_categories = list(age_dict.keys())
    ethnic_categories = list(ethnic_dict.keys())

    # Convert the encoded tensor to a list of indices
    sex_indices, age_indices, ethnic_indices = encoded_data.split(1, dim=1)
    # Convert indices back to the original categories
    sex_list = [sex_categories[int(idx.item())] for idx in sex_indices]
    age_list = [age_categories[int(idx.item())] for idx in age_indices]
    ethnic_list = [ethnic_categories[int(idx.item())] for idx in ethnic_indices]

    # Combine the decoded lists into a list of tuples
    decoded_population_data = list(zip(sex_list, age_list, ethnic_list))
    return decoded_population_data

pop = generate_population_gumbel_softmax()
# # Assuming you want to access the first person in the tensor
# person = pop[0]
# # Split the vector based on the number of categories
# # Let's assume num_sex_categories, num_age_categories, num_ethnic_categories are defined
# sex_probs, age_probs, ethnicity_probs = torch.split(person, [2,2, 3])
# most_likely_sex = torch.argmax(sex_probs)
# most_likely_age = torch.argmax(age_probs)
# most_likely_ethnicity = torch.argmax(ethnicity_probs)
# sex_categories = list(sex_dict.keys())
# age_categories = list(age_dict.keys())
# ethnic_categories = list(ethnic_dict.keys())
# print(sex_categories[int(most_likely_sex)])
# print(age_categories[int(most_likely_age)])
# print(ethnic_categories[int(most_likely_ethnicity)])



def aggregate_encoded_tensor(encoded_population):
    sex_categories = list(sex_dict.keys())
    age_categories = list(age_dict.keys())
    ethnic_categories = list(ethnic_dict.keys())

    # Get unique rows and their counts
    unique_rows, counts = torch.unique(encoded_population, dim=0, return_counts=True)

    # Initialize tensor to hold the aggregated data, matching the size of the cross_table
    aggregated_tensor = torch.zeros_like(cross_table_tensor)

    # Iterate through the unique rows and update the aggregated tensor
    for i, row in enumerate(unique_rows):
        # Construct the key for cross_table
        # key = f'{sex_dict_inv[int(row[0])]}-{age_dict_inv[int(row[1])]}-{ethnic_dict_inv[int(row[2])]}'
        sex = sex_categories[int(row[0])]
        age = age_categories[int(row[1])]
        ethnicity = ethnic_categories[int(row[2])]
        key = sex + "-" + age + "-" + ethnicity
        key_index = list(cross_table.keys()).index(key)

        # Update the aggregated tensor
        aggregated_tensor[key_index] = counts[i]
    return aggregated_tensor
#
# def compute_accuracy(target, computed):
#     """
#     Compute accuracy as the inverse of normalized Mean Absolute Error (MAE).
#     Accuracy = 1 - (MAE / max_possible_error)
#     max_possible_error is computed as the sum of all values in the target distribution.
#     """
#     total_error = sum(abs(target[k] - computed[k]) for k in target)
#     max_possible_error = sum(target.values())
#     accuracy = 1 - (total_error / max_possible_error)
#     return accuracy
#
#
# # Initialize the encoded population
# encoded_population = encode_tensor(generate_population())
# encoded_population = encoded_population.float().requires_grad_()
#
# # Before your training loop
# torch.autograd.set_detect_anomaly(True)
#
# # Define the loss function
# def loss_function(aggregated_tensor, cross_table_tensor):
#     return torch.mean((aggregated_tensor - cross_table_tensor) ** 2)
#
# # Define the maximum number of iterations
# max_iterations = 1000
# # Define a convergence threshold
# convergence_threshold = 0.00001
# # Optimizer using SGD
# optimizer = optim.SGD([encoded_population], lr=0.1)
# # optimizer = optim.Adam([encoded_population], lr=0.001)
#
# # Optimization loop
# for iteration in range(max_iterations):
#     optimizer.zero_grad()
#
#     # Aggregate the current state of the encoded population
#     aggregated_tensor = aggregate_encoded_tensor(encoded_population).requires_grad_()
#
#     # Compute loss
#     loss = loss_function(aggregated_tensor, cross_table_tensor)
#     print(loss)  # Check if loss is valid and not zero
#
#     # Backward pass and optimization
#     loss.backward()
#     optimizer.step()
#
#     # Convergence check
#     if loss < convergence_threshold:
#         print(f'Converged at iteration {iteration} with loss {loss}')
#         break
#
# # Decode the final population for inspection or use
# computed_population = decode_tensor(encoded_population.detach())
#
#
# def plot(target, computed):
#     accuracy = compute_accuracy(target, computed)
#     # Create bar plots for both dictionaries
#     fig = go.Figure()
#     fig.add_trace(go.Bar(
#         x=list(target.keys()),
#         y=list(target.values()),
#         name='Target',
#         marker_color='indianred'
#     ))
#     fig.add_trace(go.Bar(
#         x=list(computed.keys()),
#         y=list(computed.values()),
#         name='Computed',
#         marker_color='blue'
#     ))
#
#     # Update layout with accuracy
#     fig.update_layout(
#         title=f'Comparison of Target and Computed Distributions<br>Model Accuracy: {accuracy:.2%}',
#         xaxis_tickangle=-45,
#         xaxis_title='Categories',
#         yaxis_title='Counts',
#         barmode='group'
#     )
#
#     # Show plot
#     fig.show()
#
# plot(cross_table, aggregate_population(computed_population))
