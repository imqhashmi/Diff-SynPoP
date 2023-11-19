import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
import plotly.graph_objects as go
import InputData as ID
import InputCrossTables as ICT

# Data
#MSOA
area = 'E02005924'
total = ID.get_total(ID.age5ydf, area)

sex_dict = ID.getdictionary(ID.sexdf, area)
age_dict = ID.getdictionary(ID.age5ydf, area)
ethnic_dict = ID.getdictionary(ID.ethnicdf, area)
cross_table = ICT.getdictionary(ICT.ethnic_by_sex_by_age, area)

cross_table_tensor = torch.tensor(list(cross_table.values()), dtype=torch.float32)
# Initialize logits as learnable parameters
sex_logits = torch.randn(total, len(sex_dict.keys()), requires_grad=True)
age_logits = torch.randn(total, len(age_dict.keys()), requires_grad=True)
ethnicity_logits = torch.randn(total, len(ethnic_dict.keys()), requires_grad=True)

if torch.cuda.is_available():
    sex_logits = torch.randn(total, len(sex_dict.keys()), requires_grad=True, device='cuda')
    age_logits = torch.randn(total, len(age_dict.keys()), requires_grad=True, device='cuda')
    ethnicity_logits = torch.randn(total, len(ethnic_dict.keys()), requires_grad=True, device='cuda')
else:
    sex_logits = torch.randn(total, len(sex_dict.keys()), requires_grad=True)
    age_logits = torch.randn(total, len(age_dict.keys()), requires_grad=True)
    ethnicity_logits = torch.randn(total, len(ethnic_dict.keys()), requires_grad=True)

# print("Is sex_logits a leaf tensor?", sex_logits.is_leaf)
# print("Is age_logits a leaf tensor?", age_logits.is_leaf)
# print("Is ethnicity_logits a leaf tensor?", ethnicity_logits.is_leaf)


def gumbel_softmax_sample(logits, temperature):
    if torch.cuda.is_available():
        gumbel_noise = -torch.log(-torch.log(torch.rand(logits.shape, device='cuda')))
    else:
        gumbel_noise = -torch.log(-torch.log(torch.rand(logits.shape)))
    y = logits + gumbel_noise
    return torch.nn.functional.softmax(y / temperature, dim=-1)

def generate_population_gumbel_softmax(temperature=0.5):
    sex = gumbel_softmax_sample(sex_logits, temperature)
    age = gumbel_softmax_sample(age_logits, temperature)
    ethnicity = gumbel_softmax_sample(ethnicity_logits, temperature)
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

number_of_epochs = 10000 # or any other suitable number of epochs
temperature = 0.75  # Initial temperature for Gumbel-Softmax


# optimizer = torch.optim.Adam([sex_logits, age_logits, ethnicity_logits], lr=0.01)
optimizer = torch.optim.SGD([sex_logits, age_logits, ethnicity_logits], lr=0.01, momentum=0.9)
# optimizer = torch.optim.RMSprop([sex_logits, age_logits, ethnicity_logits], lr=0.001, alpha=0.99)

# Loss function
def mse_loss(aggregated_tensor, target_tensor):
    return torch.mean((aggregated_tensor - target_tensor) ** 2)

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
    encoded_population = generate_population_gumbel_softmax(temperature).cuda()
    aggregated_population = aggregate_softmax_encoded_tensor(encoded_population).cuda()

    # Compute and backpropagate loss
    loss = mse_loss(aggregated_population, cross_table_tensor.cuda())
    loss.backward()

    # Debugging: Print gradients
    if epoch % 10 == 0:  # Adjust this to print less often if needed
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        # print("Sex logits grad:", sex_logits.grad.norm().item())
        # print("Age logits grad:", age_logits.grad.norm().item())
        # print("Ethnicity logits grad:", ethnicity_logits.grad.norm().item())
    optimizer.step()

# Decode the final population
final_population = decode_tensor(encoded_population.detach())

# Aggregate the final population
final_aggregated_population = aggregate_softmax_encoded_tensor(encoded_population.detach())

# def compute_percentage_accuracy(target_tensor, computed_tensor):
#     """
#     Compute accuracy as the inverse of normalized Mean Absolute Error (MAE).
#     Accuracy is expressed as a percentage.
#     Accuracy = 100 * (1 - (MAE / max_possible_error))
#     where MAE = mean(abs(target_tensor - computed_tensor))
#     and max_possible_error is the sum of all values in the target_tensor.
#     """
#     mae = torch.mean(torch.abs(target_tensor - computed_tensor))
#     max_possible_error = torch.sum(target_tensor)
#     accuracy = 100 * (1 - (mae / max_possible_error))
#     return accuracy.item()  # Convert from tensor to a regular float

def rmse_accuracy(target_tensor, computed_tensor):
    """
    Compute accuracy based on the Root Mean Squared Error (RMSE).
    Accuracy = 1 - (RMSE / max_possible_error)
    where RMSE = sqrt(mean((target_tensor - computed_tensor) ** 2))
    and max_possible_error is computed as the square root of the sum of squares of the target distribution.
    """
    mse = torch.mean((target_tensor - computed_tensor) ** 2)
    rmse = torch.sqrt(mse)
    max_possible_error = torch.sqrt(torch.sum(target_tensor ** 2))
    accuracy = 1 - (rmse / max_possible_error)
    return accuracy.item()

def plot(target, computed):
    accuracy = rmse_accuracy(target.cpu(), computed.cpu())
    # Create bar plots for both dictionaries
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
        title= 'Model Accuracy:' + str(accuracy),
        xaxis_tickangle=-90,
        xaxis_title='Categories',
        yaxis_title='Counts',
        barmode='group',
        bargap=0.5,
        width=9000
    )

    # Show plot
    fig.show()

plot(cross_table_tensor, final_aggregated_population)
