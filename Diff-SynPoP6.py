import torch
import torch.nn as nn
import torch.optim as optim
import InputData as ID
import InputCrossTables as ICT
import plotly.graph_objects as go
import  plotly as py
import pandas as pd
import time
import os

# Define the Feedforward Neural Network
class FFNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(FFNetwork, self).__init__()
        self.layers = nn.ModuleList()
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(input_dim if i == 0 else hidden_dims[i-1], hidden_dim))
            self.layers.append(nn.ReLU())
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
path = os.path.join(os.path.dirname(os.getcwd()), 'Diff-SynPoP')

print(path)
#MSOA
area = 'E02005924'
total = ID.get_total(ID.age5ydf, area)

sex_dict = ID.getdictionary(ID.sexdf, area)
age_dict = ID.getdictionary(ID.age5ydf, area)
ethnic_dict = ID.getdictionary(ID.ethnicdf, area)
religion_dict = ID.getdictionary(ID.religiondf, area)
marital_dict = ID.getdictionary(ID.maritaldf, area)
qual_dict = ID.getdictionary(ID.qualdf, area)


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
cross_table3 = ICT.convert_marital_cross_table(ICT.getdictionary(ICT.marital_by_sex_by_age, area))
cross_table4 = ICT.convert_qualification_cross_table(ICT.getdictionary(ICT.qualification_by_sex_by_age, area))


cross_table_tensor1 = torch.tensor(list(cross_table1.values()), dtype=torch.float32).to(device)
cross_table_tensor2 = torch.tensor(list(cross_table2.values()), dtype=torch.float32).to(device)
cross_table_tensor3 = torch.tensor(list(cross_table3.values()), dtype=torch.float32).to(device)
cross_table_tensor4 = torch.tensor(list(cross_table4.values()), dtype=torch.float32).to(device)

# Instantiate networks for each characteristic
input_dim = len(sex_dict.keys()) + len(age_dict.keys()) + len(ethnic_dict.keys()) + \
            len(religion_dict.keys()) + len(marital_dict.keys()) + len(qual_dict.keys())

hidden_dims = [64, 32]

sex_net = FFNetwork(input_dim, hidden_dims, len(sex_dict)).to(device)
age_net = FFNetwork(input_dim, hidden_dims, len(age_dict)).to(device)
ethnic_net = FFNetwork(input_dim, hidden_dims, len(ethnic_dict)).to(device)
relgion_net = FFNetwork(input_dim, hidden_dims, len(religion_dict)).to(device)
marital_net = FFNetwork(input_dim, hidden_dims, len(marital_dict)).to(device)
qual_net = FFNetwork(input_dim, hidden_dims, len(qual_dict)).to(device)

# input for the networks
input_tensor = torch.randn(total, input_dim).to(device)  # Random noise as input, adjust as necessary


# Define the Gumbel-Softmax function
def gumbel_softmax_sample(logits, temperature=0.5):
    gumbel_noise = -torch.log(-torch.log(torch.rand(logits.shape, device=device)))
    y = logits + gumbel_noise
    return torch.nn.functional.softmax(y / temperature, dim=-1)

# Define a function to generate the population
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
    # Calculate split sizes based on category dictionaries
    split_sizes = [len(cat_dict) for cat_dict in category_dicts]
    # Ensure the tensor dimension matches the total category count
    if encoded_tensor.size(1) != sum(split_sizes):
        raise ValueError("Size mismatch between encoded_tensor and category_dicts")

    # Split the tensor into category-specific probabilities
    category_probs = torch.split(encoded_tensor, split_sizes, dim=1)

    # Initialize the aggregated tensor
    aggregated_tensor = torch.zeros(len(cross_table.keys()), device=device)

    # Aggregate the tensor based on the cross table
    for i, key in enumerate(cross_table.keys()):
        category_keys = key.split(' ')
        expected_count = torch.ones(encoded_tensor.size(0), device=device)  # Initialize as a vector

        # Multiply probabilities across each category
        for cat_index, cat_key in enumerate(category_keys):
            category_index = list(category_dicts[cat_index].keys()).index(cat_key)
            expected_count *= category_probs[cat_index][:, category_index]

        # Aggregate the expected counts
        aggregated_tensor[i] = torch.sum(expected_count)
    return aggregated_tensor

def decode_tensor(encoded_tensor, category_dicts):
    # Calculate the split sizes from the category dictionaries
    split_sizes = [len(cat_dict) for cat_dict in category_dicts]

    # Dynamic tensor splitting
    category_encoded_tensors = torch.split(encoded_tensor, split_sizes, dim=1)

    # Decoding each category
    decoded_categories = []
    for cat_tensor, cat_dict in zip(category_encoded_tensors, category_dicts):
        cat_labels = list(cat_dict.keys())
        decoded_cat = [cat_labels[torch.argmax(ct).item()] for ct in cat_tensor]
        decoded_categories.append(decoded_cat)

    # Combine the decoded categories
    return list(zip(*decoded_categories))

def keep_categories(encoded_tensor, category_lengths, categories_to_keep):
    # Calculate the indices for the categories to be kept
    keep_indices = []
    current_index = 0
    for category, length in category_lengths.items():
        if category in categories_to_keep:
            indices = torch.arange(start=current_index, end=current_index + length, device=encoded_tensor.device)
            keep_indices.append(indices)
        current_index += length

    # Concatenate all the keep_indices and use them to index the tensor
    keep_indices = torch.cat(keep_indices, dim=0)
    kept_tensor = encoded_tensor[:, keep_indices]
    return kept_tensor

def plot(target, computed, cross_table, name):
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
        title= name + ' [' + "RMSE:" + str(accuracy) + ']',
        xaxis_tickangle=-90,
        xaxis_title='Categories',
        yaxis_title='Counts',
        barmode='group',
        bargap=0.5,
        width=9000
    )
    # Save plot
    py.offline.plot(fig, filename= path + '/plots/' + str(name) + '.html')
    # Show plot
    # fig.show()



def rmse_accuracy(target_tensor, computed_tensor):
    mse = torch.mean((target_tensor - computed_tensor) ** 2)
    rmse = torch.sqrt(mse)
    max_possible_error = torch.sqrt(torch.sum(target_tensor ** 2))
    accuracy = 1 - (rmse / max_possible_error)
    return accuracy.item()

# encoded_population = generate_population(input_tensor).cuda()
# records = decode_tensor(encoded_population, [sex_dict, age_dict, ethnic_dict, religion_dict, marital_dict, qual_dict])
# print(records)
# categories_to_keep = ['sex', 'age', 'marital']  # Categories to keep
# kept_tensor = keep_categories(encoded_population, category_lengths, categories_to_keep)
# aggregated_tensor = aggregate(kept_tensor, cross_table3, [sex_dict, age_dict, marital_dict])

def rmse_accuracy(target_tensor, computed_tensor):
    mse = torch.mean((target_tensor - computed_tensor) ** 2)
    rmse = torch.sqrt(mse)
    max_possible_error = torch.sqrt(torch.sum(target_tensor ** 2))
    accuracy = 1 - (rmse / max_possible_error)
    return accuracy.item()

def rmse_loss(aggregated_tensor, target_tensor):
    return torch.sqrt(torch.mean((aggregated_tensor - target_tensor) ** 2))


# record execution start time

start = time.time()
# Training loop
optimizer = torch.optim.Adam([{'params': sex_net.parameters()},
                              {'params': age_net.parameters()},
                              {'params': ethnic_net.parameters()},
                              {'params': relgion_net.parameters()},
                              {'params': marital_net.parameters()},
                              {'params': qual_net.parameters()}], lr=0.001)

number_of_epochs = 70
for epoch in range(number_of_epochs):
    optimizer.zero_grad()

    # Generate and aggregate encoded population for sex, age, ethnicity
    encoded_population = generate_population(input_tensor).cuda()
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


    # Compute and backpropagate loss
    loss1 = rmse_loss(aggregated_population1, cross_table_tensor1.cuda())
    loss2 = rmse_loss(aggregated_population2, cross_table_tensor2.cuda())
    loss3 = rmse_loss(aggregated_population3, cross_table_tensor3.cuda())
    loss4 = rmse_loss(aggregated_population4, cross_table_tensor4.cuda())

    loss = loss1 + loss2 + loss3 + loss4

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

plot(cross_table_tensor1, aggregated_population1, cross_table1, 'Age-Sex-Ethnicity')
plot(cross_table_tensor2, aggregated_population2, cross_table2, 'Age-Sex-Religion')
plot(cross_table_tensor3, aggregated_population3, cross_table3, 'Age-Sex-MaritalStatus')
plot(cross_table_tensor4, aggregated_population4, cross_table4, 'Age-Sex-Qualification')

# create a dataframe from the records
records = decode_tensor(encoded_population, [sex_dict, age_dict, ethnic_dict, religion_dict, marital_dict, qual_dict])
df = pd.DataFrame(records, columns=['sex', 'age', 'ethnicity', 'religion', 'marital', 'qualification'])
df.to_csv('synthetic_population.csv', index=False)

# record execution end time
end = time.time()
duration = end - start

# Convert to hours, minutes, and seconds
hours = int(duration // 3600)
minutes = int((duration % 3600) // 60)
seconds = duration % 60
print(f"Duration: {hours} hours, {minutes} minutes, {seconds:.2f} seconds")

