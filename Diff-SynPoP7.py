import sys
sys.path.insert(1, '/kaggle/input/diffspop/Diff-SynPoP')

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
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        x = self.layers(x)
        return self.output_layer(x)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
path = os.path.join(os.path.dirname(os.getcwd()), 'Diff-SynPoP')

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

# Define the Gumbel-Softmax function
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
        expected_count = torch.ones(encoded_tensor.size(0), device=device) # Initialize as a vector

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
    #py.offline.plot(fig, filename= path + '/plots/' + str(name) + '.html')
    # Show plot
    fig.show()

# encoded_population = generate_population(input_tensor).cuda()
# records = decode_tensor(encoded_population, [sex_dict, age_dict, ethnic_dict, religion_dict, marital_dict, qual_dict])
# print(records)
# categories_to_keep = ['sex', 'age', 'marital']  # Categories to keep
# kept_tensor = keep_categories(encoded_population, category_lengths, categories_to_keep)
# aggregated_tensor = aggregate(kept_tensor, cross_table3, [sex_dict, age_dict, marital_dict])

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
    # Concatenate the target and computed tensors along the characteristic dimension (dim=1)
    concatenated_tensor = torch.cat([target_tensor1, target_tensor2, target_tensor3, target_tensor4])
    aggregated_cat_tensor = torch.cat([aggregated_tensor1, aggregated_tensor2, aggregated_tensor3, aggregated_tensor4])
    # Calculate RMSE loss on the concatenated tensor
    loss = torch.sqrt(torch.mean((aggregated_cat_tensor - concatenated_tensor) ** 2))
    return loss

generated_population = generate_population(input_tensor)
records = decode_tensor(generated_population, [sex_dict, age_dict, ethnic_dict, religion_dict, marital_dict, qual_dict])
df = pd.DataFrame(records, columns=['sex', 'age', 'ethnicity', 'religion', 'marital', 'qualification'])
print(df)

loss_history = []
accuracy_history = []

# record execution start time
start = time.time()

# Training loop
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

number_of_epochs = 300
for epoch in range(number_of_epochs):
    optimizer.zero_grad()

    # Generate and aggregate encoded population for sex, age, ethnicity
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
#age_categories_to_single = ["0_4", "5_7", "8_9", "10_14", "15"]
#df.loc[df['age'].isin(age_categories_to_single), 'marital'] = 'Single'
df.to_csv('synthetic_population.csv', index=False)

# record execution end time
end = time.time()
duration = end - start

# Convert to hours, minutes, and seconds
hours = int(duration // 3600)
minutes = int((duration % 3600) // 60)
seconds = duration % 60
print(f"Duration: {hours} hours, {minutes} minutes, {seconds:.2f} seconds")

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

data = pd.read_csv('/kaggle/working/synthetic_population.csv') # loading the generated CSV file
columns_for_clustering = ['sex', 'age', 'ethnicity', 'religion', 'marital', 'qualification']

cluster_data = data[columns_for_clustering] # extracting relevant columns for clustering
cluster_data_encoded = pd.get_dummies(cluster_data) # converting categorical columns to numerical using one-hot encoding
# k-means clustering
num_clusters = 4239
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['cluster'] = kmeans.fit_predict(cluster_data_encoded)

# sorting the obtained data by cluster for better visualization
data.sort_values('cluster', inplace=True)
data.to_csv('clustered_data.csv', index=False) # saving the results with cluster assignments

import pandas as pd

file_path = '/kaggle/working/synthetic_population.csv' # loading the synthetic_population CSV file
persons_df = pd.read_csv(file_path) # saving the loaded csv file to a pandas dataframe

persons_df['Person_ID'] = range(1, len(persons_df) + 1) # assigning a person ID to each row

import random

# creating a new dataframe for households
num_households = 4239
households_df = pd.DataFrame(index=range(1, num_households + 1), columns=['Household_ID', 'Composition', 'Assigned_Persons'])
for index, row in households_df.iterrows():
    households_df.at[index, 'Assigned_Persons'] = []
households_df['Household_ID'] = [str(i) for i in range(1, num_households + 1)]

composition_counts = {
    'SP Elder': 622,
    'SP Adult': 552,
    'OF Elder': 565,
    'OF Married NC': 553,
    'OF Married DC': 705,
    'OF Married NDC': 275,
    'OF Cohab NC': 260,
    'OF Cohab DC': 177,
    'OF Cohab NDC': 41,
    'LP DC': 201,
    'LP NDC': 155,
    'Others': 133
}

# Ensure the total number of rows is equal to the number of households
assert sum(composition_counts.values()) == num_households, "Total rows should be equal to the number of households"

# Initialize 'Composition' column
households_df['Composition'] = ''

# Populate 'Composition' column based on counts
current_row = 1
for composition, count in composition_counts.items():
    households_df.loc[current_row:current_row + count - 1, 'Composition'] = composition
    current_row += count
    
composition_counts = households_df['Composition'].value_counts()
print(composition_counts)
print()
print(composition_counts.sum())

print(households_df)
households_df = households_df.sample(frac=1).reset_index(drop=True)
print(households_df)

for _, person_row in persons_df.iterrows():
    while True:
        # randomly choosing a household from the household dataframe
        household_id = random.randint(1, num_households-1)
        household = households_df.loc[household_id]
        
        child_ages = ["0_4", "5_7", "8_9", "10_14", "15", "16_17"]
        elder_ages = ["65_69", "70_74", "75_79", "80_84", "85+"]
        
        # ********** checking the rules here *********
        
        # only assigning a single person to compositions "SP Elder" & "SP Adult"
        if (household['Composition'] in ['SP Elder', 'SP Adult']) and (len(household['Assigned_Persons']) > 0):
            continue  # skipping this household
        
        # only assigning elders to compositions "SP Elder" & "SP Adult"
        if (household['Composition'] in ['SP Elder', 'OF Elder']) and (person_row['age'] not in elder_ages):
            continue  # skipping this household
        
        # only assigning adults or elders to compositions having no children
        if (household['Composition'] in ["OF Married NC", "OF Cohab NC"]) and (person_row['age'] in child_ages):
            continue  # Skip this household, choose another one
        
        # only assigning a single person to household compositions "SP Elder" & "SP Adult"
        if (household['Composition'] not in ["OF Married DC", "OF Cohab DC", "LP DC"]) and (person_row['age'] in child_ages):
            continue  # Skip this household, choose another one
         
        else:
            households_df.at[household_id, 'Assigned_Persons'].append(person_row['Person_ID'])
            break

print(households_df.head())
print()
print(households_df.tail())

def assign_persons_to_households(persons_df, households_df):
    for _, person in persons_df.head(1).iterrows():  # Only the first person
        assigned = False
        while not assigned:
            # randomly selecting a household
            selected_household_index = np.random.choice(households_df.index)
            selected_household = households_df.loc[selected_household_index]

            # checking the rules here 
            if (selected_household['Composition'] in ['SP Elder', 'SP Adult'] and
                    len(selected_household['Assigned_Persons']) > 0):
                continue  # skipping this household, and then going on to try again
                
            # assigning the person to the selected household
            print("Adding Here: ", households_df.at[selected_household_index, 'Assigned_Persons'])
            households_df.at[selected_household_index, 'Assigned_Persons'].append(person['Person_ID'])
            assigned = True

# performing the assignment
assign_persons_to_households(persons_df, households_df)