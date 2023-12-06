import torch
import torch.nn as nn
import torch.optim as optim
import InputData as ID
import InputCrossTables as ICT
import plotly.graph_objects as go

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

#MSOA
area = 'E02005924'
total = ID.get_total(ID.age5ydf, area)

sex_dict = ID.getdictionary(ID.sexdf, area)
age_dict = ID.getdictionary(ID.age5ydf, area)
ethnic_dict = ID.getdictionary(ID.ethnicdf, area)
cross_table = ICT.getdictionary(ICT.ethnic_by_sex_by_age, area)

cross_table_tensor = torch.tensor(list(cross_table.values()), dtype=torch.float32).to(device)

# Instantiate networks for each demographic characteristic
input_dim = len(sex_dict.keys()) + len(age_dict.keys()) + len(ethnic_dict.keys())  # Set appropriate input dimension based on your data
hidden_dims = [64, 32]
sex_net = FFNetwork(input_dim, hidden_dims, len(sex_dict)).to(device)
age_net = FFNetwork(input_dim, hidden_dims, len(age_dict)).to(device)
ethnic_net = FFNetwork(input_dim, hidden_dims, len(ethnic_dict)).to(device)

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

def rmse_accuracy(target_tensor, computed_tensor):
    mse = torch.mean((target_tensor - computed_tensor) ** 2)
    rmse = torch.sqrt(mse)
    max_possible_error = torch.sqrt(torch.sum(target_tensor ** 2))
    accuracy = 1 - (rmse / max_possible_error)
    return accuracy.item()

def rmse_loss(aggregated_tensor, target_tensor):
    return torch.sqrt(torch.mean((aggregated_tensor - target_tensor) ** 2))

# Training loop
optimizer = torch.optim.Adam([{'params': sex_net.parameters()},
                              {'params': age_net.parameters()},
                              {'params': ethnic_net.parameters()}], lr=0.01)

number_of_epochs = 100
for epoch in range(number_of_epochs):
    optimizer.zero_grad()

    # Generate and aggregate encoded population
    encoded_population = generate_population(input_tensor).cuda()
    aggregated_population = aggregate_softmax_encoded_tensor(encoded_population).cuda()

    # Compute and backpropagate loss
    loss = rmse_loss(aggregated_population, cross_table_tensor.cuda())
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


# Decode the final population
final_population = decode_tensor(encoded_population.detach())

# Aggregate the final population
final_aggregated_population = aggregate_softmax_encoded_tensor(encoded_population.detach())

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
