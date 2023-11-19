import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random

# Data
total_population = 500
sex_dict = {'M': 240, 'F': 260}
age_dict = {'Child': 300, 'Adult': 200}
ethnic_dict = {'White': 200, 'Black': 100, 'Asian': 200}

cross_table = {
    'M-Child-White': 66, 'M-Child-Black': 31, 'M-Child-Asian': 46,
    'M-Adult-White': 38, 'M-Adult-Black': 14, 'M-Adult-Asian': 35,
    'F-Child-White': 67, 'F-Child-Black': 26, 'F-Child-Asian': 71,
    'F-Adult-White': 44, 'F-Adult-Black': 17, 'F-Adult-Asian': 45
}


# Updated generate_population function
def generate_population(total_population):
    population = []
    for _ in range(total_population):
        sex = random.choice(list(sex_dict.keys()))
        age = random.choice(list(age_dict.keys()))
        ethnicity = random.choice(list(ethnic_dict.keys()))
        population.append((sex, age, ethnicity))
    return population

population_data = generate_population(total_population)

def encode_tensor():
    # Convert categorical attributes to numerical values
    sex_num = torch.tensor([list(sex_dict.keys()).index(individual[0]) for individual in population_data], dtype=torch.float32)
    age_num = torch.tensor([list(age_dict.keys()).index(individual[1]) for individual in population_data], dtype=torch.float32)
    ethnic_num = torch.tensor([list(ethnic_dict.keys()).index(individual[2]) for individual in population_data],
                              dtype=torch.float32)
    return torch.stack([sex_num, age_num, ethnic_num], dim=1)

print(torch.tensor(list(cross_table.values())))

#
# # Define the neural network model
# class PopulationModel(nn.Module):
#     def __init__(self):
#         super(PopulationModel, self).__init__()
#         self.fc = nn.Linear(3, 12)
#         self.relu = nn.ReLU()
#         self.output = nn.Linear(12, len(cross_table))
#
#     def forward(self, x):
#         x = self.fc(x)
#         x = self.relu(x)
#         x = self.output(x)
#         return x
#
#
# # Create the model, loss function, and optimizer
# model = PopulationModel()
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
#
# # Train the model to fit the cross table
# epochs = 1
# losses = []
#
# for epoch in range(epochs):
#     optimizer.zero_grad()
#     population_data = generate_population(total_population)
#
#     # Convert categorical attributes to numerical values
#     input_data = encode_tensor()
#     predicted_cross_table = model(input_data)
#
#     loss = criterion(predicted_cross_table, torch.tensor(list(cross_table.values()), dtype=torch.float32))
#     loss.backward()
#     optimizer.step()
#     losses.append(loss.item())
#
# # Calculate accuracy
# predicted_cross_table = model(input_data).detach().numpy().reshape(-1)
# actual_cross_table = torch.tensor(list(cross_table.values()), dtype=torch.float32).numpy()
# accuracy = 1 - np.sum(np.square(actual_cross_table - predicted_cross_table)) / np.sum(np.square(actual_cross_table))
#
# # Plot the results
# plt.plot(losses)
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.show()
#
# print(f'Accuracy: {accuracy * 100:.2f}%')
