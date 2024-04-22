import torch
import torch.nn.functional as F
import InputData as ID
import InputCrossTables as ICT
import math

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data setup
area = 'E02005924'
total = ID.get_total(ID.age5ydf, area)

# Attribute dictionaries
sex_dict = ID.getdictionary(ID.sexdf, area)
age_dict = ID.getdictionary(ID.age5ydf, area)
ethnic_dict = ID.getdictionary(ID.ethnicdf, area)
religion_dict = ID.getdictionary(ID.religiondf, area)


# Cross tables
sex_age_ethnic = ICT.getdictionary(ICT.ethnic_by_sex_by_age, area)
sex_age_religion = ICT.getdictionary(ICT.religion_by_sex_by_age, area)

# Merge attribute dictionaries
attribute_dicts = {'sex': sex_dict, 'age': age_dict, 'ethnicity': ethnic_dict, 'religion': religion_dict}

# Function to calculate log probabilities
def calculate_proportions(dictionary):
    total_count = sum(dictionary.values())
    proportions = [count / total_count for count in dictionary.values()]
    log_probs = torch.log(torch.tensor(proportions) + 1e-10)
    return log_probs.to(device)

# Initialize logits
def initialize_logits(total, attribute_dicts):
    logits = {}
    for attr, attr_dict in attribute_dicts.items():
        logits[attr] = torch.empty(total, len(attr_dict), requires_grad=True, device=device).data = calculate_proportions(attr_dict).repeat(total, 1)
    return logits

logits = initialize_logits(total, attribute_dicts)

# Gumbel-Softmax sampling function
def gumbel_softmax_sample(logits, temperature=0.5):
    return F.gumbel_softmax(logits, tau=temperature, hard=False)

def generate_population_gumbel_softmax(logits, temperature=0.5):
    list_of_logits = []
    for attr in logits:
        list_of_logits.append(F.gumbel_softmax(logits[attr], tau=temperature, hard=False))
    return torch.cat(list_of_logits, dim=-1)

# Decoding function
def decode_tensor(encoded_tensor):
    # split encoded tensor into individual categories
    encoded_categories = torch.split(encoded_tensor, [len(list(c.keys())) for c in attribute_dicts.values()], dim=1)
    # decode each category
    decoded_categories = []
    for encoded_category, attr_dict in zip(encoded_categories, attribute_dicts.values()):
        decoded_category = [list(attr_dict.keys())[torch.argmax(s).item()] for s in encoded_category]
        decoded_categories.append(decoded_category)
    return list(zip(*decoded_categories))

def aggregate(encoded_data, cross_table, attributes):
    data = decode_tensor(encoded_data)
    result = {}
    for key in cross_table.keys():
        result[key] = 0

    for entry in data:
        key = ' '.join(entry[attr] for attr in attributes)
        result[key] += 1
    return result

def calculate_rmse(actual, computed):
    return math.sqrt(sum([(actual[key] - computed[key]) ** 2 for key in actual.keys()]))


# encoded_population = generate_population_gumbel_softmax(logits)
# data = decode_tensor(encoded_population)
# computed = aggregate(data, sex_age_religion, [0, 1, 3])

# Optimizer
optimizer = torch.optim.Adam([logits_tensor for logits_tensor in logits.values()], lr=0.01)

# Training loop
number_of_epochs = 100  # Define the number of epochs
for epoch in range(number_of_epochs):
    optimizer.zero_grad()

    # Generate population
    encoded_population = generate_population_gumbel_softmax(logits)

    # Aggregate data according to cross tables
    computed_sex_age_ethnic = aggregate(encoded_population, sex_age_ethnic, [0, 1, 2])
    computed_sex_age_religion = aggregate(encoded_population, sex_age_religion, [0, 1, 3])


    # Calculate loss and update model
    loss = calculate_rmse(sex_age_ethnic, computed_sex_age_ethnic)
    loss += calculate_rmse(sex_age_religion, computed_sex_age_religion)

    loss.backward()
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_([sex_logits, age_logits, ethnicity_logits], max_norm=1.0)

    optimizer.step()
    print('Epoch {}, Loss: {}'.format(epoch, loss))