import torch
import torch.nn.functional as F
import torch.optim as optim
import InputData as ID
import InputCrossTables as ICT

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
# mstatus_dict = ID.getdictionary(ID.mstatusdf, area)
# qual_dict = ID.getdictionary(ID.qualdf, area)

# Cross tables
sex_age_ethnic = ICT.getdictionary(ICT.ethnic_by_sex_by_age, area)
sex_age_religion = ICT.getdictionary(ICT.religion_by_sex_by_age, area)
# sex_age_mstatus = ICT.getdictionary(ICT.marital_by_sex_by_age, area)
# sex_age_qual = ICT.getdictionary(ICT.qualification_by_sex_by_age, area)

attribute_dicts = {**sex_dict, **age_dict, **ethnic_dict, **religion_dict}


# Preparing cross tables and tensors
cross_tables = {**sex_age_ethnic, **sex_age_religion}
cross_tables_tensors = torch.tensor(list(cross_tables.values()), dtype=torch.float32, device=device)

# Function to calculate log probabilities
def calculate_proportions(dictionary):
    total_count = sum(dictionary.values())
    proportions = [count / total_count for count in dictionary.values()]
    log_probs = torch.log(torch.tensor(proportions) + 1e-10)
    return log_probs.to(device)


# Initialize logits
sex_logits = torch.empty(total, len(sex_dict), requires_grad=True, device=device).data = calculate_proportions(
    sex_dict).repeat(total, 1)
age_logits = torch.empty(total, len(age_dict), requires_grad=True, device=device).data = calculate_proportions(
    age_dict).repeat(total, 1)
ethnicity_logits = torch.empty(total, len(ethnic_dict), requires_grad=True, device=device).data = calculate_proportions(
    ethnic_dict).repeat(total, 1)
religion_logits = torch.empty(total, len(religion_dict), requires_grad=True,
                              device=device).data = calculate_proportions(religion_dict).repeat(total, 1)


# Gumbel-Softmax sampling function
def generate_population_gumbel_softmax(temperature=0.5):
    sex = F.gumbel_softmax(sex_logits, tau=temperature, hard=False)
    age = F.gumbel_softmax(age_logits, tau=temperature, hard=False)
    ethnicity = F.gumbel_softmax(ethnicity_logits, tau=temperature, hard=False)
    religion = F.gumbel_softmax(religion_logits, tau=temperature, hard=False)

    return torch.cat([sex, age, ethnicity, religion], dim=-1)

def aggregate_softmax_encoded_tensor(encoded_tensor, cross_table, attribute_dicts):
    aggregated_tensor = torch.zeros(len(cross_table.keys()), device=device)
    start_index = 0
    split_sizes = len(attribute_dicts.keys())
    probabilities = torch.split(encoded_tensor, split_sizes, dim=1)
    for i, key in enumerate(cross_tables.keys()):
        keys = key.split(' ')
        expected_count = torch.ones(encoded_tensor.size(0), device=device)
        for k, key_part in enumerate(keys):
            index = attribute_dicts[k][key_part]

            # Debugging print statement
            print(f"Key part: {key_part}, Index: {index}, Size of probabilities[{k}]: {probabilities[k].shape}")

            expected_count *= probabilities[k][:, index]
        aggregated_tensor[i] = torch.sum(expected_count)
    return aggregated_tensor


# Decoding function
def decode_tensor(encoded_tensor, attribute_dicts):
    split_sizes = len(attribute_dicts.keys())
    probabilities = torch.split(encoded_tensor, split_sizes, dim=1)
    decoded_attributes = []

    for i, attr_dict in enumerate(attribute_dicts):
        categories = list(attr_dict.keys())
        decoded_attributes.append([categories[torch.argmax(prob).item()] for prob in probabilities[i]])

    return list(zip(*decoded_attributes))

# test all functions before traaining loop
# encoded_population = generate_population_gumbel_softmax(temperature=0.5)
# print(encoded_population)
# print(decode_tensor(encoded_population, attribute_dicts))
# aggregate_softmax_encoded_tensor(encoded_population, cross_tables, attribute_dicts)



# Training parameters
number_of_epochs = 10
temperature = 0.5

optimizer = optim.SGD([sex_logits, age_logits, ethnicity_logits, religion_logits],
                      lr=0.001, momentum=0.9)


# Loss function
def rmse_loss(aggregated_tensor, target_tensor):
    return torch.sqrt(torch.mean((aggregated_tensor - target_tensor) ** 2))


# Nested loss computation for multiple cross tables
def compute_total_loss(encoded_population, cross_tables, cross_tables_tensors, attribute_dicts):
    total_loss = 0
    for cross_table, target_tensor in zip(cross_tables, cross_tables_tensors):
        aggregated_population = aggregate_softmax_encoded_tensor(encoded_population, cross_tables.keys(),
                                                                 attribute_dicts)
        total_loss += rmse_loss(aggregated_population, target_tensor)
    return total_loss





# Training loop
for epoch in range(number_of_epochs):
    optimizer.zero_grad()
    encoded_population = generate_population_gumbel_softmax(temperature)
    total_loss = compute_total_loss(encoded_population, cross_tables, cross_tables_tensors, attribute_dicts)

    # Check for NaN in loss
    if torch.isnan(total_loss):
        print(f"NaN detected in loss at epoch {epoch}")
        break

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [sex_logits, age_logits, ethnicity_logits, religion_logits], max_norm=1.0)
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss.item()}")
