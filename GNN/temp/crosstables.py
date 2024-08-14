import random
import pandas as pd

def generate_population(size, age_probs, sex_probs, category_probs):
    """Generates a population with given probabilities for age, sex, and a third category."""
    population = []
    for _ in range(size):
        person = {
            'age': random.choices(['Child', 'Adult', 'Elder'], weights=age_probs)[0],
            'sex': random.choices(['Male', 'Female'], weights=sex_probs)[0],
            'category': random.choices(list(category_probs.keys()), weights=list(category_probs.values()))[0],
        }
        population.append(person)
    return population

# Probabilities (adjust these to your needs)
age_probs = [0.3, 0.4, 0.3]  # Example: 25% child, 60% adult, 15% elder
sex_probs = [0.5, 0.5]          # Example: 50% male, 50% female

# Ethnicity probabilities
ethnicity_probs = {'White': 0.6, 'Black': 0.2, 'Asian': 0.2}

# Religion probabilities
# religion_probs =  {'C': 0.5, 'J': 0.1, 'M': 0.3, 'O': 0.1}

# Generate populations
population1 = generate_population(10, age_probs, sex_probs, ethnicity_probs)
# population2 = generate_population(100, age_probs, sex_probs, religion_probs)

# Create Pandas DataFrames and cross-tabulations
df1 = pd.DataFrame(population1)
# df2 = pd.DataFrame(population2)

crosstab1 = pd.crosstab(index=df1['age'], columns=[df1['sex'], df1['category']], margins=True)
# crosstab2 = pd.crosstab(index=df2['age'], columns=[df2['sex'], df2['category']], margins=True)
# remove rows with 'All' in them
crosstab1 = crosstab1.loc[crosstab1.index != 'All']
# remove columns that has 'All' in them
crosstab1 = crosstab1.loc[:, crosstab1.columns.map(lambda x: 'All' not in x)]
age_by_sex_by_ethnicity = {}
for row in crosstab1.iterrows():
    for col in crosstab1.columns:
        age_by_sex_by_ethnicity[(row[0] +'-' + col[0] +'-' + col[1])] = row[1][col]
        print((row[0] +'-' + col[0] +'-' + col[1]),  row[1][col])

print(age_by_sex_by_ethnicity)
print(sum(list(age_by_sex_by_ethnicity.values())))
# # do the same for cross tab 2
# crosstab2 = crosstab2.loc[crosstab2.index != 'All']
# crosstab2 = crosstab2.loc[:, crosstab2.columns.map(lambda x: 'All' not in x)]
# age_by_sex_by_religion = {}
# for row in crosstab2.iterrows():
#     for col in crosstab2.columns:
#         age_by_sex_by_religion[(row[0] +'-' + col[0] +'-' + col[1])] = row[1][col]
#
# print(age_by_sex_by_religion)
#
# print(age_probs)
# print(sex_probs)
# print(ethnicity_probs)
# print(religion_probs)
