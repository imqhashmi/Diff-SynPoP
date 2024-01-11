import pandas as pd
import numpy as np

# Creating a random dataset for 100 people
np.random.seed(0)  # for reproducibility

# Define the categories
age_groups = ["0-18", "19-40", "41-100"]
sexes = ["Male", "Female"]
religions = ["Christian", "Muslim"]
ethnicities = ["White", "Black", "Asian"]
marital_statuses = ["Unmarried", "Married"]
qualifications = ["Level1", "Level2", "Level3"]

# Randomly assign people to categories
data = {
    "Age": np.random.choice(age_groups, 100),
    "Sex": np.random.choice(sexes, 100),
    "Religion": np.random.choice(religions, 100),
    "Ethnicity": np.random.choice(ethnicities, 100),
    "Marital Status": np.random.choice(marital_statuses, 100),
    "Qualification": np.random.choice(qualifications, 100),
}

df = pd.DataFrame(data)

# Function to create triple cross tables
def create_triple_cross_tables(df):
    cross_tables = {}
    columns = list(df.columns)
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            for k in range(j+1, len(columns)):
                cross_table = pd.crosstab(index=[df[columns[i]], df[columns[j]]],
                                          columns=df[columns[k]],
                                          margins=True)
                cross_tables[(columns[i], columns[j], columns[k])] = cross_table
    return cross_tables

# Generate triple cross tables
triple_cross_tables = create_triple_cross_tables(df)
# list = [('Age', 'Sex', 'Religion'),
# ('Age', 'Sex', 'Ethnicity'),
# ('Age', 'Sex', 'Marital Status'),
# ('Age', 'Sex', 'Qualification')]
