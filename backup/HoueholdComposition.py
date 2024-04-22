import os
import pandas as pd
import random
import time
import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

import sys

# appending the system path to run the file on kaggle
# not required if you are running it locally
# sys.path.insert(1, '/kaggle/input/diffspop/Diff-SynPoP')

import os
import time
import numpy as np
import plotly as py
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# imporint InputData and InputCrossTables for processing UK census data files
import InputData as ID
import InputCrossTables as ICT

# file_path = '/kaggle/working/synthetic_population.csv' # loading the synthetic_population CSV file
# create local path from current working directory
file_path = os.path.join(os.path.dirname(os.getcwd()), 'Diff-SynPoP', 'synthetic_population.csv')
persons_df = pd.read_csv(file_path) # saving the loaded csv file to a pandas dataframe

persons_df['Person_ID'] = range(1, len(persons_df) + 1) # assigning a person ID to each row

area = 'E02005924'
num_households = ID.get_HH_com_total(ID.HHcomdf, area)
composition_counts = ID.getHHcomdictionary(ID.HHcomdf, area)
hh_size_dist_org = ID.getdictionary(ID.HHsizedf, area)

households_df = pd.DataFrame(index=range(1, num_households + 1), columns=['Household_ID', 'Composition', 'Assigned_Persons'])
for index, row in households_df.iterrows():
    households_df.at[index, 'Assigned_Persons'] = []
households_df['Household_ID'] = [str(i) for i in range(1, num_households + 1)]

assert sum(composition_counts.values()) == num_households, "Total rows should be equal to the number of households"

households_df['Composition'] = ''
# populating the position' column based on counts
current_row = 1
for composition, count in composition_counts.items():
    households_df.loc[current_row:current_row + count - 1, 'Composition'] = composition
    current_row += count

composition_counts = households_df['Composition'].value_counts()
print(composition_counts)
print()
print(composition_counts.sum())
print()

print(households_df.head())
print()
print(households_df.tail())

order = ['1PA', '1PE', '1FM-0C', '1FM-1C', '1FC-0C', '1FC-1C', '1FL-1C', '1H-1C', '1FE', '1FM-nC', '1FM-nA', '1FC-nC', '1FC-nA', '1FL-nC', '1FL-nA', '1H-nC', '1H-nA', '1H-nE']
households_df['Composition'] = pd.Categorical(households_df['Composition'], categories=order, ordered=True)
households_df = households_df.sort_values('Composition')
print(households_df)

nk = '8'
ok = '8+'
hh_size_dist_org[nk] = hh_size_dist_org[ok]
del hh_size_dist_org[ok]

values_size_org, weights_size_org = zip(*hh_size_dist_org.items())
rk = ['1']
household_size_dist = {key: value for key, value in hh_size_dist_org.items() if key not in rk}
values_size, weights_size = zip(*household_size_dist.items())
rk = ['1', '2']
household_size_dist_na = {key: value for key, value in hh_size_dist_org.items() if key not in rk}
values_size_na, weights_size_na = zip(*household_size_dist_na.items())
rk = ['1', '2', '3']
household_size_dist_nc = {key: value for key, value in hh_size_dist_org.items() if key not in rk}
values_size_nc, weights_size_nc = zip(*household_size_dist_nc.items())

print(hh_size_dist_org)
print(household_size_dist)
print(household_size_dist_na)
print(household_size_dist_nc)

# individuals are in persons_df
# households are in households_df

child_ages = ["0_4", "5_7", "8_9", "10_14", "15"]
adult_ages = ["16_17", "18_19", "20_24", "25_29", "30_34", "35_39", "40_44", "45_49", "50_54", "55_59", "60_64"]
elder_ages = ["65_69", "70_74", "75_79", "80_84", "85+"]

inter_ethnic_ratio = 0.1
inter_rel_ratio = 0.015

# recording execution start time
start = time.time()

def assign_individuals(row):

    random_number = random.random()

    # Composition # 1
    # composition '1PE' One person: Pensioner (person who is above 65)
    if row['Composition'] == '1PE':
        # Filter individuals dataframe based on age category and sample one person
        eligible_individuals = persons_df[
            (persons_df['age'].isin(elder_ages))
        ]

        if not eligible_individuals.empty:
            sampled_person = eligible_individuals.sample(1)
            Person_ID = sampled_person['Person_ID'].values[0]

            # Update the assigned_persons column in the households dataframe
            row['Assigned_Persons'].append(Person_ID)

            # Remove the sampled person from individuals dataframe
            persons_df.drop(persons_df[persons_df['Person_ID'] == Person_ID].index, inplace=True)

    # Composition # 2
    # composition '1PA' One person: Other (a single person who is above 18 and below 65)
    if row['Composition'] == '1PA':
        # Filter individuals dataframe based on age category and sample one person
        eligible_individuals = persons_df[
            (persons_df['age'].isin(adult_ages))
        ]

        if not eligible_individuals.empty:
            sampled_person = eligible_individuals.sample(1)
            Person_ID = sampled_person['Person_ID'].values[0]

            # Update the assigned_persons column in the households dataframe
            row['Assigned_Persons'].append(Person_ID)

            # Remove the sampled person from individuals dataframe
            persons_df.drop(persons_df[persons_df['Person_ID'] == Person_ID].index, inplace=True)

    # Composition # 3
    # composition '1FM-0C' One family: Married Couple: No children
    if row['Composition'] == '1FM-0C':
        # Filter individuals dataframe based on criteria and sample one male person
        eligible_male_individuals = persons_df[
            (persons_df['marital'] == 'Married') &
            (persons_df['sex'] == 'M') &
            (~persons_df['age'].isin(child_ages))
        ]

        # Sample one male person
        if not eligible_male_individuals.empty:
            male_person = eligible_male_individuals.sample(1)

            # Get the male person's ethnicity and religion
            male_ethnicity = male_person['ethnicity'].values[0]
            male_religion = male_person['religion'].values[0]

            if random_number < inter_ethnic_ratio:
                if random_number < inter_rel_ratio:
                    eligible_female_individuals = persons_df[
                        (persons_df['marital'] == 'Married') &
                        (persons_df['sex'] == 'F') &
                        (persons_df['ethnicity'] != male_ethnicity) &
                        (persons_df['religion'] != male_religion) &
                        (~persons_df['age'].isin(child_ages))
                    ]
                else:
                    eligible_female_individuals = persons_df[
                        (persons_df['marital'] == 'Married') &
                        (persons_df['sex'] == 'F') &
                        (persons_df['ethnicity'] != male_ethnicity) &
                        (persons_df['religion'] == male_religion) &
                        (~persons_df['age'].isin(child_ages))
                    ]
            else:
                eligible_female_individuals = persons_df[
                        (persons_df['marital'] == 'Married') &
                        (persons_df['sex'] == 'F') &
                        (persons_df['ethnicity'] == male_ethnicity) &
                        (persons_df['religion'] == male_religion) &
                        (~persons_df['age'].isin(child_ages))
                    ]

            # Sample one female person
            if not eligible_female_individuals.empty:
                female_person = eligible_female_individuals.sample(1)

                male_person_id = male_person['Person_ID'].values[0]
                female_person_id = female_person['Person_ID'].values[0]

                row['Assigned_Persons'].extend([male_person_id, female_person_id])

                persons_df.drop(persons_df[persons_df['Person_ID'].isin([male_person_id, female_person_id])].index, inplace=True)

    # Composition # 4
    # composition '1FM-1C' One family: Married Couple: 1 dependent child
    if row['Composition'] == '1FM-1C':
        # Filter individuals dataframe based on criteria and sample one male person
        eligible_male_individuals = persons_df[
            (persons_df['marital'] == 'Married') &
            (persons_df['sex'] == 'M') &
            (~persons_df['age'].isin(child_ages))
        ]

        # Sample one male person
        if not eligible_male_individuals.empty:
            male_person = eligible_male_individuals.sample(1)

            # Get the male person's ethnicity and religion
            male_ethnicity = male_person['ethnicity'].values[0]
            male_religion = male_person['religion'].values[0]

            if random_number < inter_ethnic_ratio:
                if random_number < inter_rel_ratio:
                    eligible_female_individuals = persons_df[
                        (persons_df['marital'] == 'Married') &
                        (persons_df['sex'] == 'F') &
                        (persons_df['ethnicity'] != male_ethnicity) &
                        (persons_df['religion'] != male_religion) &
                        (~persons_df['age'].isin(child_ages))
                    ]
                else:
                    eligible_female_individuals = persons_df[
                        (persons_df['marital'] == 'Married') &
                        (persons_df['sex'] == 'F') &
                        (persons_df['ethnicity'] != male_ethnicity) &
                        (persons_df['religion'] == male_religion) &
                        (~persons_df['age'].isin(child_ages))
                    ]
            else:
                eligible_female_individuals = persons_df[
                        (persons_df['marital'] == 'Married') &
                        (persons_df['sex'] == 'F') &
                        (persons_df['ethnicity'] == male_ethnicity) &
                        (persons_df['religion'] == male_religion) &
                        (~persons_df['age'].isin(child_ages))
                    ]

            eligible_children = persons_df[
                (persons_df['ethnicity'] == male_ethnicity) &
                (persons_df['religion'] == male_religion) &
                (persons_df['age'].isin(child_ages))
            ]

            # Sample one female person
            if not eligible_female_individuals.empty:
                female_person = eligible_female_individuals.sample(1)

                if not eligible_children.empty:
                    child = eligible_children.sample(1)

                    # Get Person_IDs
                    male_person_id = male_person['Person_ID'].values[0]
                    female_person_id = female_person['Person_ID'].values[0]
                    child_id = child['Person_ID'].values[0]

                    row['Assigned_Persons'].extend([male_person_id, female_person_id, child_id])

                    persons_df.drop(persons_df[persons_df['Person_ID'].isin([male_person_id, female_person_id, child_id])].index, inplace=True)

    # Composition # 5
    # composition '1FC-0C' One family: Cohabiting Couple: No children
    if row['Composition'] == '1FC-0C':
        # Filter individuals dataframe based on criteria and sample one male person
        eligible_male_individuals = persons_df[
            (persons_df['marital'] != 'Married') &
            (persons_df['sex'] == 'M') &
            (~persons_df['age'].isin(child_ages))
        ]

        # Sample one male person
        if not eligible_male_individuals.empty:
            male_person = eligible_male_individuals.sample(1)

            # Get the male person's ethnicity and religion
            male_ethnicity = male_person['ethnicity'].values[0]
            male_religion = male_person['religion'].values[0]

            # Filter eligible females based on the same ethnicity and religion
            if random_number < inter_ethnic_ratio:
                if random_number < inter_rel_ratio:
                    eligible_female_individuals = persons_df[
                        (persons_df['marital'] == 'Married') &
                        (persons_df['sex'] == 'F') &
                        (persons_df['ethnicity'] != male_ethnicity) &
                        (persons_df['religion'] != male_religion) &
                        (~persons_df['age'].isin(child_ages))
                    ]
                else:
                    eligible_female_individuals = persons_df[
                        (persons_df['marital'] == 'Married') &
                        (persons_df['sex'] == 'F') &
                        (persons_df['ethnicity'] != male_ethnicity) &
                        (persons_df['religion'] == male_religion) &
                        (~persons_df['age'].isin(child_ages))
                    ]
            else:
                eligible_female_individuals = persons_df[
                        (persons_df['marital'] == 'Married') &
                        (persons_df['sex'] == 'F') &
                        (persons_df['ethnicity'] == male_ethnicity) &
                        (persons_df['religion'] == male_religion) &
                        (~persons_df['age'].isin(child_ages))
                    ]

            # Sample one female person
            if not eligible_female_individuals.empty:
                female_person = eligible_female_individuals.sample(1)

                male_person_id = male_person['Person_ID'].values[0]
                female_person_id = female_person['Person_ID'].values[0]

                row['Assigned_Persons'].extend([male_person_id, female_person_id])

                persons_df.drop(persons_df[persons_df['Person_ID'].isin([male_person_id, female_person_id])].index, inplace=True)

    # Composition # 6
    # composition '1FC-1C' One family: Cohabiting Couple: 1 dependent child
    if row['Composition'] == '1FC-1C':
        # Filter individuals dataframe based on criteria and sample one male person
        eligible_male_individuals = persons_df[
            (persons_df['marital'] != 'Married') &
            (persons_df['sex'] == 'M') &
            (~persons_df['age'].isin(child_ages))
        ]

        # Sample one male person
        if not eligible_male_individuals.empty:
            male_person = eligible_male_individuals.sample(1)

            # Get the male person's ethnicity and religion
            male_ethnicity = male_person['ethnicity'].values[0]
            male_religion = male_person['religion'].values[0]

            if random_number < inter_ethnic_ratio:
                if random_number < inter_rel_ratio:
                    eligible_female_individuals = persons_df[
                        (persons_df['marital'] == 'Married') &
                        (persons_df['sex'] == 'F') &
                        (persons_df['ethnicity'] != male_ethnicity) &
                        (persons_df['religion'] != male_religion) &
                        (~persons_df['age'].isin(child_ages))
                    ]
                else:
                    eligible_female_individuals = persons_df[
                        (persons_df['marital'] == 'Married') &
                        (persons_df['sex'] == 'F') &
                        (persons_df['ethnicity'] != male_ethnicity) &
                        (persons_df['religion'] == male_religion) &
                        (~persons_df['age'].isin(child_ages))
                    ]
            else:
                eligible_female_individuals = persons_df[
                        (persons_df['marital'] == 'Married') &
                        (persons_df['sex'] == 'F') &
                        (persons_df['ethnicity'] == male_ethnicity) &
                        (persons_df['religion'] == male_religion) &
                        (~persons_df['age'].isin(child_ages))
                    ]

            eligible_children = persons_df[
                (persons_df['ethnicity'] == male_ethnicity) &
                (persons_df['religion'] == male_religion) &
                (persons_df['age'].isin(child_ages))
            ]

            # Sample one female person
            if not eligible_female_individuals.empty:
                female_person = eligible_female_individuals.sample(1)

                if not eligible_children.empty:
                    child = eligible_children.sample(1)

                    # Get Person_IDs
                    male_person_id = male_person['Person_ID'].values[0]
                    female_person_id = female_person['Person_ID'].values[0]
                    child_id = child['Person_ID'].values[0]

                    row['Assigned_Persons'].extend([male_person_id, female_person_id, child_id])

                    persons_df.drop(persons_df[persons_df['Person_ID'].isin([male_person_id, female_person_id, child_id])].index, inplace=True)

    # Composition # 7
    # composition '1FL-1C' One family: Lone Parent: 1 dependent child
    if row['Composition'] == '1FL-1C':
        # Filter individuals dataframe based on criteria and sample one male person
        eligible_parent = persons_df[
            (persons_df['marital'] != 'Married') &
            (~persons_df['age'].isin(child_ages))
        ]

        # Sample one parent
        if not eligible_parent.empty:
            parent = eligible_parent.sample(1)

            # Get the male person's ethnicity and religion
            parent_ethnicity = parent['ethnicity'].values[0]
            parent_religion = parent['religion'].values[0]

            eligible_children = persons_df[
                (persons_df['ethnicity'] == parent_ethnicity) &
                (persons_df['religion'] == parent_religion) &
                (persons_df['age'].isin(child_ages))
            ]

            if not eligible_children.empty:
                child = eligible_children.sample(1)

                # Get Person_IDs
                parent_id = parent['Person_ID'].values[0]
                child_id = child['Person_ID'].values[0]

                row['Assigned_Persons'].extend([parent_id, child_id])

                persons_df.drop(persons_df[persons_df['Person_ID'].isin([parent_id, child_id])].index, inplace=True)

    # Composition # 8
    # composition '1FE' One family: All pensioner (a family consisting of persons all above 65)
    if row['Composition'] == '1FE':
        # specifying the number of individuals to sample according to distribution of household sizes
        n = int(random.choices(values_size, weights=weights_size)[0]) - 1

        eligible_individuals = persons_df[
            (persons_df['age'].isin(elder_ages))
        ]

        if not eligible_individuals.empty:
            first_person = eligible_individuals.sample(1)

            first_person_id = first_person['Person_ID'].values[0]
            row['Assigned_Persons'].append(first_person_id)
            persons_df.drop(persons_df[persons_df['Person_ID'].isin([first_person_id])].index, inplace=True)

            first_person_ethnicity = first_person['ethnicity'].values[0]
            first_person_religion = first_person['religion'].values[0]

            other_eligible_persons = persons_df[
                (persons_df['ethnicity'] == first_person_ethnicity) &
                (persons_df['religion'] == first_person_religion) &
                (persons_df['age'].isin(elder_ages))
            ]

            if len(other_eligible_persons) >= n:
                sampled_elders = other_eligible_persons.sample(n)
                sampled_elders_ids = sampled_elders['Person_ID'].tolist()

                row['Assigned_Persons'].extend(sampled_elders_ids)

                # Remove the sampled persons from the individuals dataframe
                persons_df.drop(persons_df[persons_df['Person_ID'].isin(sampled_elders['Person_ID'])].index, inplace=True)

    # Composition # 9
    # composition '1FM-nC' One family: Married Couple: 2 or more dependent child
    if row['Composition'] == '1FM-nC':
        n = int(random.choices(values_size_nc, weights=weights_size_nc)[0]) - 2

        # Filter individuals dataframe based on criteria and sample one male person
        eligible_male_individuals = persons_df[
            (persons_df['marital'] == 'Married') &
            (persons_df['sex'] == 'M') &
            (~persons_df['age'].isin(child_ages))
        ]

        # Sample one male person
        if not eligible_male_individuals.empty:
            male_person = eligible_male_individuals.sample(1)

            # Get the male person's ethnicity and religion
            male_ethnicity = male_person['ethnicity'].values[0]
            male_religion = male_person['religion'].values[0]

            if random_number < inter_ethnic_ratio:
                if random_number < inter_rel_ratio:
                    eligible_female_individuals = persons_df[
                        (persons_df['marital'] == 'Married') &
                        (persons_df['sex'] == 'F') &
                        (persons_df['ethnicity'] != male_ethnicity) &
                        (persons_df['religion'] != male_religion) &
                        (~persons_df['age'].isin(child_ages))
                    ]
                else:
                    eligible_female_individuals = persons_df[
                        (persons_df['marital'] == 'Married') &
                        (persons_df['sex'] == 'F') &
                        (persons_df['ethnicity'] != male_ethnicity) &
                        (persons_df['religion'] == male_religion) &
                        (~persons_df['age'].isin(child_ages))
                    ]
            else:
                eligible_female_individuals = persons_df[
                        (persons_df['marital'] == 'Married') &
                        (persons_df['sex'] == 'F') &
                        (persons_df['ethnicity'] == male_ethnicity) &
                        (persons_df['religion'] == male_religion) &
                        (~persons_df['age'].isin(child_ages))
                    ]

            eligible_children = persons_df[
                (persons_df['ethnicity'] == male_ethnicity) &
                (persons_df['religion'] == male_religion) &
                (persons_df['age'].isin(child_ages))
            ]

            # Sample one female person
            if not eligible_female_individuals.empty:
                female_person = eligible_female_individuals.sample(1)

                if len(eligible_children) >= n:
                    sampled_children = eligible_children.sample(n)
                    sampled_children_ids = sampled_children['Person_ID'].tolist()

                    # Get Person_IDs
                    male_person_id = male_person['Person_ID'].values[0]
                    female_person_id = female_person['Person_ID'].values[0]

                    row['Assigned_Persons'].extend([male_person_id, female_person_id])
                    row['Assigned_Persons'].extend(sampled_children_ids)

                    persons_df.drop(persons_df[persons_df['Person_ID'].isin([male_person_id, female_person_id])].index, inplace=True)
                    persons_df.drop(persons_df[persons_df['Person_ID'].isin(sampled_children['Person_ID'])].index, inplace=True)

    # Composition # 10
    # composition '1FC-nC' One family: Cohabiting Couple: 2 or more dependent child
    if row['Composition'] == '1FC-nC':
        n = int(random.choices(values_size_nc, weights=weights_size_nc)[0]) - 2

        # Filter individuals dataframe based on criteria and sample one male person
        eligible_male_individuals = persons_df[
            (persons_df['marital'] != 'Married') &
            (persons_df['sex'] == 'M') &
            (~persons_df['age'].isin(child_ages))
        ]

        # Sample one male person
        if not eligible_male_individuals.empty:
            male_person = eligible_male_individuals.sample(1)

            # Get the male person's ethnicity and religion
            male_ethnicity = male_person['ethnicity'].values[0]
            male_religion = male_person['religion'].values[0]

            if random_number < inter_ethnic_ratio:
                if random_number < inter_rel_ratio:
                    eligible_female_individuals = persons_df[
                        (persons_df['marital'] == 'Married') &
                        (persons_df['sex'] == 'F') &
                        (persons_df['ethnicity'] != male_ethnicity) &
                        (persons_df['religion'] != male_religion) &
                        (~persons_df['age'].isin(child_ages))
                    ]
                else:
                    eligible_female_individuals = persons_df[
                        (persons_df['marital'] == 'Married') &
                        (persons_df['sex'] == 'F') &
                        (persons_df['ethnicity'] != male_ethnicity) &
                        (persons_df['religion'] == male_religion) &
                        (~persons_df['age'].isin(child_ages))
                    ]
            else:
                eligible_female_individuals = persons_df[
                        (persons_df['marital'] == 'Married') &
                        (persons_df['sex'] == 'F') &
                        (persons_df['ethnicity'] == male_ethnicity) &
                        (persons_df['religion'] == male_religion) &
                        (~persons_df['age'].isin(child_ages))
                    ]

            eligible_children = persons_df[
                (persons_df['ethnicity'] == male_ethnicity) &
                (persons_df['religion'] == male_religion) &
                (persons_df['age'].isin(child_ages))
            ]

            # Sample one female person
            if not eligible_female_individuals.empty:
                female_person = eligible_female_individuals.sample(1)

                if len(eligible_children) >= n:
                    sampled_children = eligible_children.sample(n)
                    sampled_children_ids = sampled_children['Person_ID'].tolist()

                    # Get Person_IDs
                    male_person_id = male_person['Person_ID'].values[0]
                    female_person_id = female_person['Person_ID'].values[0]

                    row['Assigned_Persons'].extend([male_person_id, female_person_id])
                    row['Assigned_Persons'].extend(sampled_children_ids)

                    persons_df.drop(persons_df[persons_df['Person_ID'].isin([male_person_id, female_person_id])].index, inplace=True)
                    persons_df.drop(persons_df[persons_df['Person_ID'].isin(sampled_children['Person_ID'])].index, inplace=True)

    # Composition # 11
    # composition '1FL-nC' One family: Lone Parent: 2 or more dependent child
    if row['Composition'] == '1FL-nC':
        n = int(random.choices(values_size_nc, weights=weights_size_nc)[0]) - 2

        # Filter individuals dataframe based on criteria and sample one male person
        eligible_parent = persons_df[
            (persons_df['marital'] != 'Married') &
            (~persons_df['age'].isin(child_ages))
        ]

        # sample one parent
        if not eligible_parent.empty:
            parent = eligible_parent.sample(1)

            # Get the male person's ethnicity and religion
            parent_ethnicity = parent['ethnicity'].values[0]
            parent_religion = parent['religion'].values[0]

            eligible_children = persons_df[
                (persons_df['ethnicity'] == parent_ethnicity) &
                (persons_df['religion'] == parent_religion) &
                (persons_df['age'].isin(child_ages))
            ]

            if len(eligible_children) >= n:
                sampled_children = eligible_children.sample(n)
                sampled_children_ids = sampled_children['Person_ID'].tolist()

                # Get Person_IDs
                parent_id = parent['Person_ID'].values[0]

                row['Assigned_Persons'].extend([parent_id])
                row['Assigned_Persons'].extend(sampled_children_ids)

                persons_df.drop(persons_df[persons_df['Person_ID'].isin([parent_id])].index, inplace=True)
                persons_df.drop(persons_df[persons_df['Person_ID'].isin(sampled_children['Person_ID'])].index, inplace=True)

    # Composition # 12
    # composition '1FM-nA' One family: Married Couple: all children non-dependent
    if row['Composition'] == '1FM-nA':
        n = int(random.choices(values_size_na, weights=weights_size_na)[0]) - 2

        # Filter individuals dataframe based on criteria and sample one male person
        eligible_male_individuals = persons_df[
            (persons_df['marital'] == 'Married') &
            (persons_df['sex'] == 'M') &
            (~persons_df['age'].isin(child_ages))
        ]

        # Sample one male person
        if not eligible_male_individuals.empty:
            male_person = eligible_male_individuals.sample(1)

            # Get the male person's ethnicity and religion
            male_ethnicity = male_person['ethnicity'].values[0]
            male_religion = male_person['religion'].values[0]

            if random_number < inter_ethnic_ratio:
                if random_number < inter_rel_ratio:
                    eligible_female_individuals = persons_df[
                        (persons_df['marital'] == 'Married') &
                        (persons_df['sex'] == 'F') &
                        (persons_df['ethnicity'] != male_ethnicity) &
                        (persons_df['religion'] != male_religion) &
                        (~persons_df['age'].isin(child_ages))
                    ]
                else:
                    eligible_female_individuals = persons_df[
                        (persons_df['marital'] == 'Married') &
                        (persons_df['sex'] == 'F') &
                        (persons_df['ethnicity'] != male_ethnicity) &
                        (persons_df['religion'] == male_religion) &
                        (~persons_df['age'].isin(child_ages))
                    ]
            else:
                eligible_female_individuals = persons_df[
                        (persons_df['marital'] == 'Married') &
                        (persons_df['sex'] == 'F') &
                        (persons_df['ethnicity'] == male_ethnicity) &
                        (persons_df['religion'] == male_religion) &
                        (~persons_df['age'].isin(child_ages))
                    ]

            eligible_children = persons_df[
                (persons_df['ethnicity'] == male_ethnicity) &
                (persons_df['religion'] == male_religion) &
                (persons_df['age'].isin(adult_ages))
            ]

            # Sample one female person
            if not eligible_female_individuals.empty:
                female_person = eligible_female_individuals.sample(1)

                if len(eligible_children) >= n:
                    sampled_children = eligible_children.sample(n)
                    sampled_children_ids = sampled_children['Person_ID'].tolist()

                    # Get Person_IDs
                    male_person_id = male_person['Person_ID'].values[0]
                    female_person_id = female_person['Person_ID'].values[0]

                    row['Assigned_Persons'].extend([male_person_id, female_person_id])
                    row['Assigned_Persons'].extend(sampled_children_ids)

                    persons_df.drop(persons_df[persons_df['Person_ID'].isin([male_person_id, female_person_id])].index, inplace=True)
                    persons_df.drop(persons_df[persons_df['Person_ID'].isin(sampled_children['Person_ID'])].index, inplace=True)

    # Composition # 13
    # composition '1FC-nA' One family: Cohabiting Couple: all children non-dependent
    if row['Composition'] == '1FC-nA':
        n = int(random.choices(values_size_na, weights=weights_size_na)[0]) - 2

        # Filter individuals dataframe based on criteria and sample one male person
        eligible_male_individuals = persons_df[
            (persons_df['marital'] != 'Married') &
            (persons_df['sex'] == 'M') &
            (~persons_df['age'].isin(child_ages))
        ]

        # Sample one male person
        if not eligible_male_individuals.empty:
            male_person = eligible_male_individuals.sample(1)

            # Get the male person's ethnicity and religion
            male_ethnicity = male_person['ethnicity'].values[0]
            male_religion = male_person['religion'].values[0]

            if random_number < inter_ethnic_ratio:
                if random_number < inter_rel_ratio:
                    eligible_female_individuals = persons_df[
                        (persons_df['marital'] == 'Married') &
                        (persons_df['sex'] == 'F') &
                        (persons_df['ethnicity'] != male_ethnicity) &
                        (persons_df['religion'] != male_religion) &
                        (~persons_df['age'].isin(child_ages))
                    ]
                else:
                    eligible_female_individuals = persons_df[
                        (persons_df['marital'] == 'Married') &
                        (persons_df['sex'] == 'F') &
                        (persons_df['ethnicity'] != male_ethnicity) &
                        (persons_df['religion'] == male_religion) &
                        (~persons_df['age'].isin(child_ages))
                    ]
            else:
                eligible_female_individuals = persons_df[
                        (persons_df['marital'] == 'Married') &
                        (persons_df['sex'] == 'F') &
                        (persons_df['ethnicity'] == male_ethnicity) &
                        (persons_df['religion'] == male_religion) &
                        (~persons_df['age'].isin(child_ages))
                    ]

            eligible_children = persons_df[
                (persons_df['ethnicity'] == male_ethnicity) &
                (persons_df['religion'] == male_religion) &
                (persons_df['age'].isin(adult_ages))
            ]

            # Sample one female person
            if not eligible_female_individuals.empty:
                female_person = eligible_female_individuals.sample(1)

                if len(eligible_children) >= n:
                    sampled_children = eligible_children.sample(n)
                    sampled_children_ids = sampled_children['Person_ID'].tolist()

                    # Get Person_IDs
                    male_person_id = male_person['Person_ID'].values[0]
                    female_person_id = female_person['Person_ID'].values[0]

                    row['Assigned_Persons'].extend([male_person_id, female_person_id])
                    row['Assigned_Persons'].extend(sampled_children_ids)

                    persons_df.drop(persons_df[persons_df['Person_ID'].isin([male_person_id, female_person_id])].index, inplace=True)
                    persons_df.drop(persons_df[persons_df['Person_ID'].isin(sampled_children['Person_ID'])].index, inplace=True)

    # Composition # 14
    # composition '1FL-nA' One family: Lone parent: all children non-dependent
    if row['Composition'] == '1FL-nA':
        n = int(random.choices(values_size_na, weights=weights_size_na)[0]) - 2

        # filtering individuals dataframe based on criteria and sampling one male person
        eligible_parent = persons_df[
            (persons_df['marital'] != 'Married') &
            (~persons_df['age'].isin(child_ages))
        ]

        # sampling one parent
        if not eligible_parent.empty:
            parent = eligible_parent.sample(1)

            # getting the first person's ethnicity and religion
            parent_ethnicity = parent['ethnicity'].values[0]
            parent_religion = parent['religion'].values[0]

            eligible_children = persons_df[
                (persons_df['ethnicity'] == parent_ethnicity) &
                (persons_df['religion'] == parent_religion) &
                (persons_df['age'].isin(adult_ages))
            ]

            if len(eligible_children) >= n:
                sampled_children = eligible_children.sample(n)
                sampled_children_ids = sampled_children['Person_ID'].tolist()

                # Get Person_IDs
                parent_id = parent['Person_ID'].values[0]

                row['Assigned_Persons'].extend([parent_id])
                row['Assigned_Persons'].extend(sampled_children_ids)

                persons_df.drop(persons_df[persons_df['Person_ID'].isin([parent_id])].index, inplace=True)
                persons_df.drop(persons_df[persons_df['Person_ID'].isin(sampled_children['Person_ID'])].index, inplace=True)

    # Composition # 15
    # composition '1H-1C' Other households: With one dependent child
    if row['Composition'] == '1H-1C':
        # Filter individuals dataframe based on criteria and sample one male person
        eligible_adults = persons_df[
            (~persons_df['age'].isin(child_ages))
        ]

        # Sample one male person
        if not eligible_adults.empty:
            adult = eligible_adults.sample(1)

            # Get the male person's ethnicity and religion
            adult_ethnicity = adult['ethnicity'].values[0]
            adult_religion = adult['religion'].values[0]

            eligible_children = persons_df[
                (persons_df['ethnicity'] == adult_ethnicity) &
                (persons_df['religion'] == adult_religion) &
                (persons_df['age'].isin(child_ages))
            ]

            if not eligible_children.empty:
                child = eligible_children.sample(1)

                # Get Person_IDs
                adult_id = adult['Person_ID'].values[0]
                child_id = child['Person_ID'].values[0]

                row['Assigned_Persons'].extend([adult_id, child_id])

                persons_df.drop(persons_df[persons_df['Person_ID'].isin([adult_id, child_id])].index, inplace=True)

    # Composition # 16
    # composition '1H-nC' Other households: With two or more dependent children


    if row['Composition'] == '1H-nC':
        n = int(random.choices(values_size, weights=weights_size)[0]) - 1

        # Filter individuals dataframe based on criteria and sample one male person
        eligible_adults = persons_df[
            (~persons_df['age'].isin(child_ages))
        ]

        # Sample one male person
        if not eligible_adults.empty:
            adult = eligible_adults.sample(1)

            # Get the male person's ethnicity and religion
            adult_ethnicity = adult['ethnicity'].values[0]
            adult_religion = adult['religion'].values[0]

            eligible_children = persons_df[
                (persons_df['ethnicity'] == adult_ethnicity) &
                (persons_df['religion'] == adult_religion) &
                (persons_df['age'].isin(child_ages))
            ]

            if len(eligible_children) >= n:
                sampled_children = eligible_children.sample(n)
                sampled_children_ids = sampled_children['Person_ID'].tolist()
                adult_id = adult['Person_ID'].values[0]

                row['Assigned_Persons'].extend([adult_id])
                row['Assigned_Persons'].extend(sampled_children_ids)

                persons_df.drop(persons_df[persons_df['Person_ID'].isin([adult_id])].index, inplace=True)
                persons_df.drop(persons_df[persons_df['Person_ID'].isin(sampled_children['Person_ID'])].index, inplace=True)

    # Composition # 17
    # composition '1H-nA' Other households: All student
    if row['Composition'] == '1H-nA':
        n = int(random.choices(values_size_org, weights=weights_size_org)[0])

        eligible_members = persons_df[
            (~persons_df['age'].isin(child_ages)) &
            (persons_df['qualification'] != 'no')
        ]

        if len(eligible_members) >= n:
            sampled_members = eligible_members.sample(n)
            sampled_members_ids = sampled_members['Person_ID'].tolist()

            row['Assigned_Persons'].extend(sampled_members_ids)

            persons_df.drop(persons_df[persons_df['Person_ID'].isin(sampled_members['Person_ID'])].index, inplace=True)

    # Composition # 18
    # composition '1H-nE' Other households: All pensioner
    if row['Composition'] == '1H-nA':
        n = int(random.choices(values_size_org, weights=weights_size_org)[0])

        eligible_members = persons_df[
            (persons_df['age'].isin(elder_ages))
        ]

        if len(eligible_members) >= n:
            sampled_members = eligible_members.sample(n)
            sampled_members_ids = sampled_members['Person_ID'].tolist()

            row['Assigned_Persons'].extend(sampled_members_ids)

            persons_df.drop(persons_df[persons_df['Person_ID'].isin(sampled_members['Person_ID'])].index, inplace=True)

    return row

# applying the assign_individuals function to each row in the households dataframe
households_df = households_df.apply(assign_individuals, axis=1)

# recording execution end time
end = time.time()
duration = end - start

# converting the recordind time to hours, minutes, and seconds
hours = int(duration // 3600)
minutes = int((duration % 3600) // 60)
seconds = duration % 60
print(f"Duration: {hours} hours, {minutes} minutes, {seconds:.2f} seconds")

empty_count = households_df['Assigned_Persons'].apply(lambda x: len(x) == 0).sum()
non_empty_count = households_df['Assigned_Persons'].apply(lambda x: len(x) > 0).sum()

print(f"Rows with empty lists: {empty_count}")
print(f"Rows with non-empty lists: {non_empty_count}")

households_df = households_df.sample(frac=1).reset_index(drop=True)
print(households_df.head())
print()
print(households_df.tail())

households_df.to_csv('households.csv', index=False)