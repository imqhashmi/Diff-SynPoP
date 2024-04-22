# IMPORTING LIBRARIES
import ast
import sys

# appending the system path to run the file on kaggle
# not required if you are running it locally
# sys.path.insert(1, '/kaggle/input/diffspop/Diff-SynPoP')

import os
import time
import random
import numpy as np
import plotly as py
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import torch
import torch.nn as nn
import torch.optim as optim

# importing InputData and InputCrossTables for processing UK census data files
import InputData as ID
import InputCrossTables as ICT

from torch.nn import init
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ExponentialLR

child_ages = ["0_4", "5_7", "8_9", "10_14", "15"]
adult_ages = ["16_17", "18_19", "20_24", "25_29", "30_34", "35_39", "40_44", "45_49", "50_54", "55_59", "60_64"]
elder_ages = ["65_69", "70_74", "75_79", "80_84", "85+"]

def is_student(person):
    # Students are often in certain age groups and have not yet completed the highest education level
    return person['age'] <= 25 and person['qualification'] not in ['no']

def check_composition(composition, assigned_individuals):
    total_members = len(assigned_individuals)

    if composition == '1PE':
        # One person: Pensioner
        return total_members == 1 and all(ind['age_group'] in elder_ages for _, ind in assigned_individuals.iterrows())
    elif composition == '1PA':
        # One person: Other (adult but not a pensioner)
        return total_members == 1 and all(ind['age_group'] in adult_ages for _, ind in assigned_individuals.iterrows())
    elif composition == '1FM-0C':
        # One family: Married Couple without Children
        return total_members == 2 and all(ind['age_group'] in adult_ages for _, ind in assigned_individuals.iterrows()) and all(ind['marital'] == 'Married' for _, ind in assigned_individuals.iterrows())
    elif composition == '1FC-0C':
        # Cohabiting couple without children
        return total_members == 2 and all(ind['age_group'] in adult_ages for _, ind in assigned_individuals.iterrows())
    elif composition == '1FE':
        # All pensioners
        return all(ind['age_group'] in elder_ages for _, ind in assigned_individuals.iterrows())
    elif composition == '1FM-2C':
        # Married couple with dependent children
        return all(ind['marital'] == 'Married' for _, ind in assigned_individuals.iterrows()) and any(ind['age_group'] in child_ages for _, ind in assigned_individuals.iterrows())
    elif composition == '1FC-2C':
        # Cohabiting couple with dependent children
        return any(ind['age_group'] in child_ages for _, ind in assigned_individuals.iterrows())
    elif composition == '1FL-2C':
        # Lone parent with dependent children
        return any(ind['age_group'] in child_ages for _, ind in assigned_individuals.iterrows()) and any(ind['age_group'] in adult_ages for _, ind in assigned_individuals.iterrows())
    elif composition == '1FM-nA':
        # Married couple with all children non-dependent
        return all(ind['marital'] == 'Married' for _, ind in assigned_individuals.iterrows()) and all(ind['age_group'] in adult_ages for _, ind in assigned_individuals.iterrows())
    elif composition == '1FC-nA':
        # Cohabiting couple with all children non-dependent
        return all(ind['age_group'] in adult_ages for _, ind in assigned_individuals.iterrows())
    elif composition == '1FL-nA':
        # Lone parent with all children non-dependent
        return all(ind['age_group'] in adult_ages for _, ind in assigned_individuals.iterrows())
    elif composition == '1H-2C':
        # Other households with dependent children
        return any(ind['age_group'] in child_ages for _, ind in assigned_individuals.iterrows())
    elif composition == '1H-nS':
        # All students (assuming this is based on another attribute like 'is_student')
        return all(ind.get('is_student', False) for _, ind in assigned_individuals.iterrows())  # Assumes is_student attribute exists
    elif composition == '1H-nE':
        # All pensioners
        return all(ind['age_group'] in elder_ages for _, ind in assigned_individuals.iterrows())
    elif composition == '1H-nA':
        # All adults
        return all(ind['age_group'] in adult_ages for _, ind in assigned_individuals.iterrows())
    else:
        return False  # Default case if the composition is not recognized


# load persons data from csv
persons_df = pd.read_csv('synthetic_population.csv')

# load households data from csv
households_df = pd.read_csv('households.csv')
# convert households_df['assigned_persons'] to list
households_df['assigned_persons'] = households_df['assigned_persons'].apply(lambda x: ast.literal_eval(x))
errors = 0  # Initialize error count
sum = 0
for index, row in households_df.iterrows():
    assigned_individuals = persons_df[persons_df['Person_ID'].isin(row['assigned_persons'])]
    sum += len(assigned_individuals)
    if not check_composition(row['composition'], assigned_individuals):
        errors += 1  # Increment error count for each failed household validation

print(f"Total errors: {errors}", f"Total persons: {sum}", sep="\n")