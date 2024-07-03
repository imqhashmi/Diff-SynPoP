import random
import numpy as np
import geopandas as gpd
import pandas as pd
import os
import json
import plotly as py
import math
from collections import Counter
import itertools

def getkeys(df):
    groups = list(df.columns)[2:] #drop first two columns: areacode and total
    return groups

def get_weights(df, area):
    df = df[df['geography code'] == area]
    groups = list(df.columns)[2:] #drop first two columns: areacode and total
    values = df.values.flatten().tolist()[1:]
    total = values.pop(0)
    weights = [x / total for x in values]
    return weights
    # return np.random.choice(groups, size=size, replace=True, p=weights).tolist()

def get_weighted_samples_by_age_sex(df, age, sex, size):
    df = df[[col for col in df.columns if age in col]]
    df = df[[col for col in df.columns if sex in col]]
    groups = list(df.columns)
    values = df.values.flatten().tolist()
    total = sum(values)
    weights = [x / total for x in values]
    return np.random.choice(groups, size=size, replace=True, p=weights).tolist()

def get_weight(key, df):
    value = getdictionary(df).get(key)
    total = df.values.flatten().tolist()[1:].pop(0)
    return value/total

def getdictionary(df, area):
    df = df[df['geography code'] == area]
    if 'total' in df.columns:
        df = df.iloc[:, 1:] #drop total column
    dic = {}
    for index, row in df.iterrows():
        for index, column in enumerate(df.columns):
            if index==0:
                continue
            dic[column] = int(row[column])
    return dic

def aggregate_age(data):
    # Define age brackets
    child_ages = ['0_4', '5_7', '8_9', '10_14', '15', '16_17', '18_19']
    adult_ages = ['20_24', '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64']
    elder_ages = ['65_69', '70_74', '75_79', '80_84', '85+']

    aggregate = {}
    for key in data.keys():
        aggregate[key.split(' ')[0] + ' child ' + key.split(' ')[2]] = 0
        aggregate[key.split(' ')[0] + ' adult ' + key.split(' ')[2]] = 0
        aggregate[key.split(' ')[0] + ' elder ' + key.split(' ')[2]] = 0

    for key, val in data.items():
        k = key.split(' ')
        if k[1] in child_ages:
            aggregate[k[0] + ' child ' + k[2]] +=val
        elif k[1] in adult_ages:
            aggregate[k[0] + ' adult ' + k[2]] +=val
        elif k[1] in elder_ages:
            aggregate[k[0] + ' elder ' + k[2]] +=val
    return aggregate

def plot_crosstable(data, title):
    import plotly.express as px
    # Creating the bar plot
    fig = px.bar(x=list(data.keys()), y=list(data.values()), labels={'x': title, 'y': 'Count'})
    fig.update_layout(title=title, xaxis_tickangle=-45)
    return fig


def plot(actual, predicted, rmse):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x0='data', name='actual', y=actual, line_color='#636EFA'))
    fig.add_trace(
        go.Scatter(x0='data', name='pred', y=predicted, line_color='#EF553B', opacity=0.6))
    fig.update_layout(width=1000, title='RMSE=' + str(rmse), showlegend=False)
    py.offline.plot(fig, filename="temp.html")
    fig.show()

def generate_combinations(left, right, total):
    # generate all possible combinations
    combinations = itertools.product(left, right)
    return [combo for combo in combinations if combo[0].split(' ')[:2] == combo[1].split(' ')[:2]]


def convert_marital_cross_table(old_cross_table):
    new_age_categories = ['0_4', '5_7', '8_9', '10_14', '15', '16_17', '18_19', '20_24',
                          '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59',
                          '60_64', '65_69', '70_74', '75_79', '80_84', '85+']
    marital_statuses = ['Single', 'Married', 'Partner', 'Separated', 'Divorced', 'Widowed']

    # Initialize new cross table with all zeros
    new_cross_table = {'M ' + age + ' ' + status: 0 for age in new_age_categories for status in marital_statuses}
    new_cross_table.update({'F ' + age + ' ' + status: 0 for age in new_age_categories for status in marital_statuses})

    for key, value in old_cross_table.items():
        parts = key.split(' ')
        gender, age_range, marital_status = parts[0], parts[1], ' '.join(parts[2:])

        new_key = f'{gender} {age_range} {marital_status}'
        if new_key in new_cross_table:
            new_cross_table[new_key] += value

    return new_cross_table


def convert_qualification_cross_table(old_cross_table):
    new_age_categories = ['0_4', '5_7', '8_9', '10_14', '15', '16_17', '18_19', '20_24',
                          '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59',
                          '60_64', '65_69', '70_74', '75_79', '80_84', '85+']
    qualification_levels = ['no', 'level1', 'level2', 'apprent', 'level3', 'level4+', 'other']
    genders = ['M', 'F']

    # Initialize new cross table with all zeros
    new_cross_table = {}
    for gender in genders:
        for age in new_age_categories:
            for level in qualification_levels:
                new_cross_table[f'{gender} {age} {level}'] = 0

    # Map the old age ranges to the new age categories
    age_range_mapping = {
        '0_4': ['0_4'],
        '5_7': ['5_7'],
        '8_9': ['8_9'],
        '10_14': ['10_14'],
        '15': ['15'],
        '16_24': ['16_17', '18_19', '20_24'],
        '25_34': ['25_29', '30_34'],
        '35_49': ['35_39', '40_44', '45_49'],
        '50_64': ['50_54', '55_59', '60_64'],
        '65+': ['65_69', '70_74', '75_79', '80_84', '85+']
    }

    # Distribute counts uniformly
    for key, value in old_cross_table.items():
        parts = key.split(' ')
        gender, old_age_range, qual_level = parts[0], parts[1], parts[2]

        if old_age_range in age_range_mapping:
            new_ages = age_range_mapping[old_age_range]
            count_per_age = round(value / len(new_ages))  # Uniform distribution

            for new_age in new_ages:
                new_key = f'{gender} {new_age} {qual_level}'
                new_cross_table[new_key] += count_per_age

    return new_cross_table

def convert_composition_sex_age_cross_table(old_cross_table):
    new_age_categories = ['0_4', '5_7', '8_9', '10_14', '15', '16_17', '18_19', '20_24',
                          '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59',
                          '60_64', '65_69', '70_74', '75_79', '80_84', '85+']
    compositions = ['SP-Elder', 'SP-Adult', 'OF-Elder', 'OF-Married', 'OF-Married-0C', 'OF-Married-1C', 'OF-Married-2C', 'OF-Married-ND', 'OF-Cohabiting', 'OF-Cohabiting-0C', 'OF-Cohabiting-1C', 'OF-Cohabiting-2C', 'OF-Cohabiting-ND', 'OF-LoneParent', 'OF-Lone-1C', 'OF-Lone-2C', 'OF-Lone-ND', 'OH-1C', 'OH-2C', 'OH-Student', 'OH-Elder', 'OH-Adult']
    genders = ['M', 'F']

    # Initialize new cross table with all zeros
    new_cross_table = {}
    for gender in genders:
        for age in new_age_categories:
            for composition in compositions:
                new_cross_table[f'{gender} {age} {composition}'] = 0

    # Map the old age ranges to the new age categories
    age_range_mapping = {
        '0_15': ['0_4', '5_7', '8_9', '10_14', '15'],
        '16_24': ['16_17', '18_19', '20_24'],
        '25_34': ['25_29', '30_34'],
        '35_49': ['35_39', '40_44', '45_49'],
        '50+': ['50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85+']
    }

    # Distribute counts uniformly
    for key, value in old_cross_table.items():
        parts = key.split(' ')
        gender, old_age_range, comp = parts[0], parts[1], parts[2]

        if old_age_range in age_range_mapping:
            new_ages = age_range_mapping[old_age_range]
            count_per_age = round(value / len(new_ages))  # Uniform distribution

            for new_age in new_ages:
                new_key = f'{gender} {new_age} {comp}'
                new_cross_table[new_key] += count_per_age

    return new_cross_table


def convert_household_cross_table(old_cross_table):
    new_age_categories = ['0_4', '5_7', '8_9', '10_14', '15', '16_17', '18_19', '20_24',
                          '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59',
                          '60_64', '65_69', '70_74', '75_79', '80_84', '85+']
    household_compositions = ['1E_0C', '1A_0C', '2E_0C', '3A_1C', '3A_0C', '2A_0C', '2A_1C', '2A_3C', '1A_1C']
    genders = ['M', 'F']

    # Initialize new cross table with all zeros
    new_cross_table = {}
    for gender in genders:
        for age in new_age_categories:
            for comp in household_compositions:
                new_cross_table[f'{gender} {age} {comp}'] = 0

    # Map the old age ranges to the new age categories
    age_range_mapping = {
        '0_15': ['0_4', '5_7', '8_9', '10_14', '15'],
        '16_24': ['16_17', '18_19', '20_24'],
        '25_34': ['25_29', '30_34'],
        '35_49': ['35_39', '40_44', '45_49'],
        '50+': ['50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85+']
    }

    # Distribute counts uniformly
    for key, value in old_cross_table.items():
        parts = key.split(' ')
        gender, old_age_range, household_comp = parts[0], parts[1], parts[2]

        if old_age_range in age_range_mapping:
            new_ages = age_range_mapping[old_age_range]
            count_per_age = round(value / len(new_ages))  # Uniform distribution

            for new_age in new_ages:
                new_key = f'{gender} {new_age} {household_comp}'
                new_cross_table[new_key] += count_per_age

    return new_cross_table

path = os.path.join(os.path.dirname(os.getcwd()), 'input/diffspop/Diff-SynPoP')

sex_by_age = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'sex_by_age_5yrs.csv'))
cols = [(col.split(' ')[1] + ' ' + col.split(' ')[0]) for col in sex_by_age.columns[2:]]
sex_by_age.columns = ['geography code', 'total'] + cols

substring_mapping_rel = {
    'OR': 'O',
    'NR': 'N'
}

religion_by_sex_by_age = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'religion_by_sex_by_age.csv'))
for col in religion_by_sex_by_age.columns:
    for old_substring, new_substring in substring_mapping_rel.items():
        if old_substring in col:
            new_col = col.replace(old_substring, new_substring)
            religion_by_sex_by_age.rename(columns={col: new_col}, inplace=True)
            break
religion_by_sex_by_age = religion_by_sex_by_age.drop(columns=[col for col in religion_by_sex_by_age.columns if 'All' in col])

ethnicities = ['W0', 'M0', 'B0', 'A0', 'O0']
ethnic_by_sex_by_age = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'ethnic_by_sex_by_age.csv'))

ethnic_by_religion = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'ethnic_by_religion.csv'))
ethnic_by_religion = ethnic_by_religion.drop(columns=[col for col in ethnic_by_religion.columns[2:] if  col.split(' ')[0] not in ethnicities])
ethnic_by_religion.columns = [col.replace('0', '') for col in ethnic_by_religion.columns]

marital_by_sex_by_age = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'marital_by_sex_by_age.csv'))
# print(convert_marital_cross_table(getdictionary(marital_by_sex_by_age, 'E02005924')))

qualification_by_sex_by_age = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'qualification_by_sex_by_age.csv'))
# print(convert_qualification_cross_table(getdictionary(qualification_by_sex_by_age, 'E02005924')))

def getFinDictionary(df, area):
    return df

def getHHDictionary(df, area):
    return df

seg = {"1": 691, "2": 1523, "3": 964, "4": 594, "5": 708, "6": 1487, "7": 1242, "8": 404, "9": 451, "0": 2816}
occupation = {"1": 544, "2": 780, "3": 652, "4": 563, "5": 616, "6": 486, "7": 595, "8": 819, "9": 856, "0": 4969}
economic_act = {"1": 1027, "2": 4150, "3": 576, "4": 343, "5": 561, "6": 465, "7": 370, "8": 354, "9": 218, "0": 2816}
approx_social_grade = {"AB": 695, "C1": 1128, "C2": 837, "DE": 1391, "Not_Reference_Person": 6829}
general_health = {"Very_good_health": 5161, "Good_health": 3841, "Fair_health": 1339, "Bad_health": 434, "Very_bad_health": 105}
industry = {"A": 26, "B": 2, "C": 953, "D": 15, "E": 48, "F": 350, "G": 1135, "H": 421, "I": 340, "J": 307, "K": 158, "L": 59, "M": 291, "N": 318, "O": 199, "P": 450, "Q": 544, "R_S_T_U": 295, "Not_employed": 4969}

car_ownership = {"0": 1558, "1": 2227, "2": 851, "3": 169, "4+": 47}

# new terms for household compositions
substring_mapping = {
    'SP-Elder': '1PE',
    'SP-Adult': '1PA',
    'OF-Elder': '1FE',
    'OF-Married-0C': '1FM-0C',
    'OF-Married-2C': '1FM-2C',
    'OF-Married-ND': '1FM-nA',
    'OF-Cohabiting-0C': '1FC-0C',
    'OF-Cohabiting-2C': '1FC-2C',
    'OF-Cohabiting-ND': '1FC-nA',
    'OF-Lone-2C': '1FL-2C',
    'OF-Lone-ND': '1FL-nA',
    'OH-2C': '1H-2C',
    'OH-Student': '1H-nS',
    'OH-Elder': '1H-nE',
    'OH-Adult': '1H-nA',
}
# removing aggregated columns
substrings_to_exclude = ['OF-Married', 'OF-Cohabiting', 'OF-LoneParent']
# removing 0 in front of aggregated ethnicities
ethnicity_terms_mapping = {
    'W0': 'W',
    'M0': 'M',
    'A0': 'A',
    'B0': 'B',
    'O0': 'O'
}

# processing household composition by ethnicity cross table
HH_composition_by_Ethnicity = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'HH_composition_by_ethnicity.csv'))
filtered_columns = [col for col in HH_composition_by_Ethnicity.columns if any(substring in col for substring in ['geography code', 'total', 'W0', 'M0', 'A0', 'B0', 'O0'])]
HH_composition_by_Ethnicity = HH_composition_by_Ethnicity[filtered_columns]
for col in HH_composition_by_Ethnicity.columns:
    for old_substring, new_substring in substring_mapping.items():
        if old_substring in col:
            new_col = col.replace(old_substring, new_substring)
            HH_composition_by_Ethnicity.rename(columns={col: new_col}, inplace=True)
            break
for col in HH_composition_by_Ethnicity.columns:
    for old_substring, new_substring in ethnicity_terms_mapping.items():
        if old_substring in col:
            new_col = col.replace(old_substring, new_substring)
            HH_composition_by_Ethnicity.rename(columns={col: new_col}, inplace=True)
            break
filtered_columns = [col for col in HH_composition_by_Ethnicity.columns if not any(substring in col for substring in substrings_to_exclude)]
HH_composition_by_Ethnicity = HH_composition_by_Ethnicity[filtered_columns]

# processing household composition by religion cross table
HH_composition_by_Religion = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'HH_composition_by_religion.csv'))
for col in HH_composition_by_Religion.columns:
    for old_substring, new_substring in substring_mapping.items():
        if old_substring in col:
            new_col = col.replace(old_substring, new_substring)
            HH_composition_by_Religion.rename(columns={col: new_col}, inplace=True)
            break
filtered_columns = [col for col in HH_composition_by_Religion.columns if not any(substring in col for substring in substrings_to_exclude)]
HH_composition_by_Religion = HH_composition_by_Religion[filtered_columns]

# processing household composition by sex by age cross table
HH_composition_by_sex_by_age = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'HH_composition_by_sex_by_age.csv'))
for col in HH_composition_by_sex_by_age.columns:
    for old_substring, new_substring in substring_mapping.items():
        if old_substring in col:
            new_col = col.replace(old_substring, new_substring)
            HH_composition_by_sex_by_age.rename(columns={col: new_col}, inplace=True)
            break
filtered_columns = [col for col in HH_composition_by_sex_by_age.columns if not any(substring in col for substring in substrings_to_exclude)]
HH_composition_by_sex_by_age = HH_composition_by_sex_by_age[filtered_columns]