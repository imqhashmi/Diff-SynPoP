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
    child_ages = ['0-4', '5-7', '8-9', '10-14', '15', '16-17', '18-19']
    adult_ages = ['20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64']
    elder_ages = ['65-69', '70-74', '75-79', '80-84', '85+']

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
    new_age_categories = ['0-4', '5-7', '8-9', '10-14', '15', '16-17', '18-19', '20-24',
                          '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59',
                          '60-64', '65-69', '70-74', '75-79', '80-84', '85+']
    marital_statuses = ['Single', 'Married', 'Partner', 'Separated', 'Divorced', 'Widowed']

    # Initialize new cross table with all zeros
    new_cross_table = {'M ' + age + ' ' + status: 0 for age in new_age_categories for status in marital_statuses}
    new_cross_table.update({'F ' + age + ' ' + status: 0 for age in new_age_categories for status in marital_statuses})

    for key, value in old_cross_table.items():
        parts = key.split(' ')
        gender, age_range, marital_status = parts[0], parts[1], ' '.join(parts[2:])

        # For age below 16, count them as 'Single', rest as '0'
        if age_range in ['0-4', '5-7', '8-9', '10-14', '15']:
            new_key = f'{gender} 15 Single'  # All below 16 are considered 15 and single
            new_cross_table[new_key] += value
        else:
            new_key = f'{gender} {age_range} {marital_status}'
            if new_key in new_cross_table:
                new_cross_table[new_key] += value

    return new_cross_table


def convert_qualification_cross_table(old_cross_table):
    new_age_categories = ['0-4', '5-7', '8-9', '10-14', '15', '16-17', '18-19', '20-24',
                          '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59',
                          '60-64', '65-69', '70-74', '75-79', '80-84', '85+']
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
        '16-24': ['16-17', '18-19', '20-24'],
        '25-34': ['25-29', '30-34'],
        '35-49': ['35-39', '40-44', '45-49'],
        '50-64': ['50-54', '55-59', '60-64'],
        '65+': ['65-69', '70-74', '75-79', '80-84', '85+']
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


def convert_household_cross_table(old_cross_table):
    new_age_categories = ['0-4', '5-7', '8-9', '10-14', '15', '16-17', '18-19', '20-24',
                          '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59',
                          '60-64', '65-69', '70-74', '75-79', '80-84', '85+']
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
        '0-15': ['0-4', '5-7', '8-9', '10-14', '15'],
        '16-24': ['16-17', '18-19', '20-24'],
        '25-34': ['25-29', '30-34'],
        '35-49': ['35-39', '40-44', '45-49'],
        '50+': ['50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85+']
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

path = os.path.join(os.path.dirname(os.getcwd()), 'Diff-SynPoP')

sex_by_age = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'sex_by_age_5yrs.csv'))
cols = [(col.split(' ')[1] + ' ' + col.split(' ')[0]) for col in sex_by_age.columns[2:]]
sex_by_age.columns = ['geography code', 'total'] + cols

religion_by_sex_by_age = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'religion_by_sex_by_age.csv'))
religion_by_sex_by_age = religion_by_sex_by_age.drop(columns=[col for col in religion_by_sex_by_age.columns if 'All' in col])

ethnicities = ['W0', 'M0', 'B0', 'A0', 'O0']
ethnic_by_sex_by_age = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'ethnic_by_sex_by_age.csv'))

ethnic_by_religion = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'ethnic_by_religion.csv'))
ethnic_by_religion = ethnic_by_religion.drop(columns=[col for col in ethnic_by_religion.columns[2:] if  col.split(' ')[0] in ethnicities])
# ethnic_by_religion.columns = [col.replace('0', '') for col in ethnic_by_religion.columns]

marital_by_sex_by_age = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'marital_by_sex_by_age.csv'))
# print(convert_marital_cross_table(getdictionary(marital_by_sex_by_age, 'E02005924')))

qualification_by_sex_by_age = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'qualification_by_sex_by_age.csv'))
# print(convert_qualification_cross_table(getdictionary(qualification_by_sex_by_age, 'E02005924')))
HH_composition_by_sex_by_age = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'HH_composition_by_sex_by_age.csv'))

drop_cols = ['SP', 'OF', 'OH']
HH_composition_by_Ethnicity = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'HH_composition_by_ethnicity.csv'))
HH_composition_by_Ethnicity = HH_composition_by_Ethnicity.drop(columns=[col for col in HH_composition_by_Ethnicity.columns[2:] if  col.split(' ')[1] in ethnicities])
HH_composition_by_Ethnicity = HH_composition_by_Ethnicity.drop(columns=[col for col in HH_composition_by_Ethnicity.columns[2:] if  col.split(' ')[1] in ethnicities])
HH_composition_by_Ethnicity = HH_composition_by_Ethnicity.drop(columns=[col for col in HH_composition_by_Ethnicity.columns[2:] if  col.split(' ')[0] in drop_cols])
# HH_composition_by_Ethnicity.columns = [col.replace('0', '') for col in HH_composition_by_Ethnicity.columns]
HH_composition_by_Religion = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'HH_composition_by_religion.csv'))


# sex_by_age = sex_by_age[sex_by_age['geography code'] == 'E02005927']
# ethnic_by_religion = ethnic_by_religion[ethnic_by_religion['geography code'] == 'E02005927']
# ethnic_by_sex_by_age = ethnic_by_sex_by_age[ethnic_by_sex_by_age['geography code'] == 'E02005927']
# religion_by_sex_by_age = religion_by_sex_by_age[religion_by_sex_by_age['geography code'] == 'E02005927']
#
# # save to csv
# sex_by_age.to_csv('sex_by_age.csv', index=False)
# ethnic_by_religion.to_csv('ethnic_by_religion.csv', index=False)
# ethnic_by_sex_by_age.to_csv('ethnic_by_sex_by_age.csv', index=False)
# religion_by_sex_by_age.to_csv('religion_by_sex_by_age.csv', index=False)