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


path = os.path.join(os.path.dirname(os.getcwd()), 'Diff-SynPoP')

sex_by_age = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'sex_by_age_5yrs.csv'))

religion_by_sex_by_age = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'religion_by_sex_by_age.csv'))
religion_by_sex_by_age = religion_by_sex_by_age.drop(columns=[col for col in religion_by_sex_by_age.columns if 'All' in col])



ethnic_by_sex_by_age = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'ethnic_by_sex_by_age.csv'))

ethnic_by_religion = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'ethnic_by_religion.csv'))

marital_by_sex_by_age = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'marital_by_sex_by_age.csv'))

qualification_by_sex_by_age = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'qualification_by_sex_by_age.csv'))

HH_composition_by_sex_by_age = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'HH_composition_by_sex_by_age.csv'))

# HH_composition_by_sex_by_age['M 0-15 2A_0C'] = HH_composition_by_sex_by_age['M 0-15 2A_0C_a']+HH_composition_by_sex_by_age['M 0-15 2A_0C_b']
# HH_composition_by_sex_by_age['M 0-15 2A_1C'] = HH_composition_by_sex_by_age['M 0-15 2A_1C_a']+HH_composition_by_sex_by_age['M 0-15 2A_1C_b']
# HH_composition_by_sex_by_age['M 0-15 2A_3C'] = HH_composition_by_sex_by_age['M 0-15 2A_3C_a']+HH_composition_by_sex_by_age['M 0-15 2A_3C_b']
# HH_composition_by_sex_by_age['M 0-15 1A_1C'] = HH_composition_by_sex_by_age['M 0-15 1A_1C_a']+HH_composition_by_sex_by_age['M 0-15 1A_1C_b']
#
# HH_composition_by_sex_by_age['M 16-24 2A_0C'] = HH_composition_by_sex_by_age['M 16-24 2A_0C_a']+HH_composition_by_sex_by_age['M 16-24 2A_0C_b']
# HH_composition_by_sex_by_age['M 16-24 2A_1C'] = HH_composition_by_sex_by_age['M 16-24 2A_1C_a']+HH_composition_by_sex_by_age['M 16-24 2A_1C_b']
# HH_composition_by_sex_by_age['M 16-24 2A_3C'] = HH_composition_by_sex_by_age['M 16-24 2A_3C_a']+HH_composition_by_sex_by_age['M 16-24 2A_3C_b']
# HH_composition_by_sex_by_age['M 16-24 1A_1C'] = HH_composition_by_sex_by_age['M 16-24 1A_1C_a']+HH_composition_by_sex_by_age['M 16-24 1A_1C_b']
#
# HH_composition_by_sex_by_age['M 25-34 2A_0C'] = HH_composition_by_sex_by_age['M 25-34 2A_0C_a']+HH_composition_by_sex_by_age['M 25-34 2A_0C_b']
# HH_composition_by_sex_by_age['M 25-34 2A_1C'] = HH_composition_by_sex_by_age['M 25-34 2A_1C_a']+HH_composition_by_sex_by_age['M 25-34 2A_1C_b']
# HH_composition_by_sex_by_age['M 25-34 2A_3C'] = HH_composition_by_sex_by_age['M 25-34 2A_3C_a']+HH_composition_by_sex_by_age['M 25-34 2A_3C_b']
# HH_composition_by_sex_by_age['M 25-34 1A_1C'] = HH_composition_by_sex_by_age['M 25-34 1A_1C_a']+HH_composition_by_sex_by_age['M 25-34 1A_1C_b']
#
# HH_composition_by_sex_by_age['M 35-49 2A_0C'] = HH_composition_by_sex_by_age['M 35-49 2A_0C_a']+HH_composition_by_sex_by_age['M 35-49 2A_0C_b']
# HH_composition_by_sex_by_age['M 35-49 2A_1C'] = HH_composition_by_sex_by_age['M 35-49 2A_1C_a']+HH_composition_by_sex_by_age['M 35-49 2A_1C_b']
# HH_composition_by_sex_by_age['M 35-49 2A_3C'] = HH_composition_by_sex_by_age['M 35-49 2A_3C_a']+HH_composition_by_sex_by_age['M 35-49 2A_3C_b']
# HH_composition_by_sex_by_age['M 35-49 1A_1C'] = HH_composition_by_sex_by_age['M 35-49 1A_1C_a']+HH_composition_by_sex_by_age['M 35-49 1A_1C_b']
#
# HH_composition_by_sex_by_age['M 50+ 2A_0C'] = HH_composition_by_sex_by_age['M 50+ 2A_0C_a']+HH_composition_by_sex_by_age['M 50+ 2A_0C_b']
# HH_composition_by_sex_by_age['M 50+ 2A_1C'] = HH_composition_by_sex_by_age['M 50+ 2A_1C_a']+HH_composition_by_sex_by_age['M 50+ 2A_1C_b']
# HH_composition_by_sex_by_age['M 50+ 2A_3C'] = HH_composition_by_sex_by_age['M 50+ 2A_3C_a']+HH_composition_by_sex_by_age['M 50+ 2A_3C_b']
# HH_composition_by_sex_by_age['M 50+ 1A_1C'] = HH_composition_by_sex_by_age['M 50+ 1A_1C_a']+HH_composition_by_sex_by_age['M 50+ 1A_1C_b']
#
# HH_composition_by_sex_by_age['F 0-15 2A_0C'] = HH_composition_by_sex_by_age['F 0-15 2A_0C_a']+HH_composition_by_sex_by_age['F 0-15 2A_0C_b']
# HH_composition_by_sex_by_age['F 0-15 2A_1C'] = HH_composition_by_sex_by_age['F 0-15 2A_1C_a']+HH_composition_by_sex_by_age['F 0-15 2A_1C_b']
# HH_composition_by_sex_by_age['F 0-15 2A_3C'] = HH_composition_by_sex_by_age['F 0-15 2A_3C_a']+HH_composition_by_sex_by_age['F 0-15 2A_3C_b']
# HH_composition_by_sex_by_age['F 0-15 1A_1C'] = HH_composition_by_sex_by_age['F 0-15 1A_1C_a']+HH_composition_by_sex_by_age['F 0-15 1A_1C_b']
#
# HH_composition_by_sex_by_age['F 16-24 2A_0C'] = HH_composition_by_sex_by_age['F 16-24 2A_0C_a']+HH_composition_by_sex_by_age['F 16-24 2A_0C_b']
# HH_composition_by_sex_by_age['F 16-24 2A_1C'] = HH_composition_by_sex_by_age['F 16-24 2A_1C_a']+HH_composition_by_sex_by_age['F 16-24 2A_1C_b']
# HH_composition_by_sex_by_age['F 16-24 2A_3C'] = HH_composition_by_sex_by_age['F 16-24 2A_3C_a']+HH_composition_by_sex_by_age['F 16-24 2A_3C_b']
# HH_composition_by_sex_by_age['F 16-24 1A_1C'] = HH_composition_by_sex_by_age['F 16-24 1A_1C_a']+HH_composition_by_sex_by_age['F 16-24 1A_1C_b']
#
# HH_composition_by_sex_by_age['F 25-34 2A_0C'] = HH_composition_by_sex_by_age['F 25-34 2A_0C_a']+HH_composition_by_sex_by_age['F 25-34 2A_0C_b']
# HH_composition_by_sex_by_age['F 25-34 2A_1C'] = HH_composition_by_sex_by_age['F 25-34 2A_1C_a']+HH_composition_by_sex_by_age['F 25-34 2A_1C_b']
# HH_composition_by_sex_by_age['F 25-34 2A_3C'] = HH_composition_by_sex_by_age['F 25-34 2A_3C_a']+HH_composition_by_sex_by_age['F 25-34 2A_3C_b']
# HH_composition_by_sex_by_age['F 25-34 1A_1C'] = HH_composition_by_sex_by_age['F 25-34 1A_1C_a']+HH_composition_by_sex_by_age['F 25-34 1A_1C_b']
#
# HH_composition_by_sex_by_age['F 35-49 2A_0C'] = HH_composition_by_sex_by_age['F 35-49 2A_0C_a']+HH_composition_by_sex_by_age['F 35-49 2A_0C_b']
# HH_composition_by_sex_by_age['F 35-49 2A_1C'] = HH_composition_by_sex_by_age['F 35-49 2A_1C_a']+HH_composition_by_sex_by_age['F 35-49 2A_1C_b']
# HH_composition_by_sex_by_age['F 35-49 2A_3C'] = HH_composition_by_sex_by_age['F 35-49 2A_3C_a']+HH_composition_by_sex_by_age['F 35-49 2A_3C_b']
# HH_composition_by_sex_by_age['F 35-49 1A_1C'] = HH_composition_by_sex_by_age['F 35-49 1A_1C_a']+HH_composition_by_sex_by_age['F 35-49 1A_1C_b']
#
# HH_composition_by_sex_by_age['F 50+ 2A_0C'] = HH_composition_by_sex_by_age['F 50+ 2A_0C_a']+HH_composition_by_sex_by_age['F 50+ 2A_0C_b']
# HH_composition_by_sex_by_age['F 50+ 2A_1C'] = HH_composition_by_sex_by_age['F 50+ 2A_1C_a']+HH_composition_by_sex_by_age['F 50+ 2A_1C_b']
# HH_composition_by_sex_by_age['F 50+ 2A_3C'] = HH_composition_by_sex_by_age['F 50+ 2A_3C_a']+HH_composition_by_sex_by_age['F 50+ 2A_3C_b']
# HH_composition_by_sex_by_age['F 50+ 1A_1C'] = HH_composition_by_sex_by_age['F 50+ 1A_1C_a']+HH_composition_by_sex_by_age['F 50+ 1A_1C_b']
#
# HH_composition_by_sex_by_age = HH_composition_by_sex_by_age.drop(columns=[col for col in HH_composition_by_sex_by_age.columns if '_a' in col])
# HH_composition_by_sex_by_age = HH_composition_by_sex_by_age.drop(columns=[col for col in HH_composition_by_sex_by_age.columns if '_b' in col])