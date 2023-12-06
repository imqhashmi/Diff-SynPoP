import random
import numpy as np
import geopandas as gpd
import pandas as pd
import os
import json
import plotly.express as px
import plotly as py
import math

def getdictionary(df, area):
    #filter df by area
    df = df[df['geography code'] == area]
    if 'total' in df.columns:
        df = df.iloc[:, 1:] #drop total column
    dic = {}
    for index, row in df.iterrows():
        for index, column in enumerate(df.columns):
            if index==0:
                continue
            dic[column] = int(row[column])
    # return dict(sorted(dic.items()))
    return dic

def get_weighted_sample(df):
    groups = list(df.columns)[2:] #drop first two columns: areacode and total
    values = df.values.flatten().tolist()[1:]
    total = values.pop(0)
    weights = [x / total for x in values]
    # print(sum(weights))
    # get random values based on prob. distribution
    # result = [str(i) for i in list()]
    return np.random.choice(groups, size=1, replace=True, p=weights).tolist()[0]

def get_category_weights(df, area):
    #filter df by area
    df = df[df['geography code'] == area]
    groups = [col.strip() for col in list(df.columns)][2:] #drop first two columns: areacode and total
    values = df.values.flatten().tolist()[1:]
    total = values.pop(0)
    # divide each value by total to get probability
    weights = [x / total for x in values]
    return weights

def get_total(df, area):
    #filter df by area
    df = df[df['geography code'] == area]
    groups = [col.strip() for col in list(df.columns)][2:] #drop first two columns: areacode and total
    values = df.values.flatten().tolist()[1:]
    total = values.pop(0)
    return total


def get_weighted_samples(df, size):
    groups = [col.strip() for col in list(df.columns)][2:] #drop first two columns: areacode and total
    values = df.values.flatten().tolist()[1:]
    total = values.pop(0)
    weights = [x / total for x in values]
    # print(sum(weights))
    # get random values based on prob. distribution
    return np.random.choice(groups, size=size, replace=True, p=weights).tolist()

def get_weighted_samples_by_age_sex(df, age, sex, size):
    print(df.columns)
    df = df[[col for col in df.columns if age in col]]
    df = df[[col for col in df.columns if sex in col]]
    groups = list(df.columns)
    values = df.values.flatten().tolist()
    total = sum(values)
    weights = [x / total for x in values]
    return np.random.choice(groups, size=size, replace=True, p=weights).tolist()


path = os.path.join(os.path.dirname(os.getcwd()), 'Diff-SynPoP')

oxford_areas = ['E02005921', 'E02005922', 'E02005923', 'E02005924', 'E02005925', 'E02005926', 'E02005927', 'E02005928', 'E02005929', 'E02005930', 'E02005931', 'E02005932', 'E02005933', 'E02005934', 'E02005935', 'E02005936', 'E02005937', 'E02005938', 'E02005939', 'E02005940', 'E02005941', 'E02005942', 'E02005943', 'E02005944', 'E02005945', 'E02005946', 'E02005947', 'E02005948', 'E02005949', 'E02005950', 'E02005951', 'E02005952', 'E02005953', 'E02005954', 'E02005955', 'E02005956', 'E02005957', 'E02005958', 'E02005959', 'E02005960', 'E02005961', 'E02005962', 'E02005963', 'E02005964', 'E02005965', 'E02005966', 'E02005967', 'E02005968', 'E02005969', 'E02005970', 'E02005971', 'E02005972', 'E02005973', 'E02005974', 'E02005975', 'E02005976', 'E02005977', 'E02005978', 'E02005979', 'E02005980', 'E02005981', 'E02005982', 'E02005983', 'E02005984', 'E02005985', 'E02005986', 'E02005987', 'E02005988', 'E02005991', 'E02005992', 'E02006886', 'E02005993', 'E02005994', 'E02005995', 'E02005996', 'E02005997', 'E02005998', 'E02005999', 'E02006000', 'E02006001', 'E02006002', 'E02006003', 'E02006004', 'E02006005', 'E02006006', 'E02006007']
# Read census data
age5ydf = pd.read_csv(os.path.join(path, 'Census_2011_MSOA', 'individual', 'Age_5yrs.csv'))
age5ydf = age5ydf[age5ydf['geography code'].isin(oxford_areas)]

sexdf = pd.read_csv(os.path.join(path, 'Census_2011_MSOA', 'individual', 'Sex.csv'))
sexdf = sexdf[sexdf['geography code'].isin(oxford_areas)]

ethnicdf = pd.read_csv(os.path.join(path, 'Census_2011_MSOA', 'individual', 'Ethnic.csv'))
ethnicdf = ethnicdf[ethnicdf['geography code'].isin(oxford_areas)]
ethnicdf = ethnicdf.drop(columns=[col for col in ethnicdf.columns if '0' in col]) #remove all category columns

religiondf = pd.read_csv(os.path.join(path, 'Census_2011_MSOA', 'individual', 'Religion.csv'))
religiondf = religiondf[religiondf['geography code'].isin(oxford_areas)]
#raname columns
religiondf = religiondf.rename(columns={'N': 'NR', 'O': 'OR'})


maritaldf = pd.read_csv(os.path.join(path, 'Census_2011_MSOA', 'individual', 'Marital.csv'))
maritaldf = maritaldf[maritaldf['geography code'].isin(oxford_areas)]

qualdf = pd.read_csv(os.path.join(path, 'Census_2011_MSOA', 'individual', 'Qualification.csv'))
qualdf = qualdf[qualdf['geography code'].isin(oxford_areas)]

# HHsizedf = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'individual', 'HH_size.csv'))
# HHsizedf = HHsizedf[HHsizedf['geography code'] == 'E02005949']
#
# HHcomdf = pd.read_csv(os.path.join(path, 'SPONGE', 'Census_2011_MSOA', 'individual', 'HH_composition.csv'))
# HHcomdf = HHcomdf[HHcomdf['geography code'] == 'E02005949']
#
# HHtypedf = pd.read_csv(os.path.join(path, 'SPONGE', 'Census_2011_MSOA', 'individual', 'HH_type.csv'))
# HHtypedf = HHtypedf[HHtypedf['geography code'] == 'E02005949']

