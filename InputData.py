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

def agemap(age):
    if age in ['0_4', '5_7', '8_9', '10_14', '15']:
        return "0_15"
    elif age in ['16_17', '18_19', '20_24']:
        return "16_24"
    elif age in ['25_29', '30_34']:
        return "25_34"
    elif age in ['35_39', '40_44', '45_49']:
        return "35_49"
    elif age in ['50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85+']:
        return "50+"

def reverse_agemap(age):
    if age=="0_15":
        return ['0_4', '5_7', '8_9', '10_14', '15']
    elif age=="16_24":
        return ['16_17', '18_19', '20_24']
    elif age=="25_34":
        return ['25_29', '30_34']
    elif age=="35_49":
        return ['35_39', '40_44', '45_49']
    elif age=="50+":
        return ['50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85+']

def key_replace(key):
    temp = key.split(' ')
    return temp[0] + ' ' + temp[1].replace('0', '')

def getweights(df, area):
    df = df[df['geography code'] == area]
    dic = df.iloc[0].to_dict()
    dic.pop('geography code')
    total = dic.pop('total')
    weights = [x / total for x in dic.values()]
    weights = {key: value/total for key, value in dic.items()}

    # weights for age bands
    weights = {'0_4': 0.4516,
           '5_7': 0.167,
           '8_9': 0.084,
           '10_14': 0.254,
           '15': 0.042,
           '16_17': 0.173,
           '18_19': 0.172,
           '20_24': 0.654,
           '25_29': 0.538,
           '30_34': 0.461,
           '35_39': 0.363,
           '40_44': 0.338,
           '45_49': 0.297,
           '50_54': 0.222,
           '55_59': 0.168,
           '60_64': 0.139,
           '65_69': 0.109,
           '70_74': 0.086,
           '75_79': 0.086,
           '80_84': 0.079,
           '85+': 0.107}
    
    return weights

def getHHcomdictionary(df, area):
    #filter df by area
    # df = df[df['geography code'] == area]
    df.drop(['total', '1FM', '1FC', '1FL'], axis=1, inplace=True)
    # dic = {}
    # for index, row in df.iterrows():
    #     for index, column in enumerate(df.columns):
    #         if index==0:
    #             continue
    #         dic[column] = int(row[column])
    # # return dict(sorted(dic.items()))
    df = df[df['geography code'] == area]
    dic = df.iloc[0].to_dict()
    dic.pop('geography code')
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

def aggregate_age_groups(age_distribution):
    # Define the age categories
    categories = {
        'child': ['0_4', '5_7', '8_9', '10_14', '15', '16_17'],
        'adult': ['18_19', '20_24', '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64'],
        'elder': ['65_69', '70_74', '75_79', '80_84', '85+']
    }

    # Initialize aggregated counts
    aggregated_counts = {'child': 0, 'adult': 0, 'elder': 0}

    # Aggregate the counts
    for category, age_ranges in categories.items():
        for age_range in age_ranges:
            aggregated_counts[category] += age_distribution.get(age_range, 0)
    return aggregated_counts

path = os.path.join(os.path.dirname(os.getcwd()), 'Diff-SynPoP')

oxford_areas = ['E02005921', 'E02005922', 'E02005923', 'E02005924', 'E02005925', 'E02005926', 'E02005927', 'E02005928', 'E02005929', 'E02005930', 'E02005931', 'E02005932', 'E02005933', 'E02005934', 'E02005935', 'E02005936', 'E02005937', 'E02005938', 'E02005939', 'E02005940', 'E02005941', 'E02005942', 'E02005943', 'E02005944', 'E02005945', 'E02005946', 'E02005947', 'E02005948', 'E02005949', 'E02005950', 'E02005951', 'E02005952', 'E02005953', 'E02005954', 'E02005955', 'E02005956', 'E02005957', 'E02005958', 'E02005959', 'E02005960', 'E02005961', 'E02005962', 'E02005963', 'E02005964', 'E02005965', 'E02005966', 'E02005967', 'E02005968', 'E02005969', 'E02005970', 'E02005971', 'E02005972', 'E02005973', 'E02005974', 'E02005975', 'E02005976', 'E02005977', 'E02005978', 'E02005979', 'E02005980', 'E02005981', 'E02005982', 'E02005983', 'E02005984', 'E02005985', 'E02005986', 'E02005987', 'E02005988', 'E02005991', 'E02005992', 'E02006886', 'E02005993', 'E02005994', 'E02005995', 'E02005996', 'E02005997', 'E02005998', 'E02005999', 'E02006000', 'E02006001', 'E02006002', 'E02006003', 'E02006004', 'E02006005', 'E02006006', 'E02006007']
# Read census data
age5ydf = pd.read_csv(os.path.join(path, 'Census_2011_MSOA', 'individual', 'Age_5yrs.csv'))
age5ydf = age5ydf[age5ydf['geography code'].isin(oxford_areas)]

sexdf = pd.read_csv(os.path.join(path, 'Census_2011_MSOA', 'individual', 'Sex.csv'))
sexdf = sexdf[sexdf['geography code'].isin(oxford_areas)]

ethnicdf = pd.read_csv(os.path.join(path, 'Census_2011_MSOA', 'individual', 'Ethnic.csv'))
ethnicdf = ethnicdf[ethnicdf['geography code'].isin(oxford_areas)]
ethnicdf = ethnicdf.drop(columns=[col for col in ethnicdf.columns[2:] if '0' not in col]) #remove all category columns
ethnicdf.columns = [col.replace('0', '') for col in ethnicdf.columns]

religiondf = pd.read_csv(os.path.join(path, 'Census_2011_MSOA', 'individual', 'Religion.csv'))
religiondf = religiondf[religiondf['geography code'].isin(oxford_areas)]

maritaldf = pd.read_csv(os.path.join(path, 'Census_2011_MSOA', 'individual', 'Marital.csv'))
maritaldf = maritaldf[maritaldf['geography code'].isin(oxford_areas)]

qualdf = pd.read_csv(os.path.join(path, 'Census_2011_MSOA', 'individual', 'Qualification.csv'))
qualdf = qualdf[qualdf['geography code'].isin(oxford_areas)]

seg_df = {"1": 691, "2": 1523, "3": 964, "4": 594, "5": 708, "6": 1487, "7": 1242, "8": 404, "9": 451, "0": 2816}
occupation_df = {"1": 544, "2": 780, "3": 652, "4": 563, "5": 616, "6": 486, "7": 595, "8": 819, "9": 856, "0": 4969}
economic_act_df = {"1": 1027, "2": 4150, "3": 576, "4": 343, "5": 561, "6": 465, "7": 370, "8": 354, "9": 218, "0": 2816}
approx_social_grade_df = {"AB": 695, "C1": 1128, "C2": 837, "DE": 1391, "NA": 6829}
general_health_df = {"Very_Good": 5161, "Good": 3841, "Fair": 1339, "Bad": 434, "Poor": 105}
industry_df = {"A": 26, "B": 2, "C": 953, "D": 15, "E": 48, "F": 350, "G": 1135, "H": 421, "I": 340, "J": 307, "K": 158, "L": 59, "M": 291, "N": 318, "O": 199, "P": 450, "Q": 544, "R_S_T_U": 295, "NE": 4969}

car_ownership_df = {"0": 1558, "1": 2227, "2": 851, "3": 169, "4+": 47}
income_df = {"Male": (830.9, 2.7), "Female": (602.7, 2.5)}

# processing household size distribution
HHsizedf = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'individual', 'HH_size.csv'))
HHsizedf = HHsizedf.rename(columns={'8+': '8'})

# processing household composition distribution
HHcomdf = pd.read_csv(os.path.join(path, 'Census_2011_MSOA', 'individual', 'HH_Compositions.csv'))
HHcomdf['1FM-2C'] = HHcomdf['1FM-1C'] + HHcomdf['1FM-nC']
HHcomdf['1FC-2C'] = HHcomdf['1FC-1C'] + HHcomdf['1FC-nC']
HHcomdf['1FL-2C'] = HHcomdf['1FL-1C'] + HHcomdf['1FL-nC']
HHcomdf['1H-2C'] = HHcomdf['1H-1C'] + HHcomdf['1H-nC']
HHcomdf.drop(columns=['1FM-1C', '1FM-nC', '1FC-1C', '1FC-nC', '1FL-1C', '1FL-nC', '1H-1C', '1H-nC'], inplace=True)

HHtypedf = pd.read_csv(os.path.join(path, 'Census_2011_MSOA', 'individual','HH_type.csv'))