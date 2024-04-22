import InputData as ID
import InputCrossTables as ICT
import pandas as pd
import time
import os

#MSOA
area = 'E02005924'
sex_dict = ID.getdictionary(ID.sexdf, area)
age_dict = ID.getdictionary(ID.age5ydf, area)
ethnic_dict = ID.getdictionary(ID.ethnicdf, area)
religion_dict = ID.getdictionary(ID.religiondf, area)
marital_dict = ID.getdictionary(ID.maritaldf, area)
qual_dict = ID.getdictionary(ID.qualdf, area)

HH_size = ID.getdictionary(ID.HHsizedf, area)
HH_type = ID.getdictionary(ID.HHtypedf, area)
HH_composition = ID.getdictionary(ID.HHcomdf, area)


print('area = ', area, ' Total population = ', ID.get_total(ID.sexdf, area))
print('Sex distribution: ', sex_dict)
print('Age distribution: ', ID.aggregate_age_groups(age_dict))

print('Ethnicity distribution: ', ethnic_dict.keys())
print('Religion distribution: ', religion_dict.keys())
print('Marital status distribution: ', marital_dict.keys())
print('Qualification distribution: ', qual_dict.keys())
print(' ')
print('Household size: ', HH_size)
print('Household type: ', HH_type)
print('Household composition: ', HH_composition)

print('Cross Tables: ')

sex_by_age = ICT.getdictionary(ICT.sex_by_age, area)
child_ages = ['0_4', '5_7', '8_9', '10_14', '15', '16_17', '18_19']
adult_ages = ['20_24', '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64']
elder_ages = ['65_69', '70_74', '75_79', '80_84', '85+']
aggregate = {}
for key in sex_by_age.keys():
    aggregate[key.split(' ')[0] + ' child '] = 0
    aggregate[key.split(' ')[0] + ' adult '] = 0
    aggregate[key.split(' ')[0] + ' elder '] = 0
for key, val in sex_by_age.items():
    k = key.split(' ')
    if k[1] in child_ages:
        aggregate[k[0] + ' child '] += val
    elif k[1] in adult_ages:
        aggregate[k[0] + ' adult '] += val
    elif k[1] in elder_ages:
        aggregate[k[0] + ' elder '] += val
print('Sex by Age: ', aggregate)
print('Sex by Age by Ethnicity:', ICT.aggregate_age(ICT.getdictionary(ICT.ethnic_by_sex_by_age, area)))
print('Sex by Age by Religion:', ICT.aggregate_age(ICT.getdictionary(ICT.religion_by_sex_by_age, area)))
print('Ethnicity by Religion:',ICT.getdictionary(ICT.ethnic_by_religion, area))
print('Sex by Age by Marital Status:',ICT.aggregate_age(ICT.convert_marital_cross_table(ICT.getdictionary(ICT.marital_by_sex_by_age, area))))
print('Sex by Age by Qualification:', ICT.aggregate_age(ICT.convert_qualification_cross_table(ICT.getdictionary(ICT.qualification_by_sex_by_age, area))))
print(' ')
print('Sex by Age by Household Composition', ICT.getdictionary(ICT.HH_composition_by_sex_by_age, area))
print('Household Composition by Ethnicity', ICT.getdictionary(ICT.HH_composition_by_Ethnicity, area))
print('Household Composition by Religion', ICT.getdictionary(ICT.HH_composition_by_Religion, area))