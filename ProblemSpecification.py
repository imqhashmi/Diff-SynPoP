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

print('area = ', area, ' Total population = ', ID.get_total(ID.sexdf, area))
print('Sex distribution: ', sex_dict)
print('Age distribution: ', ID.aggregate_age_groups(age_dict))

print('Ethnicity distribution: ', ethnic_dict)
print('Religion distribution: ', religion_dict)
print('Marital status distribution: ', marital_dict)
print('Qualification distribution: ', qual_dict)


sex_by_age = ICT.getdictionary(ICT.sex_by_age, area)
child_ages = ['0-4', '5-7', '8-9', '10-14', '15', '16-17', '18-19']
adult_ages = ['20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64']
elder_ages = ['65-69', '70-74', '75-79', '80-84', '85+']
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

