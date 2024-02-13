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
print('Households: 4239')
print('Sex distribution: ', sex_dict)
print('Age distribution: ', age_dict)

print('Ethnicity distribution: ', ethnic_dict)
print('Religion distribution: ', religion_dict)
print('Marital status distribution: ', marital_dict)
print('Qualification distribution: ', qual_dict)
print(' ')
print('Household size: ', HH_size)
print('Household type: ', HH_type)
print('Household composition: ', HH_composition)

print('Cross Tables: ')
print('Sex by Age by Household Composition', ICT.getdictionary(ICT.HH_composition_by_sex_by_age, area))
print('Household Composition by Ethnicity', ICT.getdictionary(ICT.HH_composition_by_Ethnicity, area))
print('Household Composition by Religion', ICT.getdictionary(ICT.HH_composition_by_Religion, area))