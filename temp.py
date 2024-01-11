import torch
import torch.nn as nn
import torch.optim as optim
import InputData as ID
import InputCrossTables as ICT
import plotly.graph_objects as go
import  plotly as py
import pandas as pd
import time
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
path = os.path.join(os.path.dirname(os.getcwd()), 'Diff-SynPoP')

print(path)
#MSOA
area = 'E02005924'
total = ID.get_total(ID.age5ydf, area)

sex = ID.getdictionary(ID.sexdf, area)
age = ID.getdictionary(ID.age5ydf, area)
ethnic = ID.getdictionary(ID.ethnicdf, area)

print(sex)
print(age)
print(ethnic)
area = 'E02005924'
cross_table1 = ICT.getdictionary(ICT.ethnic_by_sex_by_age, area)
print(cross_table1)