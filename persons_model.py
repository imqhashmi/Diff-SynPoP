import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mpld3 as mpld3
import plotly.graph_objs as go
import plotly.io as pio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import commons as cm

num_epochs = 2000
# Use CUDA if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

sex_dict = {'M': 3587, 'F': 3622}
age_dict = {'0_4': 405, '5_7': 201, '8_9': 198, '10_14': 659, '15': 177, '16_17': 322, '18_19': 197, '20_24': 361, '25_29': 681, '30_34': 566, '35_39': 430, '40_44': 438, '45_49': 439, '50_54': 386, '55_59': 360, '60_64': 337, '65_69': 288, '70_74': 202, '75_79': 176, '80_84': 154, '85+': 232}

ethnic_dict = {'W': 6066, 'M': 272, 'A': 581, 'B': 135, 'O': 155}
religion_dict = {'C': 3628, 'B': 76, 'H': 80, 'J': 88, 'M': 293, 'S': 4, 'O': 41, 'N': 2356, 'NS': 643}
marital_dict = {'Single': 2966, 'Married': 3111, 'Partner': 12, 'Separated': 177, 'Divorced': 524, 'Widowed': 419}
qual_dict = {'no': 2707, 'level1': 961, 'level2': 896, 'apprent': 152, 'level3': 643, 'level4+': 1342, 'other': 508}
occupation_dict = {'1': 360, '2': 517, '3': 432, '4': 373, '5': 408, '6': 322, '7': 394, '8': 543, '9': 567, '0': 3293}
econ_activity_dict = {'1': 680, '2': 2750, '3': 382, '4': 227, '5': 372, '6': 308, '7': 245, '8': 235, '9': 144, '0': 1866}
health_dict = {'Very_Good': 3419, 'Good': 2545, 'Fair': 887, 'Bad': 288, 'Poor': 70}

category_dicts = {
    'sex': sex_dict,
    'age': age_dict,
    'ethnicity': ethnic_dict,
    'religion': religion_dict,
    'marital': marital_dict,
    'qual': qual_dict,
    # 'occupation': occupation_dict,
    # 'economic': econ_activity_dict,
    # 'general_health': health_dict,
}

sex_by_age = {'M 0_4': 183, 'M 5_7': 99, 'M 8_9': 136, 'M 10_14': 425, 'M 15': 106, 'M 16_17': 170, 'M 18_19': 114, 'M 20_24': 162, 'M 25_29': 334, 'M 30_34': 308, 'M 35_39': 203, 'M 40_44': 214, 'M 45_49': 221, 'M 50_54': 164, 'M 55_59': 184, 'M 60_64': 153, 'M 65_69': 135, 'M 70_74': 90, 'M 75_79': 74, 'M 80_84': 57, 'M 85+': 55, 'F 0_4': 222, 'F 5_7': 102, 'F 8_9': 62, 'F 10_14': 234, 'F 15': 71, 'F 16_17': 152, 'F 18_19': 83, 'F 20_24': 199, 'F 25_29': 347, 'F 30_34': 258, 'F 35_39': 227, 'F 40_44': 224, 'F 45_49': 218, 'F 50_54': 222, 'F 55_59': 176, 'F 60_64': 184, 'F 65_69': 153, 'F 70_74': 112, 'F 75_79': 102, 'F 80_84': 97, 'F 85+': 177}
ethnic_by_sex_by_age = {'M 0_4 W': 156, 'M 0_4 M': 8, 'M 0_4 A': 1, 'M 0_4 B': 3, 'M 0_4 O': 10, 'M 5_7 W': 98, 'M 5_7 M': 4, 'M 5_7 A': 1, 'M 5_7 B': 3, 'M 5_7 O': 1, 'M 8_9 W': 124, 'M 8_9 M': 0, 'M 8_9 A': 3, 'M 8_9 B': 11, 'M 8_9 O': 1, 'M 10_14 W': 414, 'M 10_14 M': 4, 'M 10_14 A': 4, 'M 10_14 B': 19, 'M 10_14 O': 3, 'M 15 W': 112, 'M 15 M': 0, 'M 15 A': 0, 'M 15 B': 1, 'M 15 O': 0, 'M 16_17 W': 159, 'M 16_17 M': 3, 'M 16_17 A': 5, 'M 16_17 B': 4, 'M 16_17 O': 0, 'M 18_19 W': 109, 'M 18_19 M': 0, 'M 18_19 A': 1, 'M 18_19 B': 4, 'M 18_19 O': 1, 'M 20_24 W': 148, 'M 20_24 M': 1, 'M 20_24 A': 3, 'M 20_24 B': 4, 'M 20_24 O': 0, 'M 25_29 W': 230, 'M 25_29 M': 3, 'M 25_29 A': 5, 'M 25_29 B': 4, 'M 25_29 O': 8, 'M 30_34 W': 249, 'M 30_34 M': 1, 'M 30_34 A': 16, 'M 30_34 B': 12, 'M 30_34 O': 3, 'M 35_39 W': 174, 'M 35_39 M': 1, 'M 35_39 A': 4, 'M 35_39 B': 4, 'M 35_39 O': 5, 'M 40_44 W': 183, 'M 40_44 M': 0, 'M 40_44 A': 8, 'M 40_44 B': 4, 'M 40_44 O': 10, 'M 45_49 W': 215, 'M 45_49 M': 1, 'M 45_49 A': 7, 'M 45_49 B': 7, 'M 45_49 O': 0, 'M 50_54 W': 167, 'M 50_54 M': 0, 'M 50_54 A': 7, 'M 50_54 B': 0, 'M 50_54 O': 3, 'M 55_59 W': 221, 'M 55_59 M': 0, 'M 55_59 A': 1, 'M 55_59 B': 3, 'M 55_59 O': 1, 'M 60_64 W': 187, 'M 60_64 M': 0, 'M 60_64 A': 1, 'M 60_64 B': 0, 'M 60_64 O': 0, 'M 65_69 W': 152, 'M 65_69 M': 0, 'M 65_69 A': 3, 'M 65_69 B': 0, 'M 65_69 O': 4, 'M 70_74 W': 94, 'M 70_74 M': 0, 'M 70_74 A': 4, 'M 70_74 B': 1, 'M 70_74 O': 1, 'M 75_79 W': 82, 'M 75_79 M': 0, 'M 75_79 A': 1, 'M 75_79 B': 0, 'M 75_79 O': 0, 'M 80_84 W': 71, 'M 80_84 M': 0, 'M 80_84 A': 3, 'M 80_84 B': 0, 'M 80_84 O': 0, 'M 85+ W': 67, 'M 85+ M': 0, 'M 85+ A': 1, 'M 85+ B': 0, 'M 85+ O': 0, 'F 0_4 W': 185, 'F 0_4 M': 4, 'F 0_4 A': 7, 'F 0_4 B': 8, 'F 0_4 O': 4, 'F 5_7 W': 100, 'F 5_7 M': 1, 'F 5_7 A': 0, 'F 5_7 B': 0, 'F 5_7 O': 0, 'F 8_9 W': 64, 'F 8_9 M': 0, 'F 8_9 A': 0, 'F 8_9 B': 0, 'F 8_9 O': 0, 'F 10_14 W': 219, 'F 10_14 M': 3, 'F 10_14 A': 4, 'F 10_14 B': 3, 'F 10_14 O': 1, 'F 15 W': 79, 'F 15 M': 1, 'F 15 A': 1, 'F 15 B': 0, 'F 15 O': 0, 'F 16_17 W': 144, 'F 16_17 M': 0, 'F 16_17 A': 0, 'F 16_17 B': 4, 'F 16_17 O': 0, 'F 18_19 W': 72, 'F 18_19 M': 1, 'F 18_19 A': 1, 'F 18_19 B': 0, 'F 18_19 O': 1, 'F 20_24 W': 180, 'F 20_24 M': 3, 'F 20_24 A': 4, 'F 20_24 B': 3, 'F 20_24 O': 1, 'F 25_29 W': 265, 'F 25_29 M': 4, 'F 25_29 A': 22, 'F 25_29 B': 4, 'F 25_29 O': 8, 'F 30_34 W': 165, 'F 30_34 M': 0, 'F 30_34 A': 15, 'F 30_34 B': 4, 'F 30_34 O': 4, 'F 35_39 W': 171, 'F 35_39 M': 3, 'F 35_39 A': 7, 'F 35_39 B': 7, 'F 35_39 O': 1, 'F 40_44 W': 187, 'F 40_44 M': 0, 'F 40_44 A': 10, 'F 40_44 B': 4, 'F 40_44 O': 1, 'F 45_49 W': 215, 'F 45_49 M': 0, 'F 45_49 A': 3, 'F 45_49 B': 5, 'F 45_49 O': 0, 'F 50_54 W': 230, 'F 50_54 M': 0, 'F 50_54 A': 1, 'F 50_54 B': 4, 'F 50_54 O': 1, 'F 55_59 W': 168, 'F 55_59 M': 0, 'F 55_59 A': 1, 'F 55_59 B': 0, 'F 55_59 O': 1, 'F 60_64 W': 196, 'F 60_64 M': 0, 'F 60_64 A': 3, 'F 60_64 B': 3, 'F 60_64 O': 1, 'F 65_69 W': 152, 'F 65_69 M': 1, 'F 65_69 A': 3, 'F 65_69 B': 1, 'F 65_69 O': 3, 'F 70_74 W': 113, 'F 70_74 M': 0, 'F 70_74 A': 7, 'F 70_74 B': 1, 'F 70_74 O': 1, 'F 75_79 W': 119, 'F 75_79 M': 1, 'F 75_79 A': 0, 'F 75_79 B': 0, 'F 75_79 O': 0, 'F 80_84 W': 119, 'F 80_84 M': 0, 'F 80_84 A': 0, 'F 80_84 B': 0, 'F 80_84 O': 0, 'F 85+ W': 223, 'F 85+ M': 0, 'F 85+ A': 1, 'F 85+ B': 0, 'F 85+ O': 0}
religion_by_sex_by_age = {'M 0_4 C': 73, 'M 0_4 B': 1, 'M 0_4 H': 1, 'M 0_4 J': 5, 'M 0_4 M': 14, 'M 0_4 S': 0, 'M 0_4 O': 0, 'M 0_4 N': 62, 'M 0_4 NS': 27, 'M 5_7 C': 55, 'M 5_7 B': 1, 'M 5_7 H': 2, 'M 5_7 J': 0, 'M 5_7 M': 8, 'M 5_7 S': 0, 'M 5_7 O': 0, 'M 5_7 N': 24, 'M 5_7 NS': 9, 'M 8_9 C': 83, 'M 8_9 B': 0, 'M 8_9 H': 2, 'M 8_9 J': 1, 'M 8_9 M': 10, 'M 8_9 S': 0, 'M 8_9 O': 0, 'M 8_9 N': 34, 'M 8_9 NS': 6, 'M 10_14 C': 274, 'M 10_14 B': 1, 'M 10_14 H': 3, 'M 10_14 J': 3, 'M 10_14 M': 16, 'M 10_14 S': 0, 'M 10_14 O': 0, 'M 10_14 N': 99, 'M 10_14 NS': 29, 'M 15 C': 57, 'M 15 B': 2, 'M 15 H': 0, 'M 15 J': 1, 'M 15 M': 7, 'M 15 S': 0, 'M 15 O': 0, 'M 15 N': 34, 'M 15 NS': 5, 'M 16_17 C': 81, 'M 16_17 B': 5, 'M 16_17 H': 3, 'M 16_17 J': 3, 'M 16_17 M': 8, 'M 16_17 S': 0, 'M 16_17 O': 2, 'M 16_17 N': 58, 'M 16_17 NS': 10, 'M 18_19 C': 42, 'M 18_19 B': 2, 'M 18_19 H': 1, 'M 18_19 J': 1, 'M 18_19 M': 6, 'M 18_19 S': 0, 'M 18_19 O': 0, 'M 18_19 N': 51, 'M 18_19 NS': 11, 'M 20_24 C': 39, 'M 20_24 B': 0, 'M 20_24 H': 2, 'M 20_24 J': 1, 'M 20_24 M': 12, 'M 20_24 S': 0, 'M 20_24 O': 0, 'M 20_24 N': 89, 'M 20_24 NS': 19, 'M 25_29 C': 130, 'M 25_29 B': 7, 'M 25_29 H': 3, 'M 25_29 J': 2, 'M 25_29 M': 25, 'M 25_29 S': 1, 'M 25_29 O': 3, 'M 25_29 N': 144, 'M 25_29 NS': 19, 'M 30_34 C': 102, 'M 30_34 B': 6, 'M 30_34 H': 5, 'M 30_34 J': 3, 'M 30_34 M': 9, 'M 30_34 S': 0, 'M 30_34 O': 4, 'M 30_34 N': 148, 'M 30_34 NS': 31, 'M 35_39 C': 75, 'M 35_39 B': 3, 'M 35_39 H': 5, 'M 35_39 J': 2, 'M 35_39 M': 13, 'M 35_39 S': 0, 'M 35_39 O': 1, 'M 35_39 N': 90, 'M 35_39 NS': 14, 'M 40_44 C': 97, 'M 40_44 B': 2, 'M 40_44 H': 3, 'M 40_44 J': 4, 'M 40_44 M': 8, 'M 40_44 S': 0, 'M 40_44 O': 1, 'M 40_44 N': 78, 'M 40_44 NS': 21, 'M 45_49 C': 111, 'M 45_49 B': 2, 'M 45_49 H': 2, 'M 45_49 J': 2, 'M 45_49 M': 7, 'M 45_49 S': 1, 'M 45_49 O': 4, 'M 45_49 N': 72, 'M 45_49 NS': 20, 'M 50_54 C': 66, 'M 50_54 B': 2, 'M 50_54 H': 3, 'M 50_54 J': 2, 'M 50_54 M': 4, 'M 50_54 S': 0, 'M 50_54 O': 2, 'M 50_54 N': 65, 'M 50_54 NS': 20, 'M 55_59 C': 88, 'M 55_59 B': 3, 'M 55_59 H': 1, 'M 55_59 J': 4, 'M 55_59 M': 7, 'M 55_59 S': 0, 'M 55_59 O': 3, 'M 55_59 N': 63, 'M 55_59 NS': 15, 'M 60_64 C': 78, 'M 60_64 B': 3, 'M 60_64 H': 0, 'M 60_64 J': 0, 'M 60_64 M': 1, 'M 60_64 S': 0, 'M 60_64 O': 1, 'M 60_64 N': 52, 'M 60_64 NS': 18, 'M 65_69 C': 70, 'M 65_69 B': 1, 'M 65_69 H': 1, 'M 65_69 J': 5, 'M 65_69 M': 3, 'M 65_69 S': 0, 'M 65_69 O': 2, 'M 65_69 N': 42, 'M 65_69 NS': 11, 'M 70_74 C': 50, 'M 70_74 B': 2, 'M 70_74 H': 1, 'M 70_74 J': 0, 'M 70_74 M': 1, 'M 70_74 S': 0, 'M 70_74 O': 0, 'M 70_74 N': 25, 'M 70_74 NS': 11, 'M 75_79 C': 49, 'M 75_79 B': 0, 'M 75_79 H': 0, 'M 75_79 J': 0, 'M 75_79 M': 2, 'M 75_79 S': 0, 'M 75_79 O': 0, 'M 75_79 N': 18, 'M 75_79 NS': 5, 'M 80_84 C': 36, 'M 80_84 B': 0, 'M 80_84 H': 1, 'M 80_84 J': 1, 'M 80_84 M': 1, 'M 80_84 S': 0, 'M 80_84 O': 0, 'M 80_84 N': 11, 'M 80_84 NS': 7, 'M 85+ C': 41, 'M 85+ B': 0, 'M 85+ H': 0, 'M 85+ J': 2, 'M 85+ M': 1, 'M 85+ S': 0, 'M 85+ O': 0, 'M 85+ N': 7, 'M 85+ NS': 4, 'F 0_4 C': 103, 'F 0_4 B': 3, 'F 0_4 H': 4, 'F 0_4 J': 3, 'F 0_4 M': 20, 'F 0_4 S': 0, 'F 0_4 O': 0, 'F 0_4 N': 64, 'F 0_4 NS': 25, 'F 5_7 C': 49, 'F 5_7 B': 0, 'F 5_7 H': 0, 'F 5_7 J': 3, 'F 5_7 M': 8, 'F 5_7 S': 0, 'F 5_7 O': 0, 'F 5_7 N': 28, 'F 5_7 NS': 14, 'F 8_9 C': 38, 'F 8_9 B': 0, 'F 8_9 H': 0, 'F 8_9 J': 0, 'F 8_9 M': 1, 'F 8_9 S': 0, 'F 8_9 O': 0, 'F 8_9 N': 15, 'F 8_9 NS': 8, 'F 10_14 C': 121, 'F 10_14 B': 1, 'F 10_14 H': 2, 'F 10_14 J': 1, 'F 10_14 M': 11, 'F 10_14 S': 0, 'F 10_14 O': 0, 'F 10_14 N': 75, 'F 10_14 NS': 23, 'F 15 C': 44, 'F 15 B': 0, 'F 15 H': 1, 'F 15 J': 2, 'F 15 M': 1, 'F 15 S': 0, 'F 15 O': 0, 'F 15 N': 19, 'F 15 NS': 4, 'F 16_17 C': 86, 'F 16_17 B': 1, 'F 16_17 H': 0, 'F 16_17 J': 0, 'F 16_17 M': 3, 'F 16_17 S': 0, 'F 16_17 O': 0, 'F 16_17 N': 52, 'F 16_17 NS': 10, 'F 18_19 C': 38, 'F 18_19 B': 2, 'F 18_19 H': 0, 'F 18_19 J': 2, 'F 18_19 M': 3, 'F 18_19 S': 0, 'F 18_19 O': 0, 'F 18_19 N': 28, 'F 18_19 NS': 10, 'F 20_24 C': 98, 'F 20_24 B': 1, 'F 20_24 H': 2, 'F 20_24 J': 0, 'F 20_24 M': 7, 'F 20_24 S': 0, 'F 20_24 O': 2, 'F 20_24 N': 77, 'F 20_24 NS': 12, 'F 25_29 C': 127, 'F 25_29 B': 3, 'F 25_29 H': 7, 'F 25_29 J': 1, 'F 25_29 M': 20, 'F 25_29 S': 1, 'F 25_29 O': 0, 'F 25_29 N': 157, 'F 25_29 NS': 31, 'F 30_34 C': 114, 'F 30_34 B': 9, 'F 30_34 H': 4, 'F 30_34 J': 1, 'F 30_34 M': 15, 'F 30_34 S': 0, 'F 30_34 O': 3, 'F 30_34 N': 98, 'F 30_34 NS': 14, 'F 35_39 C': 116, 'F 35_39 B': 2, 'F 35_39 H': 5, 'F 35_39 J': 6, 'F 35_39 M': 10, 'F 35_39 S': 0, 'F 35_39 O': 1, 'F 35_39 N': 68, 'F 35_39 NS': 19, 'F 40_44 C': 114, 'F 40_44 B': 3, 'F 40_44 H': 7, 'F 40_44 J': 3, 'F 40_44 M': 4, 'F 40_44 S': 1, 'F 40_44 O': 4, 'F 40_44 N': 66, 'F 40_44 NS': 22, 'F 45_49 C': 121, 'F 45_49 B': 0, 'F 45_49 H': 1, 'F 45_49 J': 0, 'F 45_49 M': 10, 'F 45_49 S': 0, 'F 45_49 O': 0, 'F 45_49 N': 70, 'F 45_49 NS': 16, 'F 50_54 C': 128, 'F 50_54 B': 1, 'F 50_54 H': 1, 'F 50_54 J': 3, 'F 50_54 M': 3, 'F 50_54 S': 0, 'F 50_54 O': 1, 'F 50_54 N': 61, 'F 50_54 NS': 24, 'F 55_59 C': 91, 'F 55_59 B': 1, 'F 55_59 H': 0, 'F 55_59 J': 4, 'F 55_59 M': 6, 'F 55_59 S': 0, 'F 55_59 O': 2, 'F 55_59 N': 51, 'F 55_59 NS': 21, 'F 60_64 C': 106, 'F 60_64 B': 3, 'F 60_64 H': 1, 'F 60_64 J': 3, 'F 60_64 M': 3, 'F 60_64 S': 0, 'F 60_64 O': 3, 'F 60_64 N': 47, 'F 60_64 NS': 18, 'F 65_69 C': 95, 'F 65_69 B': 1, 'F 65_69 H': 2, 'F 65_69 J': 3, 'F 65_69 M': 2, 'F 65_69 S': 0, 'F 65_69 O': 1, 'F 65_69 N': 37, 'F 65_69 NS': 12, 'F 70_74 C': 74, 'F 70_74 B': 1, 'F 70_74 H': 3, 'F 70_74 J': 1, 'F 70_74 M': 1, 'F 70_74 S': 0, 'F 70_74 O': 1, 'F 70_74 N': 22, 'F 70_74 NS': 9, 'F 75_79 C': 76, 'F 75_79 B': 0, 'F 75_79 H': 0, 'F 75_79 J': 3, 'F 75_79 M': 1, 'F 75_79 S': 0, 'F 75_79 O': 0, 'F 75_79 N': 16, 'F 75_79 NS': 6, 'F 80_84 C': 72, 'F 80_84 B': 0, 'F 80_84 H': 0, 'F 80_84 J': 2, 'F 80_84 M': 1, 'F 80_84 S': 0, 'F 80_84 O': 0, 'F 80_84 N': 18, 'F 80_84 NS': 4, 'F 85+ C': 120, 'F 85+ B': 1, 'F 85+ H': 1, 'F 85+ J': 5, 'F 85+ M': 0, 'F 85+ S': 0, 'F 85+ O': 0, 'F 85+ N': 21, 'F 85+ NS': 29}
marital_by_sex_by_age = {'M 0_4 Single' : 192, 'F 0_4 Single' : 257, 'M 0_4 Married' : 0, 'M 0_4 Partner' : 0, 'M 0_4 Separated' : 0, 'M 0_4 Divorced' : 0, 'M 0_4 Widowed' : 0, 'F 0_4 Married' : 0, 'F 0_4 Partner' : 0, 'F 0_4 Separated' : 0, 'F 0_4 Divorced' : 0, 'F 0_4 Widowed' : 0, 'M 5_7 Single' : 89, 'F 5_7 Single' : 82, 'M 5_7 Married' : 0, 'M 5_7 Partner' : 0, 'M 5_7 Separated' : 0, 'M 5_7 Divorced' : 0, 'M 5_7 Widowed' : 0, 'F 5_7 Married' : 0, 'F 5_7 Partner' : 0, 'F 5_7 Separated' : 0, 'F 5_7 Divorced' : 0, 'F 5_7 Widowed' : 0, 'M 8_9 Single' : 134, 'F 8_9 Single' : 46, 'M 8_9 Married' : 0, 'M 8_9 Partner' : 0, 'M 8_9 Separated' : 0, 'M 8_9 Divorced' : 0, 'M 8_9 Widowed' : 0, 'F 8_9 Married' : 0, 'F 8_9 Partner' : 0, 'F 8_9 Separated' : 0, 'F 8_9 Divorced' : 0, 'F 8_9 Widowed' : 0, 'M 10_14 Single' : 498, 'F 10_14 Single' : 222, 'M 10_14 Married' : 0, 'M 10_14 Partner' : 0, 'M 10_14 Separated' : 0, 'M 10_14 Divorced' : 0, 'M 10_14 Widowed' : 0, 'F 10_14 Married' : 0, 'F 10_14 Partner' : 0, 'F 10_14 Separated' : 0, 'F 10_14 Divorced' : 0, 'F 10_14 Widowed' : 0, 'M 15 Single' : 84, 'F 15 Single' : 55, 'M 15 Married' : 0, 'M 15 Partner' : 0, 'M 15 Separated' : 0, 'M 15 Divorced' : 0, 'M 15 Widowed' : 0, 'F 15 Married' : 0, 'F 15 Partner' : 0, 'F 15 Separated' : 0, 'F 15 Divorced' : 0, 'F 15 Widowed' : 0, 'M 16_17 Single' : 64, 'M 16_17 Married' : 23, 'M 16_17 Partner' : 11, 'M 16_17 Separated' : 18, 'M 16_17 Divorced' : 29, 'M 16_17 Widowed' : 17, 'F 16_17 Single' : 63, 'F 16_17 Married' : 21, 'F 16_17 Partner' : 16, 'F 16_17 Separated' : 16, 'F 16_17 Divorced' : 18, 'F 16_17 Widowed' : 19, 'M 18_19 Single' : 44, 'M 18_19 Married' : 25, 'M 18_19 Partner' : 15, 'M 18_19 Separated' : 12, 'M 18_19 Divorced' : 9, 'M 18_19 Widowed' : 12, 'F 18_19 Single' : 31, 'F 18_19 Married' : 11, 'F 18_19 Partner' : 9, 'F 18_19 Separated' : 11, 'F 18_19 Divorced' : 10, 'F 18_19 Widowed' : 11, 'M 20_24 Single' : 63, 'M 20_24 Married' : 25, 'M 20_24 Partner' : 21, 'M 20_24 Separated' : 14, 'M 20_24 Divorced' : 15, 'M 20_24 Widowed' : 20, 'F 20_24 Single' : 89, 'F 20_24 Married' : 34, 'F 20_24 Partner' : 32, 'F 20_24 Separated' : 20, 'F 20_24 Divorced' : 34, 'F 20_24 Widowed' : 26, 'M 25_29 Single' : 110, 'M 25_29 Married' : 73, 'M 25_29 Partner' : 32, 'M 25_29 Separated' : 26, 'M 25_29 Divorced' : 31, 'M 25_29 Widowed' : 27, 'F 25_29 Single' : 149, 'F 25_29 Married' : 106, 'F 25_29 Partner' : 34, 'F 25_29 Separated' : 46, 'F 25_29 Divorced' : 37, 'F 25_29 Widowed' : 42, 'M 30_34 Single' : 95, 'M 30_34 Married' : 140, 'M 30_34 Partner' : 21, 'M 30_34 Separated' : 25, 'M 30_34 Divorced' : 32, 'M 30_34 Widowed' : 25, 'F 30_34 Single' : 81, 'F 30_34 Married' : 118, 'F 30_34 Partner' : 14, 'F 30_34 Separated' : 10, 'F 30_34 Divorced' : 20, 'F 30_34 Widowed' : 26, 'M 35_39 Single' : 28, 'M 35_39 Married' : 125, 'M 35_39 Partner' : 14, 'M 35_39 Separated' : 14, 'M 35_39 Divorced' : 12, 'M 35_39 Widowed' : 9, 'F 35_39 Single' : 36, 'F 35_39 Married' : 153, 'F 35_39 Partner' : 11, 'F 35_39 Separated' : 4, 'F 35_39 Divorced' : 16, 'F 35_39 Widowed' : 12, 'M 40_44 Single' : 26, 'M 40_44 Married' : 121, 'M 40_44 Partner' : 10, 'M 40_44 Separated' : 5, 'M 40_44 Divorced' : 12, 'M 40_44 Widowed' : 4, 'F 40_44 Single' : 24, 'F 40_44 Married' : 191, 'F 40_44 Partner' : 7, 'F 40_44 Separated' : 5, 'F 40_44 Divorced' : 11, 'F 40_44 Widowed' : 14, 'M 45_49 Single' : 14, 'M 45_49 Married' : 175, 'M 45_49 Partner' : 4, 'M 45_49 Separated' : 9, 'M 45_49 Divorced' : 9, 'M 45_49 Widowed' : 9, 'F 45_49 Single' : 12, 'F 45_49 Married' : 153, 'F 45_49 Partner' : 6, 'F 45_49 Separated' : 3, 'F 45_49 Divorced' : 11, 'F 45_49 Widowed' : 9, 'M 50_54 Single' : 5, 'M 50_54 Married' : 126, 'M 50_54 Partner' : 5, 'M 50_54 Separated' : 4, 'M 50_54 Divorced' : 6, 'M 50_54 Widowed' : 9, 'F 50_54 Single' : 11, 'F 50_54 Married' : 141, 'F 50_54 Partner' : 4, 'F 50_54 Separated' : 3, 'F 50_54 Divorced' : 17, 'F 50_54 Widowed' : 17, 'M 55_59 Single' : 5, 'M 55_59 Married' : 151, 'M 55_59 Partner' : 5, 'M 55_59 Separated' : 7, 'M 55_59 Divorced' : 14, 'M 55_59 Widowed' : 3, 'F 55_59 Single' : 4, 'F 55_59 Married' : 138, 'F 55_59 Partner' : 2, 'F 55_59 Separated' : 1, 'F 55_59 Divorced' : 15, 'F 55_59 Widowed' : 7, 'M 60_64 Single' : 5, 'M 60_64 Married' : 140, 'M 60_64 Partner' : 3, 'M 60_64 Separated' : 2, 'M 60_64 Divorced' : 6, 'M 60_64 Widowed' : 3, 'F 60_64 Single' : 3, 'F 60_64 Married' : 140, 'F 60_64 Partner' : 2, 'F 60_64 Separated' : 5, 'F 60_64 Divorced' : 15, 'F 60_64 Widowed' : 12, 'M 65_69 Single' : 3, 'M 65_69 Married' : 94, 'M 65_69 Partner' : 3, 'M 65_69 Separated' : 4, 'M 65_69 Divorced' : 13, 'M 65_69 Widowed' : 7, 'F 65_69 Single' : 3, 'F 65_69 Married' : 90, 'F 65_69 Partner' : 3, 'F 65_69 Separated' : 3, 'F 65_69 Divorced' : 26, 'F 65_69 Widowed' : 6, 'M 70_74 Single' : 0, 'M 70_74 Married' : 58, 'M 70_74 Partner' : 2, 'M 70_74 Separated' : 1, 'M 70_74 Divorced' : 10, 'M 70_74 Widowed' : 4, 'F 70_74 Single' : 2, 'F 70_74 Married' : 73, 'F 70_74 Partner' : 1, 'F 70_74 Separated' : 4, 'F 70_74 Divorced' : 17, 'F 70_74 Widowed' : 14, 'M 75_79 Single' : 3, 'M 75_79 Married' : 46, 'M 75_79 Partner' : 5, 'M 75_79 Separated' : 1, 'M 75_79 Divorced' : 11, 'M 75_79 Widowed' : 2, 'F 75_79 Single' : 2, 'F 75_79 Married' : 48, 'F 75_79 Partner' : 3, 'F 75_79 Separated' : 2, 'F 75_79 Divorced' : 16, 'F 75_79 Widowed' : 17, 'M 80_84 Single' : 4, 'M 80_84 Married' : 22, 'M 80_84 Partner' : 1, 'M 80_84 Separated' : 3, 'M 80_84 Divorced' : 8, 'M 80_84 Widowed' : 9, 'F 80_84 Single' : 4, 'F 80_84 Married' : 43, 'F 80_84 Partner' : 6, 'F 80_84 Separated' : 2, 'F 80_84 Divorced' : 11, 'F 80_84 Widowed' : 47, 'M 85+ Single' : 0, 'M 85+ Married' : 11, 'M 85+ Partner' : 0, 'M 85+ Separated' : 1, 'M 85+ Divorced' : 7, 'M 85+ Widowed' : 23, 'F 85+ Single' : 19, 'F 85+ Married' : 23, 'F 85+ Partner' : 2, 'F 85+ Separated' : 10, 'F 85+ Divorced' : 12, 'F 85+ Widowed' : 143}
qual_by_sex_by_age = {'M 0_4 no': 307, 'M 0_4 level1': 0, 'M 0_4 level2': 0, 'M 0_4 apprent': 0, 'M 0_4 level3': 0, 'M 0_4 level4+': 0, 'M 0_4 other': 0, 'M 5_7 no': 109, 'M 5_7 level1': 0, 'M 5_7 level2': 0, 'M 5_7 apprent': 0, 'M 5_7 level3': 0, 'M 5_7 level4+': 0, 'M 5_7 other': 0, 'M 8_9 no': 60, 'M 8_9 level1': 0, 'M 8_9 level2': 0, 'M 8_9 apprent': 0, 'M 8_9 level3': 0, 'M 8_9 level4+': 0, 'M 8_9 other': 0, 'M 10_14 no': 186, 'M 10_14 level1': 0, 'M 10_14 level2': 0, 'M 10_14 apprent': 0, 'M 10_14 level3': 0, 'M 10_14 level4+': 0, 'M 10_14 other': 0, 'M 15 no': 28, 'M 15 level1': 0, 'M 15 level2': 0, 'M 15 apprent': 0, 'M 15 level3': 0, 'M 15 level4+': 0, 'M 15 other': 0, 'M 16_17 no': 13, 'M 16_17 level1': 22, 'M 16_17 level2': 19, 'M 16_17 apprent': 2, 'M 16_17 level3': 12, 'M 16_17 level4+': 11, 'M 16_17 other': 4, 'M 18_19 no': 11, 'M 18_19 level1': 19, 'M 18_19 level2': 17, 'M 18_19 apprent': 1, 'M 18_19 level3': 10, 'M 18_19 level4+': 9, 'M 18_19 other': 3, 'M 20_24 no': 42, 'M 20_24 level1': 73, 'M 20_24 level2': 64, 'M 20_24 apprent': 6, 'M 20_24 level3': 38, 'M 20_24 level4+': 34, 'M 20_24 other': 13, 'M 25_29 no': 46, 'M 25_29 level1': 63, 'M 25_29 level2': 62, 'M 25_29 apprent': 6, 'M 25_29 level3': 56, 'M 25_29 level4+': 126, 'M 25_29 other': 58, 'M 30_34 no': 46, 'M 30_34 level1': 62, 'M 30_34 level2': 62, 'M 30_34 apprent': 5, 'M 30_34 level3': 54, 'M 30_34 level4+': 123, 'M 30_34 other': 56, 'M 35_39 no': 39, 'M 35_39 level1': 58, 'M 35_39 level2': 43, 'M 35_39 apprent': 11, 'M 35_39 level3': 36, 'M 35_39 level4+': 76, 'M 35_39 other': 25, 'M 40_44 no': 39, 'M 40_44 level1': 58, 'M 40_44 level2': 43, 'M 40_44 apprent': 11, 'M 40_44 level3': 36, 'M 40_44 level4+': 76, 'M 40_44 other': 25, 'M 45_49 no': 33, 'M 45_49 level1': 49, 'M 45_49 level2': 36, 'M 45_49 apprent': 9, 'M 45_49 level3': 30, 'M 45_49 level4+': 64, 'M 45_49 other': 23, 'M 50_54 no': 57, 'M 50_54 level1': 23, 'M 50_54 level2': 24, 'M 50_54 apprent': 22, 'M 50_54 level3': 23, 'M 50_54 level4+': 42, 'M 50_54 other': 19, 'M 55_59 no': 41, 'M 55_59 level1': 17, 'M 55_59 level2': 17, 'M 55_59 apprent': 16, 'M 55_59 level3': 17, 'M 55_59 level4+': 30, 'M 55_59 other': 13, 'M 60_64 no': 33, 'M 60_64 level1': 13, 'M 60_64 level2': 14, 'M 60_64 apprent': 13, 'M 60_64 level3': 13, 'M 60_64 level4+': 24, 'M 60_64 other': 12, 'M 65_69 no': 56, 'M 65_69 level1': 6, 'M 65_69 level2': 4, 'M 65_69 apprent': 9, 'M 65_69 level3': 3, 'M 65_69 level4+': 13, 'M 65_69 other': 5, 'M 70_74 no': 40, 'M 70_74 level1': 4, 'M 70_74 level2': 3, 'M 70_74 apprent': 6, 'M 70_74 level3': 2, 'M 70_74 level4+': 9, 'M 70_74 other': 4, 'M 75_79 no': 36, 'M 75_79 level1': 4, 'M 75_79 level2': 3, 'M 75_79 apprent': 6, 'M 75_79 level3': 2, 'M 75_79 level4+': 9, 'M 75_79 other': 3, 'M 80_84 no': 27, 'M 80_84 level1': 3, 'M 80_84 level2': 2, 'M 80_84 apprent': 4, 'M 80_84 level3': 1, 'M 80_84 level4+': 6, 'M 80_84 other': 3, 'M 85+ no': 25, 'M 85+ level1': 3, 'M 85+ level2': 2, 'M 85+ apprent': 4, 'M 85+ level3': 1, 'M 85+ level4+': 6, 'M 85+ other': 1, 'F 0_4 no': 322, 'F 0_4 level1': 0, 'F 0_4 level2': 0, 'F 0_4 apprent': 0, 'F 0_4 level3': 0, 'F 0_4 level4+': 0, 'F 0_4 other': 0, 'F 5_7 no': 126, 'F 5_7 level1': 0, 'F 5_7 level2': 0, 'F 5_7 apprent': 0, 'F 5_7 level3': 0, 'F 5_7 level4+': 0, 'F 5_7 other': 0, 'F 8_9 no': 58, 'F 8_9 level1': 0, 'F 8_9 level2': 0, 'F 8_9 apprent': 0, 'F 8_9 level3': 0, 'F 8_9 level4+': 0, 'F 8_9 other': 0, 'F 10_14 no': 169, 'F 10_14 level1': 0, 'F 10_14 level2': 0, 'F 10_14 apprent': 0, 'F 10_14 level3': 0, 'F 10_14 level4+': 0, 'F 10_14 other': 0, 'F 15 no': 31, 'F 15 level1': 0, 'F 15 level2': 0, 'F 15 apprent': 0, 'F 15 level3': 0, 'F 15 level4+': 0, 'F 15 other': 0, 'F 16_17 no': 10, 'F 16_17 level1': 19, 'F 16_17 level2': 17, 'F 16_17 apprent': 1, 'F 16_17 level3': 14, 'F 16_17 level4+': 10, 'F 16_17 other': 3, 'F 18_19 no': 12, 'F 18_19 level1': 22, 'F 18_19 level2': 20, 'F 18_19 apprent': 1, 'F 18_19 level3': 16, 'F 18_19 level4+': 11, 'F 18_19 other': 4, 'F 20_24 no': 44, 'F 20_24 level1': 81, 'F 20_24 level2': 74, 'F 20_24 apprent': 3, 'F 20_24 level3': 60, 'F 20_24 level4+': 42, 'F 20_24 other': 16, 'F 25_29 no': 48, 'F 25_29 level1': 60, 'F 25_29 level2': 70, 'F 25_29 apprent': 2, 'F 25_29 level3': 52, 'F 25_29 level4+': 168, 'F 25_29 other': 54, 'F 30_34 no': 36, 'F 30_34 level1': 44, 'F 30_34 level2': 52, 'F 30_34 apprent': 2, 'F 30_34 level3': 38, 'F 30_34 level4+': 125, 'F 30_34 other': 40, 'F 35_39 no': 38, 'F 35_39 level1': 59, 'F 35_39 level2': 57, 'F 35_39 apprent': 2, 'F 35_39 level3': 34, 'F 35_39 level4+': 76, 'F 35_39 other': 21, 'F 40_44 no': 33, 'F 40_44 level1': 51, 'F 40_44 level2': 49, 'F 40_44 apprent': 1, 'F 40_44 level3': 29, 'F 40_44 level4+': 65, 'F 40_44 other': 19, 'F 45_49 no': 30, 'F 45_49 level1': 47, 'F 45_49 level2': 46, 'F 45_49 apprent': 1, 'F 45_49 level3': 27, 'F 45_49 level4+': 60, 'F 45_49 other': 17, 'F 50_54 no': 60, 'F 50_54 level1': 27, 'F 50_54 level2': 23, 'F 50_54 apprent': 2, 'F 50_54 level3': 13, 'F 50_54 level4+': 31, 'F 50_54 other': 15, 'F 55_59 no': 48, 'F 55_59 level1': 22, 'F 55_59 level2': 18, 'F 55_59 apprent': 2, 'F 55_59 level3': 10, 'F 55_59 level4+': 25, 'F 55_59 other': 13, 'F 60_64 no': 41, 'F 60_64 level1': 19, 'F 60_64 level2': 16, 'F 60_64 apprent': 1, 'F 60_64 level3': 9, 'F 60_64 level4+': 22, 'F 60_64 other': 11, 'F 65_69 no': 61, 'F 65_69 level1': 6, 'F 65_69 level2': 8, 'F 65_69 apprent': 0, 'F 65_69 level3': 1, 'F 65_69 level4+': 9, 'F 65_69 other': 5, 'F 70_74 no': 52, 'F 70_74 level1': 5, 'F 70_74 level2': 7, 'F 70_74 apprent': 0, 'F 70_74 level3': 1, 'F 70_74 level4+': 8, 'F 70_74 other': 5, 'F 75_79 no': 58, 'F 75_79 level1': 5, 'F 75_79 level2': 7, 'F 75_79 apprent': 0, 'F 75_79 level3': 1, 'F 75_79 level4+': 9, 'F 75_79 other': 5, 'F 80_84 no': 60, 'F 80_84 level1': 6, 'F 80_84 level2': 7, 'F 80_84 apprent': 0, 'F 80_84 level3': 1, 'F 80_84 level4+': 9, 'F 80_84 other': 5, 'F 85+ no': 95, 'F 85+ level1': 9, 'F 85+ level2': 12, 'F 85+ apprent': 0, 'F 85+ level3': 3, 'F 85+ level4+': 15, 'F 85+ other': 11}

# Usage Example
child_age_keys = ['0_4', '5_7', '8_9', '10_14', '15']

# Population size
total_population = 7209

# Define the neural network model
class PopulationModel(nn.Module):
    def __init__(self, input_size, hidden_layers, output_sizes):
        super(PopulationModel, self).__init__()
        layers = []
        in_dim = input_size
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.hidden_layers = nn.Sequential(*layers)
        self.sex_output = nn.Linear(in_dim, output_sizes['sex'])
        self.age_output = nn.Linear(in_dim, output_sizes['age'])
        self.ethnic_output = nn.Linear(in_dim, output_sizes['ethnic'])
        self.religion_output = nn.Linear(in_dim, output_sizes['religion'])
        self.marital_output = nn.Linear(in_dim, output_sizes['marital'])
        self.qual_output = nn.Linear(in_dim, output_sizes['qual'])

    def forward(self, x, temperature=0.75):
        x = self.hidden_layers(x)
        sex_output = F.gumbel_softmax(self.sex_output(x), tau=temperature, hard=False)
        age_output = F.gumbel_softmax(self.age_output(x), tau=temperature, hard=False)
        ethnic_output = F.gumbel_softmax(self.ethnic_output(x), tau=temperature, hard=False)
        religion_output = F.gumbel_softmax(self.religion_output(x), tau=temperature, hard=False)
        marital_output = F.gumbel_softmax(self.marital_output(x), tau=temperature, hard=False)
        qual_output = F.gumbel_softmax(self.qual_output(x), tau=temperature, hard=False)
        return sex_output, age_output, ethnic_output, religion_output, marital_output, qual_output

def aggregate(outputs, cross_table, category_dicts):
    """
    Aggregates soft counts based on the output tensors and cross table, filtering out zero values.

    Parameters:
    - outputs: List of tensors corresponding to each characteristic's probabilities.
    - cross_table: Dictionary representing the target cross table.
    - category_dicts: List of dictionaries for each category (e.g., [sex_dict, age_dict]).

    Returns:
    - Aggregated tensor of counts based on the cross table.
    """
    keys = [key for key, value in cross_table.items()]
    aggregated_tensor = torch.zeros(len(keys), device=device)

    for i, key in enumerate(keys):
        category_keys = key.split(' ')
        expected_count = torch.ones(outputs[0].size(0), device=device)
        for output, category_key, category_dict in zip(outputs, category_keys, category_dicts):
            category_index = list(category_dict.keys()).index(category_key)
            expected_count *= output[:, category_index]
        aggregated_tensor[i] = torch.sum(expected_count)

    return aggregated_tensor


# Function to filter out child ages from the aggregated and target data
def filter_aggregated_and_targets(aggregated, targets):
    """
    Filters out child age entries from aggregated tensors and target dictionaries.

    Parameters:
    - aggregated: Aggregated tensor with counts.
    - targets: Target dictionary with counts.
    - child_age_keys: List of age categories to exclude.

    Returns:
    - Filtered aggregated tensor and target dictionary.
    """
    # Filter out keys corresponding to child ages from the target dictionary
    child_age_keys = ['0_4', '5_7', '8_9', '10_14', '15']

    filtered_keys = [key for key in targets if key.split(' ')[1] not in child_age_keys]
    filtered_targets = {key: targets[key] for key in filtered_keys}

    # Create a filtered version of the aggregated tensor based on the filtered keys
    filtered_aggregated = aggregated[:len(filtered_keys)]

    return filtered_aggregated, filtered_targets

def decode_outputs(sex_output, age_output, ethnic_output, religion_output, marital_output, qual_output):
    sex_decoded = sex_output.argmax(dim=1).tolist()
    age_decoded = age_output.argmax(dim=1).tolist()
    ethnic_decoded = ethnic_output.argmax(dim=1).tolist()
    religion_decoded = religion_output.argmax(dim=1).tolist()
    marital_decoded = marital_output.argmax(dim=1).tolist()
    qual_decoded = qual_output.argmax(dim=1).tolist()

    sex_labels = [list(sex_dict.keys())[idx] for idx in sex_decoded]
    age_labels = [list(age_dict.keys())[idx] for idx in age_decoded]
    ethnic_labels = [list(ethnic_dict.keys())[idx] for idx in ethnic_decoded]
    religion_labels = [list(religion_dict.keys())[idx] for idx in religion_decoded]
    marital_labels = [list(marital_dict.keys())[idx] for idx in marital_decoded]
    qual_labels = [list(qual_dict.keys())[idx] for idx in qual_decoded]

    decoded_df = pd.DataFrame({
        'Sex': sex_labels,
        'Age': age_labels,
        'Ethnicity': ethnic_labels,
        'Religion': religion_labels,
        'MaritalStatus': marital_labels,
        'Qualification': qual_labels
    })

    return decoded_df


input_size = 10
hidden_layers = [128, 64]
output_sizes = {
    'sex': len(sex_dict),
    'age': len(age_dict),
    'ethnic': len(ethnic_dict),
    'religion': len(religion_dict),
    'marital': len(marital_dict),
    'qual': len(qual_dict)
}

model = PopulationModel(input_size, hidden_layers, output_sizes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    optimizer.zero_grad()
    x = torch.randn(total_population, input_size, device=device, requires_grad=True)
    sex_output, age_output, ethnic_output, religion_output, marital_output, qual_output = model(x)

    sex_aggregated = aggregate([sex_output, age_output], sex_by_age, [sex_dict, age_dict])
    ethnic_aggregated = aggregate([sex_output, age_output, ethnic_output], ethnic_by_sex_by_age, [sex_dict, age_dict, ethnic_dict])
    religion_aggregated = aggregate([sex_output, age_output, religion_output], religion_by_sex_by_age, [sex_dict, age_dict, religion_dict])
    marital_aggregated = aggregate([sex_output, age_output, marital_output], marital_by_sex_by_age, [sex_dict, age_dict, marital_dict])
    qual_aggregated = aggregate([sex_output, age_output, qual_output], qual_by_sex_by_age, [sex_dict, age_dict, qual_dict])

    # Filter marital and qualification data to exclude children
    # marital_aggregated, marital_by_sex_by_age = filter_aggregated_and_targets(marital_aggregated, marital_by_sex_by_age)
    # qual_aggregated, qual_by_sex_by_age = filter_aggregated_and_targets(qual_aggregated, qual_by_sex_by_age)


    sex_target = torch.tensor([value for value in sex_by_age.values() ], dtype=torch.float32, device=device)
    ethnic_target = torch.tensor([value for value in ethnic_by_sex_by_age.values() ], dtype=torch.float32,device=device)
    religion_target = torch.tensor([value for value in religion_by_sex_by_age.values() ],dtype=torch.float32, device=device)
    marital_target = torch.tensor([value for value in marital_by_sex_by_age.values() ], dtype=torch.float32,device=device)
    qual_target = torch.tensor([value for value in qual_by_sex_by_age.values() ], dtype=torch.float32,device=device)

    # Calculate RMSE for each characteristic
    sex_loss = torch.sqrt(nn.functional.mse_loss(sex_aggregated, sex_target))
    ethnic_loss = torch.sqrt(nn.functional.mse_loss(ethnic_aggregated, ethnic_target))
    religion_loss = torch.sqrt(nn.functional.mse_loss(religion_aggregated, religion_target))
    marital_loss = torch.sqrt(nn.functional.mse_loss(marital_aggregated, marital_target))
    qual_loss = torch.sqrt(nn.functional.mse_loss(qual_aggregated, qual_target))

    total_loss = sex_loss + ethnic_loss + religion_loss + marital_loss + qual_loss
    total_loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(epoch, ": ", 'Total Loss:', total_loss.item(),
              'Sex Loss:',sex_loss.item(),
              'Ethnic Loss', ethnic_loss.item(),
              'Religion Loss', religion_loss.item(),
              'Marital Loss', marital_loss.item(),
              'Qual Loss', qual_loss.item())


# Decode and visualize outputs
df = decode_outputs(sex_output, age_output, ethnic_output, religion_output, marital_output, qual_output)
# df.to_csv('generated_population2.csv', index=False)


