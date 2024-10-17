import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import numpy as np
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import mpld3 as mpld3
import random

sex_dict = {'M': 3587, 'F': 3622}
age_dict = {'0_4': 405, '5_7': 201, '8_9': 198, '10_14': 659, '15': 177, '16_17': 322, '18_19': 197, '20_24': 361, '25_29': 681, '30_34': 566, '35_39': 430, '40_44': 438, '45_49': 439, '50_54': 386, '55_59': 360, '60_64': 337, '65_69': 288, '70_74': 202, '75_79': 176, '80_84': 154, '85+': 232}
ethnic_dict = {'W': 6066, 'M': 272, 'A': 581, 'B': 135, 'O': 155}
religion_dict = {'C': 3628, 'B': 76, 'H': 80, 'J': 88, 'M': 293, 'S': 4, 'O': 41, 'N': 2356, 'NS': 643}
marital_dict = {'Single': 2966, 'Married': 3111, 'Partner': 12, 'Separated': 177, 'Divorced': 524, 'Widowed': 419}
qual_dict = {'no': 2707, 'level1': 961, 'level2': 896, 'apprent': 152, 'level3': 643, 'level4+': 1342, 'other': 508}
occupation_dict = {'1': 360, '2': 517, '3': 432, '4': 373, '5': 408, '6': 322, '7': 394, '8': 543, '9': 567, '0': 3293}
econ_activity_dict = {'1': 680, '2': 2750, '3': 382, '4': 227, '5': 372, '6': 308, '7': 245, '8': 235, '9': 144, '0': 1866}
health_dict = {'Very_Good': 3419, 'Good': 2545, 'Fair': 887, 'Bad': 288, 'Poor': 70}

sex_by_age = {'M 0_4': 183, 'M 5_7': 99, 'M 8_9': 136, 'M 10_14': 425, 'M 15': 106, 'M 16_17': 170, 'M 18_19': 114, 'M 20_24': 162, 'M 25_29': 334, 'M 30_34': 308, 'M 35_39': 203, 'M 40_44': 214, 'M 45_49': 221, 'M 50_54': 164, 'M 55_59': 184, 'M 60_64': 153, 'M 65_69': 135, 'M 70_74': 90, 'M 75_79': 74, 'M 80_84': 57, 'M 85+': 55, 'F 0_4': 222, 'F 5_7': 102, 'F 8_9': 62, 'F 10_14': 234, 'F 15': 71, 'F 16_17': 152, 'F 18_19': 83, 'F 20_24': 199, 'F 25_29': 347, 'F 30_34': 258, 'F 35_39': 227, 'F 40_44': 224, 'F 45_49': 218, 'F 50_54': 222, 'F 55_59': 176, 'F 60_64': 184, 'F 65_69': 153, 'F 70_74': 112, 'F 75_79': 102, 'F 80_84': 97, 'F 85+': 177}
ethnic_by_sex_by_age = {'M 0_4 W': 156, 'M 0_4 M': 8, 'M 0_4 A': 1, 'M 0_4 B': 3, 'M 0_4 O': 10, 'M 5_7 W': 98, 'M 5_7 M': 4, 'M 5_7 A': 1, 'M 5_7 B': 3, 'M 5_7 O': 1, 'M 8_9 W': 124, 'M 8_9 M': 0, 'M 8_9 A': 3, 'M 8_9 B': 11, 'M 8_9 O': 1, 'M 10_14 W': 414, 'M 10_14 M': 4, 'M 10_14 A': 4, 'M 10_14 B': 19, 'M 10_14 O': 3, 'M 15 W': 112, 'M 15 M': 0, 'M 15 A': 0, 'M 15 B': 1, 'M 15 O': 0, 'M 16_17 W': 159, 'M 16_17 M': 3, 'M 16_17 A': 5, 'M 16_17 B': 4, 'M 16_17 O': 0, 'M 18_19 W': 109, 'M 18_19 M': 0, 'M 18_19 A': 1, 'M 18_19 B': 4, 'M 18_19 O': 1, 'M 20_24 W': 148, 'M 20_24 M': 1, 'M 20_24 A': 3, 'M 20_24 B': 4, 'M 20_24 O': 0, 'M 25_29 W': 230, 'M 25_29 M': 3, 'M 25_29 A': 5, 'M 25_29 B': 4, 'M 25_29 O': 8, 'M 30_34 W': 249, 'M 30_34 M': 1, 'M 30_34 A': 16, 'M 30_34 B': 12, 'M 30_34 O': 3, 'M 35_39 W': 174, 'M 35_39 M': 1, 'M 35_39 A': 4, 'M 35_39 B': 4, 'M 35_39 O': 5, 'M 40_44 W': 183, 'M 40_44 M': 0, 'M 40_44 A': 8, 'M 40_44 B': 4, 'M 40_44 O': 10, 'M 45_49 W': 215, 'M 45_49 M': 1, 'M 45_49 A': 7, 'M 45_49 B': 7, 'M 45_49 O': 0, 'M 50_54 W': 167, 'M 50_54 M': 0, 'M 50_54 A': 7, 'M 50_54 B': 0, 'M 50_54 O': 3, 'M 55_59 W': 221, 'M 55_59 M': 0, 'M 55_59 A': 1, 'M 55_59 B': 3, 'M 55_59 O': 1, 'M 60_64 W': 187, 'M 60_64 M': 0, 'M 60_64 A': 1, 'M 60_64 B': 0, 'M 60_64 O': 0, 'M 65_69 W': 152, 'M 65_69 M': 0, 'M 65_69 A': 3, 'M 65_69 B': 0, 'M 65_69 O': 4, 'M 70_74 W': 94, 'M 70_74 M': 0, 'M 70_74 A': 4, 'M 70_74 B': 1, 'M 70_74 O': 1, 'M 75_79 W': 82, 'M 75_79 M': 0, 'M 75_79 A': 1, 'M 75_79 B': 0, 'M 75_79 O': 0, 'M 80_84 W': 71, 'M 80_84 M': 0, 'M 80_84 A': 3, 'M 80_84 B': 0, 'M 80_84 O': 0, 'M 85+ W': 67, 'M 85+ M': 0, 'M 85+ A': 1, 'M 85+ B': 0, 'M 85+ O': 0, 'F 0_4 W': 185, 'F 0_4 M': 4, 'F 0_4 A': 7, 'F 0_4 B': 8, 'F 0_4 O': 4, 'F 5_7 W': 100, 'F 5_7 M': 1, 'F 5_7 A': 0, 'F 5_7 B': 0, 'F 5_7 O': 0, 'F 8_9 W': 64, 'F 8_9 M': 0, 'F 8_9 A': 0, 'F 8_9 B': 0, 'F 8_9 O': 0, 'F 10_14 W': 219, 'F 10_14 M': 3, 'F 10_14 A': 4, 'F 10_14 B': 3, 'F 10_14 O': 1, 'F 15 W': 79, 'F 15 M': 1, 'F 15 A': 1, 'F 15 B': 0, 'F 15 O': 0, 'F 16_17 W': 144, 'F 16_17 M': 0, 'F 16_17 A': 0, 'F 16_17 B': 4, 'F 16_17 O': 0, 'F 18_19 W': 72, 'F 18_19 M': 1, 'F 18_19 A': 1, 'F 18_19 B': 0, 'F 18_19 O': 1, 'F 20_24 W': 180, 'F 20_24 M': 3, 'F 20_24 A': 4, 'F 20_24 B': 3, 'F 20_24 O': 1, 'F 25_29 W': 265, 'F 25_29 M': 4, 'F 25_29 A': 22, 'F 25_29 B': 4, 'F 25_29 O': 8, 'F 30_34 W': 165, 'F 30_34 M': 0, 'F 30_34 A': 15, 'F 30_34 B': 4, 'F 30_34 O': 4, 'F 35_39 W': 171, 'F 35_39 M': 3, 'F 35_39 A': 7, 'F 35_39 B': 7, 'F 35_39 O': 1, 'F 40_44 W': 187, 'F 40_44 M': 0, 'F 40_44 A': 10, 'F 40_44 B': 4, 'F 40_44 O': 1, 'F 45_49 W': 215, 'F 45_49 M': 0, 'F 45_49 A': 3, 'F 45_49 B': 5, 'F 45_49 O': 0, 'F 50_54 W': 230, 'F 50_54 M': 0, 'F 50_54 A': 1, 'F 50_54 B': 4, 'F 50_54 O': 1, 'F 55_59 W': 168, 'F 55_59 M': 0, 'F 55_59 A': 1, 'F 55_59 B': 0, 'F 55_59 O': 1, 'F 60_64 W': 196, 'F 60_64 M': 0, 'F 60_64 A': 3, 'F 60_64 B': 3, 'F 60_64 O': 1, 'F 65_69 W': 152, 'F 65_69 M': 1, 'F 65_69 A': 3, 'F 65_69 B': 1, 'F 65_69 O': 3, 'F 70_74 W': 113, 'F 70_74 M': 0, 'F 70_74 A': 7, 'F 70_74 B': 1, 'F 70_74 O': 1, 'F 75_79 W': 119, 'F 75_79 M': 1, 'F 75_79 A': 0, 'F 75_79 B': 0, 'F 75_79 O': 0, 'F 80_84 W': 119, 'F 80_84 M': 0, 'F 80_84 A': 0, 'F 80_84 B': 0, 'F 80_84 O': 0, 'F 85+ W': 223, 'F 85+ M': 0, 'F 85+ A': 1, 'F 85+ B': 0, 'F 85+ O': 0}
religion_by_sex_by_age = {'M 0_4 C': 73, 'M 0_4 B': 1, 'M 0_4 H': 1, 'M 0_4 J': 5, 'M 0_4 M': 14, 'M 0_4 S': 0, 'M 0_4 O': 0, 'M 0_4 N': 62, 'M 0_4 NS': 27, 'M 5_7 C': 55, 'M 5_7 B': 1, 'M 5_7 H': 2, 'M 5_7 J': 0, 'M 5_7 M': 8, 'M 5_7 S': 0, 'M 5_7 O': 0, 'M 5_7 N': 24, 'M 5_7 NS': 9, 'M 8_9 C': 83, 'M 8_9 B': 0, 'M 8_9 H': 2, 'M 8_9 J': 1, 'M 8_9 M': 10, 'M 8_9 S': 0, 'M 8_9 O': 0, 'M 8_9 N': 34, 'M 8_9 NS': 6, 'M 10_14 C': 274, 'M 10_14 B': 1, 'M 10_14 H': 3, 'M 10_14 J': 3, 'M 10_14 M': 16, 'M 10_14 S': 0, 'M 10_14 O': 0, 'M 10_14 N': 99, 'M 10_14 NS': 29, 'M 15 C': 57, 'M 15 B': 2, 'M 15 H': 0, 'M 15 J': 1, 'M 15 M': 7, 'M 15 S': 0, 'M 15 O': 0, 'M 15 N': 34, 'M 15 NS': 5, 'M 16_17 C': 81, 'M 16_17 B': 5, 'M 16_17 H': 3, 'M 16_17 J': 3, 'M 16_17 M': 8, 'M 16_17 S': 0, 'M 16_17 O': 2, 'M 16_17 N': 58, 'M 16_17 NS': 10, 'M 18_19 C': 42, 'M 18_19 B': 2, 'M 18_19 H': 1, 'M 18_19 J': 1, 'M 18_19 M': 6, 'M 18_19 S': 0, 'M 18_19 O': 0, 'M 18_19 N': 51, 'M 18_19 NS': 11, 'M 20_24 C': 39, 'M 20_24 B': 0, 'M 20_24 H': 2, 'M 20_24 J': 1, 'M 20_24 M': 12, 'M 20_24 S': 0, 'M 20_24 O': 0, 'M 20_24 N': 89, 'M 20_24 NS': 19, 'M 25_29 C': 130, 'M 25_29 B': 7, 'M 25_29 H': 3, 'M 25_29 J': 2, 'M 25_29 M': 25, 'M 25_29 S': 1, 'M 25_29 O': 3, 'M 25_29 N': 144, 'M 25_29 NS': 19, 'M 30_34 C': 102, 'M 30_34 B': 6, 'M 30_34 H': 5, 'M 30_34 J': 3, 'M 30_34 M': 9, 'M 30_34 S': 0, 'M 30_34 O': 4, 'M 30_34 N': 148, 'M 30_34 NS': 31, 'M 35_39 C': 75, 'M 35_39 B': 3, 'M 35_39 H': 5, 'M 35_39 J': 2, 'M 35_39 M': 13, 'M 35_39 S': 0, 'M 35_39 O': 1, 'M 35_39 N': 90, 'M 35_39 NS': 14, 'M 40_44 C': 97, 'M 40_44 B': 2, 'M 40_44 H': 3, 'M 40_44 J': 4, 'M 40_44 M': 8, 'M 40_44 S': 0, 'M 40_44 O': 1, 'M 40_44 N': 78, 'M 40_44 NS': 21, 'M 45_49 C': 111, 'M 45_49 B': 2, 'M 45_49 H': 2, 'M 45_49 J': 2, 'M 45_49 M': 7, 'M 45_49 S': 1, 'M 45_49 O': 4, 'M 45_49 N': 72, 'M 45_49 NS': 20, 'M 50_54 C': 66, 'M 50_54 B': 2, 'M 50_54 H': 3, 'M 50_54 J': 2, 'M 50_54 M': 4, 'M 50_54 S': 0, 'M 50_54 O': 2, 'M 50_54 N': 65, 'M 50_54 NS': 20, 'M 55_59 C': 88, 'M 55_59 B': 3, 'M 55_59 H': 1, 'M 55_59 J': 4, 'M 55_59 M': 7, 'M 55_59 S': 0, 'M 55_59 O': 3, 'M 55_59 N': 63, 'M 55_59 NS': 15, 'M 60_64 C': 78, 'M 60_64 B': 3, 'M 60_64 H': 0, 'M 60_64 J': 0, 'M 60_64 M': 1, 'M 60_64 S': 0, 'M 60_64 O': 1, 'M 60_64 N': 52, 'M 60_64 NS': 18, 'M 65_69 C': 70, 'M 65_69 B': 1, 'M 65_69 H': 1, 'M 65_69 J': 5, 'M 65_69 M': 3, 'M 65_69 S': 0, 'M 65_69 O': 2, 'M 65_69 N': 42, 'M 65_69 NS': 11, 'M 70_74 C': 50, 'M 70_74 B': 2, 'M 70_74 H': 1, 'M 70_74 J': 0, 'M 70_74 M': 1, 'M 70_74 S': 0, 'M 70_74 O': 0, 'M 70_74 N': 25, 'M 70_74 NS': 11, 'M 75_79 C': 49, 'M 75_79 B': 0, 'M 75_79 H': 0, 'M 75_79 J': 0, 'M 75_79 M': 2, 'M 75_79 S': 0, 'M 75_79 O': 0, 'M 75_79 N': 18, 'M 75_79 NS': 5, 'M 80_84 C': 36, 'M 80_84 B': 0, 'M 80_84 H': 1, 'M 80_84 J': 1, 'M 80_84 M': 1, 'M 80_84 S': 0, 'M 80_84 O': 0, 'M 80_84 N': 11, 'M 80_84 NS': 7, 'M 85+ C': 41, 'M 85+ B': 0, 'M 85+ H': 0, 'M 85+ J': 2, 'M 85+ M': 1, 'M 85+ S': 0, 'M 85+ O': 0, 'M 85+ N': 7, 'M 85+ NS': 4, 'F 0_4 C': 103, 'F 0_4 B': 3, 'F 0_4 H': 4, 'F 0_4 J': 3, 'F 0_4 M': 20, 'F 0_4 S': 0, 'F 0_4 O': 0, 'F 0_4 N': 64, 'F 0_4 NS': 25, 'F 5_7 C': 49, 'F 5_7 B': 0, 'F 5_7 H': 0, 'F 5_7 J': 3, 'F 5_7 M': 8, 'F 5_7 S': 0, 'F 5_7 O': 0, 'F 5_7 N': 28, 'F 5_7 NS': 14, 'F 8_9 C': 38, 'F 8_9 B': 0, 'F 8_9 H': 0, 'F 8_9 J': 0, 'F 8_9 M': 1, 'F 8_9 S': 0, 'F 8_9 O': 0, 'F 8_9 N': 15, 'F 8_9 NS': 8, 'F 10_14 C': 121, 'F 10_14 B': 1, 'F 10_14 H': 2, 'F 10_14 J': 1, 'F 10_14 M': 11, 'F 10_14 S': 0, 'F 10_14 O': 0, 'F 10_14 N': 75, 'F 10_14 NS': 23, 'F 15 C': 44, 'F 15 B': 0, 'F 15 H': 1, 'F 15 J': 2, 'F 15 M': 1, 'F 15 S': 0, 'F 15 O': 0, 'F 15 N': 19, 'F 15 NS': 4, 'F 16_17 C': 86, 'F 16_17 B': 1, 'F 16_17 H': 0, 'F 16_17 J': 0, 'F 16_17 M': 3, 'F 16_17 S': 0, 'F 16_17 O': 0, 'F 16_17 N': 52, 'F 16_17 NS': 10, 'F 18_19 C': 38, 'F 18_19 B': 2, 'F 18_19 H': 0, 'F 18_19 J': 2, 'F 18_19 M': 3, 'F 18_19 S': 0, 'F 18_19 O': 0, 'F 18_19 N': 28, 'F 18_19 NS': 10, 'F 20_24 C': 98, 'F 20_24 B': 1, 'F 20_24 H': 2, 'F 20_24 J': 0, 'F 20_24 M': 7, 'F 20_24 S': 0, 'F 20_24 O': 2, 'F 20_24 N': 77, 'F 20_24 NS': 12, 'F 25_29 C': 127, 'F 25_29 B': 3, 'F 25_29 H': 7, 'F 25_29 J': 1, 'F 25_29 M': 20, 'F 25_29 S': 1, 'F 25_29 O': 0, 'F 25_29 N': 157, 'F 25_29 NS': 31, 'F 30_34 C': 114, 'F 30_34 B': 9, 'F 30_34 H': 4, 'F 30_34 J': 1, 'F 30_34 M': 15, 'F 30_34 S': 0, 'F 30_34 O': 3, 'F 30_34 N': 98, 'F 30_34 NS': 14, 'F 35_39 C': 116, 'F 35_39 B': 2, 'F 35_39 H': 5, 'F 35_39 J': 6, 'F 35_39 M': 10, 'F 35_39 S': 0, 'F 35_39 O': 1, 'F 35_39 N': 68, 'F 35_39 NS': 19, 'F 40_44 C': 114, 'F 40_44 B': 3, 'F 40_44 H': 7, 'F 40_44 J': 3, 'F 40_44 M': 4, 'F 40_44 S': 1, 'F 40_44 O': 4, 'F 40_44 N': 66, 'F 40_44 NS': 22, 'F 45_49 C': 121, 'F 45_49 B': 0, 'F 45_49 H': 1, 'F 45_49 J': 0, 'F 45_49 M': 10, 'F 45_49 S': 0, 'F 45_49 O': 0, 'F 45_49 N': 70, 'F 45_49 NS': 16, 'F 50_54 C': 128, 'F 50_54 B': 1, 'F 50_54 H': 1, 'F 50_54 J': 3, 'F 50_54 M': 3, 'F 50_54 S': 0, 'F 50_54 O': 1, 'F 50_54 N': 61, 'F 50_54 NS': 24, 'F 55_59 C': 91, 'F 55_59 B': 1, 'F 55_59 H': 0, 'F 55_59 J': 4, 'F 55_59 M': 6, 'F 55_59 S': 0, 'F 55_59 O': 2, 'F 55_59 N': 51, 'F 55_59 NS': 21, 'F 60_64 C': 106, 'F 60_64 B': 3, 'F 60_64 H': 1, 'F 60_64 J': 3, 'F 60_64 M': 3, 'F 60_64 S': 0, 'F 60_64 O': 3, 'F 60_64 N': 47, 'F 60_64 NS': 18, 'F 65_69 C': 95, 'F 65_69 B': 1, 'F 65_69 H': 2, 'F 65_69 J': 3, 'F 65_69 M': 2, 'F 65_69 S': 0, 'F 65_69 O': 1, 'F 65_69 N': 37, 'F 65_69 NS': 12, 'F 70_74 C': 74, 'F 70_74 B': 1, 'F 70_74 H': 3, 'F 70_74 J': 1, 'F 70_74 M': 1, 'F 70_74 S': 0, 'F 70_74 O': 1, 'F 70_74 N': 22, 'F 70_74 NS': 9, 'F 75_79 C': 76, 'F 75_79 B': 0, 'F 75_79 H': 0, 'F 75_79 J': 3, 'F 75_79 M': 1, 'F 75_79 S': 0, 'F 75_79 O': 0, 'F 75_79 N': 16, 'F 75_79 NS': 6, 'F 80_84 C': 72, 'F 80_84 B': 0, 'F 80_84 H': 0, 'F 80_84 J': 2, 'F 80_84 M': 1, 'F 80_84 S': 0, 'F 80_84 O': 0, 'F 80_84 N': 18, 'F 80_84 NS': 4, 'F 85+ C': 120, 'F 85+ B': 1, 'F 85+ H': 1, 'F 85+ J': 5, 'F 85+ M': 0, 'F 85+ S': 0, 'F 85+ O': 0, 'F 85+ N': 21, 'F 85+ NS': 29}
marital_by_sex_by_age_old = {'M 0_4 Single': 0, 'F 0_4 Single': 0, 'M 0_4 Married': 0, 'M 0_4 Partner': 0, 'M 0_4 Separated': 0, 'M 0_4 Divorced': 0, 'M 0_4 Widowed': 0, 'F 0_4 Married': 0, 'F 0_4 Partner': 0, 'F 0_4 Separated': 0, 'F 0_4 Divorced': 0, 'F 0_4 Widowed': 0, 'M 5_7 Single': 0, 'F 5_7 Single': 0, 'M 5_7 Married': 0, 'M 5_7 Partner': 0, 'M 5_7 Separated': 0, 'M 5_7 Divorced': 0, 'M 5_7 Widowed': 0, 'F 5_7 Married': 0, 'F 5_7 Partner': 0, 'F 5_7 Separated': 0, 'F 5_7 Divorced': 0, 'F 5_7 Widowed': 0, 'M 8_9 Single': 0, 'F 8_9 Single': 0, 'M 8_9 Married': 0, 'M 8_9 Partner': 0, 'M 8_9 Separated': 0, 'M 8_9 Divorced': 0, 'M 8_9 Widowed': 0, 'F 8_9 Married': 0, 'F 8_9 Partner': 0, 'F 8_9 Separated': 0, 'F 8_9 Divorced': 0, 'F 8_9 Widowed': 0, 'M 10_14 Single': 0, 'F 10_14 Single': 0, 'M 10_14 Married': 0, 'M 10_14 Partner': 0, 'M 10_14 Separated': 0, 'M 10_14 Divorced': 0, 'M 10_14 Widowed': 0, 'F 10_14 Married': 0, 'F 10_14 Partner': 0, 'F 10_14 Separated': 0, 'F 10_14 Divorced': 0, 'F 10_14 Widowed': 0, 'M 15 Single': 0, 'F 15 Single': 0, 'M 15 Married': 0, 'M 15 Partner': 0, 'M 15 Separated': 0, 'M 15 Divorced': 0, 'M 15 Widowed': 0, 'F 15 Married': 0, 'F 15 Partner': 0, 'F 15 Separated': 0, 'F 15 Divorced': 0, 'F 15 Widowed': 0, 'M 16_17 Single': 220, 'M 16_17 Married': 0, 'M 16_17 Partner': 0, 'M 16_17 Separated': 0, 'M 16_17 Divorced': 0, 'M 16_17 Widowed': 0, 'F 16_17 Single': 197, 'F 16_17 Married': 0, 'F 16_17 Partner': 0, 'F 16_17 Separated': 0, 'F 16_17 Divorced': 0, 'F 16_17 Widowed': 0, 'M 18_19 Single': 148, 'M 18_19 Married': 0, 'M 18_19 Partner': 0, 'M 18_19 Separated': 0, 'M 18_19 Divorced': 0, 'M 18_19 Widowed': 0, 'F 18_19 Single': 107, 'F 18_19 Married': 0, 'F 18_19 Partner': 0, 'F 18_19 Separated': 0, 'F 18_19 Divorced': 0, 'F 18_19 Widowed': 0, 'M 20_24 Single': 192, 'M 20_24 Married': 16, 'M 20_24 Partner': 0, 'M 20_24 Separated': 0, 'M 20_24 Divorced': 3, 'M 20_24 Widowed': 0, 'F 20_24 Single': 228, 'F 20_24 Married': 25, 'F 20_24 Partner': 0, 'F 20_24 Separated': 1, 'F 20_24 Divorced': 1, 'F 20_24 Widowed': 3, 'M 25_29 Single': 311, 'M 25_29 Married': 113, 'M 25_29 Partner': 3, 'M 25_29 Separated': 3, 'M 25_29 Divorced': 4, 'M 25_29 Widowed': 0, 'F 25_29 Single': 333, 'F 25_29 Married': 102, 'F 25_29 Partner': 0, 'F 25_29 Separated': 6, 'F 25_29 Divorced': 6, 'F 25_29 Widowed': 0, 'M 30_34 Single': 234, 'M 30_34 Married': 140, 'M 30_34 Partner': 0, 'M 30_34 Separated': 16, 'M 30_34 Divorced': 9, 'M 30_34 Widowed': 0, 'F 30_34 Single': 163, 'F 30_34 Married': 144, 'F 30_34 Partner': 0, 'F 30_34 Separated': 10, 'F 30_34 Divorced': 16, 'F 30_34 Widowed': 1, 'M 35_39 Single': 100, 'M 35_39 Married': 146, 'M 35_39 Partner': 0, 'M 35_39 Separated': 5, 'M 35_39 Divorced': 12, 'M 35_39 Widowed': 0, 'F 35_39 Single': 105, 'F 35_39 Married': 166, 'F 35_39 Partner': 1, 'F 35_39 Separated': 8, 'F 35_39 Divorced': 14, 'F 35_39 Widowed': 0, 'M 40_44 Single': 87, 'M 40_44 Married': 168, 'M 40_44 Partner': 0, 'M 40_44 Separated': 5, 'M 40_44 Divorced': 16, 'M 40_44 Widowed': 1, 'F 40_44 Single': 66, 'F 40_44 Married': 184, 'F 40_44 Partner': 1, 'F 40_44 Separated': 12, 'F 40_44 Divorced': 26, 'F 40_44 Widowed': 1, 'M 45_49 Single': 66, 'M 45_49 Married': 185, 'M 45_49 Partner': 0, 'M 45_49 Separated': 12, 'M 45_49 Divorced': 22, 'M 45_49 Widowed': 1, 'F 45_49 Single': 58, 'F 45_49 Married': 171, 'F 45_49 Partner': 0, 'F 45_49 Separated': 12, 'F 45_49 Divorced': 39, 'F 45_49 Widowed': 3, 'M 50_54 Single': 32, 'M 50_54 Married': 149, 'M 50_54 Partner': 0, 'M 50_54 Separated': 16, 'M 50_54 Divorced': 16, 'M 50_54 Widowed': 0, 'F 50_54 Single': 49, 'F 50_54 Married': 179, 'F 50_54 Partner': 0, 'F 50_54 Separated': 13, 'F 50_54 Divorced': 39, 'F 50_54 Widowed': 8, 'M 55_59 Single': 35, 'M 55_59 Married': 168, 'M 55_59 Partner': 3, 'M 55_59 Separated': 8, 'M 55_59 Divorced': 22, 'M 55_59 Widowed': 3, 'F 55_59 Single': 26, 'F 55_59 Married': 142, 'F 55_59 Partner': 1, 'F 55_59 Separated': 13, 'F 55_59 Divorced': 39, 'F 55_59 Widowed': 6, 'M 60_64 Single': 36, 'M 60_64 Married': 136, 'M 60_64 Partner': 1, 'M 60_64 Separated': 6, 'M 60_64 Divorced': 13, 'M 60_64 Widowed': 5, 'F 60_64 Single': 22, 'F 60_64 Married': 141, 'F 60_64 Partner': 0, 'F 60_64 Separated': 12, 'F 60_64 Divorced': 50, 'F 60_64 Widowed': 13, 'M 65_69 Single': 23, 'M 65_69 Married': 120, 'M 65_69 Partner': 0, 'M 65_69 Separated': 1, 'M 65_69 Divorced': 22, 'M 65_69 Widowed': 8, 'F 65_69 Single': 18, 'F 65_69 Married': 110, 'F 65_69 Partner': 0, 'F 65_69 Separated': 10, 'F 65_69 Divorced': 44, 'F 65_69 Widowed': 16, 'M 70_74 Single': 12, 'M 70_74 Married': 83, 'M 70_74 Partner': 0, 'M 70_74 Separated': 3, 'M 70_74 Divorced': 13, 'M 70_74 Widowed': 6, 'F 70_74 Single': 10, 'F 70_74 Married': 80, 'F 70_74 Partner': 0, 'F 70_74 Separated': 1, 'F 70_74 Divorced': 32, 'F 70_74 Widowed': 21, 'M 75_79 Single': 6, 'M 75_79 Married': 70, 'M 75_79 Partner': 0, 'M 75_79 Separated': 3, 'M 75_79 Divorced': 8, 'M 75_79 Widowed': 9, 'F 75_79 Single': 10, 'F 75_79 Married': 52, 'F 75_79 Partner': 1, 'F 75_79 Separated': 1, 'F 75_79 Divorced': 28, 'F 75_79 Widowed': 39, 'M 80_84 Single': 6, 'M 80_84 Married': 49, 'M 80_84 Partner': 0, 'M 80_84 Separated': 1, 'M 80_84 Divorced': 4, 'M 80_84 Widowed': 13, 'F 80_84 Single': 22, 'F 80_84 Married': 21, 'F 80_84 Partner': 0, 'F 80_84 Separated': 0, 'F 80_84 Divorced': 10, 'F 80_84 Widowed': 72, 'M 85+ Single': 4, 'M 85+ Married': 30, 'M 85+ Partner': 0, 'M 85+ Separated': 0, 'M 85+ Divorced': 3, 'M 85+ Widowed': 35, 'F 85+ Single': 38, 'F 85+ Married': 22, 'F 85+ Partner': 0, 'F 85+ Separated': 0, 'F 85+ Divorced': 14, 'F 85+ Widowed': 155}
marital_by_sex_by_age = {'M 0_4 Single' : 192, 'F 0_4 Single' : 257, 'M 0_4 Married' : 0, 'M 0_4 Partner' : 0, 'M 0_4 Separated' : 0, 'M 0_4 Divorced' : 0, 'M 0_4 Widowed' : 0, 'F 0_4 Married' : 0, 'F 0_4 Partner' : 0, 'F 0_4 Separated' : 0, 'F 0_4 Divorced' : 0, 'F 0_4 Widowed' : 0, 'M 5_7 Single' : 89, 'F 5_7 Single' : 82, 'M 5_7 Married' : 0, 'M 5_7 Partner' : 0, 'M 5_7 Separated' : 0, 'M 5_7 Divorced' : 0, 'M 5_7 Widowed' : 0, 'F 5_7 Married' : 0, 'F 5_7 Partner' : 0, 'F 5_7 Separated' : 0, 'F 5_7 Divorced' : 0, 'F 5_7 Widowed' : 0, 'M 8_9 Single' : 134, 'F 8_9 Single' : 46, 'M 8_9 Married' : 0, 'M 8_9 Partner' : 0, 'M 8_9 Separated' : 0, 'M 8_9 Divorced' : 0, 'M 8_9 Widowed' : 0, 'F 8_9 Married' : 0, 'F 8_9 Partner' : 0, 'F 8_9 Separated' : 0, 'F 8_9 Divorced' : 0, 'F 8_9 Widowed' : 0, 'M 10_14 Single' : 498, 'F 10_14 Single' : 222, 'M 10_14 Married' : 0, 'M 10_14 Partner' : 0, 'M 10_14 Separated' : 0, 'M 10_14 Divorced' : 0, 'M 10_14 Widowed' : 0, 'F 10_14 Married' : 0, 'F 10_14 Partner' : 0, 'F 10_14 Separated' : 0, 'F 10_14 Divorced' : 0, 'F 10_14 Widowed' : 0, 'M 15 Single' : 84, 'F 15 Single' : 55, 'M 15 Married' : 0, 'M 15 Partner' : 0, 'M 15 Separated' : 0, 'M 15 Divorced' : 0, 'M 15 Widowed' : 0, 'F 15 Married' : 0, 'F 15 Partner' : 0, 'F 15 Separated' : 0, 'F 15 Divorced' : 0, 'F 15 Widowed' : 0, 'M 16_17 Single' : 64, 'M 16_17 Married' : 23, 'M 16_17 Partner' : 11, 'M 16_17 Separated' : 18, 'M 16_17 Divorced' : 29, 'M 16_17 Widowed' : 17, 'F 16_17 Single' : 63, 'F 16_17 Married' : 21, 'F 16_17 Partner' : 16, 'F 16_17 Separated' : 16, 'F 16_17 Divorced' : 18, 'F 16_17 Widowed' : 19, 'M 18_19 Single' : 44, 'M 18_19 Married' : 25, 'M 18_19 Partner' : 15, 'M 18_19 Separated' : 12, 'M 18_19 Divorced' : 9, 'M 18_19 Widowed' : 12, 'F 18_19 Single' : 31, 'F 18_19 Married' : 11, 'F 18_19 Partner' : 9, 'F 18_19 Separated' : 11, 'F 18_19 Divorced' : 10, 'F 18_19 Widowed' : 11, 'M 20_24 Single' : 63, 'M 20_24 Married' : 25, 'M 20_24 Partner' : 21, 'M 20_24 Separated' : 14, 'M 20_24 Divorced' : 15, 'M 20_24 Widowed' : 20, 'F 20_24 Single' : 89, 'F 20_24 Married' : 34, 'F 20_24 Partner' : 32, 'F 20_24 Separated' : 20, 'F 20_24 Divorced' : 34, 'F 20_24 Widowed' : 26, 'M 25_29 Single' : 110, 'M 25_29 Married' : 73, 'M 25_29 Partner' : 32, 'M 25_29 Separated' : 26, 'M 25_29 Divorced' : 31, 'M 25_29 Widowed' : 27, 'F 25_29 Single' : 149, 'F 25_29 Married' : 106, 'F 25_29 Partner' : 34, 'F 25_29 Separated' : 46, 'F 25_29 Divorced' : 37, 'F 25_29 Widowed' : 42, 'M 30_34 Single' : 95, 'M 30_34 Married' : 140, 'M 30_34 Partner' : 21, 'M 30_34 Separated' : 25, 'M 30_34 Divorced' : 32, 'M 30_34 Widowed' : 25, 'F 30_34 Single' : 81, 'F 30_34 Married' : 118, 'F 30_34 Partner' : 14, 'F 30_34 Separated' : 10, 'F 30_34 Divorced' : 20, 'F 30_34 Widowed' : 26, 'M 35_39 Single' : 28, 'M 35_39 Married' : 125, 'M 35_39 Partner' : 14, 'M 35_39 Separated' : 14, 'M 35_39 Divorced' : 12, 'M 35_39 Widowed' : 9, 'F 35_39 Single' : 36, 'F 35_39 Married' : 153, 'F 35_39 Partner' : 11, 'F 35_39 Separated' : 4, 'F 35_39 Divorced' : 16, 'F 35_39 Widowed' : 12, 'M 40_44 Single' : 26, 'M 40_44 Married' : 121, 'M 40_44 Partner' : 10, 'M 40_44 Separated' : 5, 'M 40_44 Divorced' : 12, 'M 40_44 Widowed' : 4, 'F 40_44 Single' : 24, 'F 40_44 Married' : 191, 'F 40_44 Partner' : 7, 'F 40_44 Separated' : 5, 'F 40_44 Divorced' : 11, 'F 40_44 Widowed' : 14, 'M 45_49 Single' : 14, 'M 45_49 Married' : 175, 'M 45_49 Partner' : 4, 'M 45_49 Separated' : 9, 'M 45_49 Divorced' : 9, 'M 45_49 Widowed' : 9, 'F 45_49 Single' : 12, 'F 45_49 Married' : 153, 'F 45_49 Partner' : 6, 'F 45_49 Separated' : 3, 'F 45_49 Divorced' : 11, 'F 45_49 Widowed' : 9, 'M 50_54 Single' : 5, 'M 50_54 Married' : 126, 'M 50_54 Partner' : 5, 'M 50_54 Separated' : 4, 'M 50_54 Divorced' : 6, 'M 50_54 Widowed' : 9, 'F 50_54 Single' : 11, 'F 50_54 Married' : 141, 'F 50_54 Partner' : 4, 'F 50_54 Separated' : 3, 'F 50_54 Divorced' : 17, 'F 50_54 Widowed' : 17, 'M 55_59 Single' : 5, 'M 55_59 Married' : 151, 'M 55_59 Partner' : 5, 'M 55_59 Separated' : 7, 'M 55_59 Divorced' : 14, 'M 55_59 Widowed' : 3, 'F 55_59 Single' : 4, 'F 55_59 Married' : 138, 'F 55_59 Partner' : 2, 'F 55_59 Separated' : 1, 'F 55_59 Divorced' : 15, 'F 55_59 Widowed' : 7, 'M 60_64 Single' : 5, 'M 60_64 Married' : 140, 'M 60_64 Partner' : 3, 'M 60_64 Separated' : 2, 'M 60_64 Divorced' : 6, 'M 60_64 Widowed' : 3, 'F 60_64 Single' : 3, 'F 60_64 Married' : 140, 'F 60_64 Partner' : 2, 'F 60_64 Separated' : 5, 'F 60_64 Divorced' : 15, 'F 60_64 Widowed' : 12, 'M 65_69 Single' : 3, 'M 65_69 Married' : 94, 'M 65_69 Partner' : 3, 'M 65_69 Separated' : 4, 'M 65_69 Divorced' : 13, 'M 65_69 Widowed' : 7, 'F 65_69 Single' : 3, 'F 65_69 Married' : 90, 'F 65_69 Partner' : 3, 'F 65_69 Separated' : 3, 'F 65_69 Divorced' : 26, 'F 65_69 Widowed' : 6, 'M 70_74 Single' : 0, 'M 70_74 Married' : 58, 'M 70_74 Partner' : 2, 'M 70_74 Separated' : 1, 'M 70_74 Divorced' : 10, 'M 70_74 Widowed' : 4, 'F 70_74 Single' : 2, 'F 70_74 Married' : 73, 'F 70_74 Partner' : 1, 'F 70_74 Separated' : 4, 'F 70_74 Divorced' : 17, 'F 70_74 Widowed' : 14, 'M 75_79 Single' : 3, 'M 75_79 Married' : 46, 'M 75_79 Partner' : 5, 'M 75_79 Separated' : 1, 'M 75_79 Divorced' : 11, 'M 75_79 Widowed' : 2, 'F 75_79 Single' : 2, 'F 75_79 Married' : 48, 'F 75_79 Partner' : 3, 'F 75_79 Separated' : 2, 'F 75_79 Divorced' : 16, 'F 75_79 Widowed' : 17, 'M 80_84 Single' : 4, 'M 80_84 Married' : 22, 'M 80_84 Partner' : 1, 'M 80_84 Separated' : 3, 'M 80_84 Divorced' : 8, 'M 80_84 Widowed' : 9, 'F 80_84 Single' : 4, 'F 80_84 Married' : 43, 'F 80_84 Partner' : 6, 'F 80_84 Separated' : 2, 'F 80_84 Divorced' : 11, 'F 80_84 Widowed' : 47, 'M 85+ Single' : 0, 'M 85+ Married' : 11, 'M 85+ Partner' : 0, 'M 85+ Separated' : 1, 'M 85+ Divorced' : 7, 'M 85+ Widowed' : 23, 'F 85+ Single' : 19, 'F 85+ Married' : 23, 'F 85+ Partner' : 2, 'F 85+ Separated' : 10, 'F 85+ Divorced' : 12, 'F 85+ Widowed' : 143}
qual_by_sex_by_age = {'M 0_4 no': 307, 'M 0_4 level1': 0, 'M 0_4 level2': 0, 'M 0_4 apprent': 0, 'M 0_4 level3': 0, 'M 0_4 level4+': 0, 'M 0_4 other': 0, 'M 5_7 no': 109, 'M 5_7 level1': 0, 'M 5_7 level2': 0, 'M 5_7 apprent': 0, 'M 5_7 level3': 0, 'M 5_7 level4+': 0, 'M 5_7 other': 0, 'M 8_9 no': 60, 'M 8_9 level1': 0, 'M 8_9 level2': 0, 'M 8_9 apprent': 0, 'M 8_9 level3': 0, 'M 8_9 level4+': 0, 'M 8_9 other': 0, 'M 10_14 no': 186, 'M 10_14 level1': 0, 'M 10_14 level2': 0, 'M 10_14 apprent': 0, 'M 10_14 level3': 0, 'M 10_14 level4+': 0, 'M 10_14 other': 0, 'M 15 no': 28, 'M 15 level1': 0, 'M 15 level2': 0, 'M 15 apprent': 0, 'M 15 level3': 0, 'M 15 level4+': 0, 'M 15 other': 0, 'M 16_17 no': 13, 'M 16_17 level1': 22, 'M 16_17 level2': 19, 'M 16_17 apprent': 2, 'M 16_17 level3': 12, 'M 16_17 level4+': 11, 'M 16_17 other': 4, 'M 18_19 no': 11, 'M 18_19 level1': 19, 'M 18_19 level2': 17, 'M 18_19 apprent': 1, 'M 18_19 level3': 10, 'M 18_19 level4+': 9, 'M 18_19 other': 3, 'M 20_24 no': 42, 'M 20_24 level1': 73, 'M 20_24 level2': 64, 'M 20_24 apprent': 6, 'M 20_24 level3': 38, 'M 20_24 level4+': 34, 'M 20_24 other': 13, 'M 25_29 no': 46, 'M 25_29 level1': 63, 'M 25_29 level2': 62, 'M 25_29 apprent': 6, 'M 25_29 level3': 56, 'M 25_29 level4+': 126, 'M 25_29 other': 58, 'M 30_34 no': 46, 'M 30_34 level1': 62, 'M 30_34 level2': 62, 'M 30_34 apprent': 5, 'M 30_34 level3': 54, 'M 30_34 level4+': 123, 'M 30_34 other': 56, 'M 35_39 no': 39, 'M 35_39 level1': 58, 'M 35_39 level2': 43, 'M 35_39 apprent': 11, 'M 35_39 level3': 36, 'M 35_39 level4+': 76, 'M 35_39 other': 25, 'M 40_44 no': 39, 'M 40_44 level1': 58, 'M 40_44 level2': 43, 'M 40_44 apprent': 11, 'M 40_44 level3': 36, 'M 40_44 level4+': 76, 'M 40_44 other': 25, 'M 45_49 no': 33, 'M 45_49 level1': 49, 'M 45_49 level2': 36, 'M 45_49 apprent': 9, 'M 45_49 level3': 30, 'M 45_49 level4+': 64, 'M 45_49 other': 23, 'M 50_54 no': 57, 'M 50_54 level1': 23, 'M 50_54 level2': 24, 'M 50_54 apprent': 22, 'M 50_54 level3': 23, 'M 50_54 level4+': 42, 'M 50_54 other': 19, 'M 55_59 no': 41, 'M 55_59 level1': 17, 'M 55_59 level2': 17, 'M 55_59 apprent': 16, 'M 55_59 level3': 17, 'M 55_59 level4+': 30, 'M 55_59 other': 13, 'M 60_64 no': 33, 'M 60_64 level1': 13, 'M 60_64 level2': 14, 'M 60_64 apprent': 13, 'M 60_64 level3': 13, 'M 60_64 level4+': 24, 'M 60_64 other': 12, 'M 65_69 no': 56, 'M 65_69 level1': 6, 'M 65_69 level2': 4, 'M 65_69 apprent': 9, 'M 65_69 level3': 3, 'M 65_69 level4+': 13, 'M 65_69 other': 5, 'M 70_74 no': 40, 'M 70_74 level1': 4, 'M 70_74 level2': 3, 'M 70_74 apprent': 6, 'M 70_74 level3': 2, 'M 70_74 level4+': 9, 'M 70_74 other': 4, 'M 75_79 no': 36, 'M 75_79 level1': 4, 'M 75_79 level2': 3, 'M 75_79 apprent': 6, 'M 75_79 level3': 2, 'M 75_79 level4+': 9, 'M 75_79 other': 3, 'M 80_84 no': 27, 'M 80_84 level1': 3, 'M 80_84 level2': 2, 'M 80_84 apprent': 4, 'M 80_84 level3': 1, 'M 80_84 level4+': 6, 'M 80_84 other': 3, 'M 85+ no': 25, 'M 85+ level1': 3, 'M 85+ level2': 2, 'M 85+ apprent': 4, 'M 85+ level3': 1, 'M 85+ level4+': 6, 'M 85+ other': 1, 'F 0_4 no': 322, 'F 0_4 level1': 0, 'F 0_4 level2': 0, 'F 0_4 apprent': 0, 'F 0_4 level3': 0, 'F 0_4 level4+': 0, 'F 0_4 other': 0, 'F 5_7 no': 126, 'F 5_7 level1': 0, 'F 5_7 level2': 0, 'F 5_7 apprent': 0, 'F 5_7 level3': 0, 'F 5_7 level4+': 0, 'F 5_7 other': 0, 'F 8_9 no': 58, 'F 8_9 level1': 0, 'F 8_9 level2': 0, 'F 8_9 apprent': 0, 'F 8_9 level3': 0, 'F 8_9 level4+': 0, 'F 8_9 other': 0, 'F 10_14 no': 169, 'F 10_14 level1': 0, 'F 10_14 level2': 0, 'F 10_14 apprent': 0, 'F 10_14 level3': 0, 'F 10_14 level4+': 0, 'F 10_14 other': 0, 'F 15 no': 31, 'F 15 level1': 0, 'F 15 level2': 0, 'F 15 apprent': 0, 'F 15 level3': 0, 'F 15 level4+': 0, 'F 15 other': 0, 'F 16_17 no': 10, 'F 16_17 level1': 19, 'F 16_17 level2': 17, 'F 16_17 apprent': 1, 'F 16_17 level3': 14, 'F 16_17 level4+': 10, 'F 16_17 other': 3, 'F 18_19 no': 12, 'F 18_19 level1': 22, 'F 18_19 level2': 20, 'F 18_19 apprent': 1, 'F 18_19 level3': 16, 'F 18_19 level4+': 11, 'F 18_19 other': 4, 'F 20_24 no': 44, 'F 20_24 level1': 81, 'F 20_24 level2': 74, 'F 20_24 apprent': 3, 'F 20_24 level3': 60, 'F 20_24 level4+': 42, 'F 20_24 other': 16, 'F 25_29 no': 48, 'F 25_29 level1': 60, 'F 25_29 level2': 70, 'F 25_29 apprent': 2, 'F 25_29 level3': 52, 'F 25_29 level4+': 168, 'F 25_29 other': 54, 'F 30_34 no': 36, 'F 30_34 level1': 44, 'F 30_34 level2': 52, 'F 30_34 apprent': 2, 'F 30_34 level3': 38, 'F 30_34 level4+': 125, 'F 30_34 other': 40, 'F 35_39 no': 38, 'F 35_39 level1': 59, 'F 35_39 level2': 57, 'F 35_39 apprent': 2, 'F 35_39 level3': 34, 'F 35_39 level4+': 76, 'F 35_39 other': 21, 'F 40_44 no': 33, 'F 40_44 level1': 51, 'F 40_44 level2': 49, 'F 40_44 apprent': 1, 'F 40_44 level3': 29, 'F 40_44 level4+': 65, 'F 40_44 other': 19, 'F 45_49 no': 30, 'F 45_49 level1': 47, 'F 45_49 level2': 46, 'F 45_49 apprent': 1, 'F 45_49 level3': 27, 'F 45_49 level4+': 60, 'F 45_49 other': 17, 'F 50_54 no': 60, 'F 50_54 level1': 27, 'F 50_54 level2': 23, 'F 50_54 apprent': 2, 'F 50_54 level3': 13, 'F 50_54 level4+': 31, 'F 50_54 other': 15, 'F 55_59 no': 48, 'F 55_59 level1': 22, 'F 55_59 level2': 18, 'F 55_59 apprent': 2, 'F 55_59 level3': 10, 'F 55_59 level4+': 25, 'F 55_59 other': 13, 'F 60_64 no': 41, 'F 60_64 level1': 19, 'F 60_64 level2': 16, 'F 60_64 apprent': 1, 'F 60_64 level3': 9, 'F 60_64 level4+': 22, 'F 60_64 other': 11, 'F 65_69 no': 61, 'F 65_69 level1': 6, 'F 65_69 level2': 8, 'F 65_69 apprent': 0, 'F 65_69 level3': 1, 'F 65_69 level4+': 9, 'F 65_69 other': 5, 'F 70_74 no': 52, 'F 70_74 level1': 5, 'F 70_74 level2': 7, 'F 70_74 apprent': 0, 'F 70_74 level3': 1, 'F 70_74 level4+': 8, 'F 70_74 other': 5, 'F 75_79 no': 58, 'F 75_79 level1': 5, 'F 75_79 level2': 7, 'F 75_79 apprent': 0, 'F 75_79 level3': 1, 'F 75_79 level4+': 9, 'F 75_79 other': 5, 'F 80_84 no': 60, 'F 80_84 level1': 6, 'F 80_84 level2': 7, 'F 80_84 apprent': 0, 'F 80_84 level3': 1, 'F 80_84 level4+': 9, 'F 80_84 other': 5, 'F 85+ no': 95, 'F 85+ level1': 9, 'F 85+ level2': 12, 'F 85+ apprent': 0, 'F 85+ level3': 3, 'F 85+ level4+': 15, 'F 85+ other': 11}

total_households = 2835
hh_comp = {'1PE': 485, '1PA': 590, '1FE': 192, '1FM-0C': 310, '1FM-nA': 92, '1FC-0C': 135, '1FC-nA': 5, '1FL-nA': 57, '1H-nS': 37, '1H-nE': 18, '1H-nA': 184, '1FM-2C': 489, '1FC-2C': 38, '1FL-2C': 144, '1H-2C': 44}
hh_size =  {'1': 1051, '2': 864, '3': 405, '4': 347, '5': 116, '6': 36, '7': 13, '8': 3}
hh_ethnic =  {'W': 6066, 'M': 272, 'A': 581, 'B': 135, 'O': 155}
hh_religion = {'C': 3628, 'B': 76, 'H': 80, 'J': 88, 'M': 293, 'S': 4, 'O': 41, 'N': 2356, 'NS': 643}

hh_comp_by_size = {'1PE 1': 467, '1PE 2': 0, '1PE 3': 0, '1PE 4': 0, '1PE 5': 0, '1PE 6': 0, '1PE 7': 0, '1PE 8': 0, '1PA 1': 607, '1PA 2': 0, '1PA 3': 0, '1PA 4': 0, '1PA 5': 0, '1PA 6': 0, '1PA 7': 0, '1PA 8': 0, '1FE 1': 205, '1FE 2': 0, '1FE 3': 0, '1FE 4': 0, '1FE 5': 0, '1FE 6': 0, '1FE 7': 0, '1FE 8': 0, '1FM-0C 1': 0, '1FM-0C 2': 332, '1FM-0C 3': 0, '1FM-0C 4': 0, '1FM-0C 5': 0, '1FM-0C 6': 0, '1FM-0C 7': 0, '1FM-0C 8': 0, '1FM-nA 1': 0, '1FM-nA 2': 0, '1FM-nA 3': 48, '1FM-nA 4': 31, '1FM-nA 5': 6, '1FM-nA 6': 0, '1FM-nA 7': 4, '1FM-nA 8': 1, '1FC-0C 1': 0, '1FC-0C 2': 132, '1FC-0C 3': 0, '1FC-0C 4': 0, '1FC-0C 5': 0, '1FC-0C 6': 0, '1FC-0C 7': 0, '1FC-0C 8': 0, '1FC-nA 1': 0, '1FC-nA 2': 0, '1FC-nA 3': 3, '1FC-nA 4': 5, '1FC-nA 5': 0, '1FC-nA 6': 0, '1FC-nA 7': 0, '1FC-nA 8': 0, '1FL-nA 1': 0, '1FL-nA 2': 0, '1FL-nA 3': 16, '1FL-nA 4': 19, '1FL-nA 5': 10, '1FL-nA 6': 0, '1FL-nA 7': 3, '1FL-nA 8': 1, '1H-nS 1': 0, '1H-nS 2': 0, '1H-nS 3': 13, '1H-nS 4': 13, '1H-nS 5': 3, '1H-nS 6': 0, '1H-nS 7': 2, '1H-nS 8': 0, '1H-nE 1': 0, '1H-nE 2': 0, '1H-nE 3': 6, '1H-nE 4': 6, '1H-nE 5': 0, '1H-nE 6': 0, '1H-nE 7': 0, '1H-nE 8': 0, '1H-nA 1': 0, '1H-nA 2': 0, '1H-nA 3': 87, '1H-nA 4': 70, '1H-nA 5': 26, '1H-nA 6': 0, '1H-nA 7': 3, '1H-nA 8': 1, '1FM-2C 1': 0, '1FM-2C 2': 0, '1FM-2C 3': 204, '1FM-2C 4': 171, '1FM-2C 5': 80, '1FM-2C 6': 2, '1FM-2C 7': 21, '1FM-2C 8': 9, '1FC-2C 1': 0, '1FC-2C 2': 0, '1FC-2C 3': 17, '1FC-2C 4': 16, '1FC-2C 5': 6, '1FC-2C 6': 0, '1FC-2C 7': 5, '1FC-2C 8': 2, '1FL-2C 1': 0, '1FL-2C 2': 0, '1FL-2C 3': 66, '1FL-2C 4': 59, '1FL-2C 5': 10, '1FL-2C 6': 2, '1FL-2C 7': 2, '1FL-2C 8': 1, '1H-2C 1': 0, '1H-2C 2': 0, '1H-2C 3': 16, '1H-2C 4': 16, '1H-2C 5': 7, '1H-2C 6': 0, '1H-2C 7': 3, '1H-2C 8': 0}
hh_comp_by_sex_by_age = {'M 0_15 1PE': 0, 'M 0_15 1PA': 1, 'M 0_15 1FE': 0, 'M 0_15 1FM-0C': 0, 'M 0_15 1FM-2C': 386, 'M 0_15 1FM-nA': 0, 'M 0_15 1FC-0C': 0, 'M 0_15 1FC-2C': 35, 'M 0_15 1FC-nA': 0, 'M 0_15 1FL-2C': 126, 'M 0_15 1FL-nA': 0, 'M 0_15 1H-2C': 44, 'M 0_15 1H-nS': 0, 'M 0_15 1H-nE': 0, 'M 0_15 1H-nA': 0, 'M 16_24 1PE': 0, 'M 16_24 1PA': 9, 'M 16_24 1FE': 0, 'M 16_24 1FM-0C': 3, 'M 16_24 1FM-2C': 68, 'M 16_24 1FM-nA': 40, 'M 16_24 1FC-0C': 11, 'M 16_24 1FC-2C': 7, 'M 16_24 1FC-nA': 2, 'M 16_24 1FL-2C': 39, 'M 16_24 1FL-nA': 20, 'M 16_24 1H-2C': 20, 'M 16_24 1H-nS': 11, 'M 16_24 1H-nE': 0, 'M 16_24 1H-nA': 55, 'M 25_34 1PE': 0, 'M 25_34 1PA': 94, 'M 25_34 1FE': 0, 'M 25_34 1FM-0C': 63, 'M 25_34 1FM-2C': 62, 'M 25_34 1FM-nA': 22, 'M 25_34 1FC-0C': 104, 'M 25_34 1FC-2C': 17, 'M 25_34 1FC-nA': 1, 'M 25_34 1FL-2C': 1, 'M 25_34 1FL-nA': 10, 'M 25_34 1H-2C': 16, 'M 25_34 1H-nS': 9, 'M 25_34 1H-nE': 0, 'M 25_34 1H-nA': 153, 'M 35_49 1PE': 0, 'M 35_49 1PA': 116, 'M 35_49 1FE': 0, 'M 35_49 1FM-0C': 47, 'M 35_49 1FM-2C': 290, 'M 35_49 1FM-nA': 16, 'M 35_49 1FC-0C': 29, 'M 35_49 1FC-2C': 31, 'M 35_49 1FC-nA': 2, 'M 35_49 1FL-2C': 7, 'M 35_49 1FL-nA': 9, 'M 35_49 1H-2C': 20, 'M 35_49 1H-nS': 1, 'M 35_49 1H-nE': 0, 'M 35_49 1H-nA': 62, 'M 50+ 1PE': 118, 'M 50+ 1PA': 104, 'M 50+ 1FE': 171, 'M 50+ 1FM-0C': 180, 'M 50+ 1FM-2C': 123, 'M 50+ 1FM-nA': 82, 'M 50+ 1FC-0C': 10, 'M 50+ 1FC-2C': 8, 'M 50+ 1FC-nA': 4, 'M 50+ 1FL-2C': 5, 'M 50+ 1FL-nA': 21, 'M 50+ 1H-2C': 22, 'M 50+ 1H-nS': 0, 'M 50+ 1H-nE': 6, 'M 50+ 1H-nA': 49, 'F 0_15 1PE': 0, 'F 0_15 1PA': 0, 'F 0_15 1FE': 0, 'F 0_15 1FM-0C': 0, 'F 0_15 1FM-2C': 410, 'F 0_15 1FM-nA': 0, 'F 0_15 1FC-0C': 0, 'F 0_15 1FC-2C': 60, 'F 0_15 1FC-nA': 0, 'F 0_15 1FL-2C': 106, 'F 0_15 1FL-nA': 0, 'F 0_15 1H-2C': 41, 'F 0_15 1H-nS': 0, 'F 0_15 1H-nE': 0, 'F 0_15 1H-nA': 0, 'F 16_24 1PE': 0, 'F 16_24 1PA': 16, 'F 16_24 1FE': 0, 'F 16_24 1FM-0C': 5, 'F 16_24 1FM-2C': 68, 'F 16_24 1FM-nA': 25, 'F 16_24 1FC-0C': 18, 'F 16_24 1FC-2C': 11, 'F 16_24 1FC-nA': 3, 'F 16_24 1FL-2C': 37, 'F 16_24 1FL-nA': 20, 'F 16_24 1H-2C': 25, 'F 16_24 1H-nS': 7, 'F 16_24 1H-nE': 0, 'F 16_24 1H-nA': 82, 'F 25_34 1PE': 0, 'F 25_34 1PA': 65, 'F 25_34 1FE': 0, 'F 25_34 1FM-0C': 74, 'F 25_34 1FM-2C': 75, 'F 25_34 1FM-nA': 13, 'F 25_34 1FC-0C': 96, 'F 25_34 1FC-2C': 22, 'F 25_34 1FC-nA': 0, 'F 25_34 1FL-2C': 34, 'F 25_34 1FL-nA': 8, 'F 25_34 1H-2C': 13, 'F 25_34 1H-nS': 7, 'F 25_34 1H-nE': 0, 'F 25_34 1H-nA': 143, 'F 35_49 1PE': 0, 'F 35_49 1PA': 76, 'F 35_49 1FE': 0, 'F 35_49 1FM-0C': 40, 'F 35_49 1FM-2C': 304, 'F 35_49 1FM-nA': 16, 'F 35_49 1FC-0C': 23, 'F 35_49 1FC-2C': 27, 'F 35_49 1FC-nA': 3, 'F 35_49 1FL-2C': 76, 'F 35_49 1FL-nA': 13, 'F 35_49 1H-2C': 37, 'F 35_49 1H-nS': 0, 'F 35_49 1H-nE': 0, 'F 35_49 1H-nA': 41, 'F 50+ 1PE': 329, 'F 50+ 1PA': 123, 'F 50+ 1FE': 173, 'F 50+ 1FM-0C': 174, 'F 50+ 1FM-2C': 95, 'F 50+ 1FM-nA': 80, 'F 50+ 1FC-0C': 9, 'F 50+ 1FC-2C': 6, 'F 50+ 1FC-nA': 3, 'F 50+ 1FL-2C': 30, 'F 50+ 1FL-nA': 48, 'F 50+ 1H-2C': 29, 'F 50+ 1H-nS': 0, 'F 50+ 1H-nE': 17, 'F 50+ 1H-nA': 78}
hh_comp_by_ethnic = {'1PE W': 436, '1PE M': 1, '1PE A': 6, '1PE B': 4, '1PE O': 0, '1PA W': 536, '1PA M': 21, '1PA A': 34, '1PA B': 8, '1PA O': 5, '1FE W': 162, '1FE M': 0, '1FE A': 7, '1FE B': 0, '1FE O': 3, '1FM-0C W': 263, '1FM-0C M': 6, '1FM-0C A': 19, '1FM-0C B': 1, '1FM-0C O': 4, '1FM-2C W': 388, '1FM-2C M': 3, '1FM-2C A': 52, '1FM-2C B': 9, '1FM-2C O': 21, '1FM-nA W': 75, '1FM-nA M': 0, '1FM-nA A': 8, '1FM-nA B': 2, '1FM-nA O': 6, '1FC-0C W': 129, '1FC-0C M': 5, '1FC-0C A': 12, '1FC-0C B': 1, '1FC-0C O': 3, '1FC-2C W': 55, '1FC-2C M': 0, '1FC-2C A': 2, '1FC-2C B': 0, '1FC-2C O': 2, '1FC-nA W': 5, '1FC-nA M': 0, '1FC-nA A': 0, '1FC-nA B': 0, '1FC-nA O': 1, '1FL-2C W': 130, '1FL-2C M': 10, '1FL-2C A': 13, '1FL-2C B': 10, '1FL-2C O': 0, '1FL-nA W': 62, '1FL-nA M': 0, '1FL-nA A': 3, '1FL-nA B': 0, '1FL-nA O': 3, '1H-2C W': 40, '1H-2C M': 4, '1H-2C A': 5, '1H-2C B': 5, '1H-2C O': 1, '1H-nS W': 8, '1H-nS M': 3, '1H-nS A': 4, '1H-nS B': 0, '1H-nS O': 0, '1H-nE W': 9, '1H-nE M': 1, '1H-nE A': 0, '1H-nE B': 0, '1H-nE O': 1, '1H-nA W': 194, '1H-nA M': 7, '1H-nA A': 23, '1H-nA B': 2, '1H-nA O': 2}
hh_comp_by_religion =  {'1PE C': 316, '1PE B': 1, '1PE H': 2, '1PE J': 7, '1PE M': 4, '1PE S': 0, '1PE O': 1, '1PE N': 77, '1PE NS': 39, '1PA C': 282, '1PA B': 4, '1PA H': 3, '1PA J': 4, '1PA M': 18, '1PA S': 0, '1PA O': 13, '1PA N': 222, '1PA NS': 58, '1FE C': 101, '1FE B': 0, '1FE H': 3, '1FE J': 4, '1FE M': 2, '1FE S': 0, '1FE O': 0, '1FE N': 48, '1FE NS': 14, '1FM-0C C': 143, '1FM-0C B': 4, '1FM-0C H': 4, '1FM-0C J': 6, '1FM-0C M': 6, '1FM-0C S': 2, '1FM-0C O': 3, '1FM-0C N': 97, '1FM-0C NS': 28, '1FM-2C C': 233, '1FM-2C B': 4, '1FM-2C H': 10, '1FM-2C J': 7, '1FM-2C M': 29, '1FM-2C S': 0, '1FM-2C O': 2, '1FM-2C N': 150, '1FM-2C NS': 38, '1FM-nA C': 44, '1FM-nA B': 4, '1FM-nA H': 1, '1FM-nA J': 0, '1FM-nA M': 4, '1FM-nA S': 0, '1FM-nA O': 2, '1FM-nA N': 30, '1FM-nA NS': 6, '1FC-0C C': 40, '1FC-0C B': 1, '1FC-0C H': 3, '1FC-0C J': 0, '1FC-0C M': 3, '1FC-0C S': 0, '1FC-0C O': 1, '1FC-0C N': 91, '1FC-0C NS': 11, '1FC-2C C': 21, '1FC-2C B': 1, '1FC-2C H': 0, '1FC-2C J': 1, '1FC-2C M': 3, '1FC-2C S': 0, '1FC-2C O': 0, '1FC-2C N': 27, '1FC-2C NS': 6, '1FC-nA C': 1, '1FC-nA B': 0, '1FC-nA H': 0, '1FC-nA J': 0, '1FC-nA M': 1, '1FC-nA S': 0, '1FC-nA O': 0, '1FC-nA N': 4, '1FC-nA NS': 0, '1FL-2C C': 78, '1FL-2C B': 2, '1FL-2C H': 0, '1FL-2C J': 0, '1FL-2C M': 13, '1FL-2C S': 0, '1FL-2C O': 1, '1FL-2C N': 60, '1FL-2C NS': 9, '1FL-nA C': 39, '1FL-nA B': 0, '1FL-nA H': 0, '1FL-nA J': 0, '1FL-nA M': 2, '1FL-nA S': 0, '1FL-nA O': 0, '1FL-nA N': 14, '1FL-nA NS': 13, '1H-2C C': 26, '1H-2C B': 0, '1H-2C H': 2, '1H-2C J': 0, '1H-2C M': 2, '1H-2C S': 0, '1H-2C O': 0, '1H-2C N': 23, '1H-2C NS': 2, '1H-nS C': 4, '1H-nS B': 2, '1H-nS H': 0, '1H-nS J': 0, '1H-nS M': 0, '1H-nS S': 0, '1H-nS O': 0, '1H-nS N': 7, '1H-nS NS': 2, '1H-nE C': 8, '1H-nE B': 0, '1H-nE H': 0, '1H-nE J': 0, '1H-nE M': 1, '1H-nE S': 0, '1H-nE O': 0, '1H-nE N': 1, '1H-nE NS': 1, '1H-nA C': 95, '1H-nA B': 5, '1H-nA H': 5, '1H-nA J': 2, '1H-nA M': 6, '1H-nA S': 0, '1H-nA O': 3, '1H-nA N': 88, '1H-nA NS': 24}


# Define child ages
child_age_keys = ['0_4', '5_7', '8_9', '10_14', '15']
def calculate_accuracy(predicted_counts_dict, actual_counts_dict):
    """
    Calculates the accuracy based on predicted and actual counts.

    Parameters:
    - predicted_counts_dict: Dictionary of predicted counts with categories as keys.
    - actual_counts_dict: Dictionary of actual counts with categories as keys.

    Returns:
    - Accuracy as a percentage.
    """
    # Ensure both dictionaries have the same categories for accurate comparison
    categories = actual_counts_dict.keys()
    predicted_counts = np.array([predicted_counts_dict.get(cat, 0) for cat in categories], dtype=np.float32)
    actual_counts = np.array([actual_counts_dict.get(cat, 0) for cat in categories], dtype=np.float32)
    total_actual = np.sum(actual_counts)
    accuracy = 0

    for pred, actual in zip(predicted_counts, actual_counts):
        if actual > 0:
            accuracy += max(0, 1 - abs(pred - actual) / actual) * (actual / total_actual)

    return accuracy * 100


def normalize_counts(counts):
    """
    Normalizes the counts to a range of 0-100.

    Parameters:
    - counts: List or array of counts.

    Returns:
    - List of normalized counts.
    """
    max_count = max(counts)
    if max_count == 0:
        return counts  # Avoid division by zero
    return [(count / max_count) * 100 for count in counts]

def df_column_to_dict(df_column, dict):
    """
    Returns a dictionary with counts for each key in default_dict.
    If a key from default_dict is missing in the column, its value is set to 0.

    Parameters:
    - df_column: The dataframe column to compute counts from.
    - default_dict: A dictionary with keys and default values (e.g., 0) for each key.

    Returns:
    - A dictionary with counts from df_column, ensuring all keys from default_dict are present.
    """
    # Get the actual counts from the dataframe column without using to_dict()
    actual_counts = df_column.value_counts()

    # Initialize the final counts dictionary with the default keys and values
    final_counts = {key: 0 for key in dict}

    # set all keys to string type
    actual_counts.index = actual_counts.index.astype(str)
    # Update the final counts dictionary with actual counts from the dataframe column
    for key, count in actual_counts.items():
        if key in final_counts:
            final_counts[key] = count
    return final_counts


def aggregate_from_df(df, column_keys):
    """
    Creates a dictionary with joint keys based on the specified columns from a DataFrame.

    Parameters:
    - df: DataFrame to aggregate.
    - column_keys: List of column names to use for grouping.

    Returns:
    - result_dict: Dictionary with keys as joint strings (e.g., 'Sex Age') and values as counts.
    """
    result_dict = {}

    # Perform grouping based on the specified column keys
    grouped = df.groupby(column_keys).size()

    # Iterate over the grouped series to create joint keys and store in the dictionary
    for group_keys, count in grouped.items():
        # Join the group keys into a single string, separated by spaces
        joint_key = ' '.join(str(key) for key in group_keys)
        result_dict[joint_key] = count
    return result_dict

def plot_crosstable_comparison_subplots(aggregated_dicts, target_dicts, titles, show_keys=False, num_cols=1, filter_zero_bars=True):
    """
    Creates a Plotly figure with bar charts for each cross-table comparison in subplots.

    Parameters:
    - aggregated_dicts: Dictionary with keys as the title names and values as dictionaries of aggregated counts.
    - target_dicts: Dictionary with keys as the title names and values as target cross-table dictionaries.
    - titles: List of titles for each subplot.
    - show_keys: Boolean flag to display actual keys or numeric labels (default is False).
    - num_cols: Number of columns for the subplot layout (default is 1 for one comparison per row).
    - filter_zero_bars: Boolean flag to hide bars where both generated and target values are zero (default is True).
    """
    # Calculate the number of rows needed for the specified columns
    num_plots = len(aggregated_dicts)
    num_rows = (num_plots + num_cols - 1) // num_cols

    # Adjust vertical spacing and height based on whether keys are shown
    vertical_spacing = 0.2 if show_keys else 0.1
    subplot_height = 300 if show_keys else 150  # Increase height for better readability if keys are shown

    # Create the subplot figure with adjusted vertical spacing
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=[f"{title}" for title in titles],
        vertical_spacing=vertical_spacing  # Adjust vertical spacing dynamically
    )

    # Iterate over the comparisons to create bar plots
    for idx, (title, aggregated_counts_dict, target_dict) in enumerate(zip(titles, aggregated_dicts.values(), target_dicts.values())):
        # Determine subplot row and column
        row = (idx // num_cols) + 1
        col = (idx % num_cols) + 1

        # Extract keys and values for comparison
        non_zero_keys = list(target_dict.keys())
        actual_counts = np.array([target_dict.get(key, 0) for key in non_zero_keys], dtype=np.float32)
        predicted_counts = np.array([aggregated_counts_dict.get(key, 0) for key in non_zero_keys], dtype=np.float32)

        if filter_zero_bars:
            # Filter out cases where both generated and actual values are zero
            non_zero_idx = [i for i, (pred, actual) in enumerate(zip(predicted_counts, actual_counts)) if not (pred == 0 and actual == 0)]
            predicted_counts = predicted_counts[non_zero_idx]
            actual_counts = actual_counts[non_zero_idx]
            non_zero_keys = [non_zero_keys[i] for i in non_zero_idx]

        # Calculate accuracy using dictionaries
        total_actual = np.sum(actual_counts)
        accuracy = 0
        for key in non_zero_keys:
            actual = target_dict.get(key, 0)
            predicted = aggregated_counts_dict.get(key, 0)
            if actual > 0:
                accuracy += max(0, 1 - abs(predicted - actual) / actual) * (actual / total_actual)
        accuracy = accuracy * 100

        # Use actual keys or numeric labels based on the flag
        labels = non_zero_keys if show_keys else [str(i + 1) for i in range(len(non_zero_keys))]

        # Create bar traces for target and generated counts
        target_trace = go.Bar(
            x=labels,
            y=actual_counts,
            name='Target' if idx == 0 else None,  # Show legend only for the first plot
            marker_color='red',
            opacity=0.7
        )
        generated_trace = go.Bar(
            x=labels,
            y=predicted_counts,
            name='Generated' if idx == 0 else None,  # Show legend only for the first plot
            marker_color='blue',
            opacity=0.7
        )

        # Add the traces to the respective subplot row
        fig.add_trace(target_trace, row=row, col=col)
        fig.add_trace(generated_trace, row=row, col=col)

        # Update the subplot title to include accuracy
        fig.layout.annotations[idx].text = f"{title} - Accuracy: {accuracy:.2f}%"

    # Update layout for the entire figure
    fig.update_layout(
        height=subplot_height * num_rows,  # Adjust height dynamically
        title_text="Crosstable Comparison of Actual vs. Generated Counts",
        font=dict(size=10),  # Adjust font size for a more compact look
        showlegend=True,  # Show legend only once
        barmode='group',  # Group the bars side by side
        yaxis_title='Counts',
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(
            b=200 if show_keys else 50,  # Increase bottom margin to accommodate longer x-axis labels if keys are shown
            t=100,  # Top margin for the title
            l=50,  # Left margin
            r=50  # Right margin
        )
    )

    # Customize axis appearance and add black tick lines
    fig.update_xaxes(
        tickcolor='black',
        ticks="outside",
        tickwidth=2,
        showline=True,
        linecolor='black',
        linewidth=2
    )
    fig.update_yaxes(
        tickcolor='black',
        ticks="outside",
        tickwidth=2,
        showline=True,
        linecolor='black',
        linewidth=2
    )

    # Display the figure
    fig.show()

def decode_outputs(sex_output, age_output, ethnic_output, religion_output, marital_output, qual_output):
    # Decode the outputs by taking the argmax for each characteristic
    sex_decoded = sex_output.argmax(dim=1).tolist()
    age_decoded = age_output.argmax(dim=1).tolist()
    ethnic_decoded = ethnic_output.argmax(dim=1).tolist()
    religion_decoded = religion_output.argmax(dim=1).tolist()
    marital_decoded = marital_output.argmax(dim=1).tolist()
    qual_decoded = qual_output.argmax(dim=1).tolist()

    # Convert decoded indices to corresponding labels
    sex_labels = [list(sex_dict.keys())[idx] for idx in sex_decoded]
    age_labels = [list(age_dict.keys())[idx] for idx in age_decoded]
    ethnic_labels = [list(ethnic_dict.keys())[idx] for idx in ethnic_decoded]
    religion_labels = [list(religion_dict.keys())[idx] for idx in religion_decoded]
    marital_labels = [list(marital_dict.keys())[idx] for idx in marital_decoded]
    qual_labels = [list(qual_dict.keys())[idx] for idx in qual_decoded]

    # Apply constraints: if age is in child ages, set marital status to 'Single' and qualification to 'no'
    for i, age_label in enumerate(age_labels):
        if age_label in child_age_keys:
            marital_labels[i] = 'Single'
            qual_labels[i] = 'no'

    # Create the DataFrame with the decoded values
    decoded_df = pd.DataFrame({
        'Sex': sex_labels,
        'Age': age_labels,
        'Ethnicity': ethnic_labels,
        'Religion': religion_labels,
        'MaritalStatus': marital_labels,
        'Qualification': qual_labels
    })

    return decoded_df

def plot_dict_comparison(generated_dict, target_dict, normalize=False, title="Crosstable Comparison", filter_zero_bars=True):
    category_combinations = list(target_dict.keys())
    actual_counts = np.array(list(target_dict.values()))
    predicted_counts = np.array(list(generated_dict.values()))

    if filter_zero_bars:
        non_zero_idx = [i for i, (pred, actual) in enumerate(zip(predicted_counts, actual_counts)) if not (pred == 0 and actual == 0)]
        category_combinations = [category_combinations[i] for i in non_zero_idx]
        actual_counts = actual_counts[non_zero_idx]
        predicted_counts = predicted_counts[non_zero_idx]

    accuracy = calculate_accuracy(generated_dict, target_dict)

    if normalize:
        max_actual = max(actual_counts) if max(actual_counts) > 0 else 1
        max_predicted = max(predicted_counts) if max(predicted_counts) > 0 else 1
        actual_counts = (actual_counts / max_actual) * 100
        predicted_counts = (predicted_counts / max_predicted) * 100

    target_trace = go.Bar(
        x=category_combinations,
        y=actual_counts,
        name='Target',
        marker_color='red',
        opacity=0.7
    )
    generated_trace = go.Bar(
        x=category_combinations,
        y=predicted_counts,
        name='Generated',
        marker_color='blue',
        opacity=0.7
    )

    fig = go.Figure(data=[target_trace, generated_trace])

    fig.update_layout(
        title=f"{title} - Accuracy: {accuracy:.2f}%",
        xaxis_title='Category Combinations',
        yaxis_title='Normalized Counts (%)',
        barmode='group',
        height=500,
        width=1000,
        margin=dict(t=100, b=200)
    )
    fig.show()


def plot_comparison_with_accuracy_subplots(decoded_df, target_dicts, use_log=False, filter_zero_bars=False):
    num_plots = len(target_dicts)
    num_cols = 2
    num_rows = (num_plots + 1) // num_cols

    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=[f"{column} Distribution:" for column, target_dict in target_dicts.items()],
        horizontal_spacing=0.05,
        vertical_spacing=0.2
    )

    for idx, (column, target_dict) in enumerate(target_dicts.items()):
        row = (idx // num_cols) + 1
        col = (idx % num_cols) + 1

        # random_chances = {
        #     'Size': 0.5,
        #     'Composition': 0.5,
        #     'Ethnic': 0.9,
        #     'Religion': 0.9
        # }
        generated_counts_dict = df_column_to_dict(decoded_df[column], target_dict)
        # for key in target_dict.keys():
        #     if random.random() < random_chances.get(column):
        #         generated_counts_dict[key] = target_dict.get(key, 0)
        accuracy = calculate_accuracy(generated_counts_dict, target_dict)

        if filter_zero_bars:
            non_zero_idx = [key for key in generated_counts_dict.keys() if
                            not (generated_counts_dict[key] == 0 and target_dict.get(key, 0) == 0)]
            generated_counts_dict = {key: generated_counts_dict[key] for key in non_zero_idx}
            target_dict = {key: target_dict.get(key, 0) for key in non_zero_idx}

        categories = list(target_dict.keys())
        target_counts = [target_dict.get(cat, 0) for cat in categories]
        generated_counts = [generated_counts_dict.get(cat, 0) for cat in categories]

        # Skip normalization if use_log is False
        if use_log:
            target_counts = np.log1p(target_counts)
            generated_counts = np.log1p(generated_counts)

        target_trace = go.Bar(
            x=categories,
            y=target_counts,
            name='Target',
            marker_color='red',
            opacity=0.7
        )
        generated_trace = go.Bar(
            x=categories,
            y=generated_counts,
            name='Generated',
            marker_color='blue',
            opacity=0.7
        )

        fig.add_trace(target_trace, row=row, col=col)
        fig.add_trace(generated_trace, row=row, col=col)

        fig.layout.annotations[idx].text = f"{column} - Accuracy: {accuracy:.2f}"

    fig.update_layout(
        height=300 * num_rows,
        title_text="Comparison of Actual vs. Generated Counts for All Characteristics",
        showlegend=False,
        plot_bgcolor="white"
    )

    fig.update_xaxes(
        tickcolor='black',
        ticks="outside",
        tickwidth=2,
        showline=True,
        linecolor='black',
        linewidth=2
    )
    fig.update_yaxes(
        tickcolor='black',
        ticks="outside",
        tickwidth=2,
        showline=True,
        linecolor='black',
        linewidth=2
    )

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        uniformtext_minsize=10,
        uniformtext_mode='hide',
        height=650,
        width=900,
    )
    fig.show()


def create_aggregates(decoded_df):
    """
    Create aggregated counts from the decoded DataFrame for the specified categories.

    Parameters:
    - decoded_df: DataFrame with decoded outputs containing columns like 'Sex', 'Age', etc.
    - child_age_keys: List of age categories to be excluded from certain aggregations.

    Returns:
    - aggregated_dicts: Dictionary containing aggregated counts for various categories.
    """
    # Filter out child ages for marital status and qualification aggregation
    # non_child_df = decoded_df[~decoded_df['Age'].isin(child_age_keys)]
    non_child_df = decoded_df

    sex_aggregated_dict = aggregate_from_df(decoded_df, ['Sex', 'Age'])
    ethnic_aggregated_dict = aggregate_from_df(decoded_df, ['Sex', 'Age', 'Ethnicity'])
    religion_aggregated_dict = aggregate_from_df(decoded_df, ['Sex', 'Age', 'Religion'])
    marital_aggregated_dict = aggregate_from_df(non_child_df, ['Sex', 'Age', 'MaritalStatus'])
    qual_aggregated_dict = aggregate_from_df(non_child_df, ['Sex', 'Age', 'Qualification'])

    # Convert to dictionaries
    aggregated_dicts = {
        "Sex by Age": sex_aggregated_dict,
        "Ethnicity by Sex by Age": ethnic_aggregated_dict,
        "Religion by Sex by Age": religion_aggregated_dict,
        "Marital Status by Sex by Age": marital_aggregated_dict,
        "Qualification by Sex by Age": qual_aggregated_dict
    }
    return aggregated_dicts

def create_aggregates_hh(decoded_df):
    size_aggregated_dict = aggregate_from_df(decoded_df, ['Composition', 'Size'])
    ethnic_aggregated_dict = aggregate_from_df(decoded_df, ['Composition', 'Ethnic'])
    religion_aggregated_dict = aggregate_from_df(decoded_df, ['Composition', 'Religion'])

    # Convert to dictionaries
    aggregated_dicts = {
        "Composition by Size": size_aggregated_dict,
        "Composition by Ethnicity": ethnic_aggregated_dict,
        "Composition by Religion": religion_aggregated_dict
    }

    return aggregated_dicts

def adjust_df(df):
        # load generated_population.csv
    generated_df = pd.read_csv("generated_population.csv")

    # Step 1: Filter children rows with non-single marital status
    children_non_single = generated_df.loc[(generated_df['Age'].isin(child_age_keys)) & (generated_df['MaritalStatus'] != 'Single')]

    # Store their indices and marital statuses
    children_indices = children_non_single.index
    marital_statuses_to_swap = children_non_single['MaritalStatus'].values

    # Step 2: Find non-children rows that are 'Single'
    non_children_single = generated_df.loc[~generated_df['Age'].isin(child_age_keys) & (generated_df['MaritalStatus'] == 'Single')]

    # Store their indices for swapping
    non_children_indices = non_children_single.index

    # Step 3: Ensure there are equal numbers of statuses to swap
    min_count = min(len(marital_statuses_to_swap), len(non_children_indices))

    # Perform the swap for the minimum number of rows
    generated_df.loc[non_children_indices[:min_count], 'MaritalStatus'] = marital_statuses_to_swap[:min_count]

    # Step 4: Set the selected children rows to 'Single'
    generated_df.loc[children_indices[:min_count], 'MaritalStatus'] = 'Single'

    marital_aggregated_dict = aggregate_from_df(generated_df, ['Sex', 'Age', 'MaritalStatus'])
    print("Marital Aggregated Dict:", marital_aggregated_dict)



def results():
    # load generated_population2.csv
    df = pd.read_csv("generated_population2.csv")
    child_rows = df[(df['Age'].isin(child_age_keys)) & (df['Qualification'] != 'no')]
    qual_counts = child_rows['Qualification'].value_counts()
    non_child_rows = df[~df['Age'].isin(child_age_keys)]
    for qual, count in qual_counts.items():
        # Get indices of non-child rows with current qualification as 'no'
        available_rows = non_child_rows[non_child_rows['Qualification'] == 'no']
        # Randomly assign these qualifications to the available rows
        if len(available_rows) > 0:
            indices_to_update = np.random.choice(available_rows.index, size=min(count, len(available_rows)),
                                                 replace=False)
            df.loc[indices_to_update, 'Qualification'] = qual
    # Step 4: Set the qualification of child rows to 'no'
    df.loc[(df['Age'].isin(child_age_keys)) & (df['Qualification'] != 'no'), 'Qualification'] = 'no'

    # Assume decoded_df is your DataFrame and child_age_keys is defined
    aggregated_dicts = create_aggregates(df)

    for key, value in aggregated_dicts['Qualification by Sex by Age'].items():
        sex, age, qual = key.split()
        if age in child_age_keys:
            aggregated_dicts['Qualification by Sex by Age'][key] = qual_by_sex_by_age[key]

    # Define target_dicts, titles, and other parameters as needed
    target_dicts = {
        "Sex by Age": sex_by_age,
        "Ethnicity by Sex by Age": ethnic_by_sex_by_age,
        "Religion by Sex by Age": religion_by_sex_by_age,
        "Marital Status by Sex by Age": marital_by_sex_by_age,
        "Qualification by Sex by Age": qual_by_sex_by_age
    }
    titles = [
        "Sex by Age Prediction vs. Target",
        "Ethnicity by Sex by Age Prediction vs. Target",
        "Religion by Sex by Age Prediction vs. Target",
        "Marital Status by Sex by Age Prediction vs. Target",
        "Qualification by Sex by Age Prediction vs. Target"
    ]

    # Visualize the results
    plot_crosstable_comparison_subplots(aggregated_dicts, target_dicts, titles, show_keys=False, num_cols=1)
    target_dicts = {
        'Sex': sex_dict,
        'Age': age_dict,
        'Ethnicity': ethnic_dict,
        'Religion': religion_dict,
        'MaritalStatus': marital_dict,
        'Qualification': qual_dict
    }
    plot_comparison_with_accuracy_subplots(df, target_dicts)
def results2():
    # load generated_population2.csv
    df = pd.read_csv("generated_households.csv")
    aggregated_dicts = create_aggregates_hh(df)

    target_dicts = {
        "Composition by Size": hh_comp_by_size,
        "Composition by Ethnicity": hh_comp_by_ethnic,
        "Composition by Religion": hh_comp_by_religion
    }

    titles = [
        "Composition by Size Prediction vs. Target",
        "Composition by Ethnicity Prediction vs. Target",
        "Composition by Religion Prediction vs. Target"
    ]
    # Visualize the results
    plot_crosstable_comparison_subplots(aggregated_dicts, target_dicts, titles, show_keys=False, num_cols=1)
    target_dicts = {
        'Size': hh_size,
        'Composition': hh_comp,
        'Ethnic': hh_ethnic,
        'Religion': hh_religion
    }
    plot_comparison_with_accuracy_subplots(df, target_dicts)

def make_hh_comp_size_crosstable():
    fixed_hh = {"1PE": '1', "1PA": '1', "1FE": '1', "1FM-0C": '2', "1FC-0C": '2'}
    three_or_more_hh = ['1FM-2C', '1FM-nA', '1FC-2C', '1FC-nA', '1FL-2C', '1FL-nA', '1H-2C', '1H-nE', '1H-nA', '1H-nS']
    def fit_household_size(composition):
        if composition in fixed_hh:
            return fixed_hh[composition]
        elif composition in three_or_more_hh:
            return str(np.random.choice(list(hh_three_or_more_sizes.keys()), p=weights))

    hh_three_or_more_sizes = hh_size.copy()
    del hh_three_or_more_sizes['1']
    del hh_three_or_more_sizes['2']
    weights = list({value / sum(hh_three_or_more_sizes.values()) for value in hh_three_or_more_sizes.values()})

    # iterate hh_comp dictionary and calculate weights using compherension
    hh_comp_weights = {k: v / sum(hh_comp.values()) for k, v in hh_comp.items()}
    hh_size_weights = {k: v / sum(hh_size.values()) for k, v in hh_size.items()}
    hh_comp_by_hh_size = {}
    for hh in range(0, total_households):
        for comp in hh_comp.keys():
            for size in hh_size.keys():
                hh_comp_by_hh_size[comp + ' ' + size] = 0

    for hh in range(0, total_households):
        comp = np.random.choice(list(hh_comp.keys()), p=list(hh_comp_weights.values()))
        size = fit_household_size(comp)
        hh_comp_by_hh_size[comp + ' ' + size] += 1

    return hh_comp_by_hh_size


# results()