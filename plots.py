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
import commons as cm

# load generated_population2.csv
df = pd.read_csv("generated_population2.csv")
aggregated_dicts = cm.create_aggregates(df)
target_dicts = {
    "Sex by Age": cm.sex_by_age,
    "Ethnicity by Sex by Age": cm.ethnic_by_sex_by_age,
    "Religion by Sex by Age": cm.religion_by_sex_by_age,
    "Marital Status by Sex by Age": cm.marital_by_sex_by_age,
    "Qualification by Sex by Age": cm.qual_by_sex_by_age
}
titles = [
    "Sex by Age Prediction vs. Target",
    "Ethnicity by Sex by Age Prediction vs. Target",
    "Religion by Sex by Age Prediction vs. Target",
    "Marital Status by Sex by Age Prediction vs. Target",
    "Qualification by Sex by Age Prediction vs. Target"
]

# Visualize the results
cm.plot_crosstable_comparison_subplots(aggregated_dicts, target_dicts, titles, show_keys=False, num_cols=1)

target_dicts = {
    'Sex': cm.sex_dict,
    'Age': cm.age_dict,
    'Ethnicity': cm.ethnic_dict,
    'Religion': cm.religion_dict,
    'MaritalStatus': cm.marital_dict,
    'Qualification': cm.qual_dict
}
cm.plot_comparison_with_accuracy_subplots(df, target_dicts)


df = pd.read_csv("generated_households.csv")
aggregated_dicts = cm.create_aggregates_hh(df)

# Define target_dicts, titles, and other parameters as needed
target_dicts = {
    "Composition by Size": cm.hh_comp_by_size,
    "Composition by Ethnicity": cm.hh_comp_by_ethnic,
    "Composition by Religion": cm.hh_comp_by_religion
}
titles = [
    "Composition by Size Prediction vs. Target",
    "Composition by Ethnicity Prediction vs. Target",
    "Composition by Religion Prediction vs. Target"
]

# Visualize the results
cm.plot_crosstable_comparison_subplots(aggregated_dicts, target_dicts, titles, show_keys=True, num_cols=1)
target_dicts = {
    'Size': cm.hh_size,
    'Composition': cm.hh_comp,
    'Ethnic': cm.hh_ethnic,
    'Religion': cm.hh_religion
}
cm.plot_comparison_with_accuracy_subplots(df, target_dicts)