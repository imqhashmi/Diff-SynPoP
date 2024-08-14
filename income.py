import plotly.graph_objects as go
import plotly as py
import math
import plotly.graph_objects as go
import pandas as pd
import os
import plotly.graph_objects as go
import math
import InputData as ID
import InputCrossTables as ICT
import plotly as py
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import math
import numpy as np

income_df = {"Male": (830.9, 2.7), "Female": (602.7, 2.5)}
def get_std(n=1000):
    # Given data
    mean_male = income_df['Male'][0]
    conf_male = income_df['Male'][1]
    mean_female = income_df['Female'][0]
    conf_female = income_df['Female'][1]
    sample_size = n

    # Confidence level and Z-score for 95% confidence
    # z_score = 1.96
    z_score = 2.054
    # Calculate the standard deviation for male
    margin_of_error_male = (conf_male / 100) * mean_male / 2
    # std_dev_male = margin_of_error_male / z_score * math.sqrt(sample_size)
    std_dev_male = margin_of_error_male * math.sqrt(sample_size) / z_score

    # Calculate the standard deviation for female
    margin_of_error_female = (conf_female / 100) * mean_female / 2
    # std_dev_female = margin_of_error_female / z_score * math.sqrt(sample_size)
    std_dev_female = margin_of_error_female * math.sqrt(sample_size) / z_score
    return [std_dev_male, std_dev_female]

#import synthetic_popluation.csv as df

path = os.path.join(os.path.dirname(os.getcwd()), 'Diff-SynPoP')
area = 'E02005924'  # geography code for one of the oxford output areas (selected for this work)
persondf = pd.read_csv(os.path.join(path, 'synthetic_population.csv'))

std_devs = get_std(len(persondf))
print(std_devs)
income_df['Male'] = income_df['Male'] + (std_devs[0],)
income_df['Female'] = income_df['Female'] + (std_devs[1],)

print(income_df)
# Generate income for each individual
np.random.seed(42)  # For reproducibility




def generate_income(sex):
    if sex == 'M':
        return np.random.normal(income_df['Male'][0], income_df['Male'][2])
    else:
        return np.random.normal(income_df['Female'][0], income_df['Female'][2])

persondf['income'] = persondf['sex'].apply(generate_income)
mean_income_male = persondf[persondf['sex'] == 'M']['income'].mean()
mean_income_female = persondf[persondf['sex'] == 'F']['income'].mean()

# Data for the plot
std_devs = get_std(len(persondf))
# Create the plot
fig = go.Figure()

fig.add_trace(go.Bar(
    name='Mean Weekly Income (Actual)',
    x=['Male', 'Female'],
    y=[income_df['Male'][0], income_df['Female'][0]],
    error_y=dict(type='data', array=std_devs),
    marker_color='red'
)),
fig.add_trace(go.Bar(
    name='Mean Weekly Income (Generated)',
    x=['Male', 'Female'],
    y=[mean_income_male, mean_income_female],
    marker_color='blue'
))
fig.update_layout(
    showlegend=False,
    title='Mean Weekly Income by Sex with Standard Deviation',
    xaxis_title='(K) Mean Weekly Income',
    yaxis=dict(
        showgrid=True,  # Show grid lines
        gridwidth=1,  # Width of grid lines
        gridcolor='lightgray'  # Color of grid lines
    ),
    # yaxis_title='Mean Weekly Income',
    barmode='group',
    width=500,
    height=300,
    margin=dict(l=0, r=0, t=0, b=0),  # Remove gaps between cells
    font=dict(size=16),
    paper_bgcolor='white',  # Background color outside the plot
    plot_bgcolor='white'    # Background color inside the plot
)
py.offline.plot(fig, filename='income.html')
fig.show()
