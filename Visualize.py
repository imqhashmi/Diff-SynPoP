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


path = os.path.join(os.path.dirname(os.getcwd()), 'Diff-SynPoP')

area = 'E02005924'  # geography code for one of the oxford output areas (selected for this work)
num_households = 4852

sex_dict = ID.getdictionary(ID.sexdf, area)  # sex
age_dict = ID.getdictionary(ID.age5ydf, area)  # age
ethnic_dict = ID.getdictionary(ID.ethnicdf, area)  # ethnicity
religion_dict = ID.getdictionary(ID.religiondf, area)  # religion
marital_dict = ID.getdictionary(ID.maritaldf, area)  # marital status
qual_dict = ID.getdictionary(ID.qualdf, area)  # highest qualification level

seg_dict = ID.getFinDictionary(ID.seg_df, area)  # socio-economic grade
occupation_dict = ID.getFinDictionary(ID.occupation_df, area)  # occupation
economic_act_dict = ID.getFinDictionary(ID.economic_act_df, area)  # economic activity
approx_social_grade_dict = ID.getFinDictionary(ID.approx_social_grade_df, area)  # approximated social grade
general_health_dict = ID.getFinDictionary(ID.general_health_df, area)  # general health
industry_dict = ID.getFinDictionary(ID.industry_df, area)  # industry of occupation

hh_comp_dict = ID.getHHcomdictionary(ID.HHcomdf, area)  # household composition
hh_id = range(1, num_households + 1)
hh_size_dict = ID.getdictionary(ID.HHsizedf, area)  # household size

hh_comp_dict_mod = {index: value for index, (_, value) in enumerate(hh_comp_dict.items())}
ethnic_dict_hh = ID.getdictionary(ID.ethnicdf, area)  # ethnicity of reference person of a household
religion_dict_hh = ID.getdictionary(ID.religiondf, area)  # religion of reference person of a household
car_ownership_dict = ID.getHHDictionary(ID.car_ownership_df, area)  # car or van ownership / availability
weekly_income_dict = ID.getHHDictionary(ID.weekly_income_df, area)  # total and net weekly income

persondf = pd.read_csv(os.path.join(path, 'synthetic_population.csv'))
householddf = pd.read_csv(os.path.join(path, 'synthetic_households.csv'))

attributes = ['age', 'ethnicity', 'religion', 'marital', 'qualification',
              'seg', 'occupation', 'economic_act',
              'general_health', 'industry']


attribute_dicts = {
    'age': age_dict,
    'ethnicity' : ethnic_dict,
    'religion': religion_dict,
    'marital': marital_dict,
    'qualification': qual_dict,
    'seg': seg_dict,
    'occupation': occupation_dict,
    'economic_act': economic_act_dict,
    'general_health': general_health_dict,
    'industry': industry_dict}

def plot_radar_grid(df, attributes, attribute_dicts):
    # Determine the number of rows and columns for the grid
    rows = 3
    cols = 4

    # Create a subplot figure with the specified number of rows and columns
    fig = make_subplots(rows=rows, cols=cols, specs=[[{'type': 'polar'}]*cols]*rows)

    for i, attribute in enumerate(attributes):
        # Calculate the row and column index for the current subplot
        row = (i // cols) + 1
        col = (i % cols) + 1

        # Extract unique categories and their counts
        categories_gen = df[attribute].unique().astype(str).tolist()
        count_gen = [df[attribute].value_counts().get(str(cat), 0) for cat in categories_gen]

        # Get actual categories and counts
        categories = list(attribute_dicts[attribute].keys())
        count_act = list(attribute_dicts[attribute].values())

        # Align generated counts with actual categories
        gen_combined = sorted(zip(categories_gen, count_gen), key=lambda x: categories.index(x[0]))
        categories_gen, count_gen = zip(*gen_combined)
        count_gen = list(count_gen)

        # Calculate range and update counts for visualization
        count_act = [max(val, 10) for val in count_act]
        count_gen = [max(val, 10) for val in count_gen]

        # Calculate accuracy
        squared_errors = [(actual - predicted) ** 2 for actual, predicted in zip(count_act, count_gen)]
        mean_squared_error = sum(squared_errors) / len(count_act)
        rmse = math.sqrt(mean_squared_error)
        max_possible_error = math.sqrt(sum(x ** 2 for x in count_act))
        accuracy = 1 - (rmse / max_possible_error)

        # Close the radar chart loop
        categories.append(categories[0])
        count_act.append(count_act[0])
        count_gen.append(count_gen[0])

        # Calculate dynamic tick values and labels based on min and max values
        max_value = max(max(count_act), max(count_gen))
        min_value = min(min(count_act), min(count_gen))
        step = (max_value - min_value) / 5
        tickvals = [min_value + i * step for i in range(6)]
        ticktext = [str(round(min_value + i * step, 2)) for i in range(6)]


        # Add traces to the subplot
        fig.add_trace(go.Scatterpolar(r=count_gen, theta=categories,
                                      name=f'Generated Population ({attribute})', line=dict(width=2, color='blue', dash='solid')), row=row, col=col)
        fig.add_trace(go.Scatterpolar(r=count_act, theta=categories,
                                      name=f'Actual Population ({attribute})', line=dict(width=2, color='red', dash='solid')), row=row, col=col)

        # Update subplot layout
        fig.update_polars(row=row, col=col,
            radialaxis=dict(
                visible=True,
                type='linear',
                # tickvals=[i for i in range(0, 1400, 200)],  # Equal divisions from 0 to 1100 in steps of 100
                # ticktext=[str(i) for i in range(0, 1400, 200)],  # Text for each tick
                tickvals=tickvals,
                ticktext=ticktext,
                tickmode='array',
                showline=True,
                showticklabels=True,  # Show tick labels
                linewidth=1,  # Set the line width
                gridcolor="rgba(0, 0, 0, 0.1)",  # Set the grid color to black
                linecolor="rgba(0, 0, 0, 0.1)"  # Set the line color to black
            ),
            angularaxis=dict(
                showline=True,
                linewidth=2,  # Set the line width for the angular lines
                gridcolor="rgba(0, 0, 0, 0.1)",  # Set light black color with low opacity for grid lines
                linecolor="black",  # Set the outer boundary line color to black
            )
        )
        # Draw grid lines between subplots
        shapes = []
        for i in range(1, rows):
            shapes.append(dict(
                type='line',
                x0=0,
                y0=i / rows,
                x1=1,
                y1=i / rows,
                line=dict(color='rgba(0, 0, 0, 0.1)', width=1)
            ))
        for j in range(1, cols):
            shapes.append(dict(
                type='line',
                x0=j / cols,
                y0=0,
                x1=j / cols,
                y1=1,
                line=dict(color='rgba(0, 0, 0, 0.1)', width=1)
            ))
        fig.add_annotation(
            text= attribute[0].capitalize() + attribute[1:],
            xref='paper',
            yref='paper',
            x=(col - 1) / cols + 0.5 / cols,
            y=(rows - row) / rows ,
            showarrow=False,
            font=dict(size=16, color="black"),
            xanchor='center',
            yanchor='bottom'
        )
    fig.update_layout(
        shapes=shapes,
        showlegend=False,  # Disable the legend
        width=1000,
        height=800,
        margin=dict(l=0, r=0, t=0, b=0),  # Remove gaps between cells
        font=dict(size=8),
        title="Radar Charts Grid"
    )

    # Save and show the plot
    file = os.path.join(os.path.dirname(os.getcwd()), 'Diff-SynPoP', 'radar_charts', 'radar_charts_grid.html')
    py.offline.plot(fig, filename=file)
    fig.show()


def plot_radar(df, attribute, attribute_dict):
    # Extract unique categories and their counts
    categories_gen = df[attribute].unique().astype(str).tolist()
    count_gen = [df[attribute].value_counts().get(str(cat), 0) for cat in categories_gen]

    # Get actual categories and counts
    categories = list(attribute_dict.keys())
    count_act = list(attribute_dict.values())

    # Align generated counts with actual categories
    gen_combined = sorted(zip(categories_gen, count_gen), key=lambda x: categories.index(x[0]))
    categories_gen, count_gen = zip(*gen_combined)
    count_gen = list(count_gen)

    # Calculate range and update counts for visualization
    count_act = [max(val, 10) for val in count_act]
    count_gen = [max(val, 10) for val in count_gen]

    # Calculate accuracy
    squared_errors = [(actual - predicted) ** 2 for actual, predicted in zip(count_act, count_gen)]
    mean_squared_error = sum(squared_errors) / len(count_act)
    rmse = math.sqrt(mean_squared_error)
    max_possible_error = math.sqrt(sum(x ** 2 for x in count_act))
    accuracy = 1 - (rmse / max_possible_error)

    # Close the radar chart loop
    categories.append(categories[0])
    count_act.append(count_act[0])
    count_gen.append(count_gen[0])

    # Calculate dynamic tick values and labels based on min and max values
    max_value = max(max(count_act), max(count_gen))
    min_value = min(min(count_act), min(count_gen))
    step = (max_value - min_value) / 5
    tickvals = [min_value + i * step for i in range(6)]
    ticktext = [str(round(min_value + i * step, 2)) for i in range(6)]

    # Create radar chart
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=count_gen, theta=categories, name='Generated Population', line=dict(width=3)))
    fig.add_trace(go.Scatterpolar(r=count_act, theta=categories, name='Actual Population', line=dict(width=3)))

    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                type='linear',
                tickvals=tickvals,
                ticktext=ticktext,
                tickmode='array',
                showline=True,
                showticklabels=True,  # Show tick labels
                linewidth=1,  # Set the line width
                gridcolor="rgba(0, 0, 0, 0.1)",  # Set the grid color to black
                linecolor="rgba(0, 0, 0, 0.1)"  # Set the line color to black
            ),
            angularaxis=dict(
                showline=True,
                # gridcolor="rgba(0, 0, 0, 0.1)",
                # linecolor="rgba(0, 0, 0, 0.1)"
                linewidth=2,  # Set the line width for the angular lines
                gridcolor="rgba(0, 0, 0, 0.1)",  # Set light black color with low opacity for grid lines
                linecolor="black",  # Set the outer boundary line color to black
            ),
        ),
        showlegend=True,
        width=800,
        height=800,
        margin=dict(l=50, r=50, t=50, b=50),
        font=dict(size=15)
    )

    # Save and show the plot
    file = os.path.join(os.path.dirname(os.getcwd()), 'Diff-SynPoP', 'radar_charts', f"{attribute}-radar-chart.html")
    py.offline.plot(fig, filename=file)
    fig.show()




plot_radar_grid(persondf, attributes, attribute_dicts)
# plot_radar(persondf, 'seg', seg_dict)
# plot_radar('religion', religion_dict, show='yes')
# plot_radar(persondf, 'ethnicity', ethnic_dict)
# plot_radar('marital', marital_dict, show='yes')
# plot_radar('qualification', qual_dict, show='yes')













