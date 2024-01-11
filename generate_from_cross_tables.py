import pandas as pd
import numpy as np

# imagine we have the following tables
# sex and age counts
sex_by_age_table = {"M 0-50": 30, "M 50-100": 20, "F 0-50": 40, "F 50-100": 10}
sex_by_age_table_df = pd.Series(sex_by_age_table).to_frame()
sex_by_age_table_df.columns = ["count"]
# ethnicity by sex and age
ethnicity_by_sex_and_age = {
    "M 0-50": {"black": 10, "white": 20},
    "M 50-100": {"black": 5, "white": 15},
    "F 0-50": {"black": 20, "white": 20},
    "F 50-100": {"black": 5, "white": 5},
}
ethnicity_by_sex_and_age_df = pd.DataFrame(ethnicity_by_sex_and_age).transpose()

# check they are constructed right
print("---- Input tables -----")
print(sex_by_age_table_df)
print(ethnicity_by_sex_and_age_df)
print("\n")

# now let's generate a population
# we first normalize the tables so that they are distributions
sex_by_age_table_df = sex_by_age_table_df / sex_by_age_table_df.sum(0)
ethnicity_by_sex_and_age_df = ethnicity_by_sex_and_age_df.div(ethnicity_by_sex_and_age_df.sum(1), 0)

# check
print("---- Normalized tables -----")
print(sex_by_age_table_df)
print(ethnicity_by_sex_and_age_df)
print("\n")

# now we generate a population of 1000 individuals.
N = 10

# we first sample sex and age of each individual
probs = sex_by_age_table_df["count"].values
values = list(sex_by_age_table_df.index)
sex_and_age_values = np.random.choice(values, N, p=probs)
# now we have generated age and sex according to the distribution
# we now generate ethnicity

# this gives a row for each person with the probabilities of being black / white
ethnicity_probs = ethnicity_by_sex_and_age_df.loc[sex_and_age_values]
# now we run random choice for each row
ethnicity_values = ethnicity_probs.apply(
    lambda x: np.random.choice(["black", "white"], p=x.values), 1
).values

# so now we have 10 people with
print("---- Generated population -----")
print("Number of people: ", N)
print(sex_and_age_values)
print(ethnicity_values)