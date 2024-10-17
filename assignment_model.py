import pandas as pd

# Age categories
child_Ages = ["0_4", "5_7", "8_9", "10_14", "15"]
adult_Ages = ["16_17", "18_19", "20_24", "25_29", "30_34", "35_39", "40_44", "45_49", "50_54", "55_59", "60_64"]
elder_Ages = ["65_69", "70_74", "75_79", "80_84", "85+"]

# Read in the data
persons = pd.read_csv('generated_population2.csv')
persons['PID'] = range(1, len(persons) + 1)

households = pd.read_csv('generated_households.csv')
households['HID'] = range(1, len(households) + 1)


# Function to sample from persons and return both the sampled PIDs and the updated dataframe
def extract_random_sample(filtered_persons, persons, size):
    # Ensure size is non-negative and handle cases where filtered_persons has fewer rows than requested
    if size <= 0 or len(filtered_persons) == 0:
        return [], persons

    size = min(size, len(filtered_persons))  # Adjust size to the number of available persons

    # Sample the available persons
    sampled_persons = filtered_persons.sample(n=size)
    sampled_persons_PIDs = sampled_persons['PID'].tolist()

    # Drop sampled persons by their PID from the original dataframe
    persons = persons[~persons['PID'].isin(sampled_persons_PIDs)]

    return sampled_persons_PIDs, persons


# Handle different household compositions with ethnic and religion parameters
def handle_composition(ethnic, religion, composition, size, persons):
    # If the persons dataframe is empty, return immediately
    if persons.empty or size <= 0:
        return [], persons

    if composition == '1PE' or composition == '1FE':
        # Handle One person: Pensioner (person who is above 65)
        elders = persons[
            (persons['Age'].isin(elder_Ages)) & (persons['Ethnicity'] == ethnic) & (persons['Religion'] == religion)]
        sampled_PIDs, persons = extract_random_sample(elders, persons, size)
        return sampled_PIDs, persons

    elif composition == '1PA':
        # Handle One person: Other (a single person who is above 18 and below 65)
        adults = persons[
            (persons['Age'].isin(adult_Ages)) & (persons['Ethnicity'] == ethnic) & (persons['Religion'] == religion)]
        sampled_PIDs, persons = extract_random_sample(adults, persons, size)
        return sampled_PIDs, persons

    elif composition == '1FM-0C':
        # Handle One family: Married Couple: No children
        married_male = persons[
            (persons['MaritalStatus'] == 'Married') & (persons['Sex'] == 'M') & (persons['Ethnicity'] == ethnic) & (
                    persons['Religion'] == religion)]
        married_female = persons[
            (persons['MaritalStatus'] == 'Married') & (persons['Sex'] == 'F') & (persons['Ethnicity'] == ethnic) & (
                    persons['Religion'] == religion)]
        pids_male, persons = extract_random_sample(married_male, persons, 1)
        pids_female, persons = extract_random_sample(married_female, persons, 1)
        return pids_male + pids_female, persons

    elif composition == '1FM-2C':
        # Handle One family: Married Couple: Having dependent children
        married_male = persons[
            (persons['MaritalStatus'] == 'Married') & (persons['Sex'] == 'M') & (persons['Ethnicity'] == ethnic) & (
                    persons['Religion'] == religion)]
        married_female = persons[
            (persons['MaritalStatus'] == 'Married') & (persons['Sex'] == 'F') & (persons['Ethnicity'] == ethnic) & (
                    persons['Religion'] == religion)]
        children = persons[
            (persons['Age'].isin(child_Ages)) & (persons['Ethnicity'] == ethnic) & (persons['Religion'] == religion)]
        pids_male, persons = extract_random_sample(married_male, persons, 1)
        pids_female, persons = extract_random_sample(married_female, persons, 1)
        pids_children, persons = extract_random_sample(children, persons, size - 2)
        return pids_male + pids_female + pids_children, persons

    elif composition == '1FM-nA':
        # Handle One family: Married Couple: all children non-dependent
        married_male = persons[
            (persons['MaritalStatus'] == 'Married') & (persons['Sex'] == 'M') & (persons['Ethnicity'] == ethnic) & (
                    persons['Religion'] == religion)]
        married_female = persons[
            (persons['MaritalStatus'] == 'Married') & (persons['Sex'] == 'F') & (persons['Ethnicity'] == ethnic) & (
                    persons['Religion'] == religion)]
        children = persons[
            (persons['Age'].isin(adult_Ages)) & (persons['Ethnicity'] == ethnic) & (persons['Religion'] == religion)]
        pids_male, persons = extract_random_sample(married_male, persons, 1)
        pids_female, persons = extract_random_sample(married_female, persons, 1)
        pids_children, persons = extract_random_sample(children, persons, size - 2)
        return pids_male + pids_female + pids_children, persons

    elif composition == '1FC-0C':
        # Handle One family: Cohabiting Couple: No children
        male = persons[
            (persons['MaritalStatus'] != 'Married') & (persons['Sex'] == 'M') & (persons['Ethnicity'] == ethnic) & (
                    persons['Religion'] == religion)]
        female = persons[
            (persons['MaritalStatus'] != 'Married') & (persons['Sex'] == 'F') & (persons['Ethnicity'] == ethnic) & (
                    persons['Religion'] == religion)]
        pids_male, persons = extract_random_sample(male, persons, 1)
        pids_female, persons = extract_random_sample(female, persons, 1)
        return pids_male + pids_female, persons

    elif composition == '1FC-2C':
        # Handle One family: Cohabiting Couple: Having dependent children
        male = persons[
            (persons['MaritalStatus'] != 'Married') & (persons['Sex'] == 'M') & (persons['Ethnicity'] == ethnic) & (
                    persons['Religion'] == religion)]
        female = persons[
            (persons['MaritalStatus'] != 'Married') & (persons['Sex'] == 'F') & (persons['Ethnicity'] == ethnic) & (
                    persons['Religion'] == religion)]
        children = persons[
            (persons['Age'].isin(child_Ages)) & (persons['Ethnicity'] == ethnic) & (persons['Religion'] == religion)]
        pids_male, persons = extract_random_sample(male, persons, 1)
        pids_female, persons = extract_random_sample(female, persons, 1)
        pids_children, persons = extract_random_sample(children, persons, size - 2)
        return pids_male + pids_female + pids_children, persons

    elif composition == '1FC-nA':
        # Handle One family: Cohabiting Couple: all children non-dependent
        male = persons[
            (persons['MaritalStatus'] != 'Married') & (persons['Sex'] == 'M') & (persons['Ethnicity'] == ethnic) & (
                    persons['Religion'] == religion)]
        female = persons[
            (persons['MaritalStatus'] != 'Married') & (persons['Sex'] == 'F') & (persons['Ethnicity'] == ethnic) & (
                    persons['Religion'] == religion)]
        children = persons[
            (persons['Age'].isin(adult_Ages)) & (persons['Ethnicity'] == ethnic) & (persons['Religion'] == religion)]
        pids_male, persons = extract_random_sample(male, persons, 1)
        pids_female, persons = extract_random_sample(female, persons, 1)
        pids_children, persons = extract_random_sample(children, persons, size - 2)
        return pids_male + pids_female + pids_children, persons

    elif composition == '1FL-2C':
        # Handle One family: Lone Parent: Having dependent children
        parent = persons[(persons['MaritalStatus'] != 'Married') & (persons['Ethnicity'] == ethnic) & (
                persons['Religion'] == religion)]
        children = persons[
            (persons['Age'].isin(child_Ages)) & (persons['Ethnicity'] == ethnic) & (persons['Religion'] == religion)]
        pids_parent, persons = extract_random_sample(parent, persons, 1)
        pids_children, persons = extract_random_sample(children, persons, size - 1)
        return pids_parent + pids_children, persons

    elif composition == '1FL-nA':
        # Handle One family: Lone parent: all children non-dependent
        parent = persons[(persons['MaritalStatus'] != 'Married') & (persons['Ethnicity'] == ethnic) & (
                persons['Religion'] == religion)]
        children = persons[
            (persons['Age'].isin(adult_Ages)) & (persons['Ethnicity'] == ethnic) & (persons['Religion'] == religion)]
        pids_parent, persons = extract_random_sample(parent, persons, 1)
        pids_children, persons = extract_random_sample(children, persons, size - 1)
        return pids_parent + pids_children, persons

    elif composition == '1H-2C':
        # Handle Other households: Having dependent children
        adults = persons[
            (persons['Age'].isin(adult_Ages)) & (persons['Ethnicity'] == ethnic) & (persons['Religion'] == religion)]
        children = persons[
            (persons['Age'].isin(child_Ages)) & (persons['Ethnicity'] == ethnic) & (persons['Religion'] == religion)]
        pids_adults, persons = extract_random_sample(adults, persons, 1)
        pids_children, persons = extract_random_sample(children, persons, size - 1)
        return pids_adults + pids_children, persons

    elif composition == '1H-nA':
        # Handle Other households: All adults
        adults_and_children = persons[
            (persons['Age'].isin(adult_Ages)) & (persons['Ethnicity'] == ethnic) & (persons['Religion'] == religion)]
        pids, persons = extract_random_sample(adults_and_children, persons, size)
        return pids, persons

    elif composition == '1H-nE':
        # Handle Other households: All pensioners
        elders = persons[
            (persons['Age'].isin(elder_Ages)) & (persons['Ethnicity'] == ethnic) & (persons['Religion'] == religion)]
        return extract_random_sample(elders, persons, size)

    elif composition == '1H-nS':
        # Handle Other households: All students
        students = persons[
            (persons['Qualification'] != 'no') & (persons['Ethnicity'] == ethnic) & (persons['Religion'] == religion)]
        return extract_random_sample(students, persons, size)


print(f"Initial number of persons: {len(persons)}")
# for i, household in households.iterrows():
#     eth = household['Ethnic']
#     rel = household['Religion']
#     comp = household['Composition']
#     size = household['Size']
#     persons_to_assign, persons = handle_composition(eth, rel, comp, size, persons)

# Create an empty DataFrame to hold the merged person-household information
child_df = pd.DataFrame()

# Loop through households and merge assigned persons with household information
for i, household in households.iterrows():
    eth = household['Ethnic']
    rel = household['Religion']
    comp = household['Composition']
    size = household['Size']

    # Get the assigned persons for this household
    persons_to_assign, persons = handle_composition(eth, rel, comp, size, persons)
    assigned_df = pd.DataFrame(persons.loc[persons['PID'].isin(persons_to_assign)])
    assigned_df['HID'] = household['HID']
    assigned_df

