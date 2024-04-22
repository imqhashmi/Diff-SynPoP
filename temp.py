# comp = []
# for key, value in hh_comp_dict.items():
#     comp.extend([key] * value)
#
# hh_df = pd.DataFrame(comp, columns=['composition'])
# hh_df['size'] = hh_df['composition'].apply(lambda x: int(household_sizes[x][:-1]) if '+' in household_sizes[x] else int(household_sizes[x]))
#
# # evaluating the household sizes
# hh_df['size_check'] = hh_df.apply(lambda x: check_size(x['composition'], x['size']), axis=1)
# print(hh_df['size_check'].value_counts())
# aggregating the household sizes
# hh_size_dict = hh_df['size'].value_counts().to_dict()
# hh_size_dict = {str(key): value for key, value in hh_size_dict.items()}
# print(hh_size_dict)
# print(ID.getdictionary(ID.HHsizedf, area))