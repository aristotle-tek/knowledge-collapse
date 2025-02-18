#05_calc_shannon.py




import json
import pandas as pd
import numpy as np
import re
import os





R_dataset = 2693 # (len of uniq entities in reflist_labels_dedup.csv )
    

def calc_shannon_diversity(individuals_lists):
    combined_list = sum(individuals_lists, [])
    entity_counts = Counter(combined_list)

    total_mentions = sum(entity_counts.values())
    proportions = [count / total_mentions for count in entity_counts.values()]

    shannons_diversity_index = -sum(p * np.log(p) for p in proportions)
    return shannons_diversity_index



def calc_pielou_evenness(individuals_lists, R=None):
    combined_list = sum(individuals_lists, [])
    entity_counts = Counter(combined_list)

    total_mentions = sum(entity_counts.values())
    proportions = [count / total_mentions for count in entity_counts.values()]

    shannons_diversity_index = -sum(p * np.log(p) for p in proportions)

    if R is None:
        R = len(entity_counts)

    H_max = np.log(R)

    pielous_evenness = shannons_diversity_index / H_max

    return pielous_evenness




data_folder = os.getcwd()
reflist_folder = data_folder + "reference_list/"
datalist_folder = data_folder + 'list_dfs/'



curr_temp = '1'

# combine with metadata (e.g. which run (seed) )
# create a list of lists
modelnames = ["gpt-3.5-turbo",'claude-3-sonnet','gemini-pro']
for whichmodel in modelnames:
    individuals_lists = []
    closest_file = data_folder + "closest/closest_" + whichmodel + "_" + promptversion + ".csv"
    dg = pd.read_csv(closest_file)
    dm = pd.read_csv(datalist_folder + "metadata_" +whichmodel + "_temp" + str(curr_temp)  + "_" +promptversion +".csv")
    assert len(dg)==len(dm)
    dg['run'] = dm.run
    runs = list(set(dg.run))
    for run in runs:
        rel = dg[dg.run==run]
        currlist = list(rel.phil)
        individuals_lists.append(currlist)



shannon = calc_shannon_diversity(individuals_lists)
pielous = calc_pielou_evenness(individuals_lists, R_dataset)



modelnames = ["gpt-3.5-turbo",'claude-3-sonnet','gemini-pro']
modelnames_plus = ["gpt-3.5-turbo",'claude-3-sonnet','gemini-pro','llama2-70b']


calcs = []




for promptversion in ['v1','v2','v3']: # ['v1','v2_diverse','v3']:
    for whichmodel in modelnames_plus:
        individuals_lists = []
        closest_file = data_folder + "closest/closest_" + whichmodel + "_" + promptversion + ".csv"
        dg = pd.read_csv(closest_file)
        dm = pd.read_csv(datalist_folder + "metadata_" +whichmodel + "_temp" + str(curr_temp)  + "_" +promptversion +".csv")
        assert len(dg)==len(dm)
        dg['run'] = dm.run
        runs = list(set(dg.run))
        for run in runs:
            rel = dg[dg.run==run]
            currlist = list(rel.phil)
            individuals_lists.append(currlist)

        shannon = calc_shannon_diversity(individuals_lists)
        pielous = calc_pielou_evenness(individuals_lists, R_dataset)
        calcs.append([promptversion, whichmodel, shannon, pielous])



modelnames = ["gpt-3.5-turbo",'claude-3-sonnet','gemini-pro']


for promptversion in  ['v4','v5']: # ['v4_list20','v5_list20_regions']:
    for whichmodel in modelnames:
        individuals_lists = []
        closest_file = data_folder + "closest/closest_" + whichmodel + "_" + promptversion[:2] + ".csv"
        dg = pd.read_csv(closest_file)
        try:
            dm = pd.read_csv(datalist_folder + "metadata_" +whichmodel + "_temp" + str(curr_temp)  + "_" +promptversion +".csv")
        except:
             dm = pd.read_csv(datalist_folder + "metadata_" +whichmodel + "_temp" + str(curr_temp)  + "_" +promptversion[:2] +".csv")
        if len(dg) !=len(dm):
            print(len(dg))
            print(len(dm))
        else:
            dg['run'] = dm.run
            runs = list(set(dg.run))
            for run in runs:
                rel = dg[dg.run==run]
                currlist = list(rel.phil)
                individuals_lists.append(currlist)

            shannon = calc_shannon_diversity(individuals_lists)
            pielous = calc_pielou_evenness(individuals_lists, R_dataset)
            calcs.append([promptversion, whichmodel, shannon, pielous])




dr = pd.DataFrame(calcs)
dr.columns= ['promptversion','whichmodel','shannon','pielous']


latex_table = dr.to_latex(index=False, caption='Shannon and Pielou\'s Indices by Model and Version', label='tab:indices', column_format='llrr', float_format="%.2f")

print(latex_table)


mean_values = dr.groupby('promptversion')[['shannon', 'pielous']].mean()



