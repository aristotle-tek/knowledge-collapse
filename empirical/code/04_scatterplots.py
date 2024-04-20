#04_scatterplots.py

# and freq plots

import json
import pandas as pd
import time
import re
import os

import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np




data_folder = os.getcwd()

plot_dir = data_folder + "plots/"



#----- plots comparing freq for v1 vs v5:

for whichmodel in ["gpt-3.5-turbo",'claude-3-sonnet','gemini-pro']:
    closest_file1 = data_folder + "closest/closest_" + whichmodel + "_v1.csv"
    closest_file5 = data_folder + "closest/closest_" + whichmodel + "_v5.csv"
    dg = pd.read_csv(closest_file1)
    label_freq = dg['closest_label'].value_counts(normalize=True)

    dg2 = pd.read_csv(closest_file5)
    label_freq2 = dg2['closest_label'].value_counts(normalize=True)


    plt.figure(figsize=(width, height))

    label_freq.plot(kind='bar', color='darkblue', alpha=0.6, label=f'v1')

    label_freq2.plot(kind='bar', color='darkred', alpha=0.6, label=f'v5')


    plt.title(f'{whichmodel}')
    plt.xlabel('')
    plt.ylabel('')
    plt.xlim(0, 400)
    plt.ylim(0, ymax)
    plt.xticks([])

    plt.legend()

    curr_plot_file = plot_dir +  "freq_1v5_" + whichmodel + ".pdf"
    plt.savefig(curr_plot_file, format='pdf', bbox_inches='tight')
    plt.close()




# file with top 10 most common plus selected others
sel = pd.read_excel(data_folder + "sel_top.xlsx")



closest_file_directory = data_folder + "closest/"

all_files = os.listdir(closest_file_directory)

csv_files = [f for f in all_files if f.endswith('.csv')]



# for reference, get corpus-wide counts

clust_name_dict = {}
name_clust_dict = {}

alldata = pd.DataFrame()
for csvfile in csv_files:
    print(csvfile)
    curr = pd.read_csv(data_folder + "closest/" + csvfile)
    for i, row in curr.iterrows():
        name = row['phil']
        lab = row['closest_label']
        clust_name_dict[lab] = name
        name_clust_dict[name] = lab

    alldata = pd.concat([alldata, curr])
    print(len(alldata))


label_counts = alldata['closest_label'].value_counts()

label_freq = alldata['closest_label'].value_counts(normalize=True)


def try_getname(clust):
    try:
        name = clust_name_dict[clust]
    except:
        return ''

dff = pd.DataFrame(label_freq)
dff['name'] = dff.index.map(clust_name_dict)

dff.to_csv(data_folder + "freq_names_overall.csv")




# first, generate a dataframe with the label, freq counts from each:

# for v1, v2, v3: (4,5 diff format below)

allrows = []
for promptversion in ['v1','v2','v3']:
    for whichmodel in ["gpt-3.5-turbo",'claude-3-sonnet','gemini-pro','llama2-70b']:
        infile = data_folder + "closest/closest_" + whichmodel + "_" + promptversion + ".csv"
        dg = pd.read_csv(infile)
        print(len(dg))
        label_counts = dg['closest_label'].value_counts()
        label_freq = dg['closest_label'].value_counts(normalize=True)
        for i, row in sel.iterrows():
            #print(row['name'])
            id1 = row['label']
            id2 = row['label2']
            try:
                freq = label_freq[id1]
                count = label_counts[id1]
            except:
                freq = 0
                count = 0
            if pd.notnull(id2):
                try:
                    freq2 = label_freq[int(id2)]
                    freq += freq2
                    count2 = label_counts[int(id2)]
                    count += count2
                except:
                    pass
            allrows.append([promptversion, whichmodel, row['name'], freq, count])


dt = pd.DataFrame(allrows, columns=['prompt','model','name','freq','count'])



# now generate the plots
# Sort dt by 'freq' 
dt = dt.sort_values(by=['freq'], ascending=[ False])


model_list = dt['model'].unique()

colors = {'gpt-3.5-turbo': 'red', 'claude-3-sonnet': 'blue', 'gemini-pro': 'green', 'llama2-70b': 'purple'}
markers = ['o', '^', 's', 'x']

# Sort based on freq to keep fixed
base_order_df = dt[dt['prompt'] == 'v1']
name_order = base_order_df.groupby('name')['freq'].sum().sort_values(ascending=False).index.tolist()


for promptversion in ['v1', 'v2', 'v3']:
    plt.figure(figsize=(20, 10)) 

    dt = dt[dt['prompt'] == promptversion]
    dt['name'] = pd.Categorical(dt['name'], categories=name_order, ordered=True)
    dt = dt.sort_values('name')
    
    for model, marker in zip(model_list, markers):
        subset = dt[dt['model'] == model]
        plt.scatter(subset['name'], subset['freq'], color=colors[model], marker=marker, s=100, label=model)
    
    plt.xlabel('')
    plt.ylim(0, 0.3)
    plt.ylabel('Frequency')
    plt.title(f'')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model')

    plt.tight_layout() 
    curr_plot_file = plot_dir + "freq_" + promptversion + ".pdf"
    plt.savefig(curr_plot_file, format='pdf', bbox_inches='tight')
    plt.close()





#------------
# v4, v5
#------------


allrows = []
for promptversion in ['v4','v5']:
    for whichmodel in ["gpt-3.5-turbo",'claude-3-sonnet','gemini-pro']:
        infile = data_folder + "closest/closest_" + whichmodel + "_" + promptversion + ".csv"
        dg = pd.read_csv(infile)
        print(len(dg))
        label_counts = dg['closest_label'].value_counts()
        label_freq = dg['closest_label'].value_counts(normalize=True)
        for i, row in sel.iterrows():
            #print(row['name'])
            id1 = row['label']
            id2 = row['label2']
            try:
                freq = label_freq[id1]
                count = label_counts[id1]
            except:
                freq = 0
                count = 0
            if pd.notnull(id2):
                try:
                    freq2 = label_freq[int(id2)]
                    freq += freq2
                    count2 = label_counts[int(id2)]
                    count += count2
                except:
                    pass
            allrows.append([promptversion, whichmodel, row['name'], freq, count])


dt = pd.DataFrame(allrows, columns=['prompt','model','name','freq','count'])


# Sort dt by 'freq'
dt = dt.sort_values(by=['freq'], ascending=[ False])


model_list = dt['model'].unique()

colors = {'gpt-3.5-turbo': 'red', 'claude-3-sonnet': 'blue', 'gemini-pro': 'green'}
markers = ['o', 's', 'x'] 


for promptversion in ['v4','v5']:
    plt.figure(figsize=(20, 10))
    dt = dt[dt['prompt'] == promptversion]
    dt['name'] = pd.Categorical(dt['name'], categories=name_order, ordered=True)
    dt = dt.sort_values('name')
    

    for model, marker in zip(model_list, markers):
        subset = dt[dt['model'] == model]
        plt.scatter(subset['name'], subset['freq'], color=colors[model], marker=marker, s=100, label=model)
    
    plt.xlabel('')
    plt.ylabel('Frequency')
    plt.ylim(0, 0.3)
    plt.title('')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model')
    plt.tight_layout()
    curr_plot_file = plot_dir + "freq_" + promptversion + ".pdf"
    plt.savefig(curr_plot_file, format='pdf', bbox_inches='tight')
    plt.close()







