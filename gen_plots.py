
import numpy as np
import os
import glob
import json
import pandas as pd
import re

import matplotlib.pyplot as plt



def find_and_load_config_files(input_folder):
    # to get a list of simulations
    pattern = os.path.join(input_folder, "*", "config.json")
    
    config_files = glob.glob(pattern)
    config_dicts = []
    
    filepaths = []
    for file_path in config_files:
        filepaths.append(file_path)
        with open(file_path, 'r') as file:
            config = json.load(file)
            config_dicts.append(config)
    
    df = pd.DataFrame(config_dicts)
    df['filepath'] = filepaths
    
    return df



cwd = os.getcwd()

input_folder = cwd + "/results/" 

analysis_folder = cwd + "/analysis/"


df = find_and_load_config_files(input_folder)



n_full_samps = []
n_tr_samps = []
means = []


for row_i in range(0, len(df)):
    print("\n----------\n")
    curr = df.iloc[row_i]
    print(curr['filepath'])
    csv_filep = re.sub(r'config\.json', 'res_df.csv', curr["filepath"])
    cdf = pd.read_csv(csv_filep)
    cdf2 = cdf[pd.notnull(cdf['t_truncated_samps'])] # remove the first (NA) row.
    tr_samps = int(sum(cdf2['t_truncated_samps']))
    n_tr_samps.append(tr_samps)
    full_samps = int(sum(cdf2['n_fullsamps']))
    n_full_samps.append(full_samps)
    mn_std = np.mean(cdf['hell'][-5:])
    print(mn_std)
    means.append(mn_std)


df['mean_last5_hell'] = means


#-----------------------
#  discount - lr
#-----------------------

df.groupby(['ai_truncated_discount','lr'])['mean_last5_hell'].mean()



grouped = df.groupby(['ai_truncated_discount', 'lr'])['mean_last5_hell'].mean().reset_index()

pivot_df = grouped.pivot(index='lr', columns='ai_truncated_discount', values='mean_last5_hell')
pivot_df.plot(kind='line', figsize=(6.5, 6.5))  # marker='o'
plt.xlabel('Learning Rate')
plt.ylabel('Hellinger Distance')
plt.title('')# Distance by Learning Rate and Discounts')
plt.legend(title='Discount Factor')
plt.tight_layout()


disc_plot_file = analysis_folder + "helldist_disc_lr.pdf"
plt.savefig(disc_plot_file, format='pdf', bbox_inches='tight')
plt.close()



#-----------------------
# discount -- sigma_tr
#-----------------------



grouped = df.groupby(['ai_truncated_discount', 'sigma_tr'])['mean_last5_hell'].mean().reset_index()

pivot_df = grouped.pivot(index='sigma_tr', columns='ai_truncated_discount', values='mean_last5_hell')
pivot_df.plot(kind='line', figsize=(6.5, 6.5))  # marker='o'
plt.xlabel('Truncation')
plt.ylabel('Hellinger Distance')
plt.title('')
plt.legend(title='Discount Factor')
plt.tight_layout()

disc_plot_file = analysis_folder + "helldist_disc_sigmatr.pdf"
plt.savefig(disc_plot_file, format='pdf', bbox_inches='tight')
plt.close()




#----------------------------
#  lr - n-rounds-gen
#----------------------------



grouped = df.groupby(['lr', 'n_rounds_generation'])['mean_last5_hell'].mean().reset_index()
pivot_df = grouped.pivot(index='lr', columns='n_rounds_generation', values='mean_last5_hell')


pivot_df.plot(kind='line', figsize=(6.5, 6.5))  # Assuming pivot_df is already defined
plt.xlabel('Learning Rate')
plt.ylabel('Hellinger Distance')
plt.title('')
handles, labels = plt.gca().get_legend_handles_labels()
labels = [label if label != '110' else '(No generations)' for label in labels]
plt.legend(handles, labels, title='Rounds per generation')
plt.tight_layout()



disc_plot_file = analysis_folder + "helldist_lr_nrounds.pdf"
plt.savefig(disc_plot_file, format='pdf', bbox_inches='tight')
plt.close()



#----------------------------
#  plot pdfs
#----------------------------

nsamples=300

h_disc = {}

samples = []


df.reset_index(drop=True, inplace=True)


df2 = df[df.sigma_tr==.75]
df2 = df2[df2.lr==.05]
df2 = df2[df2.n_rounds_generation <=20]

df2.reset_index(drop=True, inplace=True)

for row_i in range(len(df2)):
    print("\n----------\n")
    curr = df2.iloc[row_i]
    print(curr['filepath'])
    curr_disc = curr.ai_truncated_discount
    
    try:
        csv_filep = re.sub(r'config\.json', 'res_df_samples.csv', curr["filepath"])
        cdf = pd.read_csv(csv_filep)
        
        curr_samps = cdf['samples'][-nsamples:]
        samples.append(curr_samps)
        if curr_disc not in h_disc:
            h_disc[curr_disc] = [curr_samps]
        else:
            h_disc[curr_disc].append(curr_samps)
    except Exception as e:
        print(f"Error processing row {row_i}: {e}")
        samples.append([])
        continue

df2['samples'] = samples


metadata = []
pdfs = []
x_grid = np.linspace(-1.5, 1.5, 1000)
t_degrees_freedom = 10

for discount in [0.2, 0.4, 0.6, 0.8, 1.0]:
    dcurr = df2[df2.ai_truncated_discount==discount]
    sampsC = []
    for i, row in dcurr.iterrows():
        sampsC.extend(row['samples'])

    kde = gaussian_kde(sampsC)

    pdfs.append(kde(x_grid))
    metadata.append([f"Discount %.2f"% discount])



plt.figure(figsize=(8, 6))

for i in range(len(pdfs)):
    try:
        metainfo = metadata[i]
        plt.plot(x_grid, pdfs[i], label=metainfo[0]) #f'Round {i + 1}')
    except:
        print("pdf not found, skipping...")

plt.plot(x_grid, t.pdf(x_grid, t_degrees_freedom), 'r-', lw=2, label='True t-distr, 10 d.f.')

plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.xlim(-1.5, 1.5)



disc_plot_file = analysis_folder + "kde_plot_lr0.05_sigmatr75_wgen.pdf"
plt.savefig(disc_plot_file, format='pdf', bbox_inches='tight')
plt.close()



#--------------------------------------------
# get particular example for simple interpretation.
#--------------------------------------------

df2 = df[df.sigma_tr==.75]
df2 = df2[df2.lr==.05]
df2 = df2[df2.n_rounds_generation ==10]


df2.index= range(len(df2))

df2_nodiscount = df2[df2.ai_truncated_discount==1.0]
df2_disc05 = df2[df2.ai_truncated_discount==0.5]


print("Disc 1.0: %.3f" % np.mean(df2_nodiscount.mean_last5_hell))
print("Disc 0.5: %.3f" % np.mean(df2_disc05.mean_last5_hell))
# Disc 1.0: 0.091
# Disc 0.5: 0.403


ratio = np.mean(df2_disc05.mean_last5_hell)/ np.mean(df2_nodiscount.mean_last5_hell)
print("ratio no discount to 0.5: %.3f" % ratio)
# ratio no discount to 0.5: 3.202


df2_disc08 = df2[df2.ai_truncated_discount==0.8]

print("Disc 0.8: %.3f" % np.mean(df2_disc08.mean_last5_hell))
# Disc 1.0: 0.091
# Disc 0.8: .216


ratio = np.mean(df2_disc08.mean_last5_hell)/ np.mean(df2_nodiscount.mean_last5_hell)
print("ratio no discount to 0.8: %.3f" % ratio)
# ratio no discount to 0.8: 2.366



#--------------------------------------------
# For fun - add animated plot
#--------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(0, 1.4)

line, = ax.plot(x_grid, pdfs[0], 'blue')
true_t_line, = ax.plot(x_grid, t.pdf(x_grid, t_degrees_freedom), 'r-', lw=2, label='True t-distr, 10 d.f.')

def update(frame):
    line.set_ydata(pdfs[frame])
    ax.legend([line, true_t_line], [metadata[frame], 'True t-distr, 10 d.f.'])
    
    return line, true_t_line

num_rounds = len(pdfs)

#frameslist = range(num_rounds)
frameslist =  [0]*2 + list(range(num_rounds)) + [num_rounds-1]*5

ani = FuncAnimation(fig, update, frames=frameslist, blit=False)  # blit=False for legend update

plt.title('Knowledge Collapse', loc='center', fontsize=12, color='black')#, pad=0)
plt.text(1, 1.02, 'Andrew Peterson, arXiv 2404.03502', fontsize=8, color='grey', ha='right', transform=ax.transAxes)
plt.xlabel('KDE estimate of public knowledge')
plt.ylabel('')
#plt.show()


