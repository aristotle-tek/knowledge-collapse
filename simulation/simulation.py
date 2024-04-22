import os
import numpy as np
from scipy.stats import truncnorm, norm
from scipy.stats import lognorm
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy.integrate import quad
import time
import re
import json


from utils import *

cwd = os.getcwd()

import argparse


parser = argparse.ArgumentParser(description="Simulation input parameters.")


parser.add_argument("--ai_truncated_discount", type=float, default=1.0,
                    help="Discount for AI truncated sample. 1=no discount, \
                    0.2= truncated sample only costs 20 pct of full sample.")
parser.add_argument("--n_individuals", type=int, default=25,
                    help="Number of individuals. Default is 25.")
parser.add_argument("--n_rounds_generation", type=int, default=5,
                    help="Number of rounds per generation. Default is 5.")
parser.add_argument("--sigma_tr", type=float, default=1.0,
                    help="Sigma for truncation, e.g. default is 1= 1 standard deviation.")
parser.add_argument("--n_sample_hist", type=int, default=100,
                    help="n samples used to calc public pdf")
parser.add_argument("--lr", type=float, default=1,
                    help="learning rate to update on value estimates")


args = parser.parse_args()


ai_truncated_discount = args.ai_truncated_discount
n_individuals = args.n_individuals
n_rounds_generation = args.n_rounds_generation
sigma_tr = args.sigma_tr
n_sample_memory = args.n_sample_hist
lr = args.lr

print(f"AI Truncated Discount: {ai_truncated_discount}")
print(f"Number of Individuals: {n_individuals}")
print(f"Number of Rounds per generation: {n_rounds_generation}")
print(f"Sigma for Truncation: {sigma_tr}")
print(f"sample memory: {n_sample_memory}")
print(f"Learning rate: {lr}")


#--------------
#  Parameters
#--------------


num_rounds = 100
t_degrees_freedom = 10
cost_percentile = 50
mu, sigma = 1, 0.5  # for log-normal distribution- alpha, theta

fullsample_cost = 10 # just for initial choice vector - ignored if too high since min samples.
infinite_approx_for_tails = 20

enforce_nonneg =  True # don't observe null results, failed attempts at innovation that don't lead to patents...

mean_init_normal = 1 # shift so that there is some value from tail & center samples
n_init_samples = 100
std_dev_init_t1 = 5
n_samples_init_t1 = 10000
nr_init_val_rounds = 3 # to provide initial value estimates
n_init_truncated_samples_t1 = 10
n_init_full_samples_t1 = 10


start_time = time.time()
starttime = re.sub(r'\.','', str(start_time)) # for ensuring unique save filename


min_samples=3 # if no samples from truncated or full, ensure some basis for next round.

curr_id = "model"

basefolder = cwd + "/results/"

if not os.path.exists(basefolder):
    os.mkdir(basefolder)


foldername = curr_id + "_sgtr" + re.sub(r"\.", "", str(sigma_tr)) + "_dsct" + re.sub(r"\.", "", str(ai_truncated_discount)) + \
     "_sampmem" + re.sub(r"\.", "", str(n_sample_memory)) + "_lr" + re.sub(r"\.", "", str(lr)) + \
     "_rgen" + str(n_rounds_generation)

foldername =  re.sub(r'[\.\/\\]', '_', foldername)
print("new foldername: ", foldername)

newfolder = basefolder + foldername

if os.path.exists(newfolder):
    print("new folder already exists, appending suffix...")
    newfolder += "_" + starttime[-7:]

print("using folder: ", newfolder)
os.mkdir(newfolder)

filepath_out = newfolder + "/"


params = {
    "ai_truncated_discount": ai_truncated_discount,
    "n_individuals": n_individuals,
    "t_degrees_freedom": t_degrees_freedom,
    "sigma_tr":sigma_tr,
    "mu": mu,
    "sigma": sigma,
    "min_samples": min_samples,
    "n_rounds_generation":n_rounds_generation,
    "n_sample_memory":n_sample_memory,
    "lr": lr
}


filepath_config = filepath_out + "config.json"


if os.path.exists(filepath_config):
    # ensure new file
    unique_name = "config_" + starttime[-7:] + ".json"
    filepath_config = re.sub(r"config\.json", unique_name, filepath_config)

with open(filepath_config, 'w') as json_file:
    json.dump(params, json_file, indent=4)



#---------------
# initialization
#---------------



rows = []
samples_list = []   
full_samples_list = []

history_avg_tr_innov = []
history_avg_full_innov = []




t_distribution_pdf = lambda x: t.pdf(x, t_degrees_freedom) # this is the "truth"


stdev_t_dist = np.sqrt(t_degrees_freedom/ (t_degrees_freedom-2))

thetas = draw_log_normal(mu, sigma, size=n_individuals )
print("Drawn log-normal samples for thetas:", thetas)



thetas_fileout = filepath_out + "thetas.csv"
thetas_df = pd.DataFrame(thetas, columns=['thetas'])

thetas_df.to_csv(thetas_fileout, index=False)
print("saved thetas to:\n", thetas_fileout)




# Initialize public pdf
public_pdf, samples = initialize_pdf("kde_normal_nonzeromean", (mean_init_normal, n_init_samples))


init_samp_fileout = filepath_out + "init_samples.csv"
init_sampdf = pd.DataFrame(samples, columns=['samples'])

init_sampdf.to_csv(init_samp_fileout, index=False)
print("saved init samples to:\n", init_samp_fileout)



samples_list.extend(samples)

""" ----- Initialize values for samples based on nr_init_val_rounds
To decide what to do the first round, people need to have an estimate 
of the value of a truncated or full sample.
We provide an estimate based on the forward-looking values if 
n_init_truncated_samples_t1 and n_init_full_samples_t1 are taken and the pdf updated over
nr_init_val_rounds rounds.

"""

n_truncated_samples = n_init_truncated_samples_t1 
n_full_samples = n_init_full_samples_t1


full = np.full(n_full_samples, 2)  # Array of 2's, repeated n_full_samples times
tr = np.full(n_truncated_samples, 1)  # Array of 1's, repeated n_truncated_samples times
init_choice_vector = np.concatenate((full, tr))



est_fullsamp_value, est_tr_value = calc_initial_expected_sample_vals(
    nr_init_val_rounds,  init_choice_vector, public_pdf, thetas, t_distribution_pdf,
    t_degrees_freedom, sigma_tr, enforce_nonneg, infinite_approx_for_tails)

value_estimates_fullsamp = np.mean(est_fullsamp_value)
value_estimates_truncated = np.mean(est_tr_value)


#--- check inputs ....just repeated...
value_estimates_fullsamp, value_estimates_truncated, history_avg_tr_innov, history_avg_full_innov = \
    calc_from_updated_value_estimates(
        est_tr_value, est_fullsamp_value, thetas, 
        est_tr_value, est_fullsamp_value,
        lr=1
    )



stdev = calculate_pdf_std(public_pdf, eval_range=10, eval_points=10000)
print("st dev curr public: %.3f" % stdev)


kl_div = calculate_approximate_kl_divergence(t_distribution_pdf, public_pdf, -10, 10, num_points=1000)
print("Init KL-divergence: %.3f" % kl_div)


hellinger_dist = calculate_hellinger_distance(t_distribution_pdf, public_pdf, -10, 10)
print("Init hellinger dist: %.3f" % hellinger_dist)


# initial state
rows.append([0, stdev, kl_div, hellinger_dist, np.NaN, np.NaN,  np.NaN, np.NaN, np.NaN, np.NaN ])



csv_fileout = filepath_out + 'res_df.csv'

if os.path.exists(csv_fileout):
    # ensure new file
    unique_name = "_" + starttime[-7:] + ".csv"
    csv_fileout = re.sub(r"\.csv", unique_name, csv_fileout)



#-- for round 1: Just need value_estimates_fullsamp, value_estimates_truncated
popular_belief_sigma = sigma_tr # init


public_stdev = stdev

for round_i in range(1, num_rounds+1):

    if round_i % n_rounds_generation == 0:
        popular_belief_sigma = update_truncation_limits(sigma_tr, stdev, stdev_t_dist)
        print("--> New sigma est: %.3f" % popular_belief_sigma)
        public_stdev = stdev
        print("--> New stdev est: %.3f" % public_stdev)


    curr_fullsample_cost = np.percentile(
        np.concatenate([value_estimates_fullsamp, value_estimates_truncated]), cost_percentile)

    choice_vector = decide_choice_from_values(
        value_estimates_fullsamp, value_estimates_truncated, curr_fullsample_cost, ai_truncated_discount)

    # (ignore tracking of payoff values for now for individuals)
    n_truncated_samples= sum([x==1 for x in choice_vector]) # for log
    n_full_samples= sum([x==2 for x in choice_vector])

    tr_samples, full_samples = draw_est_samples_based_on_choices(
        choice_vector, t_degrees_freedom, sigma_tr, public_stdev)

    all_new_samples = np.concatenate([tr_samples, full_samples])
    samples_list.extend(all_new_samples)
    full_samples_list.extend(all_new_samples)

    tr_innovations = calc_innov_for_samples(
        tr_samples, public_pdf,  t_distribution_pdf, infinite_approx_for_tails, enforce_nonneg)
    full_innovations = calc_innov_for_samples(
        full_samples, public_pdf,  t_distribution_pdf, infinite_approx_for_tails, enforce_nonneg)


    # the following does not update the pdf, just used to estimate values:
    if n_truncated_samples < min_samples:
        addtl_tr_samples = min_samples-n_truncated_samples
        supplem_tr_samples = draw_truncated_t(t_degrees_freedom, sigma_tr, addtl_tr_samples)
        supp_tr_innovations = calc_innov_for_samples(
            supplem_tr_samples, public_pdf,  t_distribution_pdf, infinite_approx_for_tails, enforce_nonneg)
        tr_innovations.extend(supp_tr_innovations)

    if n_full_samples <min_samples:
        addtl_full_samples = min_samples-n_full_samples
        supplem_full_samples = draw_t(t_degrees_freedom, addtl_full_samples)
        supp_full_innovations = calc_innov_for_samples(
            supplem_full_samples, public_pdf,  t_distribution_pdf, infinite_approx_for_tails, enforce_nonneg)
        full_innovations.extend(supp_full_innovations)


    # calc for the next round...
    value_estimates_fullsamp, value_estimates_truncated, history_avg_tr_innov, history_avg_full_innov = \
        calc_from_updated_value_estimates(
            tr_innovations, full_innovations, thetas, 
            history_avg_tr_innov, history_avg_full_innov,
            lr= lr
        )


    if n_sample_memory is not None:
        samples_list = samples_list[-n_sample_memory:] # keep only last `n_sample_memory`

    # update the pdf:
    public_pdf =  gaussian_kde(samples_list)

    # callbacks
    stdev = calculate_pdf_std(public_pdf, eval_range=10, eval_points=10000)
    kl_div = calculate_approximate_kl_divergence(t_distribution_pdf, public_pdf, -10, 10, num_points=1000)
    hellinger_dist = calculate_hellinger_distance(t_distribution_pdf, public_pdf, -10, 10)
    rows.append([round_i, stdev, kl_div, hellinger_dist, curr_fullsample_cost, popular_belief_sigma, 
        n_truncated_samples, n_full_samples, np.mean(value_estimates_fullsamp), np.mean(value_estimates_truncated)])

    print("%d Std: %.3f. KL: %.3f. Hell: %.3f. Vtrunc: %.4f. Vfull: %.4f #tr: %d  #full: %d currcost: %.3f" % \
        (round_i, stdev, kl_div, hellinger_dist, np.mean(value_estimates_truncated), \
        np.mean(value_estimates_fullsamp), n_truncated_samples, n_full_samples, curr_fullsample_cost))




df = pd.DataFrame(rows, columns = ['idx', 'std', 'kl', 'hell', 'fcost', \
    "popular_belief_sigma",  't_truncated_samps','n_fullsamps', 'avgval_full', 'avgval_trunc'])

df.to_csv(csv_fileout, index=False)
print("saved df to:\n", csv_fileout)



samp_fileout = re.sub(r"\.csv", "_samples.csv", csv_fileout)
sampdf = pd.DataFrame(full_samples_list, columns=['samples'])
sampdf.to_csv(samp_fileout, index=False)
print("saved samples to:\n", samp_fileout)




#-------------------------
print("done.")
#-------------------------



elapsed_time_seconds = time.time() - start_time
print(f"Time: {int(elapsed_time_seconds // 60)} min, {elapsed_time_seconds % 60:.2f} s")




