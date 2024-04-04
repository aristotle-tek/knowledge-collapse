#!/bin/bash


# python=3.12.2

ai_truncated_discounts=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
sigma_trs=(0.25 0.5 0.75 1.0)
lrs=(0.001 0.005 0.01 0.05 0.1)
n_sample_hists=(100)
n_rounds_generations=(3 5 10 20 110) # 110 = no generations since only 100

MAX_JOBS=3 # max processors you want to use


function run_with_limit {
    local filename="$1"; shift
    while [ $(jobs -r | wc -l) -ge "$MAX_JOBS" ]; do
        sleep 1
    done
    # Using tee to write to both stdout and a uniquely named file
    "$@" | tee -a "$filename" &
}

# Loop over the arrays, running simulations with each combination of parameter values
for ai_truncated_discount in "${ai_truncated_discounts[@]}"; do
    for sigma_tr in "${sigma_trs[@]}"; do
        for lr in "${lrs[@]}"; do
            for n_sample_hist in "${n_sample_hists[@]}"; do
                for n_rounds_generation in "${n_rounds_generations[@]}"; do
                    # Create a unique filename for each simulation run
                    unique_filename="logs/simlog_${ai_truncated_discount}_sigtr${sigma_tr}_lr${lr}_grnds${n_rounds_generation}.txt"
                    echo "Running simulation with ai_truncated_discount=$ai_truncated_discount sigma_tr=$sigma_tr lr=$lr n_sample_hist=$n_sample_hist n_rounds_generation=$n_rounds_generation" | tee -a "$unique_filename"
                    run_with_limit "$unique_filename" python simulation.py --ai_truncated_discount "$ai_truncated_discount" --sigma_tr "$sigma_tr" --lr "$lr" --n_sample_hist "$n_sample_hist" --n_rounds_generation "$n_rounds_generation"
                done
            done
        done
    done
done

wait
