import json
import numpy as np
import subprocess
import os
import argparse
import src.config, src.utils
import re

def run_baseline(baseline, seed):
    """Run the baseline with a given seed."""
    # Replace this with your actual command to run the baseline
    command = f"python baselines/{baseline}.py --p {config_path} --d {data_path} --s {seed}"
    
    # Run the command and capture the output (assumed to be the result)
    result = subprocess.check_output(command, shell=True).decode("utf-8").strip()

    matches = re.findall(r'[^|]*$', result)

    # Convert the last three matches to integers
    real_results = [(float(match)) for match in matches[0].split()]

    # Convert the result to float (assuming it's a single numerical output)
    return real_results

def main(config):
    
    seeds = config["seeds"]
    baselines = config["baselines"]
    output_dir = config["output_dir"]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Loop through each baseline
    for baseline in baselines:
        dbis = []
        chis = []
        scs = []
        
        # Run baseline for each seed
        for seed in seeds:
            result = run_baseline(baseline, seed)
            dbis.append(result[0])
            chis.append(result[1])
            scs.append(result[2])

            print(f"Baseline: {baseline}, Seed: {seed}, Result: {result}")
        
        # Calculate average and standard deviation
        try:
            avg_dbi_result = np.mean(dbis)
            std_dbi_result = np.std(dbis)
        except:
            avg_dbi_result = 0
            std_dbi_result = 0

        try:
            avg_chi_result = np.mean(chis)
            std_chi_result = np.std(chis)
        except:
            avg_chi_result = 0
            std_chi_result = 0

        try:
            avg_scs_result = np.mean(scs)
            std_scs_result = np.std(scs)
        except:
            avg_scs_result = 0
            std_scs_result = 0

        
        
        # Save results to file
        output_file = os.path.join(output_dir, f"{baseline}_results.txt")
        with open(output_file, "w") as f:
            f.write(f"Baseline: {baseline}\n")
            f.write(f"Seeds: {seeds}\n")
            
            f.write(f"Average DBI: {avg_dbi_result:.2f}\n")
            f.write(f"Average CHI: {avg_chi_result:.2f}\n")
            f.write(f"Average SC: {avg_scs_result:.2f}\n")

            f.write(f"Standard Deviation DBI: {std_dbi_result:.2f}\n")
            f.write(f"Standard Deviation CHI: {std_chi_result:.2f}\n")
            f.write(f"Standard Deviation SC: {std_scs_result:.2f}\n")

        
        print(f"Saved results for {baseline} to {output_file}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Start All Baselines training.')
    parser.add_argument('-p', '--params_path', required=False, type=str,
                        help='params path with config.yml file',
                        default='configs/harthconfig.yml')
    parser.add_argument('-d', '--dataset_path', required=False, type=str,
                        help='path to dataset.', default='data/harth')
    
    args = parser.parse_args()
    config_path = args.params_path
    data_path = args.dataset_path

    # Read config
    config = src.config.Config(config_path)
    main(config)