"""
Visualization Script for Neural Network Experiments

This script visualizes results from training experiments with Fastfood and RKS projection layers.
It loads experiment data from JSON files, calculates statistics including confidence intervals,
and generates plots to compare performance across different configurations.

The script supports filtering by layer type (Fastfood or RKS), custom y-axis metrics,
and confidence interval visualization to analyze the reliability of results.

Results are saved as PNG files in the Plots directory, with automated naming based on
the input JSON filename.
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

def parse_all_args():
    """
    Parse command-line arguments for the visualization script.
    
    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="Plot test accuracy from pickle files.")
    parser.add_argument('--filepath', type=str, default="Results/ff_rks_l_lr.json", help="Path to json file.")
    parser.add_argument('--yaxis', type=str, default="Test Accuracies", help="What to plot on the y-axis.")
    parser.add_argument('--ylim', nargs='+', default=None, type=float, help='Upper and Lower limits for the plot.')
    parser.add_argument('--CI', action='store_true', help="Plot confidence intervals or not.")
    parser.add_argument('--rks', action='store_true', help='Plot Random Kitchen Sink trials')
    parser.add_argument('--ff', action='store_true', help='Plot Fastfood trials')
    return parser.parse_args()

def parse_json(filename, yaxis, args):
    """
    Parse JSON results file and generate plots with confidence intervals.
    
    Parameters
    ----------
    filename : str
        Path to the JSON file containing experiment results
    yaxis : str
        Metric to plot on the y-axis (e.g., "Test Accuracies", "Train Times Per Epoch")
    args : argparse.Namespace
        Parsed command-line arguments
        
    Returns
    -------
    None
        Results are saved as PNG files in the Plots directory
    """
    with open(filename, 'r') as file:
        data = json.load(file)
        alpha = 0.05
        
        CI = []
        names = []
        
        for trial in data:
            # Filter out RKS or FF runs if specified in arguments
            if args.ff and trial["Projection Layer"] == "Fastfood_Layer":
                continue
            if args.rks and trial["Projection Layer"] == "RKS_Layer":
                continue
            
            # Parse a name for the trial
            if args.filepath == "Results/learnrate.json":
                name = f"{trial['Projection Layer']} LR={trial['Projection Arguments']['lr']}"
                names.append(name)
            elif args.filepath == "Results/scale.json":
                if trial['Projection Arguments']['learn_G']:
                    name = f"{trial['Projection Layer']} Learnable, Scale={trial['Projection Arguments']['scale']}"
                else:
                    name = f"{trial['Projection Layer']} Non-Learnable, Scale={trial['Projection Arguments']['scale']}"
                names.append(name)
            else:
                name = f"{trial['Projection Layer']}"
                names.append(name)
            
            # Create a 95% confidence interval for the mean
            try:
                mat = np.array(trial[yaxis])
                mean = np.mean(mat, axis=0)
                sem = np.std(mat, axis=0, ddof=1) / np.sqrt(mat.shape[0])
                t = stats.t.ppf(1 - alpha/2, df=mat.shape[0] - 1)
                lower_ci = mean - t * sem
                upper_ci = mean + t * sem
                CI.append(np.vstack((mean, lower_ci, upper_ci)))
            except KeyError:
                raise ValueError(f"'{yaxis}' is not a saved metric. Available keys are: {trial.keys()}")
        
        # Generate output filename based on input filename
        base = os.path.basename(filename).replace('.json', '')
        outfile = f"Plots/{base}.png"
        plot(CI, names, outfile, args)

def plot(matrix, names, outfile=None, args=None):
    """
    Generate and save performance plots with optional confidence intervals.
    
    Parameters
    ----------
    matrix : list of numpy.ndarray
        List of matrices containing means and confidence intervals for each trial
    names : list of str
        Names of each trial for the legend
    outfile : str, optional
        Path where the plot should be saved
    args : argparse.Namespace
        Parsed command-line arguments
        
    Returns
    -------
    None
        Plot is displayed and optionally saved to file
    """
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    for i, table in enumerate(matrix):
        x = np.arange(1, table.shape[1]+1)
        sns.lineplot(x=x, y=table[0, :], marker='', label=names[i])
        if args.CI:
            plt.fill_between(x, table[1, :], table[2, :], alpha=0.3)
    
    plt.title(args.filepath)
    plt.xlabel('Epochs')
    plt.ylabel(args.yaxis)
    plt.xticks(x)
    if len(names) != 0:
        plt.legend()
    if args.ylim is not None:
        plt.ylim(args.ylim[0], args.ylim[1])
    if outfile is not None:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        plt.savefig(outfile, dpi=300)
    plt.show()

if __name__ == "__main__":
    args = parse_all_args()
    parse_json(args.filepath, args.yaxis, args)
