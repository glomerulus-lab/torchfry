import os
import json
import pickle
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


def parse_json(filename, yaxis):
    with open(filename, 'r') as file:
        data = json.load(file)
        alpha = 0.05

        # initize lists to store confidence intervals and parsed names.
        CI = []
        names = []
        for trial in data:
            # Filter out RKS runs
            if trial["Projection Layer"] != "RKS_Layer":
                continue

            # Parse a name for the trial
            # name = f"{trial['Projection Layer']} LR={trial['Projection Arguments']['lr']}"
            if trial['Projection Arguments']['learn_G']:
                name = f"Learnable, Scale={trial['Projection Arguments']['scale']}"
            else:
                name = f"Non-Learnable, Scale={trial['Projection Arguments']['scale']}"
            names.append(name)

            # Create a 95% confidence interval for the mean.
            try:
                mat = np.array(trial[yaxis])
                mean = np.mean(mat, axis=0)
                var = np.var(mat, axis=0)
                t = stats.t.ppf(1-alpha/2, df=mat.shape[0])
                lower_ci = mean - t*var
                upper_ci = mean + t*var
                CI.append(np.vstack((mean, lower_ci, upper_ci)))
            except KeyError:
                print(f"{yaxis} is not a saved metric. Try: {trial.keys()}")
                exit()

        base = filename.split("/")[1].split(".")[0]
        outfile = f"Plots/{base}.png"
        plot(CI, names, outfile)


def plot(matrix, names, outfile=None):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for i, table in enumerate(matrix):
        x = np.arange(1, table.shape[1]+1)
        sns.lineplot(x=x, y=table[0,:], marker='', label= names[i])
        if args.CI == "True":
            plt.fill_between(x, table[1,:], table[2,:], alpha=0.3)

    plt.title(args.filepath)
    plt.xlabel('Epochs')
    plt.ylabel(args.yaxis)
    plt.xticks(x)
    plt.legend()
    if args.ylim is not None:
        plt.ylim(args.ylim[0], args.ylim[1])
    if outfile != None:
        plt.savefig(outfile, dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot test accuracy from pickle files.")
    parser.add_argument('--filepath', type=str, default="Results/ff_rks_l_lr.json", help="Path to json file.")
    parser.add_argument('--yaxis', type=str, default="Test Accuracies", help="What to plot on the y-axis.")
    parser.add_argument('--CI', type=str, default="True", help="Plot confidence intervals or not.")
    parser.add_argument('--ylim', nargs='+', default=None, type=int, help='Upper and Lower limits for the plot.')
    args = parser.parse_args()

    parse_json(args.filepath, args.yaxis)
