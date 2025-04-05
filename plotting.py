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
            # Parse a name for the trial
            name = f"{trial['Projection Layer']} LR={trial['Projection Arguments']['lr']}"
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
        plt.fill_between(x, table[1,:], table[2,:], alpha=0.3)

    plt.title(f'Performance')
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.xticks(x)
    # plt.ylim(82, 90)
    plt.legend()
    if outfile != None:
        plt.savefig(outfile, dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot test accuracy from pickle files.")
    parser.add_argument('--filepath', type=str, nargs='+', help="Path(s) to pickle files or directories.")
    parser.add_argument('--yaxis', type=str, default="Test Accuracies", help="What to plot on the y-axis.")
    args = parser.parse_args()

    parse_json("Results/ff_rks_l_lr.json", args.yaxis)
