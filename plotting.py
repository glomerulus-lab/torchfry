import os
import pickle
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def generate_label(hyperparams):
    """Generate a label based on 'projection', 'learnable_gbs', and 'scale' rules."""
    projection = hyperparams.get("projection", "Unknown")
    scale = hyperparams.get("scale", "Unknown")
    
    if projection.lower() == "rks":
        learnable = "L" if hyperparams.get("learnable", False) else "NL"
        return f"{projection}_{learnable}_Scale{scale}"
    
    elif projection.lower() == "ff":
        learnable_gbs = hyperparams.get("learnable_gbs", [False, False, False])
        gbs_mapping = ["G", "B", "S"]
        gbs_label = "".join([gbs_mapping[i] for i in range(3) if learnable_gbs[i]])
        return f"FF_{gbs_label}_Scale{scale}" if gbs_label else f"FF_Scale{scale}"

    return f"{projection}_Scale{scale}"

def plot_test_accuracies(data_list):
    """Plots test accuracies for multiple pickle files with an interactive legend toggle."""
    plt.figure(figsize=(10, 6))

    lines = []
    labels = []
    
    for data in data_list:
        try:
            hyperparams = data.get("hyperparameters", {})
            performance = data.get("performance", {})
            test_accuracy = performance.get("test_accuracy", [])
            proj = hyperparams.get("projection", "Unknown")
            
            if not test_accuracy:
                print("Warning: Missing test accuracy data.")
                continue

            label = generate_label(hyperparams)
            line, = plt.plot(range(1, len(test_accuracy) + 1), test_accuracy, 
                             linestyle='-' if proj == "rks" else '-.', label=label)
            lines.append(line)
            labels.append(label)
        except KeyError as e:
            print(f"Skipping a file due to missing key: {e}")

    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy Over Epochs")
    legend = plt.legend(loc="best", fancybox=True, shadow=True)
    legend_lines = legend.get_lines()

    # Enable interactive legend clicking
    def on_legend_click(event):
        for i, leg_line in enumerate(legend_lines):
            if event.artist == leg_line:
                visible = not lines[i].get_visible()
                lines[i].set_visible(visible)
                leg_line.set_alpha(1.0 if visible else 0.2)
                plt.draw()
                break

    plt.gcf().canvas.mpl_connect("pick_event", on_legend_click)
    
    # Make legend entries pickable
    for leg_line in legend_lines:
        leg_line.set_picker(True)
    
    plt.grid(True)
    plt.show()

def load_pickle_files(directory):
    """Loads all pickle files from a folder. Then computes confidence 
    intervals for the test accuracy across all epochs."""
    
    names = []
    plotting_CI = []
    for filename in os.listdir(directory):
        names.append(filename)
        if filename.endswith(".pkl"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                for key, val in data['performance'].items():
                    print(key)
                    print(val)
                    print()
                    print()
                exit()
                matrix = np.array(data['performance']['test_accuracy']).T

                # use seaborn to compute a non-parametric CI for each epoch
                CI = []
                for epoch in matrix:
                    boot_ci = sns.algorithms.bootstrap(epoch, func=np.mean, n_boot=1000, ci=95)
                    ci_matrix = np.array([np.mean(epoch), np.percentile(boot_ci, 2.5), np.percentile(boot_ci, 97.5)])
                    CI.append(ci_matrix)
                CI = np.array(CI)
                plotting_CI.append(CI)
    return np.array(plotting_CI), names


def CI_plots(matrix, names, outfile= "figure.png"):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for i, table in enumerate(matrix):
        x = np.arange(1, table.shape[0]+1)
        label = names[i].split("-")[3].strip("[")
        sns.lineplot(x=x, y=table[:,0], label= label, marker='')
        plt.fill_between(x, table[:,1], table[:,2], alpha=0.2)
        
    title = names[0].split("-")[0]
    plt.title(f'{title} Nonlearnable performance')
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.xticks(x)
    # plt.ylim(82, 90)
    plt.legend()
    plt.savefig(outfile, dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot test accuracy from pickle files.")
    parser.add_argument('--filepath', type=str, nargs='+', help="Path(s) to pickle files or directories.")
    
    args = parser.parse_args()

    # Get all subdirectories
    subdirs = [f.path for f in os.scandir(*args.filepath) if f.is_dir()]

    for dir in subdirs:
        foo, names = load_pickle_files(dir)
        CI_plots(foo, names, outfile= dir + "_plot.png")
