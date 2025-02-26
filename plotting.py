import os
import pickle
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def load_pickle_files(paths):
    """Loads all pickle files from given paths (directory or individual files)."""
    data_list = []
    names = []
    
    for path in paths:
        if os.path.isdir(path):
            for filename in os.listdir(path):
                if filename.endswith(".pkl"):
                    file_path = os.path.join(path, filename)
                    with open(file_path, "rb") as f:
                        data = pickle.load(f)
                        matrix = np.array(data['performance']['test_accuracy'])
                        data_list.append(matrix)
                        names.append(file_path)
                        print(file_path)
                        # series = pd.concat([pd.Series(data["hyperparameters"]), pd.Series(data["performance"])])
                        # data_list.append(series)
        elif path.endswith(".pkl"):
            with open(path, "rb") as f:
                data = pickle.load(f)
                series = pd.concat([pd.Series(data["hyperparameters"]), pd.Series(data["performance"])])
                data_list.append(series)

    write_filename = "testing_performance/Wednesday_plots/extracted_data.csv"
    with open(write_filename, "w") as f:
        for i, matrix in enumerate(data_list):
            np.savetxt(f, matrix, delimiter=",", header=f"# {names[i]}", comments="")
            f.write("\n")  # Add blank line between matrices

    return None

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

def CI_plots(directory):
    # 
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for i, filename in enumerate(os.listdir(directory)):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            CI = np.genfromtxt(file_path, delimiter=',', skip_header=1)
            # TODO: seaborn plots
            x = np.arange(1, 11)
            sns.lineplot(x=x, y=CI[0], label=filename.split("-scale")[0], marker='o')
            plt.fill_between(x, CI[1], CI[2], alpha=0.2)
    plt.title('Performance compared with different learnables.')
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.xticks(x)
    plt.legend()
    plt.savefig(f"{directory}/figure.png", dpi=300)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot test accuracy from pickle files.")
    parser.add_argument('paths', type=str, nargs='+', help="Path(s) to pickle files or directories.")
    
    args = parser.parse_args()
    
    data_list = CI_plots(args.paths)
    
if __name__ == "__main__":
    CI_plots(r"testing_performance\Wednesday_plots\inner_folder")
