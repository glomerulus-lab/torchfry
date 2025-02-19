import os
import pickle
import pandas as pd
import argparse
import matplotlib.pyplot as plt

def load_pickle_files(paths):
    """Loads all pickle files from given paths (directory or individual files)."""
    data_list = []
    
    for path in paths:
        if os.path.isdir(path):
            for filename in os.listdir(path):
                if filename.endswith(".pkl"):
                    file_path = os.path.join(path, filename)
                    with open(file_path, "rb") as f:
                        data = pickle.load(f)
                        data_list.append(data)
        elif path.endswith(".pkl"):
            with open(path, "rb") as f:
                data = pickle.load(f)
                data_list.append(data)

    return data_list

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

def main():
    parser = argparse.ArgumentParser(description="Plot test accuracy from pickle files.")
    parser.add_argument('paths', type=str, nargs='+', help="Path(s) to pickle files or directories.")
    
    args = parser.parse_args()
    
    data_list = load_pickle_files(args.paths)
    plot_test_accuracies(data_list)

if __name__ == "__main__":
    main()
