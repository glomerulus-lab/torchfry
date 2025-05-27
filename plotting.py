"""
Simple Visualization Script for Neural Network Experiments

This script loads JSON experiment results, calculates average accuracies for each model or layer type,
and creates comparative plots. It handles both single and multiple trial runs and works with 
different JSON formats.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from scipy import stats

def parse_args():
    """
    Parse command-line arguments for the visualization script.
    
    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments
    """
    import argparse
    parser = argparse.ArgumentParser(description="Plot averaged neural network accuracies from JSON results")
    parser.add_argument('--filepath', '-f', type=str, required=True, 
                      help="Path to the JSON file with experiment results")
    parser.add_argument('--output', '-o', type=str, default=None, 
                      help="Path to save the output plot (optional)")
    parser.add_argument('--yaxis', '-y', type=str, default="Test Accuracies", 
                      choices=["Test Accuracies", "Train Accuracies", "Train Times Per Epoch"],
                      help="Metric to plot on the y-axis")
    parser.add_argument('--title', '-t', type=str, default=None,
                      help="Custom title for the plot")
    parser.add_argument('--ci', action='store_true',
                      help="Display confidence intervals in the plot")
    parser.add_argument('--ylim', nargs='+', default=None, type=float,
                      help='Upper and Lower limits for the y-axis')
    parser.add_argument('--exclude', nargs='+', default=[],
                      help='Models or layer types to exclude from the plot')
    
    return parser.parse_args()

def parse_json(filepath, yaxis, args):
    """
    Parse JSON results file and prepare data for plotting.
    
    Parameters
    ----------
    filepath : str
        Path to the JSON file containing experiment results
    yaxis : str
        Metric to plot on the y-axis (e.g., "Test Accuracies", "Train Accuracies")
    args : argparse.Namespace
        Parsed command-line arguments
        
    Returns
    -------
    tuple
        (trial_stats, trial_names) containing the processed data for plotting
    """
    with open(filepath, 'r') as file:
        data = json.load(file)
    
    alpha = 0.05  # For confidence interval calculations
    trials_data = {}
    
    # Handle different JSON formats (list or dict)
    if isinstance(data, list):
        # Process a list of trial dictionaries
        for trial in data:
            # Skip excluded models/layers
            if ('Projection Layer' in trial and trial['Projection Layer'] in args.exclude) or \
               ('Model' in trial and trial['Model'] in args.exclude):
                continue
            
            # Determine trial name
            if 'Model' in trial:
                name = trial['Model']
            elif 'Projection Layer' in trial:
                name_parts = [trial['Projection Layer']]
                
                # Add arguments to name if available
                if 'Projection Arguments' in trial:
                    proj_args = trial['Projection Arguments']
                    if 'lr' in proj_args:
                        name_parts.append(f"LR={proj_args['lr']}")
                    if 'scale' in proj_args:
                        name_parts.append(f"Scale={proj_args['scale']}")
                    if 'learn_G' in proj_args:
                        name_parts.append("Learnable" if proj_args['learn_G'] else "Non-Learnable")
                
                name = " ".join(name_parts)
            else:
                name = f"Trial {len(trials_data) + 1}"
            
            # Skip if this trial doesn't have the requested data
            if yaxis not in trial:
                print(f"Warning: No {yaxis} data found for {name}")
                continue
                
            # Get the metrics data
            metric_data = trial[yaxis]
            
            # Initialize container for this trial if not already present
            if name not in trials_data:
                trials_data[name] = []
                
            # Handle the metric data structure (single run vs multiple runs)
            if isinstance(metric_data[0], list):  # Multiple runs
                trials_data[name].extend(metric_data)
            else:  # Single run
                trials_data[name].append(metric_data)
                
    elif isinstance(data, dict):
        # Process a dict mapping model names to trial lists
        for model_name, trials in data.items():
            # Skip excluded models
            if model_name in args.exclude:
                continue
                
            trials_data[model_name] = []
            
            for trial in trials:
                # Skip if this trial doesn't have the requested data
                if yaxis not in trial:
                    continue
                    
                # Get the metrics data
                metric_data = trial[yaxis]
                
                # Handle the metric data structure (single run vs multiple runs)
                if isinstance(metric_data[0], list):  # Multiple runs
                    trials_data[model_name].extend(metric_data)
                else:  # Single run
                    trials_data[model_name].append(metric_data)
    
    # Calculate statistics for each trial
    trial_stats = []
    trial_names = []
    
    for name, runs in trials_data.items():
        # Convert to numpy array for calculations
        runs_array = np.array(runs)
        n_runs = runs_array.shape[0]
        
        # Add number of runs to the name
        trial_names.append(f"{name} (n={n_runs})")
        
        # Calculate mean and confidence intervals
        mean = np.mean(runs_array, axis=0)
        
        if args.ci and n_runs > 1:  # Only calculate CI if we have multiple runs and CI is requested
            sem = np.std(runs_array, axis=0, ddof=1) / np.sqrt(n_runs)
            t = stats.t.ppf(1 - alpha/2, df=n_runs - 1)
            lower_ci = mean - t * sem
            upper_ci = mean + t * sem
            trial_stats.append(np.vstack((mean, lower_ci, upper_ci)))
        else:
            # No CI or only one run, so just use the mean
            trial_stats.append(np.vstack((mean, mean, mean)))
    
    return trial_stats, trial_names

def plot_results(trial_stats, trial_names, args):
    """
    Generate and save performance plots with optional confidence intervals.
    
    Parameters
    ----------
    trial_stats : list of numpy.ndarray
        List of matrices containing means and confidence intervals for each trial
    trial_names : list of str
        Names of each trial for the legend
    args : argparse.Namespace
        Parsed command-line arguments
        
    Returns
    -------
    None
        Plot is displayed and optionally saved to file
    """
    # Set up the plotting style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Create a color palette
    palette = sns.color_palette("colorblind", n_colors=len(trial_stats))
    
    # Plot each trial
    for i, table in enumerate(trial_stats):
        x = np.arange(1, table.shape[1]+1)
        plt.plot(x, table[0, :], linewidth=2, label=trial_names[i], color=palette[i])
        
        if args.ci:
            plt.fill_between(x, table[1, :], table[2, :], alpha=0.2, color=palette[i])
    
    # Set title
    if args.title:
        plt.title(args.title, fontsize=16)
    else:
        title = f"Average {args.yaxis}"
        if args.filepath:
            title = f"{os.path.basename(args.filepath)} - {title}"
        plt.title(title, fontsize=16)
    
    # Add labels
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel(args.yaxis, fontsize=14)
    
    # Set y-axis limits if specified
    if args.ylim is not None:
        plt.ylim(args.ylim[0], args.ylim[1])
    
    # Add legend
    if trial_names:
        plt.legend(loc='lower right', fontsize=12)
    
    # Save if output path is provided
    if args.output:
        outfile = args.output
    else:
        base = os.path.basename(args.filepath).replace('.json', '')
        outfile = f"Plots/{base}_{args.yaxis.lower().replace(' ', '_')}.png"
    
    # Ensure the output directory exists
    if outfile is not None:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        plt.savefig(outfile, dpi=300)
        print(f"Plot saved to {outfile}")
    
    # Display the plot
    plt.show()

def main():
    """Main function to run the visualization script."""
    args = parse_args()
    trial_stats, trial_names = parse_json(args.filepath, args.yaxis, args)
    plot_results(trial_stats, trial_names, args)

if __name__ == "__main__":
    main()