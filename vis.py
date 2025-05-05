from eval_utils import *

def get_max_at_n(list_prompts, metric_name):
    """
    Calculate average maximum metric for increasing numbers of responses.
    
    For each k from 1 to maximum number of responses:
    - For each prompt, select the first k responses
    - Find the maximum value of the specified metric among these k responses
    - Average these maximum values across all prompts
    
    Args:
        list_prompts (list): List of prompt dictionaries returned by read_evaluation_results
        metric_name (str): Name of the metric to evaluate (e.g., 'clip_reward', 'dino_reward')
    
    Returns:
        dict: Dictionary mapping k values to the average maximum metric when considering k responses
    """
    # Find the maximum number of responses across all prompts
    max_responses = max(len(prompt.get("responses", [])) for prompt in list_prompts)
    
    # Initialize results dictionary
    result = {}
    
    # For each k from 1 to max_responses
    for k in range(1, max_responses + 1):
        prompt_max_values = []
        
        # For each prompt
        for prompt in list_prompts:
            responses = prompt.get("responses", [])
            first_k_responses = responses[:k]
            
            # Filter to valid responses with the metric
            valid_metrics = []
            for response in first_k_responses:
                if (response.get("valid", False) and 
                    "metrics" in response and 
                    metric_name in response["metrics"]):
                    valid_metrics.append(response["metrics"][metric_name])
            
            # If any valid metrics found, get the maximum
            if valid_metrics:
                prompt_max_values.append(max(valid_metrics))
        
        # Average the maximum values across all prompts
        if prompt_max_values:
            result[k] = sum(prompt_max_values) / len(prompt_max_values)
        else:
            result[k] = 0.0
    
    return result

def get_max_at_n_across_steps(eval_dir_root, task_name, metric_name):
    """
    For each step directory in the evaluation root, call get_max_at_n for the specified task.
    
    Args:
        eval_dir_root (str): Root directory containing step subdirectories
        task_name (str): Name of the task to analyze (e.g., "coco")
        metric_name (str): Name of the metric to evaluate
        
    Returns:
        dict: Dictionary with step numbers as keys and get_max_at_n results as values
        {
            30: {1: 0.75, 2: 0.82, ...},
            60: {1: 0.77, 2: 0.84, ...},
            ...
        }
    """
    import os
    import re
    
    # Find all step directories
    step_dirs = []
    for dir_name in os.listdir(eval_dir_root):
        if os.path.isdir(os.path.join(eval_dir_root, dir_name)) and dir_name.startswith("step_"):
            step_dirs.append(dir_name)
    
    # Sort step directories numerically
    step_dirs.sort(key=lambda x: int(re.search(r'step_(\d+)', x).group(1)))
    
    # Initialize results
    results_by_step = {}
    
    # Process each step directory
    for step_dir in step_dirs:
        # Extract step number
        step_match = re.search(r'step_(\d+)', step_dir)
        if not step_match:
            print(f"Warning: Could not extract step number from directory: {step_dir}")
            continue
        
        step_num = int(step_match.group(1))
        
        # Build path to step evaluation directory
        step_eval_dir = os.path.join(eval_dir_root, step_dir)
        
        # Read evaluation results for this step
        try:
            print(f"Processing step {step_num} from {step_eval_dir}")
            step_results = read_evaluation_results(step_eval_dir, load_images=False)
            
            # Check if the requested task exists
            if task_name in step_results:
                task_prompts = step_results[task_name]
                
                # Get max metrics for the requested metric
                max_metrics = get_max_at_n(task_prompts, metric_name)
                
                # Store the results
                results_by_step[step_num] = max_metrics
                print(f"  Added metrics for step {step_num}: {len(max_metrics)} data points")
            else:
                print(f"  Task '{task_name}' not found in step {step_num}")
        except Exception as e:
            print(f"  Error processing step {step_num}: {e}")
    
    return results_by_step


def plot_max_at_n_across_steps(step_metrics, metric_name, max_n=None, figsize=(10, 6), save_path=None, 
                              plot_diff=False, log_scale=False, plot_linear_fit=True, mark_intercepts=True):
    """
    Plot metrics across different steps on the same figure with optional linear regression lines.
    
    Args:
        step_metrics (dict): The dictionary returned by get_max_at_n_across_steps
        metric_name (str): Name of the metric being plotted (for title)
        max_n (int, optional): Maximum number of responses to plot. If None, all are plotted.
        figsize (tuple): Figure size as (width, height)
        save_path (str, optional): If provided, save the plot to this path
        plot_diff (bool): If True, plot differences from the first step instead of absolute values
        log_scale (bool): If True, use logarithmic scale for x-axis
        plot_linear_fit (bool): If True, add linear regression line for each step
        mark_intercepts (bool): If True, mark and label intercepts between regression lines
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from itertools import combinations
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set logarithmic scale for x-axis if requested
    if log_scale:
        ax.set_xscale('log')
    
    # Get sorted step numbers
    sorted_steps = sorted(step_metrics.keys())
    
    # Get baseline metrics from first step if plotting differences
    baseline_metrics = None
    if plot_diff and sorted_steps:
        first_step = sorted_steps[0]
        baseline_metrics = step_metrics[first_step]
    
    # Create color cycle for distinguishing lines
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_steps)))
    
    # Store regression lines for finding intercepts
    regression_lines = {}  # {step_num: (slope, intercept, is_log_fit)}
    
    # Plot each step's metrics
    for i, step_num in enumerate(sorted_steps):
        metrics = step_metrics[step_num]
        
        # Convert the metrics dict to lists of x and y values
        x_values = list(metrics.keys())  # Number of responses (1, 2, 3, ...)
        y_values = list(metrics.values())  # Corresponding metric values
        
        # If plotting differences, subtract baseline values
        if plot_diff and baseline_metrics:
            diff_y_values = []
            for x, y in zip(x_values, y_values):
                if x in baseline_metrics:
                    diff_y_values.append(y - baseline_metrics[x])
                else:
                    diff_y_values.append(y)
            
            y_values = diff_y_values
        
        # Limit to max_n if specified
        if max_n is not None:
            x_values = [x for x in x_values if x <= max_n]
            y_values = y_values[:len(x_values)]
        
        # Plot this step's data
        ax.plot(x_values, y_values, marker='o', markersize=4, 
                color=colors[i], linewidth=2, label=f"Step {step_num}")
        
        # Add linear regression line if requested
        if plot_linear_fit and len(x_values) > 1:
            # Use numpy for linear regression
            x_array = np.array(x_values)
            y_array = np.array(y_values)
            
            # If log scale, transform x values for fitting
            if log_scale:
                # Convert to log space for fitting
                log_x_array = np.log10(x_array)
                # Compute the linear regression on log-transformed x values
                slope, intercept = np.polyfit(log_x_array, y_array, 1)
                
                # For display and annotation purposes
                regression_lines[step_num] = (slope, intercept, True)  # Flag that this is a log fit
                
                # Create points for the fitted line in log space
                if max_n is not None:
                    x_fit_log = np.linspace(np.log10(min(x_values)), np.log10(max_n), 100)
                else:
                    x_fit_log = np.linspace(np.log10(min(x_values)), np.log10(max(x_values)), 100)
                
                y_fit = slope * x_fit_log + intercept
                x_fit = 10 ** x_fit_log  # Convert back to normal space for plotting
                
                # Plot the regression line
                ax.plot(x_fit, y_fit, '--', color=colors[i], linewidth=1.5, alpha=0.7, 
                       label=f"Fit (Step {step_num}): y = {slope:.4f}·log10(x) + {intercept:.4f}")
            else:
                # Standard linear fit
                slope, intercept = np.polyfit(x_array, y_array, 1)
                regression_lines[step_num] = (slope, intercept, False)  # Flag that this is a normal fit
                
                # Extend the line across the plot range
                if max_n is not None:
                    x_fit = np.array([min(x_values), max_n])
                else:
                    x_fit = np.array([min(x_values), max(x_values)])
                
                y_fit = slope * x_fit + intercept
                
                # Plot the regression line
                ax.plot(x_fit, y_fit, '--', color=colors[i], linewidth=1.5, 
                       alpha=0.7, label=f"Fit (Step {step_num}): y = {slope:.4f}x + {intercept:.4f}")
    
    # Mark intercepts between regression lines
    if mark_intercepts and plot_linear_fit and len(regression_lines) > 1:
        # Find and mark all intercepts between pairs of lines
        for (step_a, line_a), (step_b, line_b) in combinations(regression_lines.items(), 2):
            slope_a, intercept_a, is_log_a = line_a
            slope_b, intercept_b, is_log_b = line_b
            
            # Skip if different fit types (log vs linear)
            if is_log_a != is_log_b:
                continue
                
            # Check if lines are nearly parallel
            if abs(slope_a - slope_b) < 1e-10:
                continue
            
            # Calculate intersection point
            x_intersect = (intercept_b - intercept_a) / (slope_a - slope_b)
            
            # For log scale fits, x_intersect is in log space - convert back
            if is_log_a:
                y_intersect = slope_a * x_intersect + intercept_a
                x_intersect = 10 ** x_intersect
            else:
                y_intersect = slope_a * x_intersect + intercept_a
            
            # Only show intercepts within the visible plot range
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            
            # Note: If log scale is used, x_min and x_max will be in log space
            if log_scale:
                x_min = 10 ** x_min
                x_max = 10 ** x_max
            
            if x_min <= x_intersect <= x_max and y_min <= y_intersect <= y_max:
                # Mark the intersection point
                ax.plot(x_intersect, y_intersect, 'ro', markersize=6)
                ax.annotate(f"({x_intersect:.1f}, {y_intersect:.4f})",
                           xy=(x_intersect, y_intersect),
                           xytext=(10, 10), textcoords='offset points',
                           arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    
    # Add labels and title
    x_label = "Number of Responses (n)" + (" - Log Scale" if log_scale else "")
    ax.set_xlabel(x_label)
    
    # Different labels based on plot type
    if plot_diff:
        ax.set_ylabel(f"Δ Average Maximum {metric_name}")
        ax.set_title(f"Change in Maximum {metric_name} Relative to First Step")
        
        # Add a horizontal line at y=0 for reference in difference plots
        ax.axhline(y=0, color='red', linestyle='-', alpha=0.3, label="Baseline")
    else:
        ax.set_ylabel(f"Average Maximum {metric_name}")
        ax.set_title(f"Maximum {metric_name} at Different Response Counts")
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Set x-axis to integers if not using log scale
    if not log_scale:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Improve appearance
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        # Modify save path for different plot types
        modified_save_path = save_path
        if plot_diff:
            modified_save_path = modified_save_path.replace('.png', '_diff.png')
        if log_scale:
            modified_save_path = modified_save_path.replace('.png', '_log.png')
        if plot_linear_fit:
            modified_save_path = modified_save_path.replace('.png', '_linearfit.png')
        plt.savefig(modified_save_path, dpi=300, bbox_inches='tight')
    
    return fig



def plot_metric_histograms(eval_dir_root, task_name, metrics_list=None, normalize=False, 
                          bins=20, figsize=(10, 6), save_dir=None, display=True,
                          compare_steps=False):
    """
    Plot histograms for each metric at each step.
    
    Args:
        eval_dir_root (str): Root directory containing step subdirectories
        task_name (str): Name of the task to analyze (e.g., "coco")
        metrics_list (list, optional): List of metrics to plot. If None, all available metrics are plotted
        normalize (bool): If True, normalize metrics within each prompt by (metric_value-mean)/std
        bins (int): Number of bins for the histograms
        figsize (tuple): Figure size as (width, height) for individual plots
        save_dir (str, optional): Directory to save plots. If None, plots are not saved
        display (bool): Whether to display plots
        compare_steps (bool): If True, create plots comparing the same metric across different steps
        
    Returns:
        dict: Dictionary with step numbers as keys and dictionaries of figure objects as values
    """
    import os
    import re
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict
    
    # Find all step directories
    step_dirs = []
    for dir_name in os.listdir(eval_dir_root):
        if os.path.isdir(os.path.join(eval_dir_root, dir_name)) and dir_name.startswith("step_"):
            step_dirs.append(dir_name)
    
    # Sort step directories numerically
    step_dirs.sort(key=lambda x: int(re.search(r'step_(\d+)', x).group(1)))
    
    # Dictionary to store results and metric values across steps
    figures_by_step = {}
    all_step_metrics = defaultdict(lambda: defaultdict(list))  # Format: metric -> step -> values
    
    # Process each step directory
    for step_dir in step_dirs:
        # Extract step number
        step_match = re.search(r'step_(\d+)', step_dir)
        if not step_match:
            print(f"Warning: Could not extract step number from directory: {step_dir}")
            continue
        
        step_num = int(step_match.group(1))
        
        # Build path to step evaluation directory
        step_eval_dir = os.path.join(eval_dir_root, step_dir)
        
        # Read evaluation results for this step
        try:
            print(f"Processing step {step_num} from {step_eval_dir}")
            step_results = read_evaluation_results(step_eval_dir, load_images=False)
            
            # Check if the requested task exists
            if task_name in step_results:
                task_prompts = step_results[task_name]
                
                # Collect metrics from all responses
                metric_values = defaultdict(list)
                
                # First, determine which metrics are available if metrics_list is None
                if metrics_list is None:
                    available_metrics = set()
                    for prompt in task_prompts:
                        for response in prompt.get("responses", []):
                            if "metrics" in response and response.get("valid", False):
                                available_metrics.update(response["metrics"].keys())
                    metrics_list = list(available_metrics)
                    print(f"  Auto-detected metrics: {metrics_list}")
                
                # Now collect the metric values
                for prompt in task_prompts:
                    prompt_metrics = defaultdict(list)
                    
                    # Collect metrics for this prompt
                    for response in prompt.get("responses", []):
                        if not response.get("valid", False) or "metrics" not in response:
                            continue
                            
                        for metric in metrics_list:
                            if metric in response["metrics"]:
                                prompt_metrics[metric].append(response["metrics"][metric])
                    
                    # Normalize if requested
                    if normalize:
                        for metric in metrics_list:
                            values = prompt_metrics[metric]
                            if len(values) > 1:  # Need at least 2 values to normalize
                                mean = np.mean(values)
                                std = np.std(values)
                                if std > 0:  # Avoid division by zero
                                    normalized_values = [(v - mean) / std for v in values]
                                    prompt_metrics[metric] = normalized_values
                    
                    # Add to overall collection
                    for metric, values in prompt_metrics.items():
                        metric_values[metric].extend(values)
                        all_step_metrics[metric][step_num].extend(values)
                
                figures_by_step[step_num] = {}
                
                # Create individual plots for each metric
                for metric in metrics_list:
                    if not metric_values[metric]:
                        continue
                    
                    # Create figure
                    fig, ax = plt.subplots(figsize=figsize)
                    
                    # Create histogram
                    values = metric_values[metric]
                    ax.hist(values, bins=bins, alpha=0.75, color='skyblue', edgecolor='black')
                    
                    # Add labels and title
                    metric_label = f"{metric}" + (" (Normalized)" if normalize else "")
                    ax.set_xlabel(metric_label)
                    ax.set_ylabel("Frequency")
                    ax.set_title(f"Distribution of {metric_label} at Step {step_num}")
                    
                    # Add summary statistics
                    if values:
                        stats_text = (f"Mean: {np.mean(values):.3f}\n"
                                     f"Median: {np.median(values):.3f}\n"
                                     f"Min: {np.min(values):.3f}\n"
                                     f"Max: {np.max(values):.3f}")
                        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                               verticalalignment='top', horizontalalignment='right',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
                    
                    # Add grid
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                    # Improve appearance
                    plt.tight_layout()
                    
                    # Save if requested
                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                        norm_suffix = "_normalized" if normalize else ""
                        save_path = os.path.join(save_dir, f"hist_step{step_num}_{metric}{norm_suffix}.png")
                        plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    
                    # Store or display figure
                    if display:
                        plt.show()
                    else:
                        plt.close(fig)
                    
                    figures_by_step[step_num][metric] = fig
            else:
                print(f"  Task '{task_name}' not found in step {step_num}")
        except Exception as e:
            print(f"  Error processing step {step_num}: {e}")
    
    # Create comparison plots across steps if requested
    if compare_steps:
        for metric, step_values in all_step_metrics.items():
            if not step_values:
                continue
                
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Get all steps for this metric
            steps = sorted(step_values.keys())
            
            # Calculate common bins for consistent comparison
            all_values = []
            for step_num in steps:
                all_values.extend(step_values[step_num])
                
            if not all_values:
                continue
                
            hist_min = min(all_values)
            hist_max = max(all_values)
            hist_bins = np.linspace(hist_min, hist_max, bins + 1)
            
            # Plot histograms for each step with transparency
            colors = plt.cm.viridis(np.linspace(0, 1, len(steps)))
            for i, step_num in enumerate(steps):
                values = step_values[step_num]
                if values:
                    ax.hist(values, bins=hist_bins, alpha=0.5, 
                           color=colors[i], edgecolor='black', 
                           label=f"Step {step_num}")
            
            # Add labels and title
            metric_label = f"{metric}" + (" (Normalized)" if normalize else "")
            ax.set_xlabel(metric_label)
            ax.set_ylabel("Frequency")
            ax.set_title(f"Distribution of {metric_label} Across Steps")
            
            # Add legend
            ax.legend(loc='best')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Improve appearance
            plt.tight_layout()
            
            # Save if requested
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                norm_suffix = "_normalized" if normalize else ""
                save_path = os.path.join(save_dir, f"hist_compare_steps_{metric}{norm_suffix}.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            # Store or display figure
            if display:
                plt.show()
            else:
                plt.close(fig)
            
            # Store in a special "compare" key
            if 'compare' not in figures_by_step:
                figures_by_step['compare'] = {}
            figures_by_step['compare'][metric] = fig
    
    return figures_by_step


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    eval_dir_root = "/home/chenyamei/codes/understand-r1-zero/evaluation_results/reward_siglipsmall/"
    task_name = "coco"
    
    # Plot histograms with normalization
    plot_metric_histograms(
        eval_dir_root=eval_dir_root,
        task_name=task_name,
        metrics_list=["clip_small", "clip_large", "siglip_large", "siglip_small", "dino_small"],
        normalize=True,
        bins=30,
        save_dir="./histogram_plots",
        compare_steps=True
    )
    
    # Plot histograms without normalization
    plot_metric_histograms(
        eval_dir_root=eval_dir_root,
        task_name=task_name,
        metrics_list=["clip_small", "clip_large", "siglip_large", "siglip_small", "dino_small"],
        normalize=False,
        bins=30,
        save_dir="./histogram_plots",
        compare_steps=True
    )

# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
    
#     eval_dir_root = "/home/chenyamei/codes/understand-r1-zero/evaluation_results/old/reward_siglipsmall/"
#     task_name = "coco"
    
#     # You can also plot multiple metrics in separate figures
#     for metric in [ "siglip_small", "dino_small"]:
#         metrics = get_max_at_n_across_steps(eval_dir_root, task_name, metric)
        
#         # Standard plot with linear regression lines
#         plot_max_at_n_across_steps(
#             metrics, 
#             metric, 
#             max_n=None, 
#             save_path=f"BestatN_{metric}_progress.png",
#             plot_diff=True,
#             log_scale=True,
#             plot_linear_fit=True,
#             mark_intercepts=True
#         )
        
        

