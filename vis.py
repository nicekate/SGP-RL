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


def plot_max_at_n_across_steps(step_metrics, metric_name, max_n=None, figsize=(10, 6), save_path=None, plot_diff=False):
    """
    Plot metrics across different steps on the same figure.
    
    Args:
        step_metrics (dict): The dictionary returned by get_max_at_n_across_steps
        metric_name (str): Name of the metric being plotted (for title)
        max_n (int, optional): Maximum number of responses to plot. If None, all are plotted.
        figsize (tuple): Figure size as (width, height)
        save_path (str, optional): If provided, save the plot to this path
        plot_diff (bool): If True, plot differences from the first step instead of absolute values
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get sorted step numbers
    sorted_steps = sorted(step_metrics.keys())
    
    # Get baseline metrics from first step if plotting differences
    baseline_metrics = None
    if plot_diff and sorted_steps:
        first_step = sorted_steps[0]
        baseline_metrics = step_metrics[first_step]
    
    # Create color cycle for distinguishing lines
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_steps)))
    
    # Plot each step's metrics
    for i, step_num in enumerate(sorted_steps):
        metrics = step_metrics[step_num]
        
        # Convert the metrics dict to lists of x and y values
        x_values = list(metrics.keys())  # Number of responses (1, 2, 3, ...)
        y_values = list(metrics.values())  # Corresponding metric values
        
        # If plotting differences, subtract baseline values
        if plot_diff and baseline_metrics:
            # Create new y_values that are differences from baseline
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
    
    # Add labels and title
    ax.set_xlabel("Number of Responses (n)")
    
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
    
    # Set x-axis to integers
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Improve appearance
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        # Modify save path for difference plots
        if plot_diff:
            save_path = save_path.replace('.png', '_diff.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    eval_dir_root = "/home/chenyamei/codes/understand-r1-zero/evaluation_results/reward_siglipsmall/"
    task_name = "coco"
    
    # Get metrics across steps
    
    
    # You can also plot multiple metrics in separate figures
    for metric in ["clip_small","clip_large", "siglip_large", "siglip_small","dino_small"]:
        metrics = get_max_at_n_across_steps(eval_dir_root, task_name, metric)
        plot_max_at_n_across_steps(
            metrics, 
            metric, 
            max_n=None, 
            save_path=f"BestatN_{metric}_progress.png",
            plot_diff=True
        )


