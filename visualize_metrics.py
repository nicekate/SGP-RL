import json
import matplotlib.pyplot as plt
import numpy as np
import re
import os

def load_evaluation_data(json_path):
    """Load evaluation data from a JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)
def visualize_metrics_over_steps(evaluation_data, output_path=None, task_name='coco', 
                               figsize=(18, 6), selected_metrics=None):
    """
    Plot metrics, diversity and skewness over steps for all runs.
    
    Args:
        evaluation_data (dict): The loaded JSON data
        output_path (str): Path to save the output plot
        task_name (str): Name of the task to plot ('coco' or 'SGP-Single-9k')
        figsize (tuple): Base figure size per run
        selected_metrics (list): List of specific metrics to plot, None plots all
    """
    # Count number of runs
    runs = list(evaluation_data.keys())
    n_runs = len(runs)
    
    if n_runs == 0:
        print("No runs found in the data.")
        return
    
    # Create figure with proper size
    fig, axes = plt.subplots(n_runs, 3, figsize=(figsize[0], figsize[1] * n_runs))
    
    # If there's only one run, wrap axes in a list to maintain indexing consistency
    if n_runs == 1:
        axes = [axes]
    
    # Process each run
    for i, run_name in enumerate(runs):
        run_data = evaluation_data[run_name]
        
        # Extract step numbers and sort them
        steps = list(run_data.keys())
        step_nums = [int(re.search(r'step_(\d+)', step).group(1)) for step in steps]
        pairs = sorted(zip(step_nums, steps))
        sorted_step_nums = [pair[0] for pair in pairs]
        sorted_steps = [pair[1] for pair in pairs]
        
        # Data containers
        metrics_data = {}
        diversity_data = {}
        skewness_data = {}
        
        # Extract data for each step
        for step, step_num in zip(sorted_steps, sorted_step_nums):
            if task_name in run_data[step]:
                step_data = run_data[step][task_name]
                
                # Extract metrics
                for metric_name, value in step_data.get('metrics', {}).items():
                    if selected_metrics is None or metric_name in selected_metrics:
                        if metric_name not in metrics_data:
                            metrics_data[metric_name] = {'values': [], 'steps': []}
                        metrics_data[metric_name]['values'].append(value)
                        metrics_data[metric_name]['steps'].append(step_num)
                
                # Extract diversity
                for div_name, value in step_data.get('diversity', {}).items():
                    if div_name not in diversity_data:
                        diversity_data[div_name] = {'values': [], 'steps': []}
                    diversity_data[div_name]['values'].append(value)
                    diversity_data[div_name]['steps'].append(step_num)
                
                # Extract skewness
                for skew_name, value in step_data.get('skewness', {}).items():
                    if selected_metrics is None or skew_name in selected_metrics:
                        if skew_name not in skewness_data:
                            skewness_data[skew_name] = {'values': [], 'steps': []}
                        skewness_data[skew_name]['values'].append(value)
                        skewness_data[skew_name]['steps'].append(step_num)
        
        # Plot metrics
        for metric_name, data in metrics_data.items():
            axes[i][0].plot(data['steps'], data['values'], marker='o', label=metric_name)
        
        axes[i][0].set_title(f"{run_name}: Metrics ({task_name})")
        axes[i][0].set_xlabel("Step")
        axes[i][0].set_ylabel("Value")
        axes[i][0].grid(True, alpha=0.3)
        if metrics_data:
            axes[i][0].legend(loc='best', fontsize='small')
        
        # Plot diversity
        for div_name, data in diversity_data.items():
            axes[i][1].plot(data['steps'], data['values'], marker='o', label=div_name)
        
        axes[i][1].set_title(f"{run_name}: Diversity ({task_name})")
        axes[i][1].set_xlabel("Step")
        axes[i][1].set_ylabel("Value")
        axes[i][1].grid(True, alpha=0.3)
        if diversity_data:
            axes[i][1].legend(loc='best', fontsize='small')
        
        # Plot skewness
        for skew_name, data in skewness_data.items():
            axes[i][2].plot(data['steps'], data['values'], marker='o', label=skew_name)
        
        axes[i][2].set_title(f"{run_name}: Skewness ({task_name})")
        axes[i][2].set_xlabel("Step")
        axes[i][2].set_ylabel("Value")
        axes[i][2].grid(True, alpha=0.3)
        if skewness_data:
            axes[i][2].legend(loc='best', fontsize='small')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    return fig



def test_visualize_metrics_over_steps():
    """
    # Basic usage - visualizes coco task metrics
    python visualize_metrics.py --input evaluation_summary.json --output metrics_visualization.png

    # Visualize SGP-Single-9k task
    python visualize_metrics.py --task SGP-Single-9k --output sgp_metrics_visualization.png

    # Only plot specific metrics
    python visualize_metrics.py --metrics clip_small clip_large siglip_small

    # Customize figure size
    python visualize_metrics.py --figsize-width 20 --figsize-height 8
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize evaluation metrics over steps')
    parser.add_argument('--input', default='evaluation_summary.json',
                        help='Input JSON file containing evaluation data')
    parser.add_argument('--output', default='metrics_visualization.png',
                        help='Output path for the visualization')
    parser.add_argument('--task', default='coco', choices=['coco', 'SGP-Single-9k'],
                        help='Task to visualize metrics for')
    parser.add_argument('--figsize-width', type=int, default=18,
                        help='Figure width')
    parser.add_argument('--figsize-height', type=int, default=6,
                        help='Figure height per run')
    parser.add_argument('--metrics', nargs='+',
                        help='Specific metrics to plot (e.g., clip_small clip_large)')
    
    args = parser.parse_args()
    
    # Load data
    data = load_evaluation_data(args.input)
    
    # Generate visualization
    visualize_metrics_over_steps(
        data, 
        output_path=args.output,
        task_name=args.task,
        figsize=(args.figsize_width, args.figsize_height),
        selected_metrics=args.metrics
    )
    
def print_step_as_csv(evaluation_data, step="step_00750", task_name='coco', print_skewness=False):
    """
    Print data for a specific step in CSV format with 4 decimal precision.
    Also calculates and includes average metrics.
    
    Args:
        evaluation_data (dict): The loaded JSON data
        step (str): Step name to extract (e.g., "step_00750")
        task_name (str): Name of the task to print data for
        print_skewness (bool): If True, also include skewness values in output
    """
    # Define ordered metrics to display
    ordered_metrics = [
        "clip_small",
        "clip_large",
        "siglip_small", 
        "siglip_large",
        "dino_small",
        "dino_base",
        "dino_large",
        "dino_giant",
        "reward",
    ]
    
    # Define metric groupings for averages
    clip_metrics = ["clip_small", "clip_large"]
    siglip_metrics = ["siglip_small", "siglip_large"]
    dino_metrics = ["dino_small", "dino_base", "dino_large", "dino_giant"]
    
    # Create column headers
    header = ["run_name"]
    
    # Add ordered metric columns
    for metric in ordered_metrics:
        header.append(f"metric_{metric}")
    
    # Add calculated average columns
    header.append("metric_clip_avg")
    header.append("metric_siglip_avg")
    header.append("metric_dino_avg")
    header.append("metric_avg")
    
    # Add just average diversity
    header.append("diversity_average")
    
    # Add skewness columns if requested
    if print_skewness:
        for metric in ordered_metrics:
            header.append(f"skewness_{metric}")
    
    # Add other standard columns
    header.extend(["valid_count", "total_count", "success_rate"])
    
    # Print header
    print(','.join(header))
    
    # Print data for each run
    for run_name, run_data in evaluation_data.items():
        if step not in run_data or task_name not in run_data[step]:
            continue
            
        step_data = run_data[step][task_name]
        row = [run_name]
        
        # Store metric values for average calculations
        metric_values = {}
        
        # Add metrics in specified order
        for metric in ordered_metrics:
            if 'metrics' in step_data and metric in step_data['metrics']:
                value = step_data['metrics'][metric]
                metric_values[metric] = value
                # Format floating point values to 4 decimal places
                if isinstance(value, float):
                    row.append(f"{value:.4f}")
                else:
                    row.append(str(value))
            else:
                row.append("")
        
        # Calculate and add clip_avg
        if all(m in metric_values for m in clip_metrics):
            clip_avg = sum(metric_values[m] for m in clip_metrics) / len(clip_metrics)
            row.append(f"{clip_avg:.4f}")
        else:
            row.append("")
        
        # Calculate and add siglip_avg
        if all(m in metric_values for m in siglip_metrics):
            siglip_avg = sum(metric_values[m] for m in siglip_metrics) / len(siglip_metrics)
            row.append(f"{siglip_avg:.4f}")
        else:
            row.append("")
        
        # Calculate and add dino_avg
        if all(m in metric_values for m in dino_metrics):
            dino_avg = sum(metric_values[m] for m in dino_metrics) / len(dino_metrics)
            row.append(f"{dino_avg:.4f}")
        else:
            row.append("")
        
        # Calculate and add overall average
        if metric_values:
            overall_avg = sum(metric_values.values()) / len(metric_values)
            row.append(f"{overall_avg:.4f}")
        else:
            row.append("")
        
        # Add only average diversity
        if 'diversity' in step_data and 'average' in step_data['diversity']:
            value = step_data['diversity']['average']
            # Format floating point values to 4 decimal places
            if isinstance(value, float):
                row.append(f"{value:.4f}")
            else:
                row.append(str(value))
        else:
            row.append("")
        
        # Add skewness values if requested
        if print_skewness:
            for metric in ordered_metrics:
                if 'skewness' in step_data and metric in step_data['skewness']:
                    value = step_data['skewness'][metric]
                    if isinstance(value, float):
                        row.append(f"{value:.4f}")
                    else:
                        row.append(str(value))
                else:
                    row.append("")
        
        # Add standard columns
        if 'valid_count' in step_data:
            row.append(str(step_data['valid_count']))  # Integer, no formatting
        else:
            row.append("")
            
        if 'total_count' in step_data:
            row.append(str(step_data['total_count']))  # Integer, no formatting
        else:
            row.append("")
            
        if 'success_rate' in step_data:
            value = step_data['success_rate']
            if isinstance(value, float):
                row.append(f"{value:.4f}")
            else:
                row.append(str(value))
        else:
            row.append("")
        
        # Print row
        print(','.join(row))    
    
def test_print_step_as_csv():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize evaluation metrics over steps')
    parser.add_argument('--input', default='evaluation_summary.json',
                        help='Input JSON file containing evaluation data')
    parser.add_argument('--output', default='metrics_visualization.png',
                        help='Output path for the visualization')
    parser.add_argument('--task', default='coco', choices=['coco', 'SGP-Single-9k'],
                        help='Task to visualize metrics for')
    parser.add_argument('--figsize-width', type=int, default=18,
                        help='Figure width')
    parser.add_argument('--figsize-height', type=int, default=6,
                        help='Figure height per run')
    parser.add_argument('--metrics', nargs='+',
                        help='Specific metrics to plot (e.g., clip_small clip_large)')
    parser.add_argument('--csv-only', action='store_true',
                        help='Only print CSV data without generating plots')
    parser.add_argument('--step', default='step_00750',
                        help='Step to use for CSV output')
    
    args = parser.parse_args()
    
    # Load data
    data = load_evaluation_data(args.input)
    
    # Print CSV data for step 750
    print("\n=== CSV Format Data for Step 750 ===\n")
    print_step_as_csv(data, step=args.step, task_name=args.task)
    
    # Generate visualization if not csv-only mode
    if not args.csv_only:
        visualize_metrics_over_steps(
            data, 
            output_path=args.output,
            task_name=args.task,
            figsize=(args.figsize_width, args.figsize_height),
            selected_metrics=args.metrics
        )
        

def print_as_csv(evaluation_data, task_name='coco', print_skewness=False):
    """
    Print data for all steps in CSV format with 4 decimal precision.
    Also calculates and includes average metrics.
    
    Args:
        evaluation_data (dict): The loaded JSON data
        task_name (str): Name of the task to print data for
        print_skewness (bool): If True, also include skewness values in output
    """
    # Define ordered metrics to display
    ordered_metrics = [
        "clip_small",
        "clip_large",
        "siglip_small", 
        "siglip_large",
        "dino_small",
        "dino_base",
        "dino_large",
        "dino_giant",
        "reward",
    ]
    
    # Define metric groupings for averages
    clip_metrics = ["clip_small", "clip_large"]
    siglip_metrics = ["siglip_small", "siglip_large"]
    dino_metrics = ["dino_small", "dino_base", "dino_large", "dino_giant"]
    
    # Create column headers
    header = ["run_name", "step"]  # Added step column
    
    # Add ordered metric columns
    for metric in ordered_metrics:
        header.append(f"metric_{metric}")
    
    # Add calculated average columns
    header.append("metric_clip_avg")
    header.append("metric_siglip_avg")
    header.append("metric_dino_avg")
    header.append("metric_avg")
    
    # Add just average diversity
    header.append("diversity_average")
    
    # Add skewness columns if requested
    if print_skewness:
        for metric in ordered_metrics:
            header.append(f"skewness_{metric}")
    
    # Add other standard columns
    header.extend(["valid_count", "total_count", "success_rate"])
    
    # Print header
    print(','.join(header))
    
    # Process each run
    for run_name, run_data in evaluation_data.items():
        # Extract step numbers and sort them
        steps = list(run_data.keys())
        step_nums = [int(re.search(r'step_(\d+)', step).group(1)) for step in steps if re.search(r'step_(\d+)', step)]
        pairs = sorted(zip(step_nums, steps))
        sorted_steps = [pair[1] for pair in pairs]
        
        # Print data for each step
        for step in sorted_steps:
            if task_name not in run_data[step]:
                continue
                
            step_data = run_data[step][task_name]
            row = [run_name, step]  # Add run name and step
            
            # Store metric values for average calculations
            metric_values = {}
            
            # Add metrics in specified order
            for metric in ordered_metrics:
                if 'metrics' in step_data and metric in step_data['metrics']:
                    value = step_data['metrics'][metric]
                    metric_values[metric] = value
                    # Format floating point values to 4 decimal places
                    if isinstance(value, float):
                        row.append(f"{value:.4f}")
                    else:
                        row.append(str(value))
                else:
                    row.append("")
            
            # Calculate and add clip_avg
            if all(m in metric_values for m in clip_metrics):
                clip_avg = sum(metric_values[m] for m in clip_metrics) / len(clip_metrics)
                row.append(f"{clip_avg:.4f}")
            else:
                row.append("")
            
            # Calculate and add siglip_avg
            if all(m in metric_values for m in siglip_metrics):
                siglip_avg = sum(metric_values[m] for m in siglip_metrics) / len(siglip_metrics)
                row.append(f"{siglip_avg:.4f}")
            else:
                row.append("")
            
            # Calculate and add dino_avg
            if all(m in metric_values for m in dino_metrics):
                dino_avg = sum(metric_values[m] for m in dino_metrics) / len(dino_metrics)
                row.append(f"{dino_avg:.4f}")
            else:
                row.append("")
            
            # Calculate and add overall average
            if metric_values:
                overall_avg = sum(metric_values.values()) / len(metric_values)
                row.append(f"{overall_avg:.4f}")
            else:
                row.append("")
            
            # Add only average diversity
            if 'diversity' in step_data and 'average' in step_data['diversity']:
                value = step_data['diversity']['average']
                # Format floating point values to 4 decimal places
                if isinstance(value, float):
                    row.append(f"{value:.4f}")
                else:
                    row.append(str(value))
            else:
                row.append("")
            
            # Add skewness values if requested
            if print_skewness:
                for metric in ordered_metrics:
                    if 'skewness' in step_data and metric in step_data['skewness']:
                        value = step_data['skewness'][metric]
                        if isinstance(value, float):
                            row.append(f"{value:.4f}")
                        else:
                            row.append(str(value))
                    else:
                        row.append("")
            
            # Add standard columns
            if 'valid_count' in step_data:
                row.append(str(step_data['valid_count']))
            else:
                row.append("")
                
            if 'total_count' in step_data:
                row.append(str(step_data['total_count']))
            else:
                row.append("")
                
            if 'success_rate' in step_data:
                value = step_data['success_rate']
                if isinstance(value, float):
                    row.append(f"{value:.4f}")
                else:
                    row.append(str(value))
            else:
                row.append("")
            
            # Print row
            print(','.join(row))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize evaluation metrics')
    parser.add_argument('--input', default='./evaluation_summary.json',
                        help='Input JSON file containing evaluation data')
    parser.add_argument('--output', default=None,
                        help='Output path for the visualization')
   
    
    args = parser.parse_args()
    
    # Load data
    data = load_evaluation_data(args.input)
    for task in ['coco', 'SGP-Single-9k']:
    
        print(f"\n=== CSV Format Data for {task} ===\n")
        print_as_csv(data, task_name=task)
    
    