import os
import json
import glob
from eval_utils import read_evaluation_results, calculate_average_metrics, flatten_dict
def inplace_calculate_reward(prompt_data_list, reward_metric_names):
    """
    Calculate reward scores for each response in-place based on specified metrics.
    
    Args:
        prompt_data_list (list): List of prompt data dictionaries
        reward_metric_names (list): List of metric names to sum for the reward
    
    Returns:
        None (modifies prompt_data_list in-place)
    """
    # Process each prompt
    for prompt_data in prompt_data_list:
        # Process each response for this prompt
        for response in prompt_data.get("responses", []):
            # Skip invalid responses
            if not response.get("valid", False) or "metrics" not in response:
                continue
                
            # Calculate reward as sum of specified metrics
            reward = 0.0
            
            
            for metric_name in reward_metric_names:
                if metric_name in response["metrics"]:
                    metric_value = float(response["metrics"][metric_name])
                    reward += metric_value
                else:
                    assert False, f"Metric {metric_name} not found in response metrics"
                    
            # Store the calculated reward and which metrics were used
            response["metrics"]["reward"] = reward
            
    

def calculate_skewness(prompt_data_list):
    """
    Calculate skewness of metrics across prompts.
    For each metric, normalize values within each prompt,
    then calculate skewness of the combined distribution.
    
    Args:
        prompt_data_list (list): List of prompt data dictionaries
            
    Returns:
        dict: Dictionary mapping metric names to skewness values
    """
    import numpy as np
    from scipy import stats 
    # For skewness calculation: store normalized values for each metric
    normalized_metrics = {}
    
    # Process each prompt
    for prompt_data in prompt_data_list:
        # Get valid responses for this prompt
        valid_prompt_responses = [r for r in prompt_data.get("responses", []) 
                               if r.get("valid", False) and "metrics" in r]
        
        # If we have at least 2 responses, we can normalize metrics
        if len(valid_prompt_responses) >= 2:
            # First collect metrics for this prompt
            prompt_metrics = {}
            
            # Gather all metric values for this prompt
            for response in valid_prompt_responses:
                for metric_name, value in response["metrics"].items():
                    if metric_name not in prompt_metrics:
                        prompt_metrics[metric_name] = []
                    prompt_metrics[metric_name].append(float(value))
            
            # Normalize each metric within this prompt
            for metric_name, values in prompt_metrics.items():
                if len(values) >= 2:
                    mean = np.mean(values)
                    std = np.std(values)
                    
                    if std > 0:  # Avoid division by zero
                        normalized = [(v - mean) / std for v in values]
                        
                        if metric_name not in normalized_metrics:
                            normalized_metrics[metric_name] = []
                        normalized_metrics[metric_name].extend(normalized)
    
    # Calculate skewness for each metric
    skewness = {}
    for metric_name, values in normalized_metrics.items():
        if len(values) >= 3:  # Need at least 3 points for meaningful skewness
            skewness[metric_name] = float(stats.skew(values))
        else:
            skewness[metric_name] = 0.0
    
    return skewness
def extract_step_results(eval_root="./evaluation_results", target_step="step_00750"):
    """
    Extract evaluation metrics for a specific step from all runs.
    
    Args:
        eval_root (str): Root directory containing all evaluation runs
        target_step (str): Step directory name to extract results from
    
    Returns:
        dict: Dictionary of results organized by run and task
    """
    # Find all run directories (each run should be a separate folder in eval_root)
    run_dirs = []
    for item in os.listdir(eval_root):
        full_path = os.path.join(eval_root, item)
        if os.path.isdir(full_path):
            run_dirs.append(full_path)
    
    if not run_dirs:
        print(f"No run directories found in {eval_root}")
        return {}
    
    # Results structure
    results = {}
    
    # Process each run
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        print(f"Checking run: {run_name}")
        
        # Find step directory (could be at different levels)
        step_dir = None
        
        # Check if step is directly in the run dir
        direct_step = os.path.join(run_dir, target_step)
        if os.path.isdir(direct_step):
            step_dir = direct_step
        else:
            # Check if step is inside another directory level (e.g., model_name/step_00750)
            subdirs = [os.path.join(run_dir, d) for d in os.listdir(run_dir) 
                     if os.path.isdir(os.path.join(run_dir, d))]
            
            for subdir in subdirs:
                potential_step = os.path.join(subdir, target_step)
                if os.path.isdir(potential_step):
                    step_dir = potential_step
                    break
        
        if not step_dir:
            print(f"No {target_step} directory found in {run_dir}")
            continue
        
        print(f"Processing {run_name} - {target_step}")
        results[run_name] = {}
        
        # Use read_evaluation_results to load all task data
        task_data_by_name = read_evaluation_results(step_dir, load_images=False)
        
        # Calculate metrics for each task
        for task_name, prompt_list in task_data_by_name.items():
            print(f"  Processing task: {task_name} with {len(prompt_list)} prompts")
            
            # Use calculate_average_metrics to get the averages
            avg_metrics = calculate_average_metrics(prompt_list)
            skewness = calculate_skewness(prompt_list)
        
            # Add skewness to the metrics
            avg_metrics['skewness'] = skewness
        
            # Store the results
            results[run_name][task_name] = avg_metrics
            
    return results


def print_results_summary(results):
    """Print a clean summary of the results"""
    print("\n=== Results Summary ===\n")
    
    for run_name, run_data in results.items():
        print(f"Run: {run_name}")
        for task_name, task_data in run_data.items():
            print(f"  Task: {task_name}")
            print(f"  Valid responses: {task_data['valid_count']} / {task_data['total_count']} ({task_data['success_rate']:.2%})")
            
            print("  Metrics:")
            for metric_name, value in task_data['metrics'].items():
                print(f"    {metric_name}: {value:.4f}")
            
            print("  Diversity:")
            for div_name, value in task_data['diversity'].items():
                print(f"    {div_name}: {value:.4f}")
            
            # Add skewness display
            if 'skewness' in task_data:
                print("  Skewness:")
                for metric_name, value in task_data['skewness'].items():
                    print(f"    {metric_name}: {value:.4f}")
            
            print()
def extract_all_steps_results(eval_root="./evaluation_results", run_rewards = {}
                            #   , run_rewards = {'reward_siglipsmall': ['siglip_small'],
                            #                                                    'reward_sigliplongsmall': ['siglip_small'],
                            #                                                    'reward_clipsmall': ['clip_small'],
                            #                                                    'reward_cliplarge': ['clip_large'],
                            #                                                    'reward_siglipsmalldinopatch':[],
                            #                                                    'reward_sigliplarge': ['siglip_large'],
                            #                                                    'scale_reward_cliponly_siglip': ['siglip_small'],
                            #                                                    'data_svgonly': ['siglip_small'],
                            #                                                    'reward_siglipsmalldinolarge': ['siglip_large','dino_large'],
                            #                                                    'reward_siglipsmalldino': ['siglip_small','dino_small'],
                            #                                                    'reward_siglipsmalldinobase': ['siglip_small','dino_base'],
                            #                                                    'reward_siglipsmalldinogiant': ['siglip_small','dino_giant'],}
                            ):
    """
    Extract evaluation metrics for all steps from all runs.
    
    Args:
        eval_root (str): Root directory containing all evaluation runs
    
    Returns:
        dict: Dictionary of results organized by run, step, and task
    """
    import re
    
    # Find all run directories (each run should be a separate folder in eval_root)
    run_dirs = []
    for item in os.listdir(eval_root):
        full_path = os.path.join(eval_root, item)
        if os.path.isdir(full_path):
            run_dirs.append(full_path)
    
    if not run_dirs:
        print(f"No run directories found in {eval_root}")
        return {}
    
    # Results structure
    results = {}
    
    # Process each run
    for run_dir in run_dirs:
        
        
        run_name = os.path.basename(run_dir)
        reward_metric_names = run_rewards.get(run_name,[])
        print(f"Processing run: {run_name}")
        results[run_name] = {}
        
        # Find all step directories in this run (could be at different levels)
        step_dirs = []
        
        # Check for direct step directories in the run
        direct_steps = [d for d in os.listdir(run_dir) 
                      if os.path.isdir(os.path.join(run_dir, d)) and d.startswith("step_")]
        
        for step_dir in direct_steps:
            step_dirs.append((step_dir, os.path.join(run_dir, step_dir)))
        
        # Check for step directories in subdirectories
        if not direct_steps:
            subdirs = [os.path.join(run_dir, d) for d in os.listdir(run_dir) 
                     if os.path.isdir(os.path.join(run_dir, d))]
            
            for subdir in subdirs:
                subdir_steps = [d for d in os.listdir(subdir) 
                              if os.path.isdir(os.path.join(subdir, d)) and d.startswith("step_")]
                
                for step_dir in subdir_steps:
                    step_dirs.append((step_dir, os.path.join(subdir, step_dir)))
        
        if not step_dirs:
            print(f"No step directories found in {run_dir}")
            continue
        
        # Sort steps numerically
        step_dirs.sort(key=lambda x: int(re.search(r'step_(\d+)', x[0]).group(1)))
        
        # Process each step
        for step_name, step_path in step_dirs:
            print(f"  Processing {step_name}")
            
            # Extract step number
            step_num = int(re.search(r'step_(\d+)', step_name).group(1))
            
            # Use read_evaluation_results to load all task data
            task_data_by_name = read_evaluation_results(step_path, load_images=False)
            for task_name, prompt_list in task_data_by_name.items():
                inplace_calculate_reward(prompt_list, reward_metric_names)
                
            
            
            # Store results for this step
            results[run_name][step_name] = {}
            
            # Calculate metrics for each task
            for task_name, prompt_list in task_data_by_name.items():
                print(f"    Processing task: {task_name} with {len(prompt_list)} prompts")
                
                # Use calculate_average_metrics to get the averages
                avg_metrics = calculate_average_metrics(prompt_list)
                
                # Calculate skewness
                skewness = calculate_skewness(prompt_list)
                
                # Add skewness to the metrics
                avg_metrics['skewness'] = skewness
                
                # Store the results
                results[run_name][step_name][task_name] = avg_metrics
            
    return results


def test_extract_step_results():
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract evaluation metrics from steps')
    parser.add_argument('--eval-root', default='./evaluation_results', 
                        help='Root directory containing evaluation runs')
    parser.add_argument('--step', default=None,
                        help='Extract specific step only (e.g., "step_00750"), if omitted extracts all steps')
    parser.add_argument('--output', default='evaluation_summary.json',
                        help='Output file for the results summary')
    parser.add_argument('--flatten', action='store_true',
                        help='Flatten nested metrics for easier comparison')
    
    args = parser.parse_args()
    
    # Check whether to extract specific step or all steps
    if args.step:
        results = extract_step_results(args.eval_root, args.step)
        print_results_summary(results)
    else:
        results = extract_all_steps_results(args.eval_root)
        print(f"Extracted metrics from all steps. Saving to {args.output}...")
    
    # Save the results to a JSON file
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")
    
    # Flatten metrics if requested
    if args.flatten:
        flattened_results = {}
        
        if args.step:
            # Flatten results for single step
            for run_name, run_data in results.items():
                flattened_results[run_name] = {}
                for task_name, task_data in run_data.items():
                    flattened_results[run_name][task_name] = flatten_dict(task_data)
        else:
            # Flatten results for all steps
            for run_name, step_data in results.items():
                flattened_results[run_name] = {}
                for step_name, tasks in step_data.items():
                    flattened_results[run_name][step_name] = {}
                    for task_name, task_data in tasks.items():
                        flattened_results[run_name][step_name][task_name] = flatten_dict(task_data)
        
        # Save flattened results
        output_name = args.output.replace('.json', '_flat.json')
        with open(output_name, "w") as f:
            json.dump(flattened_results, f, indent=2)
        print(f"Flattened results saved to {output_name}")
        
if __name__ == "__main__":
    test_extract_step_results()
        
