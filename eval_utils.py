import os
import json
import glob
from PIL import Image
def calculate_average_metrics(prompt_data_list):
    """
    Calculate average metrics from the values of batch_data["prompts"].
    
    Args:
        prompt_data_list (list): List of prompt data dictionaries from batch_data["prompts"].values()
            Each prompt data has the structure:
            {
                "caption": "...",
                "responses": [...],
                "diversities_by_model": {...},
                "diversity": float
            }
            
    Returns:
        dict: Dictionary containing average metrics and diversity:
            {
                "metrics": {
                    "clip_reward": float,
                    "dino_reward": float,
                    "total_reward": float,
                    
                    "clip_small": float,
                    "clip_large": float,
                    ...
                    
                },
                "diversity": {
                    "average": float,
                    "model_specific": {
                        "dino_small": float,
                        "dino_base": float,
                        ...
                    }
                },
                "valid_count": int,
                "total_count": int,
                "success_rate": float
            }
    """
    # Initialize counters and accumulators
    valid_count = 0
    total_count = 0
    
    # Metric accumulators
    metric_sums = {
    }
    
    # Track model-specific metrics
    model_specific_sums = {}
    
    # Track all diversity values
    diversity_sum = 0.0
    diversity_by_model = {}
    prompts_with_diversity = 0
    
    # Iterate through all prompts and responses
    for prompt_data in prompt_data_list:
        # Count the prompt if it has diversity
        if "diversity" in prompt_data and prompt_data["diversity"] > 0:
            diversity_sum += prompt_data["diversity"]
            prompts_with_diversity += 1
        
        # Accumulate model-specific diversity metrics
        if "diversities_by_model" in prompt_data:
            for model_name, value in prompt_data["diversities_by_model"].items():
                if model_name not in diversity_by_model:
                    diversity_by_model[model_name] = []
                diversity_by_model[model_name].append(value)
        
        # Process each response for this prompt
        for response in prompt_data["responses"]:
            total_count += 1
            
            # Only count valid responses with metrics
            if response.get("valid", False) and "metrics" in response:
                valid_count += 1
                
                
                
                # Accumulate model-specific metrics
                for metric_name, value in response["metrics"].items():
                    
                    
                    if metric_name not in metric_sums:
                        metric_sums[metric_name] = 0.0
                    
                    metric_sums[metric_name] += value
    
    # Calculate averages
    result = {
        "metrics": {
            k: metric_sums[k] / total_count if valid_count > 0 else 0.0 for k in metric_sums
        },
        "diversity": {
            "average": diversity_sum / prompts_with_diversity if prompts_with_diversity > 0 else 0.0,
           
        },
        "valid_count": valid_count,
        "total_count": total_count,
        "success_rate": valid_count / total_count if total_count > 0 else 0.0
    }
    
    
    
    # Add model-specific diversity averages
    for model_name, values in diversity_by_model.items():
        if values:
            result["diversity"][model_name] = sum(values)/ prompts_with_diversity if prompts_with_diversity > 0 else 0.0
        else:
            result["diversity"][model_name] = 0.0
    
    return result


def average_dictionaries(dict_list):
    """
    Given a list of dictionaries with identical structure, return a single dictionary
    with each value being the average of corresponding values across all dictionaries.
    
    Args:
        dict_list (list): List of dictionaries with identical structure
        
    Returns:
        dict: A dictionary with averaged values
    """
    if not dict_list:
        return {}
    
    if len(dict_list) == 1:
        return dict_list[0].copy()
    
    # Check that all dictionaries have the same structure
    first_dict = dict_list[0]
    all_keys = set(first_dict.keys())
    
    # Verify all dictionaries have the same keys
    for i, d in enumerate(dict_list[1:], 1):
        dict_keys = set(d.keys())
        if dict_keys != all_keys:
            missing = all_keys - dict_keys
            extra = dict_keys - all_keys
            print(f"Warning: Dictionary {i} has different structure than the first dictionary.")
            if missing:
                print(f"  Missing keys: {missing}")
            if extra:
                print(f"  Extra keys: {extra}")
            assert False, "Dictionaries have different structures"
            # Will only average the common keys
    
    # Initialize result dictionary
    result = {}
    
    # Process each key present in all dictionaries
    for key in first_dict:
        # Check if this key exists in all dictionaries
        if not all(key in d for d in dict_list):
            print(f"Warning: Key '{key}' not present in all dictionaries, skipping.")
            continue
        
        # For each key, collect the values from all dictionaries
        values = [d[key] for d in dict_list if key in d]
        
        # Process based on value type
        if isinstance(first_dict[key], dict):
            # For nested dictionaries, recursively average
            nested_dicts = [d[key] for d in dict_list if key in d]
            result[key] = average_dictionaries(nested_dicts)
            
        elif isinstance(first_dict[key], (int, float)):
            # For numeric values, compute average
            result[key] = sum(values) / len(values)
            
        elif isinstance(first_dict[key], list) and all(isinstance(v, (int, float)) for v in first_dict[key]):
            # For lists of numbers, check if all have the same length
            if all(len(v) == len(first_dict[key]) for v in values):
                # Average element-wise
                result[key] = [sum(item) / len(values) for item in zip(*values)]
            else:
                # If lists have different lengths, warn and use the first value
                print(f"Warning: Lists for key '{key}' have different lengths, using first list.")
                result[key] = first_dict[key]
                
        else:
            # For other types, just use the first value
            result[key] = first_dict[key]
    
    return result

def flatten_dict(nested_dict, parent_key='', sep='_'):
    """
    Flattens a nested dictionary by concatenating keys with underscores.
    
    Args:
        nested_dict (dict): The nested dictionary to flatten
        parent_key (str): The parent key used in recursion (default: '')
        sep (str): The separator between nested keys (default: '_')
        
    Returns:
        dict: A flattened dictionary where nested keys are combined with parent keys
    """
    flat_dict = {}
    
    for key, value in nested_dict.items():
        # Create the new key by combining the parent key with the current key
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        
        # If the value is a dictionary, recursively flatten it
        if isinstance(value, dict):
            # Recursively call flatten_dict with the nested dictionary
            nested_flat = flatten_dict(value, new_key, sep)
            # Update the flat_dict with the flattened nested dictionary
            flat_dict.update(nested_flat)
        else:
            # For non-dictionary values, directly add to the flattened dictionary
            flat_dict[new_key] = value
    
    return flat_dict

def read_evaluation_results(eval_dir, load_images=False):
    """
    Read evaluation results from a directory structure created by main_multigpus.
    
    Args:
        eval_dir (str): Path to the evaluation directory containing task subdirectories
        load_images (bool): If True, load images for each valid response and store under 'image' key
    
    Returns:
        dict: Dictionary mapping task names to lists of prompt data
    """
    
    
    # Initialize task results dictionary
    task_results = {}
    
    # Get all subdirectories (tasks) in the eval_dir
    task_dirs = [d for d in os.listdir(eval_dir) if os.path.isdir(os.path.join(eval_dir, d))]
    
    for task_name in task_dirs:
        task_dir = os.path.join(eval_dir, task_name)
        task_accumulated_results = []
        
        # Find all batch_info.json files for this task
        batch_info_files = glob.glob(os.path.join(task_dir, "batch_*/batch_info.json"))
        
        if not batch_info_files:
            print(f"Warning: No batch information found for task '{task_name}'")
            continue
            
        print(f"Found {len(batch_info_files)} batch files for task '{task_name}'")
        
        # Process each batch file
        for batch_idx, batch_file in enumerate(sorted(batch_info_files)):
            try:
                # Explicitly close file after reading
                with open(batch_file, 'r') as f:
                    batch_data = json.load(f)
                
                # Extract prompt data and add to accumulated results
                if "prompts" in batch_data:
                    # Convert string keys back to prompt data objects
                    prompt_data_list = list(batch_data["prompts"].values())
                    
                    # Load images if requested, but limit how many we process at once
                    if load_images:
                        for prompt_data in prompt_data_list:
                            for response in prompt_data.get("responses", []):
                                if response.get("valid", False) and "image_path" in response:
                                    try:
                                        # Resolve path relative to task directory
                                        image_path = os.path.join(task_dir, response["image_path"])
                                        if os.path.exists(image_path):
                                            # Load image and immediately close the file
                                            with Image.open(image_path) as img:
                                                # Make a copy to keep in memory after file is closed
                                                response["image"] = img.copy()
                                        else:
                                            print(f"Warning: Image file not found: {image_path}")
                                    except Exception as e:
                                        print(f"Error loading image: {e}")
                                    
                    task_accumulated_results.extend(prompt_data_list)
                else:
                    print(f"Warning: No prompts found in batch file {batch_file}")
                    
                # Garbage collect more aggressively when processing many files
                if batch_idx % 10 == 0:
                    import gc
                    gc.collect()
                    
            except Exception as e:
                print(f"Error reading batch file {batch_file}: {e}")
        
        # Store results for this task
        task_results[task_name] = task_accumulated_results
        print(f"Accumulated {len(task_accumulated_results)} prompt results for task '{task_name}'")
        
        if load_images:
            # Count loaded images
            image_count = sum(
                1 for prompt_data in task_accumulated_results
                for response in prompt_data.get("responses", [])
                if "image" in response
            )
            print(f"Loaded {image_count} images for task '{task_name}'")
    
    return task_results




if __name__ == "__main__":
    # Example usage
    eval_dir = "/home/chenyamei/codes/understand-r1-zero/evaluation_results/reward_siglipsmall/step_00030/"
    results = read_evaluation_results(eval_dir, load_images=True)
    
    for task, data in results.items():
        print(f"Task: {task}, Number of prompts: {len(data)}")
        avg_metrics = calculate_average_metrics(data)
        print(f"Average Metrics for {task}: {avg_metrics}")
        
        # Flatten and print the average metrics
        flat_avg_metrics = flatten_dict(avg_metrics)
        print(f"Flattened Average Metrics for {task}: {flat_avg_metrics}")