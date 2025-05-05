# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import time
import os
from pathlib import Path

from dataset.registry import get_dataset_class
import fire
import numpy as np
from understand_r1_zero.dataset import render_svg_to_image
import vllm
from PIL import Image

from datasets import load_from_disk
from understand_r1_zero.svg_grader import calculate_eval_rewards, render_response_to_image, clip_name_dict, dino_name_dict
from torchvision import transforms
from understand_r1_zero.svg import (extract_svg, safe_svg_to_image)
from eval_utils import (calculate_average_metrics,average_dictionaries,flatten_dict,read_evaluation_results)

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])

def prepare_data_batched(max_eval_samples=100, batch_size=16):
    """
    Load datasets and create batched iterators.
    
    Args:
        max_eval_samples: Maximum number of samples to evaluate per dataset
        batch_size: Batch size for evaluation
        
    Returns:
        Dict mapping dataset names to batched data iterators
    """
    from torch.utils.data import DataLoader, Dataset
    
    class SimpleDataset(Dataset):
        def __init__(self, prompts, captions, images, image_type = "svg"):
            self.prompts = prompts
            self.captions = captions
            self.images = images
            self.image_type = image_type
            
        def __len__(self):
            return len(self.prompts)
            
        def __getitem__(self, idx):
            if self.image_type == "svg":
                return self.prompts[idx], self.captions[idx], self.images[idx], image_transform(render_svg_to_image(self.images[idx]))
            elif self.image_type == "image_path":
                return self.prompts[idx], self.captions[idx], self.images[idx], image_transform(Image.open(self.images[idx]).convert('RGB'))
    
    # Load raw datasets
    hq_svg_dataset = get_dataset_class('hq_svg_new')().load_dataset(
        'hq_svg_new',
        None
    )['test']
    
    coco_dataset = get_dataset_class("HuggingFaceM4/COCO")().load_dataset(
        "HuggingFaceM4/COCO",
        None,
        max_test_samples=max_eval_samples,
    )['test']
    
    # Create PyTorch datasets and dataloaders
    hq_svg_loader = DataLoader(
        SimpleDataset(
            hq_svg_dataset["prompt"],  # Input field for hq_svg
            hq_svg_dataset["solution"],       # Reference field
            hq_svg_dataset["svg"],               # Image field
            image_type="svg"                  # Image type
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    
    coco_loader = DataLoader(
        SimpleDataset(
            coco_dataset["prompt"],     # Input field for COCO
            coco_dataset["solution"],    # Reference field
            coco_dataset["image_path"],  # Reference field
            image_type="image_path"     # Image type
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    
    return {
        'SGP-Single-9k': {
            'dataloader': hq_svg_loader,
            'total_samples': len(hq_svg_dataset),
        },
        'coco': {
            'dataloader': coco_loader,
            'total_samples': len(coco_dataset),
        }
    }
    
def calculate_eval_rewards_with_diversity_batched(captions, model_responses_by_prompt, reference_images=None, 
                                                models_dict={'clip': ['clip'], 'dino': ['dino']},eval_dir=None, batch_idx=None):
    """
    Calculate rewards for SVG responses with batch processing for higher throughput.
    
    Args:
        captions (List[str]): List of text prompts/descriptions
        model_responses_by_prompt (List[List[str]]): List of k lists, each with n SVG responses
        reference_images (List[PIL.Image], optional): List of reference images
        models_dict (dict): Dictionary specifying which models to use
    
    Returns:
        dict: Dictionary containing rewards, diversity metrics, and additional information
    """
    import os
    import numpy as np
    import torch
    from torch.nn.functional import cosine_similarity
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    num_prompts = len(captions)
    save_images = eval_dir is not None and batch_idx is not None
    if save_images:
        batch_dir = os.path.join(eval_dir, f"batch_{batch_idx}")
        os.makedirs(batch_dir, exist_ok=True)
        batch_data = {
            "prompts": {},
            "metadata": {
                "batch_idx": batch_idx,
                "num_prompts": num_prompts,
                "timestamp": time.time()
            }
        }
    # Initialize results structure
    results = {
        "per_prompt": [],
        "sum_clip_reward": 0.0,
        "sum_dino_reward": 0.0,
        "sum_diversity": 0.0,
        "total_valid_count": 0,
        "overall_success_rate": 0.0,"model_specific_rewards": {}
    }
     # Initialize per-model reward tracking
    for clip_model in models_dict.get('clip', []):
        results["model_specific_rewards"][clip_model] = 0.0
    for dino_model in models_dict.get('dino', []):
        results["model_specific_rewards"][dino_model] = 0.0
    
    # Initialize empty lists for results
    for i in range(num_prompts):
        results["per_prompt"].append({
            "prompt": captions[i],
            "samples": [],
            "diversity": 0.0,
            "avg_clip_reward": 0.0,
            "avg_dino_reward": 0.0,
            "valid_count": 0,
        })
        # Initialize prompt data in the batch JSON if saving
        if save_images:
            batch_data["prompts"][str(i)] = {
                "caption": captions[i],
                "responses": []
            }
    
    # Initialize models
    clip_models = models_dict.get('clip', [])
    dino_models = models_dict.get('dino', [])
    clip_model_fns = [clip_name_dict[reward_model_name] for reward_model_name in clip_models if reward_model_name in clip_name_dict]
    dino_model_fns = [dino_name_dict[reward_model_name] for reward_model_name in dino_models if reward_model_name in dino_name_dict]
    
    # Step 1: Render all SVGs for all prompts (in batches if needed)
    all_rendered_images = []  # Will be a flattened list
    all_svg_infos = []  # Will be a flattened list
    prompt_indices = []  # Maps each rendered image to its prompt index
    response_indices = []  # Maps each rendered image to its response index within the prompt
    
    for prompt_idx in range(num_prompts):
        responses = model_responses_by_prompt[prompt_idx]
        if save_images:
            prompt_dir = os.path.join(batch_dir, f"prompt_{prompt_idx}")
            os.makedirs(prompt_dir, exist_ok=True)
        
        for resp_idx, response in enumerate(responses):
            rendered_image, info = render_response_to_image(response)
            if save_images:
                response_data = {
                    "index": resp_idx,
                    "text": response,
                    "valid": rendered_image is not None
                }
            if rendered_image is not None:
                all_rendered_images.append(rendered_image)
                all_svg_infos.append(info)
                prompt_indices.append(prompt_idx)
                response_indices.append(resp_idx)
                results["per_prompt"][prompt_idx]["valid_count"] += 1
                results["total_valid_count"] += 1
                
                # Save the image if requested
                if save_images:
                    # Create image path and save
                    image_path = os.path.join(prompt_dir, f"response_{resp_idx}.jpg")
                    if rendered_image.mode == 'RGBA':
                        rgb_image = Image.new('RGB', rendered_image.size, (255, 255, 255))
                        rgb_image.paste(rendered_image, mask=rendered_image.split()[3])  # 3 is the alpha channel
                        rendered_image = rgb_image
                    rendered_image.save(image_path)
                    
                    # Add image path to response data
                    response_data["image_path"] = os.path.relpath(image_path, eval_dir)
            # Add response data to batch data
            if save_images:
                batch_data["prompts"][str(prompt_idx)]["responses"].append(response_data)
    # Save batch JSON with prompt and response information
    if save_images:
        json_path = os.path.join(batch_dir, "batch_info.json")
        with open(json_path, "w") as f:
            json.dump(batch_data, f, indent=2)
        print(f"Saved batch info to {json_path}")
    
                
    
    # If no valid images, return empty results
    if not all_rendered_images:
        print("No valid SVGs rendered!")
        return results
     
    # Step 2: Calculate CLIP rewards for all valid rendered images at once
    all_captions = []
    for idx in prompt_indices:
        all_captions.append(captions[idx])
    
    for clip_model_name in clip_models:
        if clip_model_name in clip_name_dict:
            try:
                # Process all images at once with CLIP
                clip_model = clip_name_dict[clip_model_name]
                clip_scores = clip_model(all_captions, all_rendered_images)
                
                # Assign scores back to the right prompts and responses
                for i, (prompt_idx, resp_idx) in enumerate(zip(prompt_indices, response_indices)):
                    # Initialize sample if it doesn't exist yet
                    if resp_idx >= len(results["per_prompt"][prompt_idx]["samples"]):
                        while len(results["per_prompt"][prompt_idx]["samples"]) <= resp_idx:
                            results["per_prompt"][prompt_idx]["samples"].append({
                                "response": model_responses_by_prompt[prompt_idx][resp_idx],
                                "svg_info": all_svg_infos[i],
                                "formatted": all_svg_infos[i].get("formatted", False),
                                "clip_reward": 0.0,
                                "dino_reward": 0.0,
                                "total_reward": 0.0
                            })
                    
                    # Add CLIP reward
                    clip_reward = 1.0 - clip_scores[i]  # Convert distance to similarity
                    results["per_prompt"][prompt_idx]["samples"][resp_idx]["clip_reward"] += clip_reward
                    results["per_prompt"][prompt_idx]["samples"][resp_idx][clip_model_name] = clip_reward
                    results["per_prompt"][prompt_idx]["samples"][resp_idx]["total_reward"] += clip_reward
            except Exception as e:
                print(f"Error calculating CLIP scores with {clip_model_name}: {e}")
    
    # Step 3: Calculate DINO rewards if reference images exist
    if reference_images is not None:
        all_ref_images = []
        indices_with_refs = []
        rendered_with_refs = []
        
        for i, (prompt_idx, _) in enumerate(zip(prompt_indices, response_indices)):
            if reference_images[prompt_idx] is not None:
                all_ref_images.append(reference_images[prompt_idx])
                rendered_with_refs.append(all_rendered_images[i])
                indices_with_refs.append(i)
        if indices_with_refs:  # Only process if we have valid reference-render pairs
            for dino_model_name in dino_models:
                
                if dino_model_name in dino_name_dict:
                    try:
                        # Process batch with DINO
                        dino_model = dino_name_dict[dino_model_name]
                        dino_scores, all_features = dino_model(all_ref_images, rendered_with_refs, return_features=True)
                        
                        # Assign scores back to the right prompts and responses
                        for batch_idx, orig_idx in enumerate(indices_with_refs):
                            prompt_idx = prompt_indices[orig_idx]
                            resp_idx = response_indices[orig_idx]
                            
                            dino_reward = 1.0 - dino_scores[batch_idx]  # Convert distance to similarity
                            dino_feature = all_features[batch_idx]
                            results["per_prompt"][prompt_idx]["samples"][resp_idx]["dino_reward"] += dino_reward
                            results["per_prompt"][prompt_idx]["samples"][resp_idx]["total_reward"] += dino_reward
                            results["per_prompt"][prompt_idx]["samples"][resp_idx][dino_model_name] = dino_reward
                            if "dino_feature" not in results["per_prompt"][prompt_idx]["samples"][resp_idx]:
                                # Initialize feature storage if not present
                                results["per_prompt"][prompt_idx]["samples"][resp_idx]["dino_feature"] = {}
                            results["per_prompt"][prompt_idx]["samples"][resp_idx]["dino_feature"][dino_model_name] = dino_feature
                    except Exception as e:
                        print(f"Error calculating DINO scores with {dino_model_name}: {e}")
    
    # Step 4: Calculate diversity for each prompt using DINO features
    for prompt_idx in range(num_prompts):
        prompt_data = results["per_prompt"][prompt_idx]
        
        # Need at least 2 valid rendered images for diversity
        if prompt_data["valid_count"] < 2:
            prompt_data["diversity"] = 0.0
            prompt_data["diversity_by_model"] = {model: 0.0 for model in dino_models}
            continue
        
        # Get all valid samples for this prompt
        valid_samples = [s for s in prompt_data["samples"] if "dino_feature" in s]
        
        # Check if we already have features stored from the DINO scoring step
        if valid_samples and all("dino_feature" in sample for sample in valid_samples):
            # Calculate diversity using stored features
            diversity_by_model = {}
            
            for dino_model_name in dino_models:
                if dino_model_name in dino_name_dict:
                    try:
                        # Extract features for this model from all samples
                        features = [
                            sample["dino_feature"][dino_model_name] 
                            for sample in valid_samples 
                            if dino_model_name in sample.get("dino_feature", {})
                        ]
                        features = [f for f in features if f is not None]
                        if len(features) >= 2:
                            # Calculate pairwise similarities
                            pairwise_sims = []
                            for i in range(len(features)):
                                for j in range(i+1, len(features)):
                                    sim = cosine_similarity(
                                        features[i].unsqueeze(0),
                                        features[j].unsqueeze(0)
                                    ).item()
                                    pairwise_sims.append(sim)
                            
                            # Diversity is inverse of average similarity
                            avg_sim = np.mean(pairwise_sims) if pairwise_sims else 0
                            diversity = 1.0 - avg_sim
                            diversity_by_model[dino_model_name] = diversity
                        else:
                            diversity_by_model[dino_model_name] = 0.0
                    except Exception as e:
                        print(f"Error calculating diversity with stored features for {dino_model_name}, prompt {prompt_idx}: {e}")
                        diversity_by_model[dino_model_name] = 0.0
        else:
            assert False
        for sample in valid_samples:
            if "dino_feature" in sample:
                del sample["dino_feature"]
        import gc
        gc.collect()
        
        # Store diversity results by model
        prompt_data["diversity_by_model"] = diversity_by_model
        
        # Average diversity across DINO models
        prompt_data["diversity"] = np.mean(list(diversity_by_model.values())) if diversity_by_model else 0.0
        
        # Add to overall diversity metric
        results["sum_diversity"] += prompt_data["diversity"] if prompt_data["diversity"] else 0.0
    
    # Step 5: Calculate per-prompt and overall averages
    model_specific_sums = {model: [] for model in results["model_specific_rewards"]}
    
    for prompt_idx in range(num_prompts):
        prompt_data = results["per_prompt"][prompt_idx]
        if prompt_data["valid_count"] > 0:
            # Calculate average CLIP and DINO rewards for this prompt
            # Convert any tensors to float values first
            clip_rewards = []
            dino_rewards = []
            
            for sample in prompt_data["samples"]:
                # Handle CLIP rewards - convert tensor to float if needed
                for reward_model_name in results["model_specific_rewards"].keys():
                    if reward_model_name in sample:
                        value = sample[reward_model_name]
                        if isinstance(value, torch.Tensor):
                            value = value.cpu().item()
                        model_specific_sums[reward_model_name].append(value) 
                        
            
                if isinstance(sample["clip_reward"], torch.Tensor):
                    clip_rewards.append(sample["clip_reward"].cpu().item())
                else:
                    clip_rewards.append(float(sample["clip_reward"]))
                    
                # Handle DINO rewards - convert tensor to float if needed
                if isinstance(sample["dino_reward"], torch.Tensor):
                    dino_rewards.append(sample["dino_reward"].cpu().item())
                else:
                    dino_rewards.append(float(sample["dino_reward"]))
            
            # Now calculate means with Python floats
            prompt_data["sum_clip_reward"] = np.sum(clip_rewards) if clip_rewards else 0.0
            prompt_data["sum_dino_reward"] = np.sum(dino_rewards) if dino_rewards else 0.0
            
            # Update overall metrics
            results["sum_clip_reward"] += prompt_data["sum_clip_reward"]
            results["sum_dino_reward"] += prompt_data["sum_dino_reward"]
            results["sum_diversity"] += prompt_data["diversity"]
    
    
    # Calculate overall success rate
    total_responses = sum(len(responses) for responses in model_responses_by_prompt)
    results["total_count"] = total_responses
    results["overall_success_rate"] = results["total_valid_count"] / results["total_count"] if total_responses > 0 else 0.0
    
    
    
    
    
    for reward_model_name in results["model_specific_rewards"]:
        results["model_specific_rewards"][reward_model_name] = np.sum(model_specific_sums[reward_model_name]) if model_specific_sums[reward_model_name] else 0.0
        
    
    
    return results


def calculate_eval_rewards_with_diversity_batched(captions, model_responses_by_prompt, reference_images=None, 
                                                models_dict={'clip': ['clip'], 'dino': ['dino']}, eval_dir=None, batch_idx=None):
    """
    Calculate rewards for SVG responses with batch processing for higher throughput.
    
    Args:
        captions (List[str]): List of text prompts/descriptions
        model_responses_by_prompt (List[List[str]]): List of k lists, each with n SVG responses
        reference_images (List[ndarray|tensor], optional): List of reference images
        models_dict (dict): Dictionary specifying which models to use
    
    Returns:
        dict: Dictionary containing rewards, diversity metrics, and additional information
    """
    import os
    import numpy as np
    import torch
    from torch.nn.functional import cosine_similarity
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    num_prompts = len(captions)
    save_to_disk = eval_dir is not None and batch_idx is not None
    
    # Initialize unified data structure
    batch_data = {
        "prompts": {},
        "metadata": {
            "batch_idx": batch_idx,
            "num_prompts": num_prompts,
            "timestamp": time.time()
        },
    }
    
    
    
    # Initialize prompt data structure
    for i in range(num_prompts):
        batch_data["prompts"][str(i)] = {
            "caption": captions[i],
            "responses": [],
            "diversities_by_model": {},
            "diversity": 0.0,
            # "valid_count": 0,
        }
    
    # Initialize models
    clip_models = models_dict.get('clip', [])
    dino_models = models_dict.get('dino', [])
    clip_model_fns = [clip_name_dict[reward_model_name] for reward_model_name in clip_models if reward_model_name in clip_name_dict]
    dino_model_fns = [dino_name_dict[reward_model_name] for reward_model_name in dino_models if reward_model_name in dino_name_dict]
    
    # Step 1: Render all SVGs for all prompts (in batches if needed)
    all_rendered_images = []  # Will be a flattened list
    all_svg_infos = []  # Will be a flattened list
    prompt_indices = []  # Maps each rendered image to its prompt index
    response_indices = []  # Maps each rendered image to its response index within the prompt
    
    if save_to_disk:
        batch_dir = os.path.join(eval_dir, f"batch_{batch_idx}")
        os.makedirs(batch_dir, exist_ok=True)
    
    for prompt_idx in range(num_prompts):
        responses = model_responses_by_prompt[prompt_idx]
        if save_to_disk:
            prompt_dir = os.path.join(batch_dir, f"prompt_{prompt_idx}")
            os.makedirs(prompt_dir, exist_ok=True)
        
        for resp_idx, response in enumerate(responses):
            rendered_image, info = render_response_to_image(response)
            
            # Initialize response data structure
            response_data = {
                "index": resp_idx,
                "text": response,
                "valid": rendered_image is not None,
                "metrics": {}
            }
            
            if rendered_image is not None:
                all_rendered_images.append(rendered_image)
                all_svg_infos.append(info)
                prompt_indices.append(prompt_idx)
                response_indices.append(resp_idx)
                # batch_data["prompts"][str(prompt_idx)]["valid_count"] += 1
                
                
                # Save the image if requested
                if save_to_disk:
                    # Create image path and save
                    image_path = os.path.join(prompt_dir, f"response_{resp_idx}.jpg")
                    if rendered_image.mode == 'RGBA':
                        rgb_image = Image.new('RGB', rendered_image.size, (255, 255, 255))
                        rgb_image.paste(rendered_image, mask=rendered_image.split()[3])  # 3 is the alpha channel
                        rendered_image = rgb_image
                    rendered_image.save(image_path)
                    
                    # Add image path to response data
                    response_data["image_path"] = os.path.relpath(image_path, eval_dir)
            
            # Add SVG information
            if info:
                # response_data["svg_info"] = info
                response_data["formatted"] = info.get("formatted", False)
            
            # Add response data to batch data
            batch_data["prompts"][str(prompt_idx)]["responses"].append(response_data)
    
    # Save batch JSON with prompt and response information if requested
    if save_to_disk:
        json_path = os.path.join(batch_dir, "batch_info.json")
        with open(json_path, "w") as f:
            json.dump(batch_data, f, indent=2)
        print(f"Saved batch info before eval to {json_path}")
    
    # If no valid images, return empty results
    if not all_rendered_images:
        print("No valid SVGs rendered!")
        return batch_data
     
    # Step 2: Calculate CLIP rewards for all valid rendered images at once
    all_captions = []
    for idx in prompt_indices:
        all_captions.append(captions[idx])
    
    for clip_model_name in clip_models:
        if clip_model_name in clip_name_dict:
            try:
                # Process all images at once with CLIP
                clip_model = clip_name_dict[clip_model_name]
                clip_scores = clip_model(all_captions, all_rendered_images)
                
                # Assign scores back to the right prompts and responses
                for i, (prompt_idx, resp_idx) in enumerate(zip(prompt_indices, response_indices)):
                    prompt_str = str(prompt_idx)
                    
                    # Add CLIP reward
                    clip_reward = 1.0 - clip_scores[i]  # Convert distance to similarity
                    if not isinstance(clip_reward, float):
                        clip_reward = float(clip_reward)
                    
                    # Store the metric in the response
                    batch_data["prompts"][prompt_str]["responses"][resp_idx]["metrics"][clip_model_name] = clip_reward
                    
                    # Also track total clip reward
                    if "clip_reward" not in batch_data["prompts"][prompt_str]["responses"][resp_idx]["metrics"]:
                        batch_data["prompts"][prompt_str]["responses"][resp_idx]["metrics"]["clip_reward"] = 0.0
                        batch_data["prompts"][prompt_str]["responses"][resp_idx]["metrics"]["total_reward"] = 0.0
                    
                    batch_data["prompts"][prompt_str]["responses"][resp_idx]["metrics"]["clip_reward"] += clip_reward
                    batch_data["prompts"][prompt_str]["responses"][resp_idx]["metrics"]["total_reward"] += clip_reward
                    
            except Exception as e:
            
                assert False, f"Error calculating CLIP scores with {clip_model_name}: {e}"
    
    # Step 3: Calculate DINO rewards if reference images exist
    if reference_images is not None:
        all_ref_images = []
        indices_with_refs = []
        rendered_with_refs = []
        
        for i, (prompt_idx, _) in enumerate(zip(prompt_indices, response_indices)):
            if reference_images[prompt_idx] is not None:
                all_ref_images.append(reference_images[prompt_idx])
                rendered_with_refs.append(all_rendered_images[i])
                indices_with_refs.append(i)
                
        if indices_with_refs:  # Only process if we have valid reference-render pairs
            for dino_model_name in dino_models:
                if dino_model_name in dino_name_dict:
                    try:
                        # Process batch with DINO
                        dino_model = dino_name_dict[dino_model_name]
                        dino_scores, all_features = dino_model(all_ref_images, rendered_with_refs, return_features=True)
                        
                        # Assign scores back to the right prompts and responses
                        for batch_idx, orig_idx in enumerate(indices_with_refs):
                            prompt_idx = prompt_indices[orig_idx]
                            resp_idx = response_indices[orig_idx]
                            prompt_str = str(prompt_idx)
                            
                            dino_reward = 1.0 - dino_scores[batch_idx]  # Convert distance to similarity
                            if not isinstance(dino_reward, float):
                                dino_reward = float(dino_reward)
                            dino_feature = all_features[batch_idx]
                            
                            # Store the metric in the response
                            batch_data["prompts"][prompt_str]["responses"][resp_idx]["metrics"][dino_model_name] = dino_reward
                            
                            # Initialize dino reward if needed
                            if "dino_reward" not in batch_data["prompts"][prompt_str]["responses"][resp_idx]["metrics"]:
                                batch_data["prompts"][prompt_str]["responses"][resp_idx]["metrics"]["dino_reward"] = 0.0
                                
                            batch_data["prompts"][prompt_str]["responses"][resp_idx]["metrics"]["dino_reward"] += dino_reward
                            batch_data["prompts"][prompt_str]["responses"][resp_idx]["metrics"]["total_reward"] += dino_reward
                            
                            # Store feature for diversity calculation
                            if "dino_feature" not in batch_data["prompts"][prompt_str]["responses"][resp_idx]:
                                batch_data["prompts"][prompt_str]["responses"][resp_idx]["dino_feature"] = {}
                                
                            batch_data["prompts"][prompt_str]["responses"][resp_idx]["dino_feature"][dino_model_name] = dino_feature
                            
                    except Exception as e:
                        assert False, f"Error calculating DINO scores with {dino_model_name}: {e}"
                        print(f"Error calculating DINO scores with {dino_model_name}: {e}")
    
    # Step 4: Calculate diversity for each prompt using DINO features
    for prompt_idx in range(num_prompts):
        prompt_str = str(prompt_idx)
        prompt_data = batch_data["prompts"][prompt_str]
        # Get all valid samples for this prompt
        valid_samples = [s for s in prompt_data["responses"] if "dino_feature" in s]
        
        # Need at least 2 valid rendered images for diversity
        if len(valid_samples) < 2:
            prompt_data["diversity"] = 0.0
            prompt_data["diversities"] = {model: 0.0 for model in dino_models}
            batch_data["prompts"][prompt_str] = prompt_data
            # Clean up dino_feature to save memory
            for sample in valid_samples:    
                if "dino_feature" in sample:
                    del sample["dino_feature"]
            continue
        
        
        
        # Check if we already have features stored from the DINO scoring step
        if valid_samples and all("dino_feature" in sample for sample in valid_samples):
            # Calculate diversity using stored features
            diversity_by_model = {}
            
            for dino_model_name in dino_models:
                if dino_model_name in dino_name_dict:
                    try:
                        # Extract features for this model from all samples
                        features = [
                            sample["dino_feature"][dino_model_name] 
                            for sample in valid_samples 
                            if dino_model_name in sample.get("dino_feature", {})
                        ]
                        features = [f for f in features if f is not None]
                        
                        if len(features) >= 2:
                            # Calculate pairwise similarities
                            pairwise_sims = []
                            for i in range(len(features)):
                                for j in range(i+1, len(features)):
                                    sim = cosine_similarity(
                                        features[i].unsqueeze(0),
                                        features[j].unsqueeze(0)
                                    ).item()
                                    pairwise_sims.append(sim)
                            
                            # Diversity is inverse of average similarity
                            avg_sim = np.mean(pairwise_sims) if pairwise_sims else 0
                            diversity = 1.0 - avg_sim
                            if not isinstance(diversity, float):
                                diversity = float(diversity)
                            diversity_by_model[dino_model_name] = diversity
                        else:
                            diversity_by_model[dino_model_name] = 0.0
                    except Exception as e:
                        
                        assert False, f"Error calculating diversity with stored features for {dino_model_name}, prompt {prompt_idx}: {e}"
                        diversity_by_model[dino_model_name] = 0.0
        else:
            assert False
            
        # Clean up dino_feature to save memory
        for sample in valid_samples:
            if "dino_feature" in sample:
                del sample["dino_feature"]
                
        import gc
        gc.collect()
        
        
        
        # Average diversity across DINO models
        prompt_data["diversity"] = np.mean(list(diversity_by_model.values())) if diversity_by_model else 0.0
        prompt_data["diversities_by_model"] = diversity_by_model
        batch_data["prompts"][prompt_str] = prompt_data
        
        
    
    
    
    
    
    
    # Save batch JSON with prompt and response information if requested
    if save_to_disk:
        json_path = os.path.join(batch_dir, "batch_info.json")
        with open(json_path, "w") as f:
            json.dump(batch_data, f, indent=2)
        print(f"Saved batch info after eval to {json_path}")
    
    return batch_data



import ray

ray.init()

@ray.remote(num_gpus=1)
class ModelWorker:
    def __init__(self, model_path, max_model_len):
        import torch
        import vllm
        import os
        
        # Configure PyTorch to better manage memory
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        
        # Each ray worker gets assigned to a specific GPU automatically
        gpu_id = ray.get_gpu_ids()[0]
        print(f"Worker initializing model on GPU {gpu_id}")
        
        # Lower GPU memory utilization to prevent OOM errors
        self.model = vllm.LLM(
            model_path,
            tensor_parallel_size=1,
            swap_space=16,
            max_model_len=max_model_len,
            dtype="bfloat16",
            # enable_prefix_caching=False,  # Disable prefix caching to prevent memory accumulation
            gpu_memory_utilization=0.7,   # Reduced to prevent OOM
            # enforce_eager=True,           # Better memory management
            
        )
    
    def generate(self, prompts, sampling_params):
        import torch
        
        # Clear cache before generating
        torch.cuda.empty_cache()
        
        # Ensure we're not carrying over previous requests
        result = self.model.generate(prompts, sampling_params)
        
        # Clear cache after generating
        torch.cuda.empty_cache()
        
        return result
    
    def evaluate_batch(self, batch_captions, batch_prompts, batch_ref_images, sampling_params, models_dict, eval_dir=None, batch_idx=None):
        import torch
        import gc
        
        try:
            # Clear memory before starting
            torch.cuda.empty_cache()
            gc.collect()
            
            # Generate responses
            batch_generated_responses = []
            outputs = self.model.generate(batch_prompts, sampling_params)
            
            # Force memory cleanup
            torch.cuda.empty_cache()
            gc.collect()
            
            # Verify we're generating the expected number of responses
            expected_responses = len(batch_prompts) * sampling_params.n
            actual_responses = sum(len(o.outputs) for o in outputs)
            
            print(f"Worker generated {actual_responses} responses (expected {expected_responses})")
            if actual_responses > expected_responses * 1.1:  # Allow small variation
                print("WARNING: Generated more responses than expected!")
            
            # Extract responses
            for output in outputs:
                batch_generated_responses.append([o.text for o in output.outputs])
            
            # Evaluate - breaking this into sub-steps to manage memory
            try:
                # First render SVGs to images (most memory-intensive step)
                batch_results = calculate_eval_rewards_with_diversity_batched(
                    batch_captions,
                    batch_generated_responses,
                    batch_ref_images,
                    models_dict=models_dict,
                    eval_dir=eval_dir,  # No need to save images in this worker
                    batch_idx=batch_idx
                )
                
                # del batch_results["dino_features"]  # If storing features
                del outputs
                
                # Force aggressive memory cleanup
                torch.cuda.empty_cache()
                gc.collect()
                
                
                return batch_results
                
            except Exception as e:
                import traceback
                error_msg = f"Evaluation error: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)
                return {"error": error_msg, "total_valid_count": 0}
            
        except Exception as e:
            import traceback
            error_msg = f"Worker error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return {"error": error_msg, "total_valid_count": 0}

def main_multigpus(
    model_name: str = "/home/share/oat-output/scale_reward_cliponly_small_0419T08:08:32/saved_models/step_00300/",  
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 2048,
    max_model_len: int = 4096,
    n_samples: int = 4,  # Reduced from 8
    max_eval_samples: int = 512,
    batch_size: int = 8,  # Reduced from 16
    save: bool = True,
    output_dir: str = "./evaluation_results",
    num_gpus: int = 8,  # Total number of GPUs to use
):
    """Evaluate SVG generation model with controlled data parallelism across GPUs using Ray"""
    import ray
    import time
    import numpy as np
    from tqdm import tqdm
    
    # Initialize Ray if not already started
    if not ray.is_initialized():
        ray.init()
    
    # Check available GPUs via Ray
    available_gpus = int(ray.cluster_resources().get("GPU", 0))
    if num_gpus > available_gpus:
        print(f"Warning: Requested {num_gpus} GPUs but only {available_gpus} are available via Ray")
        num_gpus = available_gpus
    
    print(f"Using {num_gpus} GPUs for data parallel evaluation via Ray")
    
    # Initialize directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure model inference settings with reduced parameters
    sampling_params = vllm.SamplingParams(
        n=n_samples,  # Reduced n_samples
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        logprobs=1,
        seed=int(time.time_ns()),
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # Initialize a Ray worker for each GPU
    print(f"Initializing {num_gpus} Ray workers with model: {model_name}")
    workers = [ModelWorker.remote(model_name, max_model_len) for _ in range(num_gpus)]
    
    # Load datasets
    datasets = prepare_data_batched(max_eval_samples, batch_size)
    
    # Setup model evaluation dictionary
    models_dict = {
        'clip': ['clip_small', 'clip_large', 'siglip_large', 'siglip_small'], 
        'dino': ['dino_small', 'dino_base', 'dino_large', 'dino_giant']
    }
    
    # Results storage
    task_results = {}
    
    # Process each dataset
    for task_name, dataset_info in datasets.items():
        eval_dir = os.path.join(output_dir, task_name)
        os.makedirs(eval_dir, exist_ok=True)
        dataloader = dataset_info['dataloader']
        print(f"Evaluating task: {task_name} with {dataset_info['total_samples']} examples")
        
        # Initialize accumulated results for this task
        task_accumulated_results = []
        
        # Track worker availability
        worker_status = [True] * num_gpus  # True means worker is available
        
        # Process batches with controlled concurrency
        total_batches = len(dataloader)
        completed_batches = 0
        active_tasks = {}  # Maps worker_idx -> (task_ref, batch_idx)
        
        with tqdm(total=total_batches, desc="Processing batches") as pbar:
            batch_iter = iter(dataloader)
            next_batch_idx = 0
            
            # Initial submission - assign one batch per available worker
            for worker_idx in range(min(num_gpus, total_batches)):
                try:
                    batch = next(batch_iter)
                    batch_prompts, batch_captions, batch_ref, batch_ref_images = batch
                    
                    print(f"Submitting initial batch {next_batch_idx+1}/{total_batches} to worker {worker_idx}")
                    
                    # Submit task to worker
                    task_ref = workers[worker_idx].evaluate_batch.remote(
                        batch_captions,
                        batch_prompts, 
                        batch_ref_images,
                        sampling_params,
                        models_dict,
                        eval_dir=eval_dir,
                        batch_idx=next_batch_idx
                    )
                    
                    # Track this task
                    active_tasks[worker_idx] = (task_ref, next_batch_idx)
                    worker_status[worker_idx] = False  # Worker is busy
                    next_batch_idx += 1
                    
                except StopIteration:
                    # No more batches
                    break
            
            # Process results and submit new tasks as workers become available
            while active_tasks:
                # Wait for any task to complete
                ready_refs = [task_ref for task_ref, _ in active_tasks.values()]
                done_refs, _ = ray.wait(ready_refs, num_returns=1, timeout=3000)
                
                if not done_refs:
                    print("Timeout waiting for tasks. Continuing...")
                    assert False, "Timeout waiting for tasks"
                    continue
                
                # Find which worker completed its task
                done_ref = done_refs[0]
                completed_worker_idx = None
                completed_batch_idx = None
                
                for worker_idx, (task_ref, batch_idx) in active_tasks.items():
                    if task_ref == done_ref:
                        completed_worker_idx = worker_idx
                        completed_batch_idx = batch_idx
                        break
                
                if completed_worker_idx is None:
                    print("Error: Could not find completed task in active tasks")
                    continue
                
                # Process result
                try:
                    batch_results = ray.get(done_ref)
                    task_accumulated_results += batch_results["prompts"].values()
                    
                    completed_batches += 1
                    pbar.update(1)
                    print(f"Completed batch {completed_batch_idx+1}/{total_batches} "
                          f"on worker {completed_worker_idx} ({completed_batches}/{total_batches} total)")
                    
                    # Important: Force Ray to clean up the result reference to free memory
                    del batch_results
                    
                    # Force Python garbage collection
                    import gc
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error processing batch {completed_batch_idx+1}: {e}")
                    # Consider restarting the worker if needed
                
                # Worker is now available
                worker_status[completed_worker_idx] = True
                del active_tasks[completed_worker_idx]
                
                # Submit next batch if available to this worker
                try:
                    batch = next(batch_iter)
                    batch_prompts, batch_captions, batch_ref, batch_ref_images = batch
                    
                    print(f"Submitting next batch {next_batch_idx+1}/{total_batches} to worker {completed_worker_idx}")
                    
                    # Submit task to worker
                    task_ref = workers[completed_worker_idx].evaluate_batch.remote(
                        batch_captions,
                        batch_prompts, 
                        batch_ref_images,
                        sampling_params,
                        models_dict,
                        eval_dir=eval_dir,
                        batch_idx=next_batch_idx
                    )
                    
                    # Track this task
                    active_tasks[completed_worker_idx] = (task_ref, next_batch_idx)
                    worker_status[completed_worker_idx] = False  # Worker is busy again
                    next_batch_idx += 1
                    
                except StopIteration:
                    # No more batches to process
                    pass
        




       
        # Store aggregated task results
        task_results[task_name] = calculate_average_metrics(task_accumulated_results)
        
        # Report task summary
        
        
        print(f"\n--- {task_name} Results Summary ---")
        print(f"Valid responses: {task_results[task_name]['valid_count']} / {task_results[task_name]['total_count']} ({task_results[task_name]['success_rate']:.2%})")
        print(f"Average metrics:")
        
        # Print CLIP and DINO rewards
        for metric_name, value in task_results[task_name]['metrics'].items():
            print(f"  {metric_name}: {value:.4f}")
            
        # Print diversity metrics
        print(f"Diversity metrics:")
        print(f"  Average diversity: {task_results[task_name]['diversity']['average']:.4f}")
        for model_name, value in task_results[task_name]['diversity'].items():
            if model_name != "average":
                print(f"  {model_name}: {value:.4f}")
        
        
        
    
    # Calculate overall metrics across tasks
    overall_results = average_dictionaries(list(task_results.values()))
 
    
    #Print overall summary
    print("\n=== OVERALL EVALUATION RESULTS ===")
    print(f"Total valid responses: {overall_results['valid_count']} / {overall_results['total_count']} ({overall_results['success_rate']:.2%})")
    
    print("\nAverage metrics across all tasks:")
    for metric_name, value in overall_results['metrics'].items():
        print(f"  {metric_name}: {value:.4f}")
    
    print("\nDiversity metrics:")
    print(f"  Average diversity: {overall_results['diversity']['average']:.4f}")
    
    for model_name, value in overall_results['diversity'].items():
        if model_name != "average":
            print(f"  {model_name}: {value:.4f}")
    
    
    # Save detailed results if requested
    if save:
        timestamp = int(time.time())
        model_short_name = os.path.basename(model_name.rstrip("/"))
        output_file = Path(output_dir) / f"svg_eval_{model_short_name}_{timestamp}.json"
        
        print(f"Saving detailed evaluation results to {output_file}")
        with open(output_file, "w") as f:
            json.dump(overall_results, f, indent=2)
    
    # Clean up Ray resources
    print("Shutting down Ray workers...")
    for worker in workers:
        ray.kill(worker)
    
    return overall_results


import re
def extract_model_name(path):
    # Strip trailing slash if present
    path = path.rstrip('/')
    
    # Get the grandparent directory (which contains the timestamp)
    parent_dir = os.path.dirname(path)  # This gives .../scale_reward_cliponly_small_0419T08:08:32
    
    # Get the directory name containing the timestamp
    dir_name = os.path.basename(parent_dir)
    
    # Use regex to extract the part before the timestamp
    match = re.match(r'(.+?)_\d{4}T\d{2}:\d{2}:\d{2}', dir_name)
    
    if match:
        return match.group(1)
    else:
        # Fallback in case the pattern doesn't match
        return dir_name

def select_checkpoints(model_path: str, num_checkpoints: int = 5, keep_steps = None):
    """
    Select checkpoint directories to evaluate, with evenly spaced distribution
    and ensuring specific steps are included.
    
    Args:
        model_path (str): Base directory containing checkpoint directories named step_XXXXX
        num_checkpoints (int): Maximum number of checkpoints to select
        keep_steps (list): List of specific step numbers that must be included
        
    Returns:
        list: Selected checkpoint directory paths
    """
    import os
    import re
    import glob
    
    # Initialize keep_steps if None
    if keep_steps is None:
        keep_steps = []
    elif isinstance(keep_steps, int):
        keep_steps = [keep_steps]
    
    # Find all checkpoint directories
    checkpoint_dirs = glob.glob(os.path.join(model_path, "step_*"))
    checkpoint_dirs.sort()
    
    if not checkpoint_dirs:
        print(f"No checkpoint directories found in {model_path}")
        return []
    
    # Extract step numbers for all checkpoints
    step_numbers = []
    step_to_dir = {}
    for cp in checkpoint_dirs:
        step_match = re.search(r'step_(\d+)', cp)
        if step_match:
            step = int(step_match.group(1))
            step_numbers.append(step)
            step_to_dir[step] = cp
    
    # Always include the last checkpoint
    last_step = step_numbers[-1]
    mandatory_steps = [last_step]
    
    # Add requested keep_steps
    for step in keep_steps:
        if step in step_to_dir and step not in mandatory_steps:
            mandatory_steps.append(step)
    
    # If we need fewer checkpoints than our mandatory ones,
    # just return the mandatory checkpoints
    if num_checkpoints <= len(mandatory_steps):
        selected_steps = sorted(mandatory_steps)[:num_checkpoints]
        return [step_to_dir[step] for step in selected_steps]
    
    # We need to select additional checkpoints for the remaining slots
    remaining_slots = num_checkpoints - len(mandatory_steps)
    
    # Get available steps (excluding mandatory ones)
    available_steps = [step for step in step_numbers if step not in mandatory_steps]
    
    if not available_steps:
        # If no more checkpoints available, return what we have
        return [step_to_dir[step] for step in sorted(mandatory_steps)]
    
    if remaining_slots >= len(available_steps):
        # If we need all available steps, add them all
        selected_steps = available_steps + mandatory_steps
    else:
        # Select evenly spaced checkpoints from available ones
        indices = [i * (len(available_steps) - 1) // (remaining_slots - 1) for i in range(remaining_slots)]
        evenly_spaced_steps = [available_steps[i] for i in indices]
        selected_steps = evenly_spaced_steps + mandatory_steps
    selected_steps = list(set(selected_steps))  # Remove duplicates
    # Sort steps and convert to directory paths
    selected_steps = sorted(selected_steps)
    selected_checkpoints = [step_to_dir[step] for step in selected_steps]
    
    return selected_checkpoints
    
    

def eval_checkpoints(
    model_path: str = "/home/share/oat-output/scale_reward_cliponly_small_0419T08:08:32/saved_models/",
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 3000,
    max_model_len: int = 4096,
    n_samples: int = 8,
    max_eval_samples: int = 1024,
    batch_size: int = 16,
    save: bool = True,
    output_dir: str = "./evaluation_results",
    wandb_project: str = "svg-model-evaluation",
    wandb_entity: str = None,
    num_checkpoints: int = 5,  # Evaluate only 5 checkpoints
    num_gpus: int = 8,  # Number of GPUs to use for evaluation
):
    """
    Evaluate SVG generation models across 5 evenly-spaced checkpoints and log to wandb.
    The last checkpoint is always included in the evaluation.
    
    Args:
        model_path: Base directory containing checkpoint directories named step_XXXXX
        temperature: Sampling temperature for generation
        top_p: Top p for sampling
        max_tokens: Maximum tokens to generate
        max_model_len: Maximum model input length
        n_samples: Number of samples to generate per prompt
        max_eval_samples: Maximum number of evaluation samples per dataset
        batch_size: Number of prompts to process simultaneously
        save: Whether to save detailed results to files
        output_dir: Directory to save evaluation results
        wandb_project: WandB project name for logging
        wandb_entity: WandB entity (username or team name)
        num_checkpoints: Number of checkpoints to evaluate (evenly spaced)
    """
    import wandb
    import re
    import glob
    import os
    from pathlib import Path
    
    # Initialize directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all checkpoint directories
    checkpoint_dirs = glob.glob(os.path.join(model_path, "step_*"))
    checkpoint_dirs.sort()
    
    if not checkpoint_dirs:
        print(f"No checkpoint directories found in {model_path}")
        return

    selected_checkpoints = select_checkpoints(
        model_path,
        num_checkpoints=num_checkpoints,
        keep_steps=[30,750]  # Always include the last checkpoint
    )
    print(f"Selected {len(selected_checkpoints)} checkpoints to evaluate from {len(checkpoint_dirs)} available")
    for cp in selected_checkpoints:
        step_match = re.search(r'step_(\d+)', cp)
        step = int(step_match.group(1)) if step_match else "unknown"
        if cp == checkpoint_dirs[-1]:
            print(f"  - Step {step}: {cp} (LAST CHECKPOINT)")
        else:
            print(f"  - Step {step}: {cp}")
    
    # Extract model name from base path for WandB run naming
    model_name = extract_model_name(model_path)
    
    # Initialize WandB
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=f"{model_name}",
        config={
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n_samples": n_samples,
            "max_eval_samples": max_eval_samples,
            "batch_size": batch_size,
            "model_path": model_path,
            "evaluated_checkpoints": len(selected_checkpoints),
        }
    )
    
    
    # Evaluate each checkpoint
    all_results = {}
    for checkpoint_dir in selected_checkpoints:
        # Extract step number from directory name
        step_match = re.search(r'step_(\d+)', checkpoint_dir)
        if not step_match:
            print(f"Couldn't extract step number from {checkpoint_dir}, skipping")
            continue
        
        step = int(step_match.group(1))
        print(f"\n\n{'='*80}")
        print(f"Evaluating checkpoint at step {step}: {checkpoint_dir}")
        print(f"{'='*80}")
        
        try:
            # Run evaluation on this checkpoint using the main function
            eval_dir = os.path.join(output_dir,model_name, f"step_{step:05d}")
            os.makedirs(eval_dir, exist_ok=True)
            checkpoint_result = main_multigpus(
                model_name=checkpoint_dir,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                max_model_len=max_model_len,
                n_samples=n_samples,
                max_eval_samples=max_eval_samples,
                batch_size=batch_size,
                save=save,
                output_dir=eval_dir,
                num_gpus=num_gpus,
            )
            
            # Store results
            all_results[step] = checkpoint_result
            
            # Log results to WandB
            wandb_log_dict = flatten_dict(checkpoint_result)
            
            
            # Log to wandb with the step number
            wandb.log(wandb_log_dict, step=step)
            
        except Exception as e:
            print(f"Error evaluating checkpoint {checkpoint_dir}: {e}")
            import traceback
            print(traceback.format_exc())
            
            # Log the error to wandb
            wandb.log({
                "error": str(e),
                "checkpoint": checkpoint_dir
            }, step=step)
    
    
    # Finish wandb run
    wandb.finish()
    
    return all_results


if __name__ == "__main__":
    # print(extract_model_name("/home/share/oat-output/scale_reward_cliponly_small_continue_0419T08:08:32/saved_models/"))
    # fire.Fire(main_multigpus)
    fire.Fire(eval_checkpoints)