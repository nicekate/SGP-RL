from .clips import (clip_text_image_distances_batch,
                    dinov2_image_image_distances_batch,
                    dinov2_image_image_patch_distances_batch,
                    siglip_text_image_distances_batch,siglip2_text_image_distances_batch)
from .svg import (extract_svg, safe_svg_to_image, get_svg_code_length, is_sketch_style, is_greyscale)
from functools import partial



clip_name_dict = {
    "clip": clip_text_image_distances_batch,
    "clip_small": partial(clip_text_image_distances_batch, model_name = "ViT-B/32"),
    "clip_large": partial(clip_text_image_distances_batch, model_name = "ViT-L/14"),
    "siglip": siglip_text_image_distances_batch,
    "siglip_small": partial(siglip_text_image_distances_batch, model_name = "google/siglip-base-patch16-384"),
    "siglip_large": partial(siglip_text_image_distances_batch, model_name = "google/siglip-large-patch16-384"),
    "siglip2_giant": partial(siglip2_text_image_distances_batch, model_name = "google/siglip2-giant-opt-patch16-384"),
    "siglip2_large": partial(siglip2_text_image_distances_batch, model_name = "google/siglip2-large-patch16-384"),}

dino_name_dict = {
    "dino": dinov2_image_image_distances_batch,
    "dino_small": partial(dinov2_image_image_distances_batch, model_name = "dinov2_vits14"),
    "dino_base": partial(dinov2_image_image_distances_batch, model_name = "dinov2_vitb14"),
    "dino_large": partial(dinov2_image_image_distances_batch, model_name = "dinov2_vitl14"),
    "dino_giant": partial(dinov2_image_image_distances_batch, model_name = "dinov2_vitg14"),
    "dino_patchmax": partial(dinov2_image_image_patch_distances_batch, reduction = "max"),
    "dino_patchposition": partial(dinov2_image_image_patch_distances_batch, reduction = "position"),
    "dino_patchinversemax": partial(dinov2_image_image_patch_distances_batch, reduction = "inversemax"),
    "dino_patchdebug": partial(dinov2_image_image_patch_distances_batch, reduction = "debug"),}

def length_penalty(length: int) -> float:
    """
    Calculate a length penalty that decreases linearly from 1 to 0
    as length increases from 0 to 6000.
    
    Args:
        length (int): The length value to penalize
        
    Returns:
        float: Penalty value between 0 and 1 (1 = no penalty, 0 = maximum penalty)
    """
    # Linear penalty from 1 (at length=0) to 0 (at length=6000)
    penalty = 1.0 - (length / 6000.0)
    
    # Ensure the penalty is clamped between 0 and 1
    return max(0.0, min(1.0, penalty))   


def is_format_correct(response):
    if "</think> <answer>" in response and "</answer>" in response:
        return True
    return False
def no_text_in_response(response):
    if "</text>" in response:
        return False
    if "</tspan>" in response:
        return False
    if "</textPath>" in response:
        return False
    return True


def render_response_to_image(response, args=None):
    """
    Extract SVG from a response and render it to an image.
    
    Args:
        response (str): Model response text potentially containing SVG code
        
    Returns:
        tuple: (rendered_image, info_dict) where:
            - rendered_image: PIL Image or None if extraction/rendering failed
            - info_dict: Dictionary with processing information and status
    """
    info = {"success": False, "formatted": False}
    
    # Check for proper format with <think>/<answer> tags
    
    
    if is_format_correct(response) and no_text_in_response(response):
        info["formatted"] = True
        # Extract content between <answer> tags
        answer_content = response.split("<answer>")[-1].replace("</answer>", "")
        svg_content = extract_svg(answer_content)
        
    else:
        # Format is incorrect - missing proper tags
        info["error"] = "format error"
        # Extract content between <answer> tags
        svg_content = extract_svg(response)
    
    if not svg_content:
        info["error"] = "No SVG found in answer"
        return None, info
    if args.require_sketch:
        if not is_sketch_style(svg_content):
            info["error"] = "Not a sketch style SVG"
            return None, info
    if args.require_greystyle:
        if not is_greyscale(svg_content):
            info["error"] = "Not a grayscale SVG"
            return None, info
    
    try:
        
        image = safe_svg_to_image(svg_content)
        
        
        if image is None:
            info["error"] = "Failed to render SVG"
            return None, info
            
        info["success"] = True
        info["length"] = get_svg_code_length(svg_content)
        return image, info
        
    except Exception as e:
        info["error"] = str(e)
        return None, info

    
    

def answer_tag_reward_fn(model_responses, prompts, images=None, rewards_dict = {'clip':1, 'dino':1, 'length':0, 'format':0}, models_dict = {'clip': 'clip', 'dino': 'dino'}, offset = 0.0, args = None):
    """
    Calculate rewards for SVG responses based on text-image and image-image similarity,
    enforcing the proper format structure with <think>/<answer> tags.
    
    Args:
        model_responses (List[str]): Model generated responses containing SVG code
        prompts (List[str]): Text prompts/descriptions of the desired images
        images (List[PIL.Image], optional): Reference images to compare against
    
    Returns:
        dict: Dictionary containing rewards and additional information
    """
    dino_model = dino_name_dict[models_dict['dino']]
    clip_model = clip_name_dict[models_dict['clip']]
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    num_examples = len(model_responses)
    
    # Initialize results structure
    results = {
        "rewards": [0.0] * num_examples,
        "clip_reward": [0.0] * num_examples,
        "dino_reward": [0.0] * num_examples,
        "length_reward": [0.0] * num_examples,
        "svg_info": [{"success": False} for _ in range(num_examples)],
        "rendered_images": [None] * num_examples,
        "formatted": [False] * num_examples
    }
    
    # Step 1: Check format and extract SVG content
    for i, response in enumerate(model_responses):
        
        rendered_image, info = render_response_to_image(response, args)
        results["rendered_images"][i] = rendered_image
        results["svg_info"][i] = info
        results["formatted"][i] = info.get("formatted", False)
        if info["success"]:
                
            results["length_reward"][i] = rewards_dict['length'] * length_penalty(info["length"])
            
            results["rewards"][i] += results["length_reward"][i]
            if results["formatted"][i]:
                results["rewards"][i] += rewards_dict['format']
    
    
    # Step 2: Calculate CLIP text-image distances (text-to-image similarity)
    valid_indices = []
    valid_rendered_images = []
    valid_prompts = []
    
    for i, rendered_img in enumerate(results["rendered_images"]):
        if rendered_img is not None:
            valid_indices.append(i)
            valid_rendered_images.append(rendered_img)
            valid_prompts.append(prompts[i])
    if valid_indices:
        
        clip_scores = clip_model(valid_prompts, valid_rendered_images)
        # Update results with CLIP scores
        for i, idx in enumerate(valid_indices):
            score = clip_scores[i]
            clip_reward = ((1.0 - score) + 1.0) /2.0  # Convert distance to similarity
            results["rewards"][idx] += clip_reward * rewards_dict['clip']
            results["clip_reward"][idx] = float(clip_reward)
    
    # Step 3: Calculate DINOv2 image-image distances (image similarity)
    if images is not None:
        img_valid_indices = []
        img_rendered = []
        img_references = []
        assert len(images) == num_examples, "Number of images must match number of responses"
        for i, (rendered, ref_img) in enumerate(zip(results["rendered_images"], images)):
            
                
            if rendered is not None and ref_img is not None:
                img_valid_indices.append(i)
                img_rendered.append(rendered)
                img_references.append(ref_img)
        
        if img_valid_indices:
            dino_scores = dino_model( img_references, img_rendered)
            # Update results with DINOv2 scores
            for i, idx in enumerate(img_valid_indices):
                score = dino_scores[i]
                dino_reward = (1.0 - score + 1.0)/2.0 # Convert distance to similarity
                results["rewards"][idx] += dino_reward * rewards_dict['dino'] # Add to existing CLIP reward
                results["dino_reward"][idx] = float(dino_reward)
    
    # Adjust rewards based on formatting - only give positive rewards if properly formatted
    for i in range(num_examples):
        if not results["formatted"][i] and  rewards_dict['format'] == 0:
            results["rewards"][i] = -offset
    
    return results


def calculate_eval_rewards(model_responses, prompts, images=None, models_dict={'clip': ['clip'], 'dino': ['dino']}):
    """
    Calculate rewards for SVG responses based on multiple text-image and image-image similarity models.
    
    Args:
        model_responses (List[str]): Model generated responses containing SVG code
        prompts (List[str]): Text prompts/descriptions of the desired images
        images (List[PIL.Image], optional): Reference images to compare against
        models_dict (dict): Dictionary specifying which clip and dino models to use
                            {'clip': ['clip_name1', 'clip_name2'], 'dino': ['dino_name1', 'dino_name2']}
    
    Returns:
        dict: Dictionary containing rewards and additional information
    """
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    num_examples = len(model_responses)
    
    # Initialize results structure
    results = {
        "rewards": [0.0] * num_examples,
        "svg_info": [{"success": False} for _ in range(num_examples)],
        "rendered_images": [None] * num_examples,
        "formatted": [False] * num_examples
    }
    
    # Initialize reward dictionaries for each model
    clip_models = models_dict.get('clip', [])
    dino_models = models_dict.get('dino', [])
    
    for model_name in clip_models:
        results[f"clip_{model_name}_reward"] = [0.0] * num_examples
    
    for model_name in dino_models:
        results[f"dino_{model_name}_reward"] = [0.0] * num_examples
    
    # Step 1: Check format and extract SVG content
    for i, response in enumerate(model_responses):
        rendered_image, info = render_response_to_image(response)
        results["rendered_images"][i] = rendered_image
        results["svg_info"][i] = info
        results["formatted"][i] = info.get("formatted", False)
    
    # Step 2: Calculate CLIP text-image distances for each CLIP model
    valid_indices = []
    valid_rendered_images = []
    valid_prompts = []
    
    for i, rendered_img in enumerate(results["rendered_images"]):
        if rendered_img is not None:
            valid_indices.append(i)
            valid_rendered_images.append(rendered_img)
            valid_prompts.append(prompts[i])
    
    if valid_indices:
        for clip_model_name in clip_models:
            if clip_model_name in clip_name_dict:
                clip_model = clip_name_dict[clip_model_name]
                try:
                    clip_scores = clip_model(valid_prompts, valid_rendered_images)
                    
                    # Update results with CLIP scores
                    for i, idx in enumerate(valid_indices):
                        score = clip_scores[i]
                        clip_reward = 1.0 - score  # Convert distance to similarity
                        results[f"clip_{clip_model_name}_reward"][idx] = float(clip_reward)
                        
                        # Add to total rewards
                        results["rewards"][idx] += clip_reward
                except Exception as e:
                    print(f"Error calculating {clip_model_name} scores: {e}")
    
    # Step 3: Calculate DINOv2 image-image distances for each DINO model
    if images is not None:
        img_valid_indices = []
        img_rendered = []
        img_references = []
        
        assert len(images) == num_examples, "Number of images must match number of responses"
        
        for i, (rendered, ref_img) in enumerate(zip(results["rendered_images"], images)):
            if rendered is not None and ref_img is not None:
                img_valid_indices.append(i)
                img_rendered.append(rendered)
                img_references.append(ref_img)
        
        if img_valid_indices:
            for dino_model_name in dino_models:
                if dino_model_name in dino_name_dict:
                    dino_model = dino_name_dict[dino_model_name]
                    try:
                        dino_scores = dino_model(img_references, img_rendered)
                        
                        # Update results with DINOv2 scores
                        for i, idx in enumerate(img_valid_indices):
                            score = dino_scores[i]
                            dino_reward = 1.0 - score  # Convert distance to similarity
                            results[f"dino_{dino_model_name}_reward"][idx] = float(dino_reward)
                            
                            # Add to total rewards
                            results["rewards"][idx] += dino_reward
                    except Exception as e:
                        print(f"Error calculating {dino_model_name} scores: {e}")
    
    # Add summary statistics
    for i in range(num_examples):
        # Calculate average CLIP reward
        clip_rewards = [results[f"clip_{model}_reward"][i] for model in clip_models]
        if clip_rewards:
            results["avg_clip_reward"] = [sum(clip_rewards)/len(clip_rewards) for _ in range(num_examples)]
        
        # Calculate average DINO reward
        dino_rewards = [results[f"dino_{model}_reward"][i] for model in dino_models]
        if dino_rewards:
            results["avg_dino_reward"] = [sum(dino_rewards)/len(dino_rewards) for _ in range(num_examples)]
    
    return results



