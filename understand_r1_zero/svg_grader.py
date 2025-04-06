from .clips import (clip_text_image_distances_batch,
                    dinov2_image_image_distances_batch)
from .svg import (extract_svg, safe_svg_to_image)



def render_response_to_image(response):
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
    if "</think> <answer>" in response and "</answer>" in response and "</text>" not in response:
        info["formatted"] = True
        # Extract content between <answer> tags
        answer_content = response.split("<answer>")[-1].replace("</answer>", "")
        svg_content = extract_svg(answer_content)
        
        if not svg_content:
            info["error"] = "No SVG found in answer"
            return None, info
        
        try:
            image = safe_svg_to_image(svg_content)
            
            if image is None:
                info["error"] = "Failed to render SVG"
                return None, info
                
            info["success"] = True
            return image, info
            
        except Exception as e:
            info["error"] = str(e)
            return None, info
    else:
        # Format is incorrect - missing proper tags
        info["error"] = "Missing <think>/<answer> tags or contains <text>"
        return None, info
    
    

def answer_tag_reward_fn(model_responses, prompts, images=None):
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
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    num_examples = len(model_responses)
    
    # Initialize results structure
    results = {
        "rewards": [0.0] * num_examples,
        "clip_reward": [0.0] * num_examples,
        "dino_reward": [0.0] * num_examples,
        "svg_info": [{"success": False} for _ in range(num_examples)],
        "rendered_images": [None] * num_examples,
        "formatted": [False] * num_examples
    }
    
    # Step 1: Check format and extract SVG content
    for i, response in enumerate(model_responses):
        
        rendered_image, info = render_response_to_image(response)
        results["rendered_images"][i] = rendered_image
        results["svg_info"][i] = info
        results["formatted"][i] = info.get("formatted", False)
    
    
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
        
        clip_scores = clip_text_image_distances_batch(valid_prompts, valid_rendered_images)
        # Update results with CLIP scores
        for i, idx in enumerate(valid_indices):
            score = clip_scores[i]
            clip_reward = 1.0 - score  # Convert distance to similarity
            results["rewards"][idx] += clip_reward
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
            dino_scores = dinov2_image_image_distances_batch( img_references, img_rendered)
            # Update results with DINOv2 scores
            for i, idx in enumerate(img_valid_indices):
                score = dino_scores[i]
                dino_reward = 1.0 - score  # Convert distance to similarity
                results["rewards"][idx] += dino_reward  # Add to existing CLIP reward
                results["dino_reward"][idx] = float(dino_reward)
    
    # Adjust rewards based on formatting - only give positive rewards if properly formatted
    for i in range(num_examples):
        if not results["formatted"][i]:
            results["rewards"][i] = 0.0
    
    return results