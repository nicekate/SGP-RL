from multiprocessing import Pool, TimeoutError
from typing import Any, List, Tuple, Dict
import torch

from PIL import Image
from oat.oracles.base import PreferenceOracleBase, RewardOracleBase
from oat.types import Metric

# from .clips import (clip_text_image_distances_batch,
#                                      dinov2_image_image_distances_batch)
# from .svg import (extract_svg, safe_svg_to_image)
from .svg_grader import answer_tag_reward_fn
class SVGOracle(RewardOracleBase, PreferenceOracleBase):
    """Defines the verification rules for SVG generation."""

    def __init__(self) -> None:
        super().__init__()
        self.svg_reward_fn = answer_tag_reward_fn
        
        # Process pool is used to enable the timeout mechanism for answer grading in our distributed training setup.
        self.mp_pool = Pool(2)
        
    def get_reward(
        self,
        responses: List[str],
        prompts: List[str],
        images: List[Image.Image] = None,
        batch_size: int = 128,
    ) -> Tuple[torch.Tensor, Metric]:
        """Process reward calculation in batches for better efficiency and memory usage."""
        num_samples = len(responses)
        all_rewards = []
        all_infos = []
        
        # Process in batches
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            
            # Extract current batch
            batch_responses = responses[batch_start:batch_end]
            batch_prompts = prompts[batch_start:batch_end]
            batch_images = images[batch_start:batch_end] if images is not None else None
            
            # Process batch asynchronously
            batch_results = self.svg_reward_fn(batch_responses, batch_prompts, batch_images)
            
           
            batch_rewards = batch_results["rewards"]
            batch_infos = []
            
            # Organize batch info dictionaries
            for i in range(len(batch_rewards)):
                info_dict = {k: v[i] if isinstance(v, list) and i < len(v) else None 
                            for k, v in batch_results.items() if k !="rewards" }
                batch_infos.append(info_dict)
            
            # Add to overall results
            all_rewards.extend(batch_rewards)
            all_infos.extend(batch_infos)
                
            # except TimeoutError:
            #     # Handle timeout for this batch
            #     batch_size = batch_end - batch_start
            #     all_rewards.extend([0.0] * batch_size)
            #     all_infos.extend([{"formatted": False, "error": "Batch processing timeout"}] * batch_size)
        
        return torch.tensor(all_rewards), all_infos
    def compare(
        self,
        inputs: List[str],
        candidates_A: List[str],
        candidates_B: List[str],
        batch_size: int = 4,
        return_probs: bool = False,
        disable_tqdm: bool = False,
    ) -> Tuple[List[Any], Metric]:
        """Facilitates easier evaluation, returning accuracy as winning probability."""
        return 0, {}