import time
from typing import List, Union, Dict, Any

import torch
import tree
import Levenshtein
import numpy as np

from oat.collectors.base import FeedbackCollector
from oat.types import PreferenceData, TrajectoryData

class FlexFeedbackCollector(FeedbackCollector):
    """
    A more flexible feedback collector that allows passing additional arguments
    to the actor's step method.
    """
    
    def get_metrics(
        self,
        actor_time: float,
        feedback_data: List[Union[PreferenceData, TrajectoryData]],
    ):
        metric = {
            "actor/total_time": actor_time,
        }
        if isinstance(feedback_data[0], PreferenceData):
            metric.update(
                {
                    "actor/chosen_avg_str_len": np.mean(
                        [len(p.chosen_response) for p in feedback_data]
                    ),
                    "actor/rejected_avg_str_len": np.mean(
                        [len(p.rejected_response) for p in feedback_data]
                    ),
                    "actor/init_clash_ratio": np.mean(
                        [p.init_clash for p in feedback_data]
                    ),
                    "actor/loss_mask": np.mean([p.loss_mask for p in feedback_data]),
                    "actor/pair_edit_dist": np.mean(
                        [
                            Levenshtein.distance(p.chosen_response, p.rejected_response)
                            for p in feedback_data
                        ]
                    ),
                    "actor/chosen_id": np.mean([p.chosen_id for p in feedback_data]),
                }
            )
        elif isinstance(feedback_data[0], TrajectoryData):
            metric.update(
                {
                    "actor/generate_avg_str_len": np.mean(
                        [len(t.response) for t in feedback_data]
                    )
                }
            )
        else:
            raise ValueError("Invalid feedback data type.")
        
        filtered_infos = []
        for p in feedback_data:
            # Create a filtered copy of the info dictionary without logging_data
            filtered_info = {k: v for k, v in p.info.items() if  'actor/logging_data' not in k}
            filtered_infos.append(filtered_info)
        
        # Compute means only on the filtered info dictionaries
        if filtered_infos:
            mean_info = tree.map_structure(
                lambda *x: np.mean(x), *filtered_infos
            )
            metric.update(mean_info)

        return metric

 
    
    def collect_feedback(
        self,
        prompts: Union[str, List[str]],
        formatted_prompts: List[str],
        refs: Union[str, List[str]],
        **kwargs
    ):
        """
        Collect feedback from actors, allowing for additional parameters to be passed to actor.step.
        
        Args:
            prompts: The original prompts
            formatted_prompts: The formatted prompts ready for model input
            refs: Reference data for evaluation
            **kwargs: Additional keyword arguments to pass to actor.step
        
        Returns:
            Tuple of (feedback_data, metrics)
        """
        # Start timing
        st_time = time.time()

        # Get the actor for the current rank
        rank = torch.distributed.get_rank()
        actor = self.actors[rank % len(self.actors)]
        
        # # Extract wandb configuration for serialization
        
        # wandb_info = self._extract_wandb_info(kwargs)
        
        # # Replace non-serializable objects with serializable info
        # if wandb_info and "logging_tools" in kwargs:
        #     # Remove the actual wandb object and replace with serializable info
        #     del kwargs["logging_tools"]
            
        #     # Add wandb info to kwargs
        #     kwargs["wandb_info"] = wandb_info
        # else:
        #     kwargs["wandb_info"] = None
        #     del kwargs["logging_tools"]
        
        # Call step with the additional arguments
        if self.args.online_evaluation:
            handle = actor.step(prompts, formatted_prompts, refs, **kwargs)
        else:
            handle = actor.step(prompts, formatted_prompts, **kwargs)
        
        # Deserialize the feedback data
        feedback_data: List[Union[PreferenceData, TrajectoryData]] = (
            self.ipc_client.deserialize_ipc(handle)
        )

        # Calculate metrics
        actor_time = time.time() - st_time
        return feedback_data, self.get_metrics(actor_time, feedback_data)
    
    def _extract_wandb_info(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract wandb configuration from kwargs in a serializable format
        """
        wandb_info = {}
        if not "logging_tools" in kwargs:
            return wandb_info
        if kwargs["logging_tools"] is None:
            return wandb_info
        
        # Check for wandb in logging_tools
        if "logging_tools" in kwargs and "wb" in kwargs["logging_tools"]:
            wb = kwargs["logging_tools"]["wb"]
            
            # Extract run ID if available
            if hasattr(wb, "run") and wb.run:
                wandb_info["run_id"] = wb.run.id
                wandb_info["project"] = getattr(wb.run, "project", self.args.wb_project)
                wandb_info["name"] = getattr(wb.run, "name", self.args.wb_run_name)
            
            # Include step information
            if "steps" in kwargs["logging_tools"]:
                wandb_info["steps"] = kwargs["logging_tools"]["steps"]
            
            # Add config
            if hasattr(self.args, "__dict__"):
                wandb_info["config"] = self.args.__dict__
        
        return wandb_info