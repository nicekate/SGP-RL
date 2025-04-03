
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
import logging
import wandb
import math
import os
import socket
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

import deepspeed
import launchpad as lp
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import tree
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers.trainer import get_scheduler

from oat.actors.base import ActorBase

from oat.types import PreferenceData, TrajectoryData
from oat.utils.data import get_datasets, get_tokenizer
from oat.utils.deepspeed import get_strategy
from oat.utils.distributed import (
    init_process_group,
    node_ip_address_from_perspective,
    torch_type_codec,
)
from oat.utils.ipc import PlasmaShmClient, PlasmaShmServer
from oat.utils.launcher import DistributedLauncher
from oat.utils.ops import disable_dropout
from PIL import Image
from understand_r1_zero.oracle import SVGOracle

import functools
import itertools
import logging
import time
from dataclasses import dataclass, field
from multiprocessing import Pool, TimeoutError
from typing import Any, List, Literal, Tuple, Dict

import numpy as np
import torch
import tree
from oat.utils.launcher import DistributedLauncher
from tqdm import tqdm

from oat.actors.base import ActorBase
from oat.algorithms.ppo import PPOActor, PPOArgs, PPOLearner
from oat.args import default_args_validation, get_default_args
from oat.interface import get_program, lp
from oat.oracles.base import PreferenceOracleBase, RewardOracleBase
from oat.types import Metric, TrajectoryData
from oat.utils.data import load_data_from_disk_or_hf
from understand_r1_zero.dataset import PromptImageDataset
from dataset.registry import get_dataset_class

from oat.utils.ops import masked_mean, masked_sum
from torch.utils.data import DataLoader

from datasets import load_from_disk


from .svg_grader import render_response_to_image
from .collector import FlexFeedbackCollector
from .args import ZeroSVGArgs
class ZeroSVGActor(PPOActor):
    def __init__(self, ipc_server, vllm_args, args: ZeroSVGArgs) -> None:
        super().__init__(ipc_server, vllm_args, args)
        self.wandb_initialized = False

        self.oracle = SVGOracle(
        )

        # if args.prompt_template in ["qwen_math", "no"]:
        #     # These two templates are better used for Qwen models, which can themselves stop generation. Hence we unset all external stopping conditions.
        #     self.sampling_params.stop = None
        #     self.sampling_params.stop_token_ids = None
        #     self.eval_sampling_params.stop = None
        #     self.eval_sampling_params.stop_token_ids = None
        # elif args.prompt_template == "r1":
        # Let's stop when the model completes its answer.
        self.sampling_params.stop = ["</answer>"]
        self.sampling_params.include_stop_str_in_output = True
        self.eval_sampling_params.stop = ["</answer>"]
        self.eval_sampling_params.include_stop_str_in_output = True
    def _init_wandb(self, wandb_info):
        """Initialize wandb with the configuration from the learner"""
        if not self.wandb_initialized:
            import wandb
            import os
            
            # Prevent tokenizer parallelism warnings
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            
            wandb.init(
                id=wandb_info["run_id"],
                project=wandb_info.get("project", self.args.wb_project),
                name=wandb_info.get("name", self.args.wb_run_name),
                config=wandb_info.get("config", vars(self.args)),
                resume="allow"
            )
               
                
            self.wandb_initialized = True
            return wandb
                
            
        return None
    
    
    
    def step(
        self,
        prompts: List[str],
        formatted_prompts: List[str],
        references: Dict[str, Any] = None,
        log_completions = False,
    ) -> List[TrajectoryData]:
        """Main logic for the actor to generate trajectories (reasoning traces)."""
        assert not self.eval_mode
        info = {}
        logging.info(f"actor start")
        
    
        # step 1. generate
        st = time.time()
        outputs = self.generate(formatted_prompts, self.sampling_params)

        candidates = []
        prompt_token_ids = []
        no_eos = []
        response_ids = []
        response_logprobs = []
        resp_lens = []
        seq_logps = []
        for i in range(len(outputs)):
            # for each prompt
            prompt_token_ids.append(outputs[i].prompt_token_ids)
            candidates.append([])
            response_logprobs.append([])
            response_ids.append([])
            for k in range(self.sampling_params.n):
                # for each response
                candidates[i].append(outputs[i].outputs[k].text)
                no_eos.append(outputs[i].outputs[k].finish_reason == "length")
                token_ids = outputs[i].outputs[k].token_ids
                logps = outputs[i].outputs[k].logprobs
                logps = [item[token_ids[i]].logprob for i, item in enumerate(logps)]
                response_logprobs[i].append(logps)
                response_ids[i].append(token_ids)
                resp_lens.append(len(token_ids))
                seq_logps.append(np.sum(logps))
        

        info["actor/generate_time"] = time.time() - st
        info['actor/entropy'] = -np.mean(seq_logps)

        # step 2. verify
        st = time.time()
        rewards, oracle_infos = self.oracle.get_reward(
            tree.flatten(candidates),
            list(
                itertools.chain.from_iterable(
                    itertools.repeat(x, self.sampling_params.n) for x in prompts
                )
            ),
             list(
                itertools.chain.from_iterable(
                    itertools.repeat(x, self.sampling_params.n) for x in references
                )
            )
            
        )

        info["actor/verify_time"] = time.time() - st
        logging.info(f"actor reward {rewards.mean()}")
        info["actor/rewards"] = rewards.mean().item()
        info["actor/num_data"] = rewards.numel()
        info["actor/formatted"] = np.mean([i["formatted"] for i in oracle_infos])
        info["actor/response_tok_len"] = np.mean(resp_lens)
        info["actor/sampling_max_tokens"] = self.sampling_params.max_tokens
        info["actor/sampling_temperature"] = self.sampling_params.temperature

        rewards = rewards.reshape(len(prompts), -1)
        no_eos = np.array(no_eos).reshape(len(prompts), -1)
        info["actor/no_eos_count"] = no_eos.sum()
        prompts_to_log = []
        responses_to_log = []
        images_to_log = [x["rendered_images"] for x in oracle_infos]
        rewards_to_log = []
        dino_to_log = torch.tensor([x["dino_reward"] for x in oracle_infos]).reshape(len(prompts), -1)
        clip_to_log = torch.tensor([x["clip_reward"] for x in oracle_infos]).reshape(len(prompts), -1)
        info["actor/dino_rewards"] = dino_to_log.mean().item()
        info["actor/clip_rewards"] = clip_to_log.mean().item()
        info["actor/valid_images_ratio"] = len([x for x in images_to_log if x is not None]) / len(images_to_log)
        trajectory_data = []
        count = 0
        for i in range(len(candidates)):
            prompt = prompts[i]
            
            candidates_per_prompt = candidates[i]
            for j in range(len(candidates_per_prompt)):
                trajectory_info = info.copy()
                
                prompts_to_log.append(prompt)
                responses_to_log.append(candidates_per_prompt[j])
                rewards_to_log.append(rewards[i][j].item())
                
                reward = rewards[i][j].item()
                if no_eos[i][j]:
                    # Set zero reward for truncated outputs.
                    reward = 0
                dense_rewards = [0] * len(response_ids[i][j])
                dense_rewards[-1] = reward
                if log_completions:
                    
                    image_to_log = images_to_log[count]
                    count += 1 
                    logging_data = {
                        "prompt": prompt,
                        "completion": candidates_per_prompt[j],
                        "reward": rewards[i][j].item(),
                        "dino_reward": dino_to_log[i][j].item(),
                        "clip_reward": clip_to_log[i][j].item(),
                    }
                    if image_to_log:
                        # Convert image to base64 string for transmission
                        import io
                        import base64
                        buffered = io.BytesIO()
                        image_to_log.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        logging_data["rendered_image"]=img_str
                    else:
                        logging_data["rendered_image"]=None
                    trajectory_info['actor/logging_data'] = logging_data
                    
                    
                trajectory_data.append(
                    TrajectoryData(
                        prompt=prompt,
                        prompt_ids=prompt_token_ids[i],
                        response=candidates_per_prompt[j],
                        response_ids=response_ids[i][j],
                        response_logprobs=response_logprobs[i][j],
                        rewards=dense_rewards,
                        loss_mask=not no_eos[i][j] if self.args.ignore_no_eos else True,
                        info=trajectory_info,
                    )
                )
        
                    

            
        logging.info(f"actor finished data_len={len(trajectory_data)}")
        handle = self.ipc_client.serialize_ipc(trajectory_data)
        return handle
