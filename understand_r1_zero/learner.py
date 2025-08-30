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
import gc
import logging
import wandb
import math
import os
import socket
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

from collections import defaultdict


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
from understand_r1_zero.dataset import PromptImageDataset, PromptSVGDataset, PromptImageSVGDataset, PureTextDataset
from dataset.registry import get_dataset_class

from oat.utils.ops import masked_mean, masked_sum
from torch.utils.data import DataLoader

from datasets import load_from_disk


from .svg_grader import render_response_to_image
from .collector import FlexFeedbackCollector
from .args import ZeroSVGArgs

def apply_r1_template(question: str):
    return (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: "
        + question
        + "\nAssistant: <think>"
    )
    
def apply_r1_svg_template(question: str):
    return (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: Please write SVG code for generating the image corresponding to the following description: "
        + question
        + "\nAssistant: <think>"
    )




TEMPLATE_FACTORY = {
    # "qwen_math": apply_qwen_math_template,
    "r1": apply_r1_template,
    "r1_svg": apply_r1_svg_template,
    # "no": apply_no_template,
}

svg_prompts = {
    "normal": "Please write SVG code for generating the image corresponding to the following description:",
    "sketch": "Please write SVG code for generating a sketch style image without color fills corresponding to the following description:",
    "bw": "Please write SVG code for generating a black and white image corresponding to the following description:",
    
}

class ZeroSVGLearner(PPOLearner):
    def _init(self, args: ZeroSVGArgs, actors: List[ActorBase]) -> None:
        super()._init(args, actors)
        self.collector = FlexFeedbackCollector(
                    args, actors, PlasmaShmClient(self.ipc_server)
                )
        # self.eval_math_dataset_dict = load_from_disk(args.eval_data)  # TODO: get fro HF.
        
        # if args.test_split != "all":
        #     self.eval_math_dataset_dict = {
        #         k: v for k, v in self.eval_math_dataset_dict.items() if k in args.test_split
        #     }
        self.args = args
        # Dr. GRPO Modification 1: Remove length bias by using masked_sum with a constant normalizer:
        self.masked_aggregator = (
            functools.partial(masked_sum, constant_normalizer=args.generate_max_length)
            if args.critic_type == "drgrpo"
            else masked_mean
        )
    def compute_sum_alllength_per_prompt(self, response_masks, loss_masks):
        """
        Calculate the sum of response lengths for each prompt, accounting for loss_masks.
        
        Args:
            response_masks: Boolean tensor of shape [batch_size, seq_len] where
                            True indicates response tokens
            loss_masks: Float tensor of shape [batch_size] indicating whether to include
                        each response in the calculations (1.0 = include, 0.0 = exclude)
        
        Returns:
            Tensor containing the sum of all valid response lengths for each prompt,
            repeated to match the original batch size
        """
        # Calculate length of each individual response
        response_lengths = response_masks.sum(dim=1)  # Shape: [batch_size]
        
        # Apply loss masks to zero out responses that shouldn't be counted
        masked_lengths = response_lengths * loss_masks  # Element-wise multiplication
        
        # Reshape to group by prompts (each prompt has self.args.num_samples responses)
        grouped_lengths = masked_lengths.view(-1, self.args.num_samples)
        
        # Sum the lengths for each prompt group
        sum_lengths_per_prompt = grouped_lengths.sum(dim=1)
        
        # Repeat each sum value num_samples times to match original batch shape
        repeated_sums = sum_lengths_per_prompt.repeat_interleave(self.args.num_samples, dim=0)
        
        # Ensure we don't divide by zero (replace zeros with 1.0)
        repeated_sums = torch.clamp(repeated_sums, min=1.0)
        
        return repeated_sums


    def compute_lsc_transformation(self, advantages):
        """
        Compute the LSC transformation for rewards.
        
        Args:
            rewards: Tensor of shape [batch_size, seq_len]
        
        Returns:
            Tensor of transformed rewards with shape [batch_size]
        """
        
        
        # Get lambda parameter from args
        lam = self.args.lsc_lam
        
        # Apply the smoothing function
        if lam == 0:
            return advantages
        else:
            # Implement f_smooth(x, lam) = (2/lam) * (log(1 + exp(lam * x)) - log(2))
            advantages = (2.0 / lam) * (torch.log1p(torch.exp(lam * advantages)) - torch.log(torch.tensor(2.0, device=advantages.device)))
            values = advantages.view(-1, self.args.num_samples).mean(dim=1)
            values = values.repeat_interleave(self.args.num_samples, dim=0)
            advantages = advantages - values
            std_grouped_rewards = advantages.view(-1, self.args.num_samples).std(dim=1)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(
                self.args.num_samples, dim=0
            )
            advantages = advantages / (std_grouped_rewards + 1e-8)
            return advantages
            
    def compute_exp_transformation(self, advantages):
        """
        Compute the exponential transformation for advantages.
        
        Args:
            advantages: Tensor of shape [batch_size]
        
        Returns:
            Tensor of transformed advantages with shape [batch_size]
        """
        # Get alpha parameter from args
        alpha = self.args.exp_alpha
        
        # Apply the exponential function
        if alpha == 0:
            return advantages
        else:
            # Apply sign(a) * exp(a * x) transformation
            
            signs = 1 if alpha>0 else -1
           
            transformed_advantages = signs * torch.exp(alpha * advantages)
            
            # Whiten the data (similar to LSC transformation)
            # Group by prompts and compute statistics
            values = transformed_advantages.view(-1, self.args.num_samples).mean(dim=1)
            values = values.repeat_interleave(self.args.num_samples, dim=0)
            
            # Subtract mean
            transformed_advantages = transformed_advantages - values
            
            # Normalize by standard deviation
            std_grouped_rewards = transformed_advantages.view(-1, self.args.num_samples).std(dim=1)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(
                self.args.num_samples, dim=0
            )
            transformed_advantages = transformed_advantages / (std_grouped_rewards + 1e-8)
            
            return transformed_advantages

    
    # Dr. GRPO Modification 2: Remove difficulty bias by just computing the MC advantage without dividing by std:
    def compute_monte_carlo_advantages(self, rewards):
        rewards = rewards.sum(-1)
        # Compute monte carlo trajectory-level advantage
        values = rewards.view(-1, self.args.num_samples).mean(dim=1)
        values = values.repeat_interleave(self.args.num_samples, dim=0)
        advantages = rewards - values
        if self.args.critic_type == "grpo" and not self.args.adv_no_std:
            # Additionally normalize by std.
            std_grouped_rewards = rewards.view(-1, self.args.num_samples).std(dim=1)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(
                self.args.num_samples, dim=0
            )
            advantages = advantages / (std_grouped_rewards + 1e-8)
            advantages = self.compute_lsc_transformation(advantages)
            advantages = self.compute_exp_transformation(advantages)
        
        return advantages
    # Add this method to ZeroSVGLearner class
    def compute_batch_normalized_advantages(self, rewards):
        """
        Compute advantages by normalizing rewards across the entire batch.
        
        Args:
            rewards: Reward tensor with shape [batch_size, seq_len]
        
        Returns:
            Tensor of batch-normalized advantages with shape [batch_size]
        """
        # Sum rewards across sequence length dimension
        rewards = rewards.sum(-1)
        
        # Compute batch-level statistics
        batch_mean = rewards.mean()
        batch_std = rewards.std()
        
        # Normalize using batch statistics
        advantages = (rewards - batch_mean) / (batch_std + 1e-8)
        
        return advantages
    def _apply_template(self, example):
        problem = example[self.args.input_key]
        example[self.args.input_key] = TEMPLATE_FACTORY[self.args.prompt_template](problem)
        return example

    def prepare_data(self, strategy, tokenizer):
        
        svg_prompt_dataset = get_dataset_class(self.args.prompt_data_svg)().load_dataset(
            self.args.prompt_data_svg, 
            None, 
            max_train_samples=self.args.max_train_svg,
            max_test_samples=500,
            instruction_prompt = svg_prompts[self.args.prompt_type],
        )
        # math_prompt_dataset = load_data_from_disk_or_hf(self.args.prompt_data_math)
        
        # prompt_dataset = load_data_from_disk_or_hf(self.args.prompt_data)
        svg_prompts_data = svg_prompt_dataset[self.args.train_split_svg]
        # math_prompts_data = math_prompt_dataset[self.args.train_split_math].select(
        #     range(min(self.args.max_train_math, len(math_prompt_dataset[self.args.train_split_math])))
        # )
        
        
        # Prepare the data: templated questions & gt final answers.
        # prompts_data = prompts_data.map(self._apply_template)
        # print("prompts_data_apply_template", prompts_data[0])
        
        if self.args.prompt_data_svg == 'hq_svg':
            self.prompts_dataset = PromptSVGDataset(
                svg_prompts_data,
                tokenizer,
                strategy,
                input_key="solution",
                output_key="svg",
                apply_chat_template=False,  # Because we have applied already.
                get_reference=True,
            )
        elif self.args.prompt_data_svg == 'coco_mix':
            self.prompts_dataset = PromptImageSVGDataset(
                svg_prompts_data,
                tokenizer,
                strategy,
                input_key="solution",
                apply_chat_template=False,  # Because we have applied already.
                get_reference=True,
            )
        elif self.args.prompt_data_svg == 'puretext':
            self.prompts_dataset = PureTextDataset(
                svg_prompts_data,
                tokenizer,
                strategy,
                input_key="solution",
                apply_chat_template=False,  # Because we have applied already.
                get_reference=True,
            )
        else:

            self.prompts_dataset = PromptImageDataset(
                svg_prompts_data,
                tokenizer,
                strategy,
                input_key="solution",
                output_key="image_path",
                apply_chat_template=False,  # Because we have applied already.
                get_reference=True,
            )
        # self.prompts_dataset = prompts_data
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset,
            strategy.args.rollout_batch_size_per_device,
            pin_memory=True,
            shuffle=True,
        )
        self.eval_prompts_dataset = self.eval_prompts_dataloader = (
            None  # We use our own `self.eval_dataset_dict`.
        )
        
        
        # svg_eval_dataset = get_dataset_class("uwunion/instruct_svg")().load_dataset(
        #     "uwunion/instruct_svg", 
        #     None, 
        #     max_test_samples=100,
        # )['train']
        
        
        # svg_eval_dataset = PromptSVGDataset(
        #     svg_eval_dataset,
        #     tokenizer,
        #     strategy,
        #     input_key="solution",
        #     output_key="svg",
        #     apply_chat_template=False,  # Because we have applied already.
        #     get_reference=True,
        # )
        # self.eval_svg_dataset_dict = {"instruct_svg":   svg_eval_dataset  }
        svg_eval_dataset = svg_prompt_dataset['test']
        if self.args.prompt_data_svg == 'hq_svg':
            svg_eval_dataset = PromptSVGDataset(
                svg_eval_dataset,
                tokenizer,
                strategy,
                input_key="solution",
                output_key="svg",
                apply_chat_template=False,  # Because we have applied already.
                get_reference=True,
            )
        elif self.args.prompt_data_svg == 'coco_mix':
            svg_eval_dataset = PromptImageSVGDataset(
                svg_eval_dataset,
                tokenizer,
                strategy,
                input_key="solution",
                apply_chat_template=False,  # Because we have applied already.
                get_reference=True,
            )
        elif self.args.prompt_data_svg == 'puretext':
            svg_eval_dataset = PureTextDataset(
                svg_eval_dataset,
                tokenizer,
                strategy,
                input_key="solution",
                apply_chat_template=False,  # Because we have applied already.
                get_reference=True,
            )
        else:
            svg_eval_dataset = PromptImageDataset(
                svg_eval_dataset,
                tokenizer,
                strategy,
                input_key="solution",
                output_key="image_path",
                apply_chat_template=False,  # Because we have applied already.
                get_reference=True,
            )
        self.eval_svg_dataset_dict = {self.args.prompt_data_svg:  svg_eval_dataset }
    
        

    def eval_math_dataloader_collate_fn(self, item_list):
        problems = []
        formatted_problems = []
        answers = []
        for item in item_list:
            problems.append(item["problem"])
            formatted_problems.append(
                TEMPLATE_FACTORY[self.args.prompt_template](item["problem"])
            )
            answers.append(item["answer"])
        return formatted_problems, problems, answers

    def evaluate_math(self, dataloader, steps):
        # Discard the default eval dataloader, and run eval on multiple benchmarks.
        del dataloader
        all_metrics = {}
        accuracies = []
        scores = []
        lens = []
        for benchmark_name, dataset in self.eval_math_dataset_dict.items():
            eval_prompts_dataloader = DataLoader(
                dataset,
                batch_size=self.args.eval_batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=self.eval_math_dataloader_collate_fn,
            )
            metrics = self.do_evaluate(
                eval_prompts_dataloader, f"{steps}_{benchmark_name}"
            )
            for k in metrics.keys():
                logging_data = None
                if "logging_data" in k:
                    logging_data = metrics[k]
            metrics = {k: v for k, v in metrics.items() if  "logging_data" not in k}    
            if logging_data is not None:
                self._log_eval_completions_to_wandb(logging_data, benchmark_name)
            all_metrics.update(
                {
                    k.replace("eval/", f"eval_math/{benchmark_name}/"): v
                    for k, v in metrics.items()
                }
            )
            accuracies.append(metrics["eval/accuracy"])
            scores.append(metrics["eval/score"])
            lens.append(metrics["eval/response_tok_len"])
        all_metrics.update(
            {
                "eval_math/average/accuracy": np.mean(accuracies),
                "eval_math/average/score": np.mean(scores),
                "eval_math/average/response_tok_len": np.mean(lens),
            }
        )
        
        return all_metrics

    def evaluate_svg(self, dataloader, steps):
        # Discard the default eval dataloader, and run eval on multiple benchmarks.
        del dataloader
        all_metrics = {}
        accuracies = []
        scores = []
        lens = []
        for benchmark_name, dataset in self.eval_svg_dataset_dict.items():
            eval_prompts_dataloader = DataLoader(
                dataset,
                batch_size=self.args.eval_batch_size,
                shuffle=False,
                drop_last=False,
            )
            metrics = self.do_evaluate(
                eval_prompts_dataloader, f"{steps}_{benchmark_name}"
            )
            for k in metrics.keys():
                logging_data = None
                additional_metrics = None
                if "logging_data" in k:
                    logging_data = metrics[k]
                    additional_metrics =self._prepare_additional_metrics(logging_data)
                    
            metrics = {k: v for k, v in metrics.items() if  "logging_data" not in k}    
            if logging_data is not None and self.strategy.is_rank_0():
                self._log_eval_completions_to_wandb(logging_data, benchmark_name)
                if additional_metrics is not None:
                    additional_metrics = {
                        k.replace("eval/", f"eval_svg/{benchmark_name}/"): v
                        for k, v in additional_metrics.items()
                    }
                    self._wandb.log(additional_metrics,step=self.steps)
                
            all_metrics.update(
                {
                    k.replace("eval/", f"eval_svg/{benchmark_name}/"): v
                    for k, v in metrics.items()
                }
            )
            accuracies.append(metrics["eval/accuracy"])
            scores.append(metrics["eval/score"])
            lens.append(metrics["eval/response_tok_len"])
        all_metrics.update(
            {
                "eval_svg/average/accuracy": np.mean(accuracies),
                "eval_svg/average/score": np.mean(scores),
                "eval_svg/average/response_tok_len": np.mean(lens),
            }
        )
        return all_metrics




    def run(self):
        self._init(self.args, self.actors)

        self.steps = 0
        early_stop = False
        self.start_time = time.time()

        self.actor_info = {}

        if not self.strategy.args.debug:
            self.eval_and_log({}, eval=True, save=False)

        self.steps = 1
        self.gradient_update_st = time.time()
        for p_ep in range(self.args.num_prompt_epoch):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(p_ep)
            progress_bar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Prompt epoch [{p_ep + 1}/{self.args.num_prompt_epoch}]",
                disable=not self.strategy.is_rank_0(),
            )

            for processed_prompts, raw_prompts, refs in self.prompts_dataloader:
                if self.steps <= self.args.skip_steps:
                    progress_bar.update()
                    self.steps += 1
                    continue
                    
                if early_stop:
                    break
                # Call actor.step remotely to generate rollout & collect feedback.
                if self.strategy.is_rank_0() and self.steps % self.args.log_completion_steps == 0 and self._wandb is not None:
                    log_completions = True
                else:
                    log_completions = False
                feedback_data, self.actor_info = self.collector.collect_feedback(
                    raw_prompts, processed_prompts, refs, log_completions = log_completions                    
                )
                

                if feedback_data is None:
                    # Asynchronous prefilling, data is stored in collector's buffer.
                    continue
                if self.strategy.is_rank_0() and self.steps % self.args.log_completion_steps == 0 and self._wandb is not None:
                    self._log_completions_to_wandb(feedback_data)
                self.prompt_consumed += len(feedback_data)

                self.process_feedback_data(feedback_data)

                if (
                    self.args.dump_replay_every > 0
                    and self.steps % self.args.dump_replay_every == 0
                ):
                    if not self.strategy.is_rank_0():
                        dist.gather_object(self.pi_buffer)
                    else:
                        gather_all_buffer = [None] * self.strategy.world_size
                        dist.gather_object(self.pi_buffer, gather_all_buffer)
                        pd.to_pickle(
                            (processed_prompts, refs, gather_all_buffer),
                            os.path.join(
                                self.save_path, f"buffer_step{self.steps:05}.pkl"
                            ),
                        )

                if self.steps % self.update_interval == 0:
                    self._pre_learning()
                    train_info = self.learn(self.steps // self.update_interval)
                    self._post_learning()

                    self.eval_and_log(train_info)

                    if (
                        self.steps // self.update_interval
                    ) % self.args.sync_params_every == 0:
                        self.sync_params_to_actors()
                    if (
                        self.steps // self.update_interval
                    ) % self.args.buffer_clear_every == 0:
                        self.pi_buffer.clear()

                progress_bar.update()
                self.steps += 1

                # if self.get_current_query() > self.args.max_queries:
                #     early_stop = True

            self.prompt_epoch = p_ep + 1

        self.eval_and_log(train_info, eval=True, save=True)
        

        if self.args.dump_all_buffer:  # For debug purpose.
            if not self.strategy.is_rank_0():
                dist.gather_object(self.all_buffer)
            else:
                gather_all_buffer = [None] * self.strategy.world_size
                dist.gather_object(self.all_buffer, gather_all_buffer)
                pd.to_pickle(
                    gather_all_buffer, os.path.join(self.save_path, "all_buffer.pkl")
                )

        if self.strategy.is_rank_0():
            self._wandb.finish() if self._wandb else None
            lp.stop()
    
    def eval_and_log(self, train_info, eval=False, save=False, log_completions = False):
        # eval
        eval_info = {}
        if (self.args.eval_steps > 0 and eval) or self._should_do(self.args.eval_steps):
            eval_info = self.evaluate_svg(None, self.steps)

        # save
        
        if (self.args.save_steps > 0 and save) or (
            self.steps > 0
            and self._should_do(self.args.save_steps)
            and self.steps >= self.args.save_from
        ):
            self.strategy.save_model(
                self.model,
                self.tokenizer,
                os.path.join(self.save_path, "saved_models"),
                tag="step_{:05d}".format(self.steps),
                max_num=self.args.max_save_num,
                max_mem=self.args.max_save_mem,
            )

        # logs
        if eval_info or self.steps % self.args.logging_steps == 0:
            misc_info = self.get_misc_info()
            last_lr = self.scheduler.get_last_lr()[0]
            misc_info["lr"] = last_lr

            misc_info = {
                "misc/%s" % k: v
                for k, v in {
                    **misc_info,
                }.items()
            }
            logs_dict = {**train_info, **eval_info, **self.actor_info, **misc_info}
            logs_dict = self.strategy.all_reduce(logs_dict)
            logs_dict.update(
                self.strategy.all_reduce(
                    {
                        "misc/query_step": self.query_step,
                        "misc/prompt_consumed": self.prompt_consumed,
                    },
                    op="sum",
                )
            )

            if self.strategy.is_rank_0():
                if self.pi_buffer:
                    self.strategy.print(np.random.choice(self.pi_buffer))
                    
                self.strategy.pprint(logs_dict)
                if self._wandb is not None:
                    self._wandb.log(logs_dict,step=self.steps)
                   
   
   
    def _prepare_additional_metrics(self, all_logging_data):
        additional_metrics = {}
        if all_logging_data:
            assert "formatted" in all_logging_data and all_logging_data["formatted"]
            formatted_mask = np.array(all_logging_data["formatted"], dtype=bool)
            formatted_count = formatted_mask.sum()
            additional_metrics["eval/formatted_rate"] = formatted_mask.mean()
            additional_metrics["eval/formatted_count"] = formatted_count
            
        
            
            # Extract clip and dino model-specific rewards, but only average over formatted samples
            clip_model_keys = [key for key in all_logging_data.keys() if key.startswith("clip_") and key.endswith("_reward")]
            for key in clip_model_keys:
                if all_logging_data[key]:
                    values = np.array(all_logging_data[key])
                    # Average only over formatted samples
                    additional_metrics[f"eval/{key}"] = values[formatted_mask].mean()
                    # # Also keep the full average for comparison
                    # additional_metrics[f"eval/{key}_all"] = values.mean()
            
            dino_model_keys = [key for key in all_logging_data.keys() if key.startswith("dino_") and key.endswith("_reward")]
            for key in dino_model_keys:
                if all_logging_data[key]:
                    values = np.array(all_logging_data[key])
                    # Average only over formatted samples
                    additional_metrics[f"eval/{key}"] = values[formatted_mask].mean()
                    # # Also keep the full average for comparison
                    # additional_metrics[f"eval/{key}_all"] = values.mean()
            
            # Extract aggregated rewards if present
            if "clip_reward" in all_logging_data and all_logging_data["clip_reward"]:
                values = np.array(all_logging_data["clip_reward"])
                additional_metrics["eval/clip_reward"] = values[formatted_mask].mean()
                # additional_metrics["eval/clip_reward_all"] = values.mean()
            
            if "dino_reward" in all_logging_data and all_logging_data["dino_reward"]:
                values = np.array(all_logging_data["dino_reward"])
                additional_metrics["eval/dino_reward"] = values[formatted_mask].mean()
                # additional_metrics["eval/dino_reward_all"] = values.mean() 
            return additional_metrics             
    def _log_completions_to_wandb(self, feedback_data):
        """Process and log completion data to wandb from actor feedback."""
        import wandb
        import pandas as pd
        from PIL import Image
        import io
        import base64
        import numpy as np
        
        # Collect logging data from all trajectories
        all_logging_data = []
        for traj in feedback_data:
            if "actor/logging_data" in traj.info:
                all_logging_data.append(traj.info["actor/logging_data"])
        
        if not all_logging_data:
            return
        
        # Combine all logging data
        combined_data = {
            "prompt": [],
            "completion": [],
            "reward": [],
            "dino_reward": [],
            "clip_reward": [],
            "rendered_image": []
        }
        
        for data in all_logging_data:
            combined_data["prompt"].append(data["prompt"])
            combined_data["completion"].append(data["completion"])
            combined_data["reward"].append(data["reward"])
            
            combined_data["dino_reward"].append(data["dino_reward"])
            
            combined_data["clip_reward"].append(data["clip_reward"])
            img_str = data["rendered_image"]
            
            # Convert base64 strings back to images
            
            if img_str:
                try:
                    img_bytes = base64.b64decode(img_str)
                    img = Image.open(io.BytesIO(img_bytes))
                    combined_data["rendered_image"].append(wandb.Image(img))
                except:
                    placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
                    combined_data["rendered_image"].append(wandb.Image(Image.fromarray(placeholder)))
            else:
                placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
                combined_data["rendered_image"].append(wandb.Image(Image.fromarray(placeholder)))
        
        # Create and log the table
        df = pd.DataFrame(combined_data)
        step_name = f"train_completions/step_{self.steps}"
        
        
        wandb.log({step_name: wandb.Table(dataframe=df)}, step=self.steps)
        
    def _log_eval_completions_to_wandb(self, logging_data, benchmark_name):
        """Process and log completion data to wandb from actor feedback."""
        import wandb
        import pandas as pd
        from PIL import Image
        import io
        import base64
        import numpy as np
        
        # Collect logging data from all trajectories
        
        
        # Combine all logging data
        
        
        
        if "rendered_image"  in logging_data:
            imgs = logging_data["rendered_image"]
            images_to_log = []
            for image in imgs:
                if image:
                   images_to_log.append(wandb.Image(image))
                else:
                    placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
                    images_to_log.append(wandb.Image(Image.fromarray(placeholder)))
            logging_data["rendered_image"] = images_to_log
                    
            
        
        # Create and log the table
        df = pd.DataFrame(logging_data)
        step_name = f"eval_completions/{benchmark_name}/step_{self.steps}"
        
        
        wandb.log({step_name: wandb.Table(dataframe=df)}, step=self.steps)
                


    def do_evaluate(self, dataloader, steps):
        self.strategy.print(f"Start generating evaluation responses at step {steps}")
        st_time = time.time()
        # 1) Let Actors cache the current behavior policy.
        if self.strategy.is_rank_0():
            done = [actor.futures.notify_eval_start() for actor in self.actors]
            _ = [d.result() for d in done]

        # 2) Push the latest policy for fast vLLM generation.
        dist.barrier()
        self._broadcast_to_vllm()

        # 3) Generate and process results
        win_rate = 0
        scores = 0
        accuracy = 0
        response_len = 0
        eval_count = 0
        all_logging_data = {}
        additional_metrics = {}
        if self.strategy.is_rank_0():
            processed_prompts = []
            prompts = []
            responses = []
            references = []
            futs = []
            scores = []
            wins = []
            accuracies = []
            
            progress_bar = tqdm(range(len(dataloader)), desc="Evaluating")
            for i, (batch_processed_prompts, batch_prompts, refs) in enumerate(
                dataloader
            ):
                eval_count += len(batch_prompts)
                processed_prompts.extend(batch_processed_prompts)
                prompts.extend(batch_prompts)
                refs_list = [x for x in refs]
                references.extend(refs_list)

                actor = self.actors[i % len(self.actors)]
            
                fut = actor.futures.generate_and_maybe_eval(
                    batch_prompts, batch_processed_prompts, refs_list
                    
                )
                futs.append(fut)
                if len(futs) == len(self.actors) or i == len(dataloader) - 1:
                    for fut in futs:
                        resp, score, logging_data = fut.result()
                        if logging_data:
                            for key, values in logging_data.items():
                                if key not in all_logging_data:
                                    all_logging_data[key] = []
                                all_logging_data[key].extend(values)
                        responses.extend(resp)
                        wins.extend(score > 0.5)  # For preference learning.
                        accuracies.extend(score == 1)  # For RL with verifiable rewards.
                        scores.extend(score)
                    futs.clear()
                progress_bar.update()

            eval_res_path = os.path.join(self.save_path, "eval_results")
            os.makedirs(eval_res_path, exist_ok=True)
            # pd.DataFrame(
            #     {
            #         "prompts": prompts,
            #         "output": responses,
            #         "scores": scores,
            #         f"format_prompts": processed_prompts,
            #         "reference": references,
            #         "generator": self.args.wb_run_name,
            #     }
            # ).to_json(
            #     os.path.join(eval_res_path, f"{steps}.json"),
            #     orient="records",
            #     indent=4,
            # )
            win_rate = np.mean(wins).item()
            scores = np.mean(scores).item()
            accuracy = np.mean(accuracies).item()
            response_len = np.mean(
                tree.map_structure(lambda x: len(self.tokenizer.encode(x)), responses)
            )
        
                
            
            
                

        dist.barrier()
        win_rate = self.strategy.broadcast(win_rate)
        scores = self.strategy.broadcast(scores)
        accuracy = self.strategy.broadcast(accuracy)
        response_len = self.strategy.broadcast(response_len)
        eval_count = self.strategy.broadcast(eval_count)
        
        # all_logging_data = self.strategy.broadcast(all_logging_data)
        # 4) Recover Actors' original behavior policy.
        if self.strategy.is_rank_0():
            done = [actor.futures.notify_eval_done() for actor in self.actors]
            _ = [d.result() for d in done]

        dist.barrier()
        

       
        return {
            "eval/rm_win_rate": win_rate,
            "eval/score": scores,
            "eval/accuracy": accuracy,
            "eval/eval_count": eval_count,
            "eval/elapse": time.time() - st_time,
            "eval/response_tok_len": response_len,
            "eval/logging_data": all_logging_data,
        }
        
        
        
        
    def learning_step(self, trajectory):
        args: PPOArgs = self.args
        infos = {}
        device = torch.cuda.current_device()
        input_ids = trajectory["input_ids"].to(device)
        att_mask = trajectory["attention_mask"].to(device)
        final_rewards = (
            torch.tensor([r[-1] for r in trajectory["rewards"]])
            .to(device)
            .reshape(-1, 1)
        ).float() * args.reward_scale
        prompt_id_lens = trajectory["prompt_ids_lens"]
        # action_logprobs = [
        #     torch.tensor(lp).to(device) for lp in trajectory["action_logprobs"]
        # ]
        loss_masks = torch.tensor(trajectory["loss_masks"]).float().to(device)
        completion_masks = self.get_completion_mask(att_mask, prompt_id_lens)
        response_masks = completion_masks[:, 1:]

        logging.info(f"learn data size {input_ids.shape}")

        indices = torch.arange(
            response_masks.size(1), device=response_masks.device
        ).expand_as(response_masks)
        masked_indices = torch.where(
            response_masks, indices, torch.full_like(indices, -1)
        )
        eos_indices = masked_indices.max(dim=1).values

        # Forward old models.
        ## 1) (Option 1) Policy log probabilities are directly from actors (vLLM).
        # logps = torch.zeros_like(response_masks).float()
        # for i in range(len(logps)):
        #     logps[i, torch.where(response_masks[i])[0]] = action_logprobs[i]
        ## 2) (Option 2) Reevaluate log probabilities using learner model.
        logps = torch.zeros(
            input_ids.shape[0], input_ids.shape[1] - 1, device=input_ids.device
        )
        with torch.no_grad():
            for i in range(0, len(input_ids), args.mini_train_batch_size_per_device):
                mini_batch_inds = torch.arange(
                    i, i + args.mini_train_batch_size_per_device
                )
                mb_input_ids = input_ids[mini_batch_inds]
                mb_att_mask = att_mask[mini_batch_inds]
                mb_response_masks = response_masks[mini_batch_inds]

                # Remove unnecessary padding introduced by the large PPO batch.
                mb_valid_token_count_per_pos = mb_att_mask.sum(0)
                mb_last_valid_token_pos = torch.where(
                    mb_valid_token_count_per_pos == 0
                )[0]
                if len(mb_last_valid_token_pos) >= 1:
                    mb_last_valid_token_pos = mb_last_valid_token_pos[0]
                else:
                    mb_last_valid_token_pos = mb_att_mask.shape[1]
                mb_input_ids = mb_input_ids[:, :mb_last_valid_token_pos]
                mb_att_mask = mb_att_mask[:, :mb_last_valid_token_pos]
                mb_response_masks = mb_response_masks[:, : mb_last_valid_token_pos - 1]

                batch_logits = self.model(mb_input_ids, attention_mask=mb_att_mask)[
                    "logits"
                ].float()
                batch_logits /= args.temperature
                batch_logps = self.get_batch_logps(
                    batch_logits,
                    mb_input_ids,
                    mb_response_masks,
                )
                logps[mini_batch_inds, : mb_last_valid_token_pos - 1] = batch_logps

        ## 2) Reference.
        if self.ref_model is not None:
            all_ref_logps = []
            with torch.no_grad():
                for i in range(
                    0, len(input_ids), args.mini_train_batch_size_per_device
                ):
                    batch_inds = torch.arange(
                        i, i + args.mini_train_batch_size_per_device
                    )

                    batch_ref_logits = self.ref_model(
                        input_ids[batch_inds], attention_mask=att_mask[batch_inds]
                    )["logits"].float()
                    batch_ref_logits /= args.temperature
                    batch_ref_logps = self.get_batch_logps(
                        batch_ref_logits,
                        input_ids[batch_inds],
                        response_masks[batch_inds],
                    )
                    all_ref_logps.append(batch_ref_logps)
            ref_logps = torch.cat(all_ref_logps)

            # Combine final reward and kl penalty as rewards.
            kl_rewards = -args.kl_penalty_coef * (logps - ref_logps) * response_masks
            rewards = kl_rewards.clone()
            del all_ref_logps
            torch.cuda.empty_cache()
            gc.collect()
        else:
            rewards = torch.zeros_like(response_masks).float()

        rewards[torch.arange(len(rewards)), eos_indices] += final_rewards.squeeze()

        if self.args.critic_type == "ppo":
            advantages, returns, values = self.compute_ppo_advantages(
                rewards, input_ids, att_mask, response_masks
            )
        elif self.args.critic_type in ["grpo", "drgrpo"]:
            advantages = self.compute_monte_carlo_advantages(rewards)[:, None]
        
        elif self.args.critic_type == "rfpp":
            advantages = self.compute_batch_normalized_advantages(rewards)[:, None]
        
        sum_alllength_per_prompt = self.compute_sum_alllength_per_prompt(response_masks, loss_masks)
        # Compute losses and update models for multiple PPO epochs.
        stats = defaultdict(list)
        for _ in range(args.num_ppo_epochs):
            batch_inds = np.random.permutation(len(input_ids))
            for b_st in range(0, len(input_ids), args.mini_train_batch_size_per_device):
                mini_batch_inds = batch_inds[
                    b_st : b_st + args.mini_train_batch_size_per_device
                ]
                mb_advantage = advantages[mini_batch_inds]
                mb_input_ids = input_ids[mini_batch_inds]
                mb_att_mask = att_mask[mini_batch_inds]
                mb_response_masks = response_masks[mini_batch_inds]
                mb_logps = logps[mini_batch_inds]
                mb_loss_masks = loss_masks[mini_batch_inds]

                # Remove unnecessary padding introduced by the large PPO batch.
                mb_valid_token_count_per_pos = mb_att_mask.sum(0)
                mb_last_valid_token_pos = torch.where(
                    mb_valid_token_count_per_pos == 0
                )[0]
                if len(mb_last_valid_token_pos) >= 1:
                    mb_last_valid_token_pos = mb_last_valid_token_pos[0]
                else:
                    mb_last_valid_token_pos = mb_att_mask.shape[1]
                if self.args.dapo_length_normalizer:
                    mb_sum_alllength_per_prompt = sum_alllength_per_prompt[mini_batch_inds]
                # # Further reduce valid token num to speed up IF:
                # ## 1. We only have PG loss, i.e., args.beta == 0.
                # ## 2. Advantage is zero in bandit case (e.g., GRPO).
                # ## 3. mini_train_batch_size_per_device is 1.
                # if (
                #     args.beta == 0
                #     and self.args.critic_type == "grpo"
                #     and len(mb_advantage) == 1
                # ):
                #     zero_adv = (mb_advantage == 0).item()  # bool
                #     if zero_adv:
                #         mb_last_valid_token_pos = 7  # An unimportant magic number.
                mb_input_ids = mb_input_ids[:, :mb_last_valid_token_pos]
                mb_att_mask = mb_att_mask[:, :mb_last_valid_token_pos]
                mb_response_masks = mb_response_masks[:, : mb_last_valid_token_pos - 1]
                mb_logps = mb_logps[:, : mb_last_valid_token_pos - 1]

                if self.args.critic_type == "ppo":
                    mb_return = returns[mini_batch_inds, : mb_last_valid_token_pos - 1]
                    mb_values = values[mini_batch_inds, : mb_last_valid_token_pos - 1]
                    mb_advantage = mb_advantage[:, : mb_last_valid_token_pos - 1]

                # Policy learning.
                logits = self.model(mb_input_ids, attention_mask=mb_att_mask)[
                    "logits"
                ].float()
                logits /= args.temperature
                new_logps = self.get_batch_logps(
                    logits,
                    mb_input_ids,
                    mb_response_masks,
                )
                
                if args.reinforce_update:
                    pg_loss_max = -mb_advantage * new_logps
                else:
                    logprobs_diff = new_logps - mb_logps
                    ratio = torch.exp(logprobs_diff)
                    pg_losses = -mb_advantage * ratio
                    pg_losses2 = -mb_advantage * torch.clamp(
                        ratio, 1.0 - args.cliprange_low, 1.0 + args.cliprange_high
                    )
                    pg_loss_max = torch.max(pg_losses, pg_losses2)

                    stats["logprobs_diff_max"].append(
                        torch.amax(logprobs_diff.detach() * mb_response_masks).item()
                    )
                    stats["logprobs_diff_min"].append(
                        torch.amin(logprobs_diff.detach() * mb_response_masks).item()
                    )
                    stats["zero_pg_loss_count"].append(
                        (pg_loss_max == 0).detach().sum().item()
                    )
                    
                if self.args.dapo_length_normalizer:
                    pg_loss = masked_sum(
                        pg_loss_max ,
                        mb_response_masks,
                        axis=1,
                    ) / mb_sum_alllength_per_prompt
                    pg_loss = (pg_loss * mb_loss_masks).sum()
                
                else:

                    pg_loss = self.masked_aggregator(pg_loss_max, mb_response_masks, axis=1)
                    pg_loss = (pg_loss * mb_loss_masks).mean()
                    infos["pg_loss"] = pg_loss.detach()
                
                
                    
                
                loss = pg_loss
                
                if args.entropy_coeff != 0:
                    entropy_loss =(-masked_sum(new_logps, mb_response_masks, axis=1) * mb_loss_masks).mean()
                    infos["entropy_loss"] = entropy_loss.detach()
                    loss -= args.entropy_coeff * entropy_loss
                
                if args.beta > 0:
                    mb_ref_logps = ref_logps[mini_batch_inds]
                    mb_ref_logps = mb_ref_logps[:, : mb_last_valid_token_pos - 1]
                    # k3 kl: http://joschu.net/blog/kl-approx.html.
                    # clamp to avoid numerical instability.
                    log_ratio = (mb_ref_logps - new_logps).clamp(-40.0, 40.0)
                    kl3 = torch.expm1(log_ratio) - log_ratio  # expm1 is more stable.
                    infos["kl3"] = (kl3 * mb_response_masks).detach().sum(1).mean()

                    reg_loss = self.masked_aggregator(kl3, mb_response_masks, axis=1)
                    reg_loss = args.beta * (reg_loss * mb_loss_masks).mean()
                    infos["reg_loss"] = reg_loss.detach()
                    loss += reg_loss

                self.strategy.backward(loss, self.model, self.optimizer)
                stats["policy_grad_norm"].append(
                    self.strategy.get_gradient_norm(self.model)
                )
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                gc.collect()
                torch.cuda.empty_cache()
                
                if self.args.critic_type == "ppo":
                    # torch.cuda.empty_cache()
                    # gc.collect()
                    self.strategy.print("start value_pred")

                    # Critic learning.
                    value_pred = self.critic(
                        input_ids=mb_input_ids, attention_mask=mb_att_mask
                    )[:, :-1]
                    self.strategy.print("finish value_pred")

                    value_pred_clipped = torch.clamp(
                        value_pred,
                        mb_values - args.cliprange_value,
                        mb_values + args.cliprange_value,
                    )
                    vf_losses1 = torch.square(value_pred - mb_return)
                    vf_losses2 = torch.square(value_pred_clipped - mb_return)
                    vf_loss_max = torch.max(vf_losses1, vf_losses2)

                    vf_loss = 0.5 * self.masked_aggregator(
                        vf_loss_max, mb_response_masks, axis=1
                    )
                    critic_loss = args.vf_coef * (vf_loss * mb_loss_masks).mean()
                    self.strategy.print("start critic backward")

                    self.strategy.backward(
                        critic_loss, self.critic, self.critic_optimizer
                    )
                    self.strategy.print("finish critic backward")
                    del mb_advantage, mb_input_ids, mb_att_mask, mb_logps, mb_loss_masks, logits, new_logps
                    if args.beta > 0:
                        del mb_ref_logps, log_ratio, kl3
                    torch.cuda.empty_cache()
                    gc.collect()
                    self.strategy.optimizer_step(
                        self.critic_optimizer, self.critic, self.critic_scheduler
                    )
                    self.strategy.print("finish critic step")
                    infos["critic_loss"] = critic_loss.detach()
                    infos["vf_clipfrac"] = masked_mean(
                        (vf_losses2 > vf_losses1).float(), mb_response_masks
                    ).detach()

                with torch.no_grad():
                    if not args.reinforce_update:
                        pg_clipfrac = masked_mean(
                            (pg_losses2 > pg_losses).float(), mb_response_masks, axis=1
                        )
                        stats["pg_clipfrac"].append(pg_clipfrac.mean().min().item())

        infos.update(
            {f"{k}_nan": torch.tensor(stats[k]).isnan().sum() for k in stats.keys()}
        )
        infos.update(
            {f"{k}_inf": torch.tensor(stats[k]).isinf().sum() for k in stats.keys()}
        )
        infos["policy_grad_norm"] = torch.tensor(stats["policy_grad_norm"]).max()
        if not args.reinforce_update:
            infos["logprobs_diff_max"] = torch.tensor(stats["logprobs_diff_max"]).max()
            infos["logprobs_diff_min"] = torch.tensor(stats["logprobs_diff_min"]).min()
            infos["zero_pg_loss_count"] = (
                torch.tensor(stats["zero_pg_loss_count"]).float().mean()
            )
            infos["pg_clipfrac"] = torch.tensor(stats["pg_clipfrac"]).mean()
        infos["adv_mean"] = advantages.mean().cpu()
        infos["adv_min"] = advantages.min().cpu()
        infos["adv_max"] = advantages.max().cpu()
        infos["all_zero_rewards_count"] = (final_rewards.mean(-1) == 0).sum().cpu()
        infos["all_one_rewards_count"] = (final_rewards.mean(-1) == 1).sum().cpu()

        return infos
