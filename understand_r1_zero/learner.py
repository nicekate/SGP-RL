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
from understand_r1_zero.dataset import PromptImageDataset, PromptSVGDataset
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

class ZeroSVGLearner(PPOLearner):
    def _init(self, args: ZeroSVGArgs, actors: List[ActorBase]) -> None:
        super()._init(args, actors)
        self.collector = FlexFeedbackCollector(
                    args, actors, PlasmaShmClient(self.ipc_server)
                )
        self.eval_math_dataset_dict = load_from_disk(args.eval_data)  # TODO: get fro HF.
        
        if args.test_split != "all":
            self.eval_math_dataset_dict = {
                k: v for k, v in self.eval_math_dataset_dict.items() if k in args.test_split
            }
        self.args = args
        # Dr. GRPO Modification 1: Remove length bias by using masked_sum with a constant normalizer:
        self.masked_aggregator = (
            functools.partial(masked_sum, constant_normalizer=args.generate_max_length)
            if args.critic_type == "drgrpo"
            else masked_mean
        )

    # Dr. GRPO Modification 2: Remove difficulty bias by just computing the MC advantage without dividing by std:
    def compute_monte_carlo_advantages(self, rewards):
        rewards = rewards.sum(-1)
        # Compute monte carlo trajectory-level advantage
        values = rewards.view(-1, self.args.num_samples).mean(dim=1)
        values = values.repeat_interleave(self.args.num_samples, dim=0)
        advantages = rewards - values
        if self.args.critic_type == "grpo":
            # Additionally normalize by std.
            std_grouped_rewards = rewards.view(-1, self.args.num_samples).std(dim=1)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(
                self.args.num_samples, dim=0
            )
            advantages = advantages / (std_grouped_rewards + 1e-8)
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
        )
        math_prompt_dataset = load_data_from_disk_or_hf(self.args.prompt_data_math)
        
        # prompt_dataset = load_data_from_disk_or_hf(self.args.prompt_data)
        svg_prompts_data = svg_prompt_dataset[self.args.train_split_svg]
        math_prompts_data = math_prompt_dataset[self.args.train_split_math].select(
            range(min(self.args.max_train_math, len(math_prompt_dataset[self.args.train_split_math])))
        )
        
        
        # Prepare the data: templated questions & gt final answers.
        # prompts_data = prompts_data.map(self._apply_template)
        # print("prompts_data_apply_template", prompts_data[0])

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
        
        
        svg_eval_dataset = get_dataset_class("uwunion/instruct_svg")().load_dataset(
            "uwunion/instruct_svg", 
            None, 
            max_test_samples=100,
        )['train']
        svg_eval_dataset = PromptSVGDataset(
            svg_eval_dataset,
            tokenizer,
            strategy,
            input_key="solution",
            output_key="svg",
            apply_chat_template=False,  # Because we have applied already.
            get_reference=True,
        )
        self.eval_svg_dataset_dict = {"instruct_svg":   svg_eval_dataset  }

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
                if "logging_data" in k:
                    logging_data = metrics[k]
            metrics = {k: v for k, v in metrics.items() if  "logging_data" not in k}    
            if logging_data is not None and self.strategy.is_rank_0():
                self._log_eval_completions_to_wandb(logging_data, benchmark_name)
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
            "eval/logging_data": all_logging_data
        }