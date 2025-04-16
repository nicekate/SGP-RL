
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


from understand_r1_zero.oracle import SVGOracle, SVGEvalOracle

import itertools
import logging
import time
from typing import Any, List, Dict

import numpy as np
import torch
import tree



from oat.algorithms.ppo import PPOActor

from oat.types import  TrajectoryData



from understand_r1_zero.oracle import SVGOracle
from understand_r1_zero.oracle import MATHOracle
from .args import ZeroSVGArgs
class ZeroSVGActor(PPOActor):
    def __init__(self, ipc_server, vllm_args, args: ZeroSVGArgs) -> None:
        super().__init__(ipc_server, vllm_args, args)
        self.math_oracle = MATHOracle(
            template="r1", verifier_version="math_verify"
        )
        rewards_dict = {"clip": args.clip_coeff,
                         "dino": args.dino_coeff,
                         "length": args.length_coeff,
                         "format": args.format_coeff}
        reward_models_dict = {"clip": args.clip_model,
                              "dino": args.dino_model}
        self.svg_oracle = SVGOracle(rewards_dict = rewards_dict, 
                                   models_dict = reward_models_dict)
        self.svg_eval_oracle = SVGEvalOracle()

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
        self.sampling_params.seed = args.seed
        self.eval_sampling_params.stop = ["</answer>"]
        self.eval_sampling_params.include_stop_str_in_output = True
        self.eval_sampling_params.seed = args.seed
        self.step_mode = "svg"
        

    def step(self,
        prompts: List[str],
        formatted_prompts: List[str],
        references: Dict[str, Any] = None,
        log_completions = False,
    ) -> List[TrajectoryData]:
        if self.step_mode == "svg":
            return self.step_svg(prompts = prompts,
                                 formatted_prompts = formatted_prompts,
                                 references = references,
                                 log_completions = log_completions)
        elif self.step_mode == "math":
            return self.step_math(prompts = prompts,
                                 formatted_prompts = formatted_prompts,
                                 references = references,
                                 log_completions = log_completions)
            
        else:
            assert False

    def step_math(
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
        rewards, oracle_infos = self.math_oracle.get_reward(
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
        
        
        
        trajectory_data = []
        for i in range(len(candidates)):
            prompt = prompts[i]
            
            candidates_per_prompt = candidates[i]
            for j in range(len(candidates_per_prompt)):
                trajectory_info = info.copy()
                
            
                
                reward = rewards[i][j].item()
                if no_eos[i][j]:
                    # Set zero reward for truncated outputs.
                    reward = 0
                dense_rewards = [0] * len(response_ids[i][j])
                dense_rewards[-1] = reward
                if log_completions:
                    
                    
                    logging_data = {
                        "prompt": prompt,
                        "completion": candidates_per_prompt[j],
                        "reward": rewards[i][j].item(),
                    }
                    
                    trajectory_info['actor/logging_data'] = logging_data
                    
                trajectory_info = {f"math/{k}" : v
                    for k, v in trajectory_info
                    }  
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
            
            

    def step_svg(
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
        rewards, oracle_infos = self.svg_oracle.get_reward(
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
        len_to_log = torch.tensor([x["length_reward"] for x in oracle_infos]).reshape(len(prompts), -1)
        
        
        info["actor/dino_rewards"] = dino_to_log.mean().item()
        info["actor/clip_rewards"] = clip_to_log.mean().item()
        info["actor/length_reward"] = len_to_log.mean().item()
        
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
                        "length_reward": len_to_log[i][j].item(),
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
                    
                # trajectory_info = {k.replace("actor", "actor_svg") : v
                #     for k, v in trajectory_info.items()
                #     }     
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



    def generate_and_maybe_eval(
        self,
        prompts: List[str],
        formatted_prompts: List[str],
        references: List[str] = None,
    ):
        assert self.eval_mode
        outputs = self.generate(formatted_prompts, self.eval_sampling_params)
        candidates = self.extract_candidates_from_output(
            outputs, self.eval_sampling_params
        )
        responses = []
        for j in range(self.eval_sampling_params.n):
            responses.extend([candidates[i][j] for i in range(len(prompts))])
       
        
        
        
        

        win_probs = None
        if self.step_mode == "svg":
            oracle = self.svg_eval_oracle
            # oracle = self.svg_oracle
        elif self.step_mode == "math":
            oracle = self.math_oracle
        else:
            raise ValueError(f"Unknown step mode {self.step_mode}")
        if references is not None:
            logging.debug(f"Evaluating using oracle {self.oracle}")
            st = time.time()
            win_probs, eval_info = oracle.compare(
                prompts * self.eval_sampling_params.n,
                responses,
                references * self.eval_sampling_params.n,
                batch_size=self.oracle_batch_size,
                return_probs=True,
                disable_tqdm=True,
            )
            logging.debug(f"Time elapse {time.time() - st}")
        reshaped_responses = []
        for x_i in range(len(prompts)):
            reshaped_responses.append(
                [responses[y_i] for y_i in range(x_i, len(responses), len(prompts))]
            )
        reshaped_win_probs = win_probs.reshape(
            self.eval_sampling_params.n, len(prompts)
        ).transpose(1, 0)
        
        
        prompts_to_log = [prompts[i] for i in range(len(prompts)) for j in range(self.eval_sampling_params.n)]
        responses_to_log = [candidates[i][j] for i in range(len(prompts)) for j in range(self.eval_sampling_params.n)]
        
        rewards_to_log = [reshaped_win_probs[i][j].item() for i in range(len(prompts)) for j in range(self.eval_sampling_params.n)]
        logging_data = {
            "prompt": prompts_to_log,
            "completion": responses_to_log,
            "reward": rewards_to_log,
        }
        reshaped_index = torch.arange(len(prompts) * self.eval_sampling_params.n).reshape(self.eval_sampling_params.n, len(prompts))
        reorder_index = [reshaped_index[i][j].item() for j in range(len(prompts))  for i in range(self.eval_sampling_params.n) ]
        # Replace the hardcoded reward collection with this dynamic approach
        if "rendered_images" in eval_info[0].keys():
            # Add rendered images to logging data
            images_to_log = [x["rendered_images"] for x in eval_info]
            images_to_log = [images_to_log[i] for i in reorder_index]
            logging_data["rendered_image"] = images_to_log
            
            # Dynamically collect all DINO model rewards
            dino_model_keys = [key for key in eval_info[0].keys() if key.startswith("dino_") and key.endswith("_reward")]
            for key in dino_model_keys:
                values_to_log = [x.get(key, 0.0) for x in eval_info]
                values_to_log = [values_to_log[i] for i in reorder_index]
                logging_data[key] = values_to_log
            
            # Dynamically collect all CLIP model rewards
            clip_model_keys = [key for key in eval_info[0].keys() if key.startswith("clip_") and key.endswith("_reward")]
            for key in clip_model_keys:
                values_to_log = [x.get(key, 0.0) for x in eval_info]
                values_to_log = [values_to_log[i] for i in reorder_index]
                logging_data[key] = values_to_log
            # Check if pre-computed averages exist in eval_info
            if "avg_dino_reward" in eval_info[0]:
                # Use pre-computed average DINO rewards
                avg_dino_values = [x.get("avg_dino_reward", 0.0) for x in eval_info]
                avg_dino_values = [avg_dino_values[i] for i in reorder_index]
                logging_data["dino_reward"] = avg_dino_values
            
            if "avg_clip_reward" in eval_info[0]:
                # Use pre-computed average CLIP rewards
                avg_clip_values = [x.get("avg_clip_reward", 0.0) for x in eval_info]
                avg_clip_values = [avg_clip_values[i] for i in reorder_index]
                logging_data["clip_reward"] = avg_clip_values
            
            # Add formatted flag to logging data
            if "formatted" in eval_info[0]:
                formatted_values = [x.get("formatted", False) for x in eval_info]
                formatted_values = [formatted_values[i] for i in reorder_index]
                logging_data["formatted"] = formatted_values
                
            
            
            
        return reshaped_responses, reshaped_win_probs, logging_data