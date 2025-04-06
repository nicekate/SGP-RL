
from oat.algorithms.ppo import PPOActor, PPOArgs
from dataclasses import dataclass, field
from typing import Any, List, Literal, Tuple, Dict




@dataclass
class ZeroSVGArgs(PPOArgs):
    # Template.
    prompt_template: Literal[ "r1", "r1_svg"] = field(default="r1")
    # Evaluation benchmarks used.
    test_split: str = ""  # Use "aime,math" to only evaluate on selected benchmarks.
    log_completion_steps: int = -1
    prompt_data_svg: str = "HuggingFaceM4/COCO"
    prompt_data_math: str = "./datasets/train/math_lvl3to5_8k"
    max_train_svg: int = 1000000
    max_train_math: int = 1000000
    train_split_svg: str = "train"
    train_split_math: str = "train"
    # Verifier.
    # verifier_version: Literal["fast", "math_verify"] = field(default="fast")
