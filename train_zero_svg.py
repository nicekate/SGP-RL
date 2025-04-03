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


import launchpad as lp



from oat.args import default_args_validation, get_default_args
from oat.interface import get_program, lp

from understand_r1_zero.learner import ZeroSVGLearner
from understand_r1_zero.actor import ZeroSVGActor
from understand_r1_zero.args import ZeroSVGArgs




def run_zero_math_rl(args: ZeroSVGArgs):
    # Define a distributed program that composes Actors and Learners.
    program, local_resources = get_program(
        args, learner_cls=ZeroSVGLearner, actor_cls=ZeroSVGActor
    )
    # Launch the program in a local, multi-processing way!
    lp.launch(
        program,
        launch_type=args.launch_type,
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    args: ZeroSVGArgs = get_default_args(ZeroSVGArgs)
    # Customization:
    args.algo = "PPO"
    args.online_evaluation = True  # Use GT answer for online verification.

    args = default_args_validation(args)
    run_zero_math_rl(args)
