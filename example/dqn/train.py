# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
DQN training example.
"""

#pylint: disable=C0413
import os
import argparse
from src import config
from src.dqn_trainer import DQNTrainer
from mindspore import context
from mindspore_rl.core import Session

parser = argparse.ArgumentParser(description='MindSpore Reinforcement DQN')
parser.add_argument('--episode', type=int, default=650, help='total episode numbers.')
parser.add_argument('--device_target', type=str, default='Auto', choices=['Ascend', 'CPU', 'GPU', 'Auto'],
                    help='Choose a device to run the dqn example(Default: Auto).')
parser.add_argument('--save_ckpt', type=int, default=0, choices=[0, 1], help='Whether to save the checkpoint file.')
parser.add_argument('--ckpt_path', type=str, default='./ckpt', help='Path to save ckpt file in train.\
                    default:./ckpt')
options, _ = parser.parse_known_args()

def train(episode=options.episode):
    if options.device_target != 'Auto':
        context.set_context(device_target=options.device_target)
    context.set_context(mode=context.GRAPH_MODE)
    config.trainer_params.update({'save_ckpt': options.save_ckpt})
    config.trainer_params.update({'ckpt_path': os.path.realpath(options.ckpt_path)})
    dqn_session = Session(config.algorithm_config)
    dqn_session.run(class_type=DQNTrainer, episode=episode, params=config.trainer_params)

if __name__ == "__main__":
    train()
