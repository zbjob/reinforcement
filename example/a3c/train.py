# Copyright 2022 Huawei Technologies Co., Ltd
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
A3C training example.
"""

#pylint: disable=C0413
import argparse
from src import config
from src.a3c_trainer import A3CTrainer
from mindspore import context
from mindspore_rl.core import Session

parser = argparse.ArgumentParser(description='MindSpore Reinforcement A3C')
parser.add_argument('--episode', type=int, default=1000, help='total episode numbers.')
parser.add_argument('--device_target', type=str, default='Auto', choices=['CPU', 'GPU', 'Auto'],
                    help='Choose a device to run the ac example(Default: Auto).')
parser.add_argument('--actor_num', type=int, default=3, help='actor number')
options, _ = parser.parse_known_args()


def train(episode=options.episode):
    if options.device_target != 'Auto':
        context.set_context(device_target=options.device_target)
    context.set_context(mode=context.GRAPH_MODE)
    config.algorithm_config['actor']['number'] = options.actor_num
    ac_session = Session(config.algorithm_config)
    ac_session.run(class_type=A3CTrainer, episode=episode)

if __name__ == "__main__":
    train()
