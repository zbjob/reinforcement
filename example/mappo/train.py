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
MAPPO training example.
"""

import argparse

from mindspore import context

from mindspore_rl.algorithm.mappo.mappo_session import MAPPOSession
from mindspore_rl.algorithm.mappo.mappo_trainer import MAPPOTrainer


parser = argparse.ArgumentParser(description='MindSpore Reinforcement MAPPO')
parser.add_argument('--episode', type=int, default=1500,
                    help='total episode numbers.')
parser.add_argument('--device_target', type=str, default='Auto', choices=['Auto', 'GPU', 'CPU'],
                    help='Choose a device to run the mappo example(Default: GPU).')
parser.add_argument('--env_yaml', type=str, default='../env_yaml/simple_spread.yaml',
                    help='Choose an environment yaml to update the mappo example(Default: simple_spread.yaml).')
parser.add_argument('--algo_yaml', type=str, default=None,
                    help='Choose an algo yaml to update the mappo example(Default: None).')
options, _ = parser.parse_known_args()


def train(episode=options.episode):
    '''MAPPO train entry.'''
    if options.device_target != 'Auto':
        context.set_context(device_target=options.device_target)
    if context.get_context('device_target') in ['GPU']:
        context.set_context(enable_graph_kernel=True)

    context.set_context(mode=context.GRAPH_MODE, max_call_depth=100000)
    mappo_session = MAPPOSession(options.env_yaml, options.algo_yaml)
    mappo_session.run(class_type=MAPPOTrainer, episode=episode)


if __name__ == "__main__":
    train()
