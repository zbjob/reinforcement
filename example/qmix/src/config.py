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
QMIX config.
"""

# pylint: disable=E0402
import mindspore as ms
from mindspore_rl.environment import StarCraft2Environment
from mindspore_rl.core.replay_buffer import ReplayBuffer
from .qmix import QMIXActor, QMIXLearner, QMIXPolicy

BATCH_SIZE = 32
collect_env_params = {'sc2_args': {'map_name': '2s3z',
                                   'seed': 1}}

eval_env_params = {'sc2_args': {'map_name': '2s3z'}}

policy_params = {
    'epsi_high': 1.0,
    'epsi_low': 0.05,
    'decay': 200,
    'state_space_dim': 0,
    'action_space_dim': 0,
    'hidden_size': 64,
    'embed_dim': 32,
    'hypernet_embed': 64,
    'time_length': 50000,
    'batch_size': BATCH_SIZE,
}

learner_params = {
    'lr': 0.0005,
    'gamma': 0.99,
    'optim_alpha': 0.99,
    'epsilon': 1e-5,
    'batch_size': BATCH_SIZE,
}

trainer_params = {
    'batch_size': BATCH_SIZE,
    'ckpt_path': './ckpt'
}

algorithm_config = {
    'actor': {
        'number': 1,
        'type': QMIXActor,
        'policies': ['collect_policy', 'eval_policy'],
    },
    'learner': {
        'number': 1,
        'type': QMIXLearner,
        'params': learner_params,
        'networks': ['policy_net', 'mixer_net']
    },
    'policy_and_network': {
        'type': QMIXPolicy,
        'params': policy_params
    },
    'collect_environment': {
        'number': 1,
        'type': StarCraft2Environment,
        'params': collect_env_params
    },
    'eval_environment': {
        'number': 1,
        'type': StarCraft2Environment,
        'params': eval_env_params
    },
    'replay_buffer': {
        'number': 1,
        'type': ReplayBuffer,
        'capacity': 5000,
        'data_shape': [(121, 5, 96), (121, 120), (121, 5, 1), (121, 5, 11), (121, 1), (121, 1), (121, 1), (121, 5, 64)],
        'data_type': [
            ms.float32, ms.float32, ms.int32, ms.int32,
            ms.float32, ms.int32, ms.int32, ms.float32
        ],
        'sample_size': BATCH_SIZE,
    }
}
