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
PPO config.
"""

import mindspore
from mindspore_rl.core.replay_buffer import ReplayBuffer
from mindspore_rl.environment import MsEnvironment
from .ppo import PPOActor, PPOLearner, PPOPolicy

env_params = {
    'name': 'Tag',
    'environment_num': 2000,
    'predator_num': 4,
    'prey_num': 1
}
eval_env_params = {
    'name': 'Tag',
    'environment_num': 2000,
    'predator_num': 4,
    'prey_num': 1
}

policy_params = {
    'epsilon': 0.2,
    'lr': 1e-3,
    'grad_clip': 3.0,
    'state_space_dim': 0,
    'action_space_dim': 0,
    'hidden_size1': 32,
    'hidden_size2': 32,
    'critic_coef': 1,
    'entropy_coef': 0.05,
}

learner_params = {
    'gamma': 0.98,
    'state_space_dim': 0,
    'action_space_dim': 0,
    'iter_times': 1
}

trainer_params = {
    'duration': 100,
    'batch_size': 1,
    'eval_interval': 20,
    'num_eval_episode': 1,
    'keep_checkpoint_max': 5,
    'metrics': False
}

algorithm_config = {
    'actor': {
        'number': 1,
        'type': PPOActor,
        'params': None,
        'policies': [],
        'networks': ['actor_critic_net'],
        'environment': True,
        'eval_environment': True,
    },
    'learner': {
        'number': 1,
        'type': PPOLearner,
        'params': learner_params,
        'networks': ['actor_critic_net', 'ppo_net_train']
    },
    'policy_and_network': {
        'type': PPOPolicy,
        'params': policy_params
    },
    'collect_environment': {
        'type': MsEnvironment,
        'params': env_params
    },
    'eval_environment': {
        'type': MsEnvironment,
        'params': eval_env_params
    },
    'replay_buffer': {
        'type': ReplayBuffer,
        'capacity': 101,
        # state, action, reward, done, probs, values
        'data_shape': [(2000, 5, 21), (2000, 5), (2000, 5), (2000,), (2000, 5, 5), (2000, 5)],
        'data_type': [
            mindspore.float32, mindspore.int32, mindspore.float32,
            mindspore.bool_, mindspore.float32, mindspore.float32,
        ],
    }
}
