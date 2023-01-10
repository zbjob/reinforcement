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
CQL session.
"""
import time
import numpy as np
from mindspore_rl.core import Session
from mindspore_rl.utils.utils import update_config
from mindspore_rl.utils.callback import CheckpointCallback, EvaluateCallback, Callback
from mindspore_rl.algorithm.cql import config


class MyLossCallback(Callback):
    """Two loss callback for CQL"""

    def __init__(self, freq):
        self.interval = freq
        self.epoch_time = time.time()

    def episode_begin(self, params):
        self.epoch_time = time.time()

    def episode_end(self, params):
        """Step info stats during training"""
        epoch_ms = (time.time() - self.epoch_time) * 1000
        losses = params.loss
        losses_out = []
        for loss in losses:
            losses_out.append(round(float(np.mean(loss.asnumpy())), 3))
        if (params.cur_episode % self.interval) == 0:
            print("Episode {}: critic_loss is {}, actor_loss is {}, per_step_time {:5.3f} ms".format(\
                params.cur_episode, losses_out[0], losses_out[1], epoch_ms), flush=True)


class CQLSession(Session):
    '''CQL session'''
    def __init__(self, env_yaml=None, algo_yaml=None):
        update_config(config, env_yaml, algo_yaml)
        env_config = config.algorithm_config.get('collect_environment')
        env = env_config.get('type')(env_config.get('params'))
        config.policy_params['state_space_dim'] = env.observation_space.shape[0]
        config.policy_params['action_space_dim'] = env.action_space.shape[0]
        config.learner_params['state_space_dim'] = env.observation_space.shape[0]
        config.learner_params['action_space_dim'] = env.action_space.shape[0]
        # Replay buffer
        config.algorithm_config['replay_buffer']['data_shape'] = [env.observation_space.shape,
                                                                  env.action_space.shape, (1,),
                                                                  env.observation_space.shape, (1,)]
        config.algorithm_config['replay_buffer']['data_type'] = [env.observation_space.ms_dtype,
                                                                 env.observation_space.ms_dtype,
                                                                 env.observation_space.ms_dtype,
                                                                 env.observation_space.ms_dtype,
                                                                 env.observation_space.ms_dtype]

        ckpt_cb = CheckpointCallback(config.trainer_params.get('save_per_episode'),
                                     config.trainer_params.get('ckpt_path'),
                                     config.trainer_params.get('max_ckpt_num'))
        loss_cb = MyLossCallback(config.trainer_params.get('loss_freq'))
        eval_cb = EvaluateCallback(config.trainer_params.get('eval_per_episode'))
        params = config.trainer_params
        cbs = [ckpt_cb, loss_cb, eval_cb]
        super().__init__(config.algorithm_config, None, params=params, callbacks=cbs)
