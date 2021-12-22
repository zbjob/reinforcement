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
"""AC Trainer"""
from mindspore_rl.agent.trainer import Trainer
import mindspore
import mindspore.nn as nn
import mindspore.numpy as msnp
from mindspore.common.api import ms_function
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter

class ACTrainer(Trainer):
    '''ACTrainer'''
    def __init__(self, msrl, params):
        nn.Cell.__init__(self, auto_prefix=False)
        self.num_evaluate_episode = params['num_evaluate_episode']
        self.zero = Parameter(Tensor(0, mindspore.float32), name='zero')
        self.done_r = Parameter(Tensor([-20.0], mindspore.float32), name='done_r')
        self.zero_value = Tensor(0, mindspore.float32)
        self.squeeze = P.Squeeze()
        super(ACTrainer, self).__init__(msrl)

    def trainable_variables(self):
        '''Trainable variables for saving.'''
        trainable_variables = {"actor_net": self.msrl.actors.actor_net}
        return trainable_variables

    @ms_function
    def train_one_episode(self):
        '''Train one episode'''
        state, _ = self.msrl.agent_reset_collect()
        total_reward = self.zero
        steps = self.zero
        loss = self.zero_value
        while True:
            done, r, state_, a = self.msrl.agent_act(state)
            r = self.squeeze(r)
            total_reward += r
            if done:
                r = self.done_r
            loss = self.msrl.agent_learn([state, r, state_, a])
            state = state_
            steps += 1
            if done:
                break
        return loss, total_reward, steps

    @ms_function
    def evaluate(self):
        '''evaluate'''
        total_reward = self.zero_value
        for _ in msnp.arange(self.num_evaluate_episode):
            episode_reward = self.zero_value
            state, done = self.msrl.agent_reset_eval()
            while not done:
                done, r, state = self.msrl.agent_evaluate(state)
                r = self.squeeze(r)
                episode_reward += r
            total_reward += episode_reward
        avg_reward = total_reward / self.num_evaluate_episode
        return avg_reward
