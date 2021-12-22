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
"""DQN Trainer"""
import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as msnp
from mindspore.common.api import ms_function
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P
from mindspore_rl.agent.trainer import Trainer


class DQNTrainer(Trainer):
    """DQN Trainer"""

    def __init__(self, msrl, params):
        nn.Cell.__init__(self, auto_prefix=False)
        self.zero = Tensor(0, ms.float32)
        self.squeeze = P.Squeeze()
        self.less = P.Less()
        self.zero_value = Tensor(0, ms.float32)
        self.fill_value = Tensor(1000, ms.float32)
        self.inited = Parameter(Tensor(False, ms.bool_), name='init_flag')
        self.mod = P.Mod()
        self.num_evaluate_episode = params['num_evaluate_episode']
        self.update_period = Tensor(5, ms.float32)
        super(DQNTrainer, self).__init__(msrl)

    def trainable_variables(self):
        """Trainable variables for saving."""
        trainable_variables = {"policy_net": self.msrl.actors.policy_network}
        return trainable_variables

    @ms_function
    def init_training(self):
        """Initialize training"""
        state, done = self.msrl.agent_reset_collect()
        i = self.zero_value
        while self.less(i, self.fill_value):
            done, _, new_state, action, my_reward = self.msrl.agent_act_init(state)
            self.msrl.replay_buffer_insert([state, action, my_reward, new_state])
            state = new_state
            if done:
                state, done = self.msrl.agent_reset_collect()
            i += 1
        return done

    @ms_function
    def train_one_episode(self):
        """Train one episode"""
        if not self.inited:
            self.init_training()
            self.inited = True
        state, done = self.msrl.agent_reset_collect()
        total_reward = self.zero
        steps = self.zero
        loss = self.zero
        while not done:
            done, r, new_state, action, my_reward = self.msrl.agent_act(state)
            self.msrl.replay_buffer_insert([state, action, my_reward, new_state])
            state = new_state
            r = self.squeeze(r)
            loss = self.msrl.agent_learn(self.msrl.replay_buffer_sample())
            total_reward += r
            steps += 1
            if not self.mod(steps, self.update_period):
                self.msrl.agent_update()
        return loss, total_reward, steps

    @ms_function
    def evaluate(self):
        """Policy evaluate"""
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
