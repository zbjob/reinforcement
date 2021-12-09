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
"""PPO Tag Trainer"""
import time
import numpy as np

import mindspore
import mindspore.nn as nn
from mindspore.common.api import ms_function
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore_rl.agent.trainer import Trainer

class PPOTrainer(Trainer):
    """This is the trainer class of PPO algorithm. It arranges the PPO algorithm"""
    def __init__(self, msrl, params=None):
        nn.Cell.__init__(self, auto_prefix=False)
        self.all_ep_r = []
        self.all_eval_ep_r = []
        self.zero = Tensor(0, mindspore.float32)
        self.assign = P.Assign()
        self.squeeze = P.Squeeze()
        self.mod = P.Mod()
        self.equal = P.Equal()
        self.less = P.Less()
        self.reduce_sum = P.ReduceSum()
        self.transpose = P.Transpose()
        self.duration = params['duration']
        self.batch_size = params['batch_size']
        self.eval_interval = params['eval_interval']
        self.num_eval_episode = params['num_eval_episode']
        self.save_ckpt_path = params['ckpt_path']
        self.keep_checkpoint_max = params['keep_checkpoint_max']
        self.metrics = params['metrics']

        # 'Starred' is not supported in graph mode. Create the dummy tensors one by one.
        buffer = msrl.buffers.buffer
        self.dummy_action = Tensor(np.zeros(buffer[1].shape[1:]), buffer[1].dtype)
        self.dummy_reward = Tensor(np.zeros(buffer[2].shape[1:]), buffer[2].dtype)
        self.dummy_done = Tensor(np.zeros(buffer[3].shape[1:]), buffer[3].dtype)
        self.dummy_probs = Tensor(np.zeros(buffer[4].shape[1:]), buffer[4].dtype)
        self.dummy_value = Tensor(np.zeros(buffer[5].shape[1:]), buffer[5].dtype)

        super(PPOTrainer, self).__init__(msrl)

    def train(self, episode):
        """The main function which arranges the algorithm"""
        for i in range(episode):
            start = time.time()
            loss = self.train_one_episode()
            cost = time.time() - start
            print(f"Episode {i}, steps: {self.duration}, loss: {loss.asnumpy():.3f}, time: {cost:.3f}")

    @ms_function
    def train_one_episode(self):
        """the algorithm in one episode"""
        training_loss = self.zero
        j = self.zero
        state, _ = self.msrl.agent_reset_collect()

        while self.less(j, self.duration):
            reward, new_state, action, done, probs, value = self.msrl.agent_act(state)
            self.msrl.replay_buffer_insert([state, action, reward, done, probs, value])
            state = new_state
            j += 1

        # Store an additional record that only state is valid.
        self.msrl.replay_buffer_insert([state, self.dummy_action, self.dummy_reward,
                                        self.dummy_done, self.dummy_probs, self.dummy_value])

        replay_buffer_elements = self.msrl.get_replay_buffer_elements()

        training_loss += self.msrl.agent_learn(replay_buffer_elements)
        self.msrl.replay_buffer_reset()
        return training_loss
