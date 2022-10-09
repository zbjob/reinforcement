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
"""MAPPO Trainer"""
#pylint: disable=W0613
import mindspore as ms
import mindspore.nn as nn
from mindspore.common.api import ms_function
from mindspore import Tensor, Parameter, set_seed
from mindspore.ops import operations as P

from mindspore_rl.utils.callback import CallbackParam, CallbackManager
from mindspore_rl.agent.trainer import Trainer

set_seed(19)


class MAPPOTrainer(Trainer):
    """This is the trainer class of MAPPO algorithm. It arranges the MAPPO algorithm"""

    def __init__(self, msrl, params):
        nn.Cell.__init__(self, auto_prefix=False)
        self.msrl = msrl
        self.params = params
        self.zero = Tensor(0, ms.int32)
        self.duration_config = Tensor(25, ms.int32)
        self.assign = P.Assign()
        self.squeeze = P.Squeeze()
        self.less = P.Less()
        self.reduce_mean = P.ReduceMean()

        self.reshape = P.Reshape()
        self.squeeze = P.Squeeze()
        self.zeros = P.Zeros()
        self.ones = P.Ones()
        self.zero_float = Tensor(0, ms.float32)
        self.one_float = Tensor(1, ms.float32)
        self.false = Tensor(False, ms.bool_)
        self.true = Tensor(True, ms.bool_)

        self.num_agent = self.msrl.num_agent
        self.local_obs_dim = self.msrl.collect_environment.observation_space.shape[-1]
        self.global_obs_dim = self.msrl.collect_environment.config['global_observation_dim']

        self.concated_action = Parameter(self.zeros(
            (self.num_agent, 128, 1), ms.int32), requires_grad=False)
        self.concated_log_prob = Parameter(self.zeros(
            (self.num_agent, 128, 1), ms.float32), requires_grad=False)
        self.concated_ht_actor = Parameter(self.zeros(
            (self.num_agent, 128, 1, 64), ms.float32), requires_grad=False)
        self.concated_value_prediction = Parameter(self.zeros(
            (self.num_agent, 128, 1), ms.float32), requires_grad=False)
        self.concated_ht_critic = Parameter(self.zeros(
            (self.num_agent, 128, 1, 64), ms.float32), requires_grad=False)
        self.onehot_action = Parameter(self.zeros(
            (128, self.num_agent, 5), ms.float32), requires_grad=False)

        self.agent_last_local_obs = Parameter(self.zeros(
            (128, self.num_agent, self.local_obs_dim), ms.float32), requires_grad=False)
        self.agent_last_global_obs = Parameter(self.zeros(
            (128, self.global_obs_dim), ms.float32), requires_grad=False)
        self.agent_last_hn_actor = Parameter(self.zeros(
            (self.num_agent, 128, 1, 64), ms.float32), requires_grad=False)
        self.agent_last_hn_critic = Parameter(self.zeros(
            (self.num_agent, 128, 1, 64), ms.float32), requires_grad=False)
        self.agent_last_mask = Parameter(
            self.ones((self.num_agent, 128, 1), ms.float32), requires_grad=False)

        super(MAPPOTrainer, self).__init__(self.msrl)

    def train(self, episodes, callbacks=None, ckpt_path=None):
        """
        The interface of the train function. User will implement
        this function.

        Args:
            episodes(int): the number of training episodes.
            callbacks(Optional[list[Callback]]): List of callback objects. Default: None
            ckpt_path(Optional[string]): The checkpoint file to init or restore net. Default: None.
        """

        cb_params = CallbackParam()
        cb_params.episodes_num = episodes

        # 1 Using `CallbackManager` to traverse each callback.
        with CallbackManager(callbacks) as callback_list:

            # 2 Init or restore the variables if the checkpoint files exist.
            cb_params.cur_episode = 0
            if self.vars:
                cb_params.vars = self.vars

            callback_list.begin(cb_params)

            # 3 Get `evaluate` function if meet the conditions.
            if 'eval_rate' in cb_params and cb_params.eval_rate > 0:
                cb_params.evaluate = self.evaluate

            self.init()
            self.agent_last_local_obs = Tensor(
                self.agent_last_local_obs.asnumpy(), ms.float32)
            self.agent_last_global_obs = Tensor(
                self.agent_last_global_obs.asnumpy(), ms.float32)

            for i in range(episodes):
                callback_list.episode_begin(cb_params)

                # 4 Get the result of `train_one_episode` func, and deal with three situation:
                #   a) Default using: Three objects in tuple, each stand for `loss`, `rewards` and `steps`.
                #   b) User defined: Four objects in tuple, the first three is same as default using, the last
                #       one `others` can be tuple or single one as user defined.
                #   c) Other situation: Runtime error.
                ans = self.train_one_episode()
                loss, rewards, steps, others = [], [], [], []
                if len(ans) == 3:
                    loss, rewards, steps = ans
                elif len(ans) == 4:
                    loss, rewards, steps, others = ans
                else:
                    raise RuntimeError("The output number of function `train_one_episode` must be 3 or 4, \
                        and represent for `loss, rewards, steps, [optional]others.` in order")

                cb_params.loss = loss
                cb_params.total_rewards = rewards
                cb_params.steps = steps
                cb_params.others = others
                callback_list.episode_end(cb_params)
                cb_params.cur_episode = i + 1

                for j in range(self.num_agent):
                    self.msrl.buffers['local_replaybuffer'][j].reset()
                self.msrl.buffers['global_replaybuffer'].reset()

            callback_list.end(cb_params)
            for collect_env in self.msrl.collect_environment.mpe_env_procs:
                collect_env.terminate()

            for eval_env in self.msrl.eval_environment.mpe_env_procs:
                eval_env.terminate()

    @ms_function
    def init(self):
        """Init method, it will be called once"""
        # ---------------------- initialize ------------------------- #
        state = self.msrl.collect_environment.reset()
        self.agent_last_local_obs = state
        self.agent_last_global_obs = self.reshape(
            state, (128, self.global_obs_dim))
        # ------------------------------------------------------------ #
        return self.true

    @ms_function
    def train_one_episode(self):
        """the algorithm in one episode"""
        # ----------------------------------------- actor -------------------------------------------
        self.assign(self.concated_action, self.zeros(
            (self.num_agent, 128, 1), ms.int32))
        self.assign(self.concated_log_prob, self.zeros(
            (self.num_agent, 128, 1), ms.float32))
        self.assign(self.concated_ht_actor, self.zeros(
            (self.num_agent, 128, 1, 64), ms.float32))
        self.assign(self.concated_value_prediction, self.zeros(
            (self.num_agent, 128, 1), ms.float32))
        self.assign(self.concated_ht_critic, self.zeros(
            (self.num_agent, 128, 1, 64), ms.float32))
        self.assign(self.onehot_action, self.zeros(
            (128, self.num_agent, 5), ms.float32))

        zero_reward = self.zeros((self.num_agent, 128, 1), ms.float32)

        local_replaybuffer = self.msrl.buffers['local_replaybuffer']
        global_replaybuffer = self.msrl.buffers['global_replaybuffer']

        agent_num = 0
        while agent_num < self.num_agent:
            local_replaybuffer[agent_num]([self.agent_last_local_obs[:, agent_num],
                                           self.concated_ht_actor[agent_num, :],
                                           self.concated_ht_critic[agent_num, :],
                                           self.agent_last_mask[agent_num, :],
                                           self.concated_action[agent_num, :],
                                           self.concated_log_prob[agent_num, :],
                                           self.concated_value_prediction[agent_num, :],
                                           zero_reward[agent_num, :]], self.true)
            agent_num += 1
        global_replaybuffer([self.agent_last_global_obs], self.true)

        training_reward = self.zero_float
        duration = self.zero
        while self.less(duration, 25):
            num_agent = 0
            while num_agent < self.num_agent:
                samples = [self.agent_last_local_obs[:, num_agent], self.agent_last_global_obs,
                           self.agent_last_hn_actor[num_agent], self.agent_last_hn_critic[num_agent],
                           self.agent_last_mask[num_agent], num_agent, self.onehot_action,
                           self.concated_action, self.concated_log_prob, self.concated_ht_actor,
                           self.concated_value_prediction, self.concated_ht_critic]

                self.msrl.agent[num_agent](1, samples)

                num_agent += 1

            new_local_obs, rewards, dones = self.msrl.collect_environment.step(
                (self.onehot_action).astype(ms.int32))

            dones = self.reshape(
                (1-dones).astype(ms.float32), (self.num_agent, 128, 1, 1))

            new_global_obs = self.reshape(
                new_local_obs, (128, self.global_obs_dim))

            masked_concated_ht_actor = self.concated_ht_actor * dones
            masked_concated_ht_critic = self.concated_ht_critic * dones
            episode_masks = self.ones((self.num_agent, 128, 1), ms.float32)
            episode_masks = episode_masks * dones.squeeze(-1)

            agent_num = 0
            while agent_num < self.num_agent:
                local_replaybuffer[agent_num]([new_local_obs[:, agent_num],
                                               masked_concated_ht_actor[agent_num, :],
                                               masked_concated_ht_critic[agent_num, :],
                                               episode_masks[agent_num, :],
                                               self.concated_action[agent_num, :],
                                               self.concated_log_prob[agent_num, :],
                                               self.concated_value_prediction[agent_num, :],
                                               rewards[:, agent_num]], self.true)
                agent_num += 1
            global_replaybuffer([new_global_obs], self.true)

            self.agent_last_local_obs = new_local_obs
            self.agent_last_global_obs = new_global_obs
            self.agent_last_hn_actor = masked_concated_ht_actor
            self.agent_last_hn_critic = masked_concated_ht_critic
            self.agent_last_mask = episode_masks

            duration += 1

        # ----------------------------------------- learner -------------------------------------------
        dummy = [self.agent_last_local_obs[:, 0],
                 self.concated_ht_actor[0, :],
                 self.concated_ht_critic[0, :],
                 self.agent_last_mask[0, :],
                 self.concated_action[0, :],
                 self.concated_log_prob[0, :],
                 self.concated_value_prediction[0, :],
                 zero_reward[0, :]]
        dummy_2 = [self.agent_last_global_obs]

        agent_id = 0
        loss_ac = self.zero_float
        global_obs_exp = global_replaybuffer(dummy_2, self.false)
        while agent_id < self.num_agent:
            buffer = local_replaybuffer[agent_id](dummy, self.false)
            new_buffer = buffer + global_obs_exp
            output_loss = self.msrl.agent[agent_id](2, new_buffer)
            loss_ac += output_loss
            agent_id += 1
            training_reward = self.reduce_mean(buffer[-1]) * 25
        return loss_ac, training_reward, duration

    def trainable_variables(self):
        """Trainable variables for saving."""
        return

    def evaluate(self):
        return
