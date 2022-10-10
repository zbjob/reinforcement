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
"""QMIX"""

import copy
import math

import mindspore.nn as nn
import mindspore as ms
from mindspore.common.api import cells_compile_cache
from mindspore.ops import operations as P
from mindspore import Tensor
import mindspore.nn.probability.distribution as msd
from mindspore.common.initializer import initializer, Uniform, HeUniform

from mindspore_rl.agent import Actor
from mindspore_rl.agent import Learner
from mindspore_rl.network import GruNet
from mindspore_rl.utils import SoftUpdate


class QMIXPolicy:
    """
    QMIXPolicy class has the implementation of policy net, mixer net and the policy(how to obtain action).
    """

    class QMIXPolicyNet(nn.Cell):
        """The policy net implementation"""

        def __init__(self, params):
            super().__init__()
            self.hidden_size = params['hidden_size']
            self.output_size = params['action_space_dim']
            self.num_agent = params['environment_config']['num_agent']
            self.input_size = (params['state_space_dim'] + self.num_agent + self.output_size)

            self.fc1 = nn.Dense(self.input_size, self.hidden_size, activation=nn.ReLU(),
                                weight_init=initializer(HeUniform(negative_slope=math.sqrt(5)),
                                                        [self.hidden_size, self.input_size], ms.float32))
            self.gru_net = GruNet(self.hidden_size, self.hidden_size, weight_init=initializer(
                Uniform(scale=(1 / math.sqrt(self.hidden_size))), [24960, 1, 1], ms.float32))
            self.fc2 = nn.Dense(self.hidden_size, self.output_size, weight_init=initializer(
                HeUniform(negative_slope=math.sqrt(5)), [self.output_size, self.hidden_size], ms.float32))

            self.concat = P.Concat()
            self.reshape = P.Reshape()
            self.expand_dims = P.ExpandDims()
            self.squeeze = P.Squeeze(axis=0)

        def construct(self, x, hx):
            x = self.fc1(x)
            x = self.expand_dims(x, 0)
            hx = self.expand_dims(hx, 0)
            hy, _ = self.gru_net(x, hx)
            hy = self.squeeze(hy)
            x = self.fc2(hy)
            return x, hy

        def clone(self):
            new_obj = copy.deepcopy(self)
            cells_compile_cache[id(new_obj)] = new_obj.compile_cache
            return new_obj

    class QMIXMixerNet(nn.Cell):
        """The mixer net implementation"""

        def __init__(self, params):
            super().__init__()
            self.embed_dim = params['embed_dim']
            self.global_obs_dim = params['environment_config']['global_observation_dim']
            self.num_agent = params['environment_config']['num_agent']
            self.batch = params['batch_size']
            hypernet_embed = params['hypernet_embed']

            self.hyper_w_1 = nn.SequentialCell([nn.Dense(self.global_obs_dim, hypernet_embed, activation=nn.ReLU(),
                                                         weight_init=initializer(HeUniform(negative_slope=math.sqrt(5)),
                                                                                 [hypernet_embed, self.global_obs_dim],
                                                                                 ms.float32)),
                                                nn.Dense(hypernet_embed, self.embed_dim * self.num_agent,
                                                         weight_init=initializer(HeUniform(negative_slope=math.sqrt(5)),
                                                                                 [self.embed_dim * self.num_agent,
                                                                                  hypernet_embed], ms.float32))])
            self.hyper_b_1 = nn.Dense(self.global_obs_dim, self.embed_dim, weight_init=initializer(
                HeUniform(negative_slope=math.sqrt(5)), [self.embed_dim, self.global_obs_dim], ms.float32))
            self.hyper_w_2 = nn.SequentialCell([nn.Dense(self.global_obs_dim, hypernet_embed, activation=nn.ReLU(),
                                                         weight_init=initializer(HeUniform(negative_slope=math.sqrt(5)),
                                                                                 [hypernet_embed, self.global_obs_dim],
                                                                                 ms.float32),),
                                                nn.Dense(hypernet_embed, self.embed_dim,
                                                         weight_init=initializer(HeUniform(negative_slope=math.sqrt(5)),
                                                                                 [self.embed_dim, hypernet_embed],
                                                                                 ms.float32))])
            self.hyper_out = nn.SequentialCell([nn.Dense(self.global_obs_dim, self.embed_dim, activation=nn.ReLU(),
                                                         weight_init=initializer(HeUniform(negative_slope=math.sqrt(5)),
                                                                                 [self.embed_dim, self.global_obs_dim],
                                                                                 ms.float32),),
                                                nn.Dense(self.embed_dim, 1,
                                                         weight_init=initializer(HeUniform(negative_slope=math.sqrt(5)),
                                                                                 [1, self.embed_dim], ms.float32))])

            self.reshape = P.Reshape()
            self.abs = P.Abs()
            self.elu = nn.ELU()
            self.batch_matmul = P.BatchMatMul()

        def construct(self, q_val, global_obs):
            """The mixer net calculation"""
            global_obs = self.reshape(global_obs, (-1, self.global_obs_dim))
            q_val = self.reshape(q_val, (-1, 1, self.num_agent))
            w1 = self.abs(self.hyper_w_1(global_obs))
            b1 = self.hyper_b_1(global_obs)
            w1 = self.reshape(w1, (-1, self.num_agent, self.embed_dim))
            b1 = self.reshape(b1, (-1, 1, self.embed_dim))
            out1 = self.elu(self.batch_matmul(q_val, w1) + b1)
            w2 = self.abs(self.hyper_w_2(global_obs))
            w2 = self.reshape(w2, (-1, self.embed_dim, 1))
            b2 = self.reshape(self.hyper_out(global_obs), (-1, 1, 1))
            q_tot = self.reshape((self.batch_matmul(out1, w2) + b2), (self.batch, -1, 1))
            return q_tot

        def clone(self):
            new_obj = copy.deepcopy(self)
            cells_compile_cache[id(new_obj)] = new_obj.compile_cache
            return new_obj

    class CollectPolicy(nn.Cell):
        """The collect policy implementation (how to obtain actions)"""

        def __init__(self, network, params):
            super().__init__()
            self.epsi_high = Tensor([params['epsi_high']], ms.float32)
            self.epsi_low = Tensor([params['epsi_low']], ms.float32)
            time_length = params['time_length']
            self.delta = (self.epsi_high - self.epsi_low) / time_length
            self.network = network

            self.expand_dims = P.ExpandDims()
            self.categorical = msd.Categorical()
            self.maximum = P.Maximum()
            self.randreal = P.UniformReal()
            self.select = P.Select()

        def construct(self, params, step):
            """How to choose an action in QMIX"""
            x, hx, avail_action = params
            x, hy = self.network(x, hx)
            x[avail_action == 0] = -9999999

            greedy_action = self.categorical.mode(x)
            random_action = self.categorical.sample((), avail_action)
            decayed_value = self.epsi_high - self.delta * step
            epsilon = self.maximum(self.epsi_low, decayed_value)
            real_num = self.randreal((5,))
            cond = (real_num < epsilon)
            output_action = self.select(cond, random_action, greedy_action)
            output_action = self.expand_dims(output_action, 1)
            return output_action, hy

    class EvalPolicy(nn.Cell):
        """The evaluation policy implementation (how to obtain actions)"""

        def __init__(self, network):
            super().__init__()
            self.network = network
            self.expand_dims = P.ExpandDims()
            self.categorical = msd.Categorical()

        def construct(self, params):
            x, hx, avail_action = params
            x, hy = self.network(x, hx)
            x[avail_action == 0] = -9999999

            greedy_action = self.categorical.mode(x)
            greedy_action = self.expand_dims(greedy_action, 1)

            return greedy_action, hy

    def __init__(self, params):
        self.policy_net = self.QMIXPolicyNet(params)
        self.mixer_net = self.QMIXMixerNet(params)
        self.collect_policy = self.CollectPolicy(self.policy_net, params)
        self.eval_policy = self.EvalPolicy(self.policy_net)


class QMIXActor(Actor):
    """The actor implementation"""

    def __init__(self, params):
        super().__init__()
        self.collect_policy = params['collect_policy']
        self.eval_policy = params['eval_policy']
        self.collect_environment = params['collect_environment']
        self.eval_environment = params['eval_environment']
        self.ones = P.Ones()
        self.zeros = P.Zeros()
        self.expand_dims = P.ExpandDims()
        self.assign = P.Assign()
        self.zero_int = Tensor(0, ms.int32)
        self.true = Tensor(True, ms.bool_)

    def act(self, phase, params):
        """How to get action and interact with environment"""
        local_obs, hx, avail_action, epsilon_steps = params
        if phase == 2:
            action, hx = self.collect_policy((local_obs, hx, avail_action), epsilon_steps)
            new_state, reward, done, global_obs, new_avail_action = self.collect_environment.step(action)
            return new_state, done, reward, action, hx, global_obs, new_avail_action
        if phase == 3:
            action, hx = self.eval_policy((local_obs, hx, avail_action))
            new_state, reward, done, global_obs, new_avail_action = self.eval_environment.step(action)
            return new_state, done, reward, action, hx, global_obs, new_avail_action
        return 0

    def get_action(self, phase, params):
        local_obs, hx, avail_action, epsilon_steps = params
        if phase == 2:
            action, hx = self.collect_policy((local_obs, hx, avail_action), epsilon_steps)
            return action, hx
        if phase == 3:
            action, hx = self.eval_policy((local_obs, hx, avail_action))
            return action, hx
        return 0


class QMIXLearner(Learner):
    """The learner implementation"""

    class QMIXLossCell(nn.Cell):
        """The loss cell implementation of QMIX"""

        def __init__(self, params, policy_net, mixer_net, target_mixer_net):
            super().__init__()
            self.policy_net = policy_net
            self.mixer_net = mixer_net
            self.target_mixer_net = target_mixer_net
            self.zero_int = Tensor(0, ms.int32)
            self.zero_float = Tensor(0, ms.float32)
            self.square = P.Square()
            self.zeros = P.Zeros()
            self.zeros_like = P.ZerosLike()
            self.reshape = P.Reshape()
            self.gather = P.GatherD()
            self.expand_dims = P.ExpandDims()
            self.argmax = P.Argmax()
            self.stack = P.Stack(axis=1)
            self.gamma = params['gamma']
            self.batch = params['batch_size']

        def construct(self, episode_local_obs, global_obs, next_global_obs, action, avail_action, reward, mask, filled,
                      q_val_target, episode_hy):
            """Calculate the loss, which is used for backpropagation"""
            reshaped_local_obs = self.reshape(episode_local_obs, (-1, episode_local_obs.shape[-1]))
            reshaped_episode_hy = self.reshape(episode_hy, (-1, 64))
            gru_out, _ = self.policy_net(reshaped_local_obs, reshaped_episode_hy)
            q_val_policy = self.reshape(gru_out, (
                self.batch, avail_action.shape[1], avail_action.shape[2], avail_action.shape[3]))
            temp_chosen_q = self.gather(q_val_policy, 3, action)
            chosen_q = (temp_chosen_q[:, :-1]).squeeze(-1)

            next_q_val_policy = q_val_policy[:, 1:]
            next_q_val_policy[avail_action[:, 1:] == 0] = -9999999
            max_q_action = self.expand_dims(self.argmax(next_q_val_policy), -1)
            max_q = self.gather(q_val_target, 3, max_q_action).squeeze(-1)

            q_tot_policy = self.mixer_net(chosen_q, global_obs)
            q_tot_target = self.target_mixer_net(max_q, next_global_obs)
            y_true = reward + self.gamma * mask * q_tot_target
            diff = (q_tot_policy - y_true) * filled
            loss = self.square(diff).sum() / filled.sum()
            return loss

    def __init__(self, params):
        super().__init__()
        self.zero_int = Tensor(0, ms.int32)
        self.one_float = Tensor(1, ms.float32)
        self.policy_net = params['policy_net']
        self.target_net = self.policy_net.clone()
        self.mixer_net = params['mixer_net']
        self.batch = params['batch_size']
        self.target_mixer_net = self.mixer_net.clone()
        self.expand_dims = P.ExpandDims()
        self.reshape = P.Reshape()
        self.stack = P.Stack(axis=1)
        self.gather = P.GatherD()
        self.zeros = P.Zeros()
        self.zeros_like = P.ZerosLike()
        self.less = P.Less()
        self.transpose = P.Transpose()

        trainable_params = self.policy_net.trainable_params() + self.mixer_net.trainable_params()
        optimizer = nn.RMSProp(trainable_params, learning_rate=params['lr'])

        qmix_loss_cell = self.QMIXLossCell(params, self.policy_net, self.mixer_net, self.target_mixer_net)
        self.qmix_net_train = nn.TrainOneStepCell(qmix_loss_cell, optimizer)
        self.qmix_net_train.set_train(mode=True)

        policy_param = self.policy_net.trainable_params() + self.mixer_net.trainable_params()
        target_param = self.target_net.trainable_params() + self.target_mixer_net.trainable_params()
        self.target_soft_updater = SoftUpdate(1, 200, policy_param, target_param)

    def learn(self, experience):
        """Prepare the data and do the backpropagation"""
        episode_local_obs, episode_global_obs, episode_action, \
        episode_avail_action, episode_reward, episode_done, \
        episode_filled, episode_hy = experience

        target_hy = self.zeros((episode_local_obs.shape[2] * self.batch, 64), ms.float32)
        global_obs = episode_global_obs[:, :-1]
        next_global_obs = episode_global_obs[:, 1:]
        reward = episode_reward[:, :-1]
        done = episode_done[:, :-1]
        filled = episode_filled[:, :-1]

        i = 0
        q_val_target = []
        transposed = self.transpose(episode_local_obs, (1, 0, 2, 3))
        while i < episode_local_obs.shape[1]:
            step_next_state = transposed[i]
            step_next_state = self.reshape(step_next_state, (step_next_state.shape[0] * step_next_state.shape[1], -1))
            q_out_target, target_hy = self.target_net(step_next_state, target_hy)
            q_out_target = self.reshape(q_out_target,
                                        (-1, episode_avail_action.shape[-2], episode_avail_action.shape[-1]))
            q_val_target.append(q_out_target)
            i += 1

        mask = 1 - done
        q_val_target = self.stack(q_val_target)[:, 1:]
        q_val_target[episode_avail_action[:, 1:] == 0] = -9999999
        loss = self.qmix_net_train(episode_local_obs, global_obs, next_global_obs, episode_action, episode_avail_action,
                                   reward, mask, filled, q_val_target, episode_hy)
        self.target_soft_updater()
        return loss
