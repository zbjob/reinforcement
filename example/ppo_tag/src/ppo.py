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
"""PPO Tag"""

import mindspore
from mindspore import Tensor
import mindspore.nn.probability.distribution as msd
import mindspore.nn as nn
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from mindspore_rl.agent.actor import Actor
from mindspore_rl.agent.learner import Learner
from mindspore_rl.utils.discounted_return import DiscountedReturn


class PPOPolicy():
    """This is PPOPolicy class. You should define your networks (PPOActorNet and PPOCriticNet here)
     which you prepare to use in the algorithm. Moreover, you should also define you loss function
     (PPOLossCell here) which calculates the loss between policy and your ground truth value.
    """
    class PPOActorCritorNet(nn.Cell):
        """PPOActorNet is the actor network of PPO algorithm. It takes a set of state as input
         and outputs miu, sigma of a normal distribution"""
        def __init__(self, state_size, hidden_size1, hidden_size2, action_size):
            super(PPOPolicy.PPOActorCritorNet, self).__init__()
            self.dense1 = nn.Dense(state_size, hidden_size1, weight_init='XavierUniform')
            self.dense2 = nn.Dense(hidden_size1, hidden_size2, weight_init='XavierUniform')
            self.dense3 = nn.Dense(hidden_size2, action_size, weight_init='XavierUniform')
            self.dense4 = nn.Dense(hidden_size2, 1, weight_init='XavierUniform')
            self.relu = P.ReLU()
            self.softmax = P.Softmax()
            self.squeeze = P.Squeeze()

        def construct(self, x):
            """calculate actioon probs and state value"""
            x = self.relu(self.dense1(x))
            x = self.relu(self.dense2(x))
            probs = self.softmax(self.dense3(x))
            value = self.dense4(x)
            value = self.squeeze(value)
            return probs, value


    class PPOLossCell(nn.Cell):
        """PPOLossCell calculates the loss of PPO algorithm"""
        def __init__(self, actor_critor_net, epsilon, critic_coef, entropy_coef):
            super(PPOPolicy.PPOLossCell, self).__init__(auto_prefix=False)
            self.actor_critor_net = actor_critor_net
            self.epsilon = epsilon
            self.critic_coef = critic_coef
            self.entropy_coef = entropy_coef

            self.reduce_mean = P.ReduceMean()
            self.minimum = P.Minimum()
            self.exp = P.Exp()
            self.squeeze = P.Squeeze()
            self.categorical = msd.Categorical()
            self.mse = nn.MSELoss()

        def construct(self, actions, states, advantage, log_prob_old, discounted_r):
            """calculate the total loss"""
            probs, value = self.actor_critor_net(states)

            # actor loss
            log_prob_new = self.categorical.log_prob(actions, probs)

            ratio = self.exp(log_prob_new - log_prob_old)[:-1]
            surr = ratio * advantage
            clip_surr = C.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon) *  advantage
            actor_loss = self.reduce_mean(-self.minimum(surr, clip_surr))

            # Critic Loss
            critic_loss = self.mse(discounted_r, value) * self.critic_coef

            # Entropy Loss
            entropy = self.categorical.entropy(probs)
            entropy_loss = self.reduce_mean(entropy) * self.entropy_coef

            # Total Loss
            total_loss = actor_loss + critic_loss - entropy_loss
            return total_loss

    class GradNorm(nn.Cell):
        """Gradient normalization"""
        def __init__(self, clip_value):
            super(PPOPolicy.GradNorm, self).__init__(auto_prefix=False)
            self.stack = P.Stack()
            self.norm = nn.Norm()
            self.hyper_map = C.HyperMap()
            self.mul = P.Mul()
            self.clip_value = Tensor(clip_value, mindspore.float32)
            self.eps = Tensor(1e-7, mindspore.float32)

        def construct(self, grads):
            """calculate gradient normlization"""
            grads_norm = self.hyper_map(self.norm, grads)
            total_norm = self.norm(self.stack(grads_norm))
            if total_norm > self.clip_value:
                coef = self.clip_value / (total_norm + self.eps)
                grads = self.hyper_map(F.partial(self.mul, coef), grads)
            return grads

    class TrainOneStepCell(nn.Cell):
        """Train one step cell"""
        def __init__(self, network, optimizer, clip_value, sens=1.0):
            super(PPOPolicy.TrainOneStepCell, self).__init__(auto_prefix=False)
            self.network = network
            self.network.set_grad()
            self.optimizer = optimizer
            self.weights = self.optimizer.parameters
            self.grad = C.GradOperation(get_by_list=True, sens_param=True)
            self.grad_norm = PPOPolicy.GradNorm(clip_value)
            self.sens = sens

        def construct(self, *inputs):
            """Train one step"""
            loss = self.network(*inputs)
            sens = F.fill(loss.dtype, loss.shape, self.sens)
            grads = self.grad(self.network, self.weights)(*inputs, sens)
            grads = self.grad_norm(grads)
            loss = F.depend(loss, self.optimizer(grads))
            return loss

    def __init__(self, params):
        self.actor_critic_net = self.PPOActorCritorNet(params['state_space_dim'],
                                                       params['hidden_size1'],
                                                       params['hidden_size2'],
                                                       params['action_space_dim'])
        trainable_parameter = self.actor_critic_net.trainable_params()
        optimizer_ppo = nn.Adam(trainable_parameter, learning_rate=params['lr'])
        ppo_loss_net = self.PPOLossCell(
            self.actor_critic_net,
            Tensor(params['epsilon'], mindspore.float32),
            Tensor(params['critic_coef'], mindspore.float32),
            Tensor(params['entropy_coef'], mindspore.float32))
        clip_value = Tensor(Tensor(params['grad_clip'], mindspore.float32))
        self.ppo_net_train = self.TrainOneStepCell(ppo_loss_net, optimizer_ppo, clip_value)
        self.ppo_net_train.set_train(mode=True)


class PPOActor(Actor):
    """This is an actor class of PPO algorithm, which is used to interact with environment, and
    generate/insert experience (data) """
    def __init__(self, params=None):
        super(PPOActor, self).__init__()
        self._params_config = params
        self._environment = params['collect_environment']
        self._eval_env = params['eval_environment']
        self._buffer = params['replay_buffer']
        self.actor_critic_net = params['actor_critic_net']
        self.categorical = msd.Categorical()

    def act(self, state):
        """collect experience and insert to replay buffer (used during training)"""
        probs, value = self.actor_critic_net(state)
        action = self.categorical.sample((), probs)
        new_state, reward, done = self._environment.step(action)
        return reward, new_state, action, done, probs, value

    def evaluate(self, state):
        """collect experience (used during evaluation)"""
        probs, _ = self.actor_critic_net(state)
        action = self.categorical.sample((), probs)
        new_state, reward, _ = self._eval_env.step(action)
        return reward, new_state


class PPOLearner(Learner):
    """This is the learner class of PPO algorithm, which is used to update the policy net"""
    def __init__(self, params):
        super(PPOLearner, self).__init__()
        self._params_config = params
        self.gamma = Tensor(self._params_config['gamma'], mindspore.float32)
        self.iter_times = params['iter_times']
        self.actor_critic_net = params['actor_critic_net']
        self._ppo_net_train = params['ppo_net_train']

        self.zeros_like = P.ZerosLike()
        self.zero = Tensor(0, mindspore.float32)
        self.zero_int = Tensor(0, mindspore.int32)
        self.categorical = msd.Categorical()
        self.discounted_return = DiscountedReturn(self._params_config['gamma'])
        self.expand_dims = P.ExpandDims()

    def learn(self, samples):
        """prepare for the value (advantage, discounted reward), which is used to calculate
        the loss"""
        # states: [100, 2000, 5, 21]
        # actions: [100, 2000, 5]
        # rewards: [100, 2000, 5]
        # done: [100, 2000]
        # probs: [100, 2000, 5, 5]
        # values: [100, 2000, 5]
        states, actions, rewards, done, probs, values = samples

        valid_step_num = states.shape[0] - 1
        last_state = states[-1]
        states = states[:valid_step_num]
        actions = actions[:valid_step_num]
        rewards = rewards[:valid_step_num]
        done = done[:valid_step_num]
        probs = probs[:valid_step_num]
        values = values[:valid_step_num]

        _, last_state_value = self.actor_critic_net(last_state)
        # Discounted return.
        done = self.expand_dims(done, -1)
        discounted_r = self.discounted_return(rewards, done, last_state_value)
        # Single step advantage: reward_i + gamma * value_{i+1} - value_i
        advantage = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        # Old policy log probs.
        log_prob_old = self.categorical.log_prob(actions, probs)

        i = self.zero
        loss = self.zero
        while i < self.iter_times:
            loss += self._ppo_net_train(actions, states, advantage, log_prob_old, discounted_r)
            i += 1

        return loss / self.iter_times
