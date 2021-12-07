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
"""GymMultiEnvironment class."""

import gym
from gym import spaces
import numpy as np
import mindspore as ms
from mindspore.ops import operations as P
from mindspore_rl.environment.environment import Environment
from mindspore_rl.environment.space import Space


class GymMultiEnvironment(Environment):
    """
    The GymMultiEnvironment class provides the functions to interact with
    different environments. It is the multi-environment version of GymEnvironment.

    Args:
        params (dict): A dictionary contains all the parameters which are used to create the
            instance of GymEnvironment, such as name of environment, number of environment.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> env_params = {'name': 'CartPole-v0', 'env_nums': 10}
        >>> environment = GymMultiEnvironment(env_params)
        >>> print(environment)
        GymMultiEnvironment<>
    """

    def __init__(self,
                 params):
        super(GymMultiEnvironment, self).__init__()
        self.params = params
        self._name = params['name']
        self._nums = params['env_nums']
        self._envs = []
        for i in range(self._nums):
            self._envs.append(gym.make(self._name))
            if 'seed' in params:
                self._envs[i].seed(params['seed'] + i)
            elif 'seeds' in params:
                seeds = params['seeds']
                if seeds.size() != self._nums:
                    raise ValueError("Seeds size should equal to envirenment numbers.")
                self._envs[i].seed(seeds[i])

        self._observation_space = self._space_adapter(self._envs[0].observation_space, self._nums)
        self._action_space = self._space_adapter(self._envs[0].action_space, self._nums)
        self._reward_space = Space((1,), np.float32, batch_shape=(self._nums,))
        self._done_space = Space((1,), np.int32, low=0, high=2, batch_shape=(self._nums,))

        self._step_op = P.PyFunc(self._step,
                                 [self.action_space.ms_dtype,],
                                 [self.action_space.shape,],
                                 [self.observation_space.ms_dtype, ms.float32, ms.bool_],
                                 [self.observation_space.shape, (self._nums, 1), (self._nums, 1)])
        self._reset_op = P.PyFunc(self._reset, [], [],
                                  [self.observation_space.ms_dtype,],
                                  [self.observation_space.shape,])

    def reset(self):
        """
        Reset the environment to the initial state. It is always used at the beginning of each
        episode. It will return the value of initial state of each environment.

        Returns:
            A list of tensor which states for all the initial states of each environment.

        """

        return self._reset_op()[0]

    def step(self, action):
        """
        Execute the environment step, which means that interact with environment once.

        Args:
            action (Tensor): A tensor that contains the action information.

        Returns:
            - state (Tensor), a list of environment state after performing the action.
            - reward (Tensor), a list of reward after performing the action.
            - done (Tensor), whether the simulations of each environment finishes or not
        """

        return self._step_op(action)


    @property
    def observation_space(self):
        """
        Get the state space dim of the environment.

        Returns:
            A tuple which states for the space dimension of state
        """

        return self._observation_space

    @property
    def action_space(self):
        """
        Get the action space dim of the environment.

        Returns:
            A tuple which states for the space dimension of action
        """

        return self._action_space

    @property
    def reward_space(self):
        return self._reward_space

    @property
    def done_space(self):
        return self._done_space

    @property
    def config(self):
        return {}

    def _reset(self):
        """
        The python(can not be interpreted by mindspore interpreter) code of resetting the
        environment. It is the main body of reset function. Due to Pyfunc, we need to
        capsule python code into a function.

        Returns:
            A list of numpy array which states for the initial state of each environment.
        """

        self._done = False
        s0 = [env.reset().astype(self.observation_space.np_dtype) for env in self._envs]
        return s0

    def _step(self, action):
        """
        The python(can not be interpreted by mindspore interpreter) code of interacting with the
        environment. It is the main body of step function. Due to Pyfunc, we need to
        capsule python code into a function.

        Args:
            action(List[numpy.dtype]): The action which is calculated by policy net.
            It could be List[int] or List[float] or other else, according to different environment.

        Returns:
            - s1 (List[numpy.array]), a list of environment state after performing the action.
            - r1 (List[numpy.array]), a list of reward after performing the action.
            - done (List[boolean]), whether the simulations of each environment finishes or not
        """
        s1, r1, done = [], [], []
        for i in range(self._nums):
            s, r, d, _ = self._envs[i].step(action[i])
            s1.append(s.astype(self.observation_space.np_dtype))
            r1.append(np.array([r]).astype(np.float32))
            done.append(np.array([d]))
        s1 = np.stack(s1)
        r1 = np.stack(r1)
        done = np.stack(done)
        return s1, r1, done

    def _space_adapter(self, gym_space, environment_num):
        shape = gym_space.shape
        # The dtype get from gym.space is np.int64, but step() accept np.int32 actually.
        dtype = np.int32 if gym_space.dtype.type == np.int64 else gym_space.dtype.type
        # The float64 is not supported, cast to float32
        dtype = np.float32 if gym_space.dtype.type == np.float64 else gym_space.dtype.type
        if isinstance(gym_space, spaces.Discrete):
            return Space(shape, dtype, low=0, high=gym_space.n, batch_shape=(environment_num,))

        return Space(shape, dtype, low=gym_space.low, high=gym_space.high, batch_shape=(environment_num,))
