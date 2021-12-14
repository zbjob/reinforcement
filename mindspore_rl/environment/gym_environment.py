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
The GymEnvironment base class.
"""

import gym
from gym import spaces
import numpy as np
from mindspore.ops import operations as P
from mindspore_rl.environment.environment import Environment
from mindspore_rl.environment.space import Space


class GymEnvironment(Environment):
    """
    The GymEnvironment class provides the functions to interact with
    different environments.

    Args:
        params (dict): A dictionary contains all the parameters which are used to create the
            instance of GymEnvironment, such as name of environment.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> env_params = {'name': 'CartPole-v0'}
        >>> environment = GymEnvironment(env_params)
        >>> print(environment)
        GymEnvironment<>
    """

    def __init__(self, params):
        super(GymEnvironment, self).__init__()
        self.params = params
        self._name = params['name']
        self._env = gym.make(self._name)
        if 'seed' in params:
            self._env.seed(params['seed'])
        self._observation_space = self._space_adapter(self._env.observation_space)
        self._action_space = self._space_adapter(self._env.action_space)
        self._reward_space = Space((1,), np.float32)
        self._done_space = Space((1,), np.bool_, low=0, high=2)

        # reset op
        reset_input_type = []
        reset_input_shape = []
        reset_output_type = [self._observation_space.ms_dtype,]
        reset_output_shape = [self._observation_space.shape,]
        self._reset_op = P.PyFunc(self._reset, reset_input_type,
                                  reset_input_shape, reset_output_type, reset_output_shape)

        # step op
        step_input_type = (self._action_space.ms_dtype,)
        step_input_shape = (self._action_space.shape,)
        step_output_type = (self.observation_space.ms_dtype, self._reward_space.ms_dtype, self._done_space.ms_dtype)
        step_output_shape = (self._observation_space.shape, self._reward_space.shape, self._done_space.shape)
        self._step_op = P.PyFunc(
            self._step, step_input_type, step_input_shape, step_output_type, step_output_shape)

    def reset(self):
        """
        Reset the environment to the initial state. It is always used at the beginning of each
        episode. It will return the value of initial state.

        Returns:
            A tensor which states for the initial state of environment.

        """

        return self._reset_op()[0]

    def step(self, action):
        r"""
        Execute the environment step, which means that interact with environment once.

        Args:
            action (Tensor): A tensor that contains the action information.

        Returns:
            - state (Tensor), the environment state after performing the action.
            - reward (Tensor), the reward after performing the action.
            - done (mindspore.bool\_), whether the simulation finishes or not.
        """

        return self._step_op(action)

    @property
    def observation_space(self):
        """
        Get the state space of the environment.

        Returns:
            A tuple which states for the space of state
        """

        return self._observation_space

    @property
    def action_space(self):
        """
        Get the action space of the environment.

        Returns:
            A tuple which states for the space of action
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
            A numpy array which states for the initial state of environment.
        """

        s0 = self._env.reset()
        # In same gym version, the obvervation space is announced to be float32, but get float64 from reset and step.
        s0 = s0.astype(self.observation_space.np_dtype)
        return s0

    def _step(self, action):
        """
        The python(can not be interpreted by mindspore interpreter) code of interacting with the
        environment. It is the main body of step function. Due to Pyfunc, we need to
        capsule python code into a function.

        Args:
            action(int or float): The action which is calculated by policy net. It could be integer
            or float, according to different environment

        Returns:
            - s1 (numpy.array), the environment state after performing the action.
            - r1 (numpy.array), the reward after performing the action.
            - done (boolean), whether the simulation finishes or not.
        """

        s, r, done, _ = self._env.step(action)
        # In same gym version, the obvervation space is announced to be float32, but get float64 from reset and step.
        s = s.astype(self.observation_space.np_dtype)
        r = np.array([r]).astype(np.float32)
        done = np.array([done])
        return s, r, done

    def _space_adapter(self, gym_space):
        shape = gym_space.shape
        # The dtype get from gym.space is np.int64, but step() accept np.int32 actually.
        dtype = np.int32 if gym_space.dtype.type == np.int64 else gym_space.dtype.type
        if isinstance(gym_space, spaces.Discrete):
            return Space(shape, dtype, low=0, high=gym_space.n)

        return Space(shape, dtype, low=gym_space.low, high=gym_space.high)
