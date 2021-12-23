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
Implementation of Actor base class.
"""

import mindspore.nn as nn


class Actor(nn.Cell):
    r"""
    Base class for all actors.

    Examples:
        >>> from mindspore_rl.agent.actor import Actor
        >>> from mindspore_rl.network import FullyConnectedNet
        >>> from mindspore_rl.environment import GymEnvironment
        >>> class MyActor(Actor):
        ...   def __init__(self):
        ...     super(MyActor, self).__init__()
        ...     self.argmax = P.Argmax()
        ...     self.actor_net = FullyConnectedNet(4, 10, 2)
        ...     self.env = GymEnvironment({'name': 'CartPole-v0'})
        >>> my_actor = MyActor()
        >>> print(my_actor)
        MyActor<
        (actor_net): FullyConnectedNet<
        (linear1): Dense<input_channels=4, output_channels=10, has_bias=True>
        (linear2): Dense<input_channels=10, output_channels=2, has_bias=True>
        (relu): ReLU<>
        >
        (environment): GymEnvironment<>
    """

    def __init__(self):
        super(Actor, self).__init__(auto_prefix=False)

    def get_action(self, phase, params):
        """
        The interface of the act function.
        User will need to overload this function according to
        the algorithm. But argument of this function should be
        phase and params. This interface will not interact with
        environment

        Args:
            phase (enum): A enumerate value states for init, collect, eval or other user-defined stage.
            params (tuple(Tensor)): A tuple of tensor as input, which is used to calculate action

        Returns:
            observation (tuple(Tensor)): A tuple of tensor as output, which states for experience
        """

        raise NotImplementedError("Method should be overridden by subclass.")

    def act(self, phase, params):
        """
        The interface of the act function.
        User will need to overload this function according to
        the algorithm. But argument of this function should be
        phase and params. This interface will interact with
        environment

        Args:
            phase (enum): A enumerate value states for init, collect, eval or other user-defined stage.
            params (tuple(Tensor)): A tuple of tensor as input, which is used to calculate action

        Returns:
            observation (tuple(Tensor)): A tuple of tensor as output, which states for experience
        """

        raise NotImplementedError("Method should be overridden by subclass.")
