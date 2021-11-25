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
The environment base class.
"""

import mindspore.nn as nn

class Environment(nn.Cell):
    r"""
    The virtual base class for the environment. This class should be overridden before calling in the model.
    """

    def __init__(self):
        super(Environment, self).__init__(auto_prefix=False)

    def reset(self):
        raise NotImplementedError("Method should be overridden by subclass.")

    def step(self, action):
        raise NotImplementedError("Method should be overridden by subclass.")

    @property
    def action_space(self):
        raise NotImplementedError("Method should be overridden by subclass.")

    @property
    def observation_space(self):
        raise NotImplementedError("Method should be overridden by subclass.")

    @property
    def reward_space(self):
        raise NotImplementedError("Method should be overridden by subclass.")

    @property
    def done_space(self):
        raise NotImplementedError("Method should be overridden by subclass.")

    @property
    def config(self):
        raise NotImplementedError("Method should be overridden by subclass.")
