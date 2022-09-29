# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
'''
Test case for DQN training.
'''
import pytest
from mindspore import context
from mindspore_rl.algorithm.dqn import DQNSession
from mindspore_rl.algorithm.dqn import DQNTrainer


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_train_dqn():
    '''
    Feature: Test dqn algorithm.
    Description: Test dqn algorithm.
    Expectation: success.
    '''
    context.set_context(mode=context.GRAPH_MODE)
    ac_session = DQNSession()
    ac_session.run(class_type=DQNTrainer, episode=5)
    assert True
