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
'''
Test case for AC training.
'''

#pylint: disable=C0413
#pylint: disable=C0411
#pylint: disable=W0611
import os
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(ROOT_PATH, 'example')
sys.path.insert(0, MODEL_PATH)

import pytest
from ac.src import config
from ac.src.ac_trainer import ACTrainer
from mindspore import context
from mindspore_rl.core import Session
from mindspore_rl.utils.callback import LossCallback


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_train_ac():
    '''
    Train the AC.
    '''
    context.set_context(mode=context.GRAPH_MODE)
    loss_cb = LossCallback()
    ac_session = Session(alg_config=config.algorithm_config, params=config.trainer_params, callbacks=[loss_cb])
    ac_session.run(class_type=ACTrainer, episode=5)
    assert True
