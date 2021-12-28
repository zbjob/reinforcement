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
Implementation of the session class.
"""
from mindspore_rl.core import MSRL

class Session():
    """
    The Session is a class for running MindSpore RL algorithms.

    Args:
        config (dict): the algorithm configuration or the deployment configuration of the algorithm.
            For more details of configuration of algorithm, please have a look at
            https://www.mindspore.cn/reinforcement/docs/zh-CN/master/custom_config_info.html
    """

    def __init__(self, config):
        self.msrl = MSRL(config)

    def run(self, class_type=None, is_train=True, episode=0, params=None, callbacks=None):
        """
        Execute the reinforcement learning algorithm.

        Args:
            class_type (class type): The class type of the algorithm's trainer class. Default: None.
            is_train (boolean): Run the algorithm in train mode or eval mode. Default: True
            episode (int): The number of episode of the training. Default: 0.
            params (dict): The algorithm specific training parameters. Default: None.
            callbacks (list[Callback]): The callback list. Default: None.
        """

        if class_type:
            if params is None:
                trainer = class_type(self.msrl)
            else:
                trainer = class_type(self.msrl, params)
            ckpt_path = None
            if params and 'ckpt_path' in params:
                ckpt_path = params['ckpt_path']
            if is_train:
                trainer.train(episode, callbacks, ckpt_path)
                print('training end')
            else:
                if ckpt_path:
                    trainer.load_and_eval(ckpt_path)
                    print('eval end')
                else:
                    print('Please provide a ckpt_path for eval.')
