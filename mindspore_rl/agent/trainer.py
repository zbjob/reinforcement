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
Implementation of trainer base class.
"""

import os
import mindspore.nn as nn
from mindspore.train.serialization import save_checkpoint


class Trainer(nn.Cell):
    r"""
    The trainer base class.

    Note:
        Reference to `dqn_trainer.py
        <https://gitee.com/mindspore/docs/blob/master/docs/reinforcement/docs/source_en/dqn.md
        #defining-the-dqntrainer-class>`_.

    Args:
        msrl(object): the function handler class.
    """

    def __init__(self, msrl):
        self.msrl = msrl

    def train(self, episode):
        """
        The interface of the train function. User will implement
        this function.

        Args:
            episode(int): the number of training episode.
        """

        raise NotImplementedError("Method should be overridden by subclass.")

    def save_ckpt(self, path, model, episode, max_ckpt_nums=5):
        """
        Save the checkpoint file for all the model weights. And keep the latest `max_ckpt_nums` checkpoint files.

        Args:
            path (str): The checkpoint path.
            model (Union[Cell, list]): The cell object or data list to be saved.
            episode (int): The episode number of this checkpoint.
            max_ckpt_nums (int): Numbers of how many checkpoint files to be kept. Default:5.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        ckpt_name = path + '/checkpoint_' + str(episode) + '.ckpt'
        save_checkpoint(model, ckpt_name)
        files = os.listdir(path)
        ckpt_list = []
        nums = 0
        for filename in files:
            if os.path.splitext(filename)[-1] == '.ckpt':
                nums += 1
                ckpt_list.append(path + "/" + filename)
            if nums > max_ckpt_nums:
                ckpt_files = sorted(ckpt_list, key=os.path.getmtime)
                os.remove(ckpt_files[0])


    def eval(self):
        """
        The interface of the eval function.
        """
        raise NotImplementedError("Method should be overridden by subclass.")
