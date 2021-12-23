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
Cudnn Gru network.
"""

import mindspore.nn as nn
from mindspore import context
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore._checkparam import Validator as validator
from mindspore.ops.operations import _rl_inner_ops as rl_ops


class GruNet(nn.Cell):
    """
    A basic fully connected neural network.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 weight_init='normal',
                 num_layers=1,
                 has_bias=True,
                 batch_first=False,
                 dropout=0.0,
                 bidirectional=False):
        super().__init__()
        validator.check_positive_int(hidden_size, "hidden_size", self.cls_name)
        validator.check_positive_int(input_size, "input_size", self.cls_name)
        validator.check_positive_int(num_layers, "num_layers", self.cls_name)
        validator.check_is_float(dropout, "dropout", self.cls_name)
        validator.check_value_type("has_bias", has_bias, [bool], self.cls_name)
        validator.check_value_type(
            "batch_first", batch_first, [bool], self.cls_name)
        validator.check_value_type(
            "bidirectional", bidirectional, [bool], self.cls_name)

        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.enable_cudnn = context.get_context('device_target') in ['GPU']
        if self.enable_cudnn:
            weight_size = 0
            gate_size = 3 * hidden_size
            num_directions = 2 if bidirectional else 1
            for layer in range(num_layers):
                input_layer_size = input_size if layer == 0 else hidden_size * num_directions
                increment_size = gate_size * input_layer_size
                increment_size += gate_size * hidden_size
                if has_bias:
                    increment_size += 2 * gate_size
                weight_size += increment_size * num_directions
            self.weight = Parameter(initializer(
                weight_init, [weight_size, 1, 1]), name="cudnn_weight")
            self.gru = rl_ops.CudnnGRU(input_size=input_size,
                                       hidden_size=hidden_size,
                                       num_layers=num_layers,
                                       has_bias=has_bias,
                                       bidirectional=bidirectional,
                                       dropout=float(dropout))
        else:
            self.gru = nn.GRU(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              has_bias=has_bias,
                              bidirectional=bidirectional,
                              dropout=float(dropout))

    def construct(self, x_in, h_in):
        """
        The forward calculation of gru net
        """
        x_out = None
        h_out = None
        if self.enable_cudnn:
            x_out, h_out, _, _ = self.gru(x_in, h_in, self.weight)
        else:
            x_out, h_out = self.gru(x_in, h_in)
        return x_out, h_out
