"""noisy layer"""
import mindspore.nn as nn
from mindspore import ops
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore._checkparam import Validator
from mindspore.common.initializer import initializer
from mindspore._extends import cell_attr_register
import mindspore.numpy as mnp


class NoisyLinear(nn.Cell):
    """noisy layer"""

    @cell_attr_register(attrs=['weight_epsilon', 'bias_epsilon'])
    def __init__(self, in_channels, out_channels, weight_init='normal', bias_init='zeros'):
        super(NoisyLinear, self).__init__()
        self.in_channels = Validator.check_positive_int(in_channels, "in_channels", self.cls_name)
        self.out_channels = Validator.check_positive_int(out_channels, "out_channels", self.cls_name)

        self.weight_mu = Parameter(initializer(weight_init, [out_channels, in_channels]), name="weight_mu")
        self.weight_sigma = Parameter(initializer(weight_init, [out_channels, in_channels]), name="weight_sigma")

        self.bias_mu = Parameter(initializer(bias_init, [out_channels]), name="bias_mu")
        self.bias_sigma = Parameter(initializer(bias_init, [out_channels]), name="bias_sigma")

        self.weight_epsilon = Parameter(initializer('zeros', [out_channels, in_channels]), name="weight_epsilon")
        self.bias_epsilon = Parameter(initializer('zeros', [out_channels]), name="bias_epsilon")

        self.matmul = P.MatMul(transpose_b=True)
        self.mul = P.Mul()
        self.bias_add = P.BiasAdd()

        self.reset_noise()

    def construct(self, x):
        if self.training:
            weight = self.weight_mu + self.mul(self.weight_sigma, self.weight_epsilon)
            bias = self.bias_mu + self.mul(self.bias_sigma, self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        x = self.matmul(x, weight)
        x = self.bias_add(x, bias)
        return x

    def reset_noise(self):
        """reset noise"""

        epsilon_in = self._scale_noise(self.in_channels)
        epsilon_out = self._scale_noise(self.out_channels)

        self.weight_epsilon = ops.tensor_dot(epsilon_out, epsilon_in, 0).copy()
        self.bias_epsilon = self._scale_noise(self.out_channels).copy()
        return True

    def _scale_noise(self, size):
        """initialize noise"""

        noise = mnp.randn(size)
        noise = self.mul(mnp.sign(noise), mnp.sqrt(noise.abs()))
        return noise
