"""c51 full connect layer"""

import mindspore.nn as nn
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from noisy_layer import NoisyLinear


class FullyConnectedNet(nn.Cell):
    """full connect layer with noisy option"""

    def __init__(self, input_size, hidden_size, output_size, action_dim, atoms_num, compute_type=mstype.float32,
                 noisy=False):
        super(FullyConnectedNet, self).__init__()
        self.linear1 = nn.Dense(
            input_size,
            hidden_size, weight_init="XavierUniform"
        ).to_float(compute_type)
        self.linear2 = nn.Dense(
            hidden_size,
            output_size, weight_init="XavierUniform"
        ).to_float(compute_type)
        self.use_noisy = noisy
        if noisy:
            self.linear2 = nn.Dense(
                hidden_size,
                hidden_size, weight_init="XavierUniform"
            ).to_float(compute_type)
            self.noisy1 = NoisyLinear(hidden_size, hidden_size)
            self.noisy2 = NoisyLinear(hidden_size, output_size)
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()
            self.reset_nosiy()
        else:
            self.linear2 = nn.Dense(
                hidden_size,
                output_size, weight_init="XavierUniform"
            ).to_float(compute_type)
        self.relu1 = nn.ReLU()
        self.softmax = nn.Softmax(1)
        self.cast = P.Cast()
        self.action_dim = action_dim
        self.atoms_num = atoms_num

    def construct(self, x):
        """
        Returns output of Dense layer.

        Args:
            x (Tensor): Tensor as the input of network.

        Returns:
            The output of the Dense layer.
        """
        x = self.relu1(self.linear1(x))
        if self.use_noisy:
            x = self.relu2(self.linear2(x))
            x = self.relu3(self.noisy1(x))
            x = self.noisy2(x)
        else:
            x = self.linear2(x)
        x = self.softmax(x.view(-1, self.atoms_num)).view(-1, self.action_dim, self.atoms_num)
        x = self.cast(x, mstype.float32)
        return x

    def reset_nosiy(self):
        """Default reset nosiy"""

        self.noisy1.reset_noise()
        self.noisy2.reset_noise()
        return True
