
.. py:class:: mindspore_rl.utils.OUNoise(input_network)

    在action上加入Ornstein-Uhlenbeck (OU)噪声。

    参数：
        - **stddev** (float) - Ornstein-Uhlenbeck (OU) 噪声标准差。
        - **damping** (float) - Ornstein-Uhlenbeck (OU) 噪声阻尼。
        - **action_shape** (tuple) - 动作的维度。

    .. py:method:: construct(actions)

        参数:
            - **actions** (Tensor) - 添加OU噪声之前的动作。

        返回：
            - **actions** (Tensor) - 添加OU噪声之后的动作。

