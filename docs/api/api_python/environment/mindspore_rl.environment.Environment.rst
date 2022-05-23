.. py:class:: mindspore_rl.environment.Environment

    环境的虚基类。在调用此类之前，请重写其中的方法。

    .. py:method:: reset()

        将环境重置为初始状态。reset方法一般在每个episode开始时使用，并返回环境的初始状态值以及其reset方法初始信息。

        **返回: **
        
        表示环境初始状态的Tensor或者Tuple包含初始信息。

    .. py:method:: step(action)

        执行环境Step函数来和环境交互一次。

        **参数: **

        - **action** (Tensor) - 包含动作信息的Tensor。

        **返回: **

        tuple，包含和环境交互后的信息，如新的状态，动作，奖励等。

    .. py:method:: action_space
        :property:

        返回环境的动作空间。

    .. py:method:: config
        :property:

        返回一个包含环境信息的字典。

    .. py:method:: done_space
        :property:

        返回环境的终止空间。

    .. py:method:: observation_space
        :property:

        返回环境的状态空间。

    .. py:method:: reward_space
        :property:

        返回环境的奖励空间。
