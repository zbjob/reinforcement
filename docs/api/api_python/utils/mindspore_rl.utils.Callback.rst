
.. py:class:: mindspore_rl.utils.CallbackParam(dict)
    
    回调函数的参数。


.. py:class:: mindspore_rl.utils.CallbackManager(Callback)

    依次执行回调函数。

    参数：
        - **callbacks** (Callback) - 回调函数。

    .. py:method:: begin()

        在训练执行开始调用，仅执行一次。

    .. py:method:: end()

        在训练执行结束调用，仅执行一次。

    .. py:method:: episode_begin()

        在每个episode执行前调用。

    .. py:method:: episode_end()

        在每个episode执行后调用。


.. py:class:: mindspore_rl.utils.LossCallback(Callback)

    在每个episode结束时打印loss值。

    参数：
        - **print_rate** (int) - 打印loss的频率。

    .. py:method:: episode_end()

        在每个episode执行后调用，打印loss值。

        参数:
            - **params** (CallbackParam) - 训练参数，用于获取结果。


.. py:class:: mindspore_rl.utils.TimeCallback(Callback)

    在每个episode结束时打印耗时。

    参数：
        - **print_rate** (int) - 打印耗时的频率。
        - **fixed_steps_in_episode** (Optional[int]) - 如果每个episode的steps是固定的，则提供一个固定步长值，否则将取实际步长。默认：None。

    .. py:method:: episode_begin()

        在每个episode执行前记录时间。

        参数:
            - **params** (CallbackParam) - 训练参数，用于获取结果。

    .. py:method:: episode_end()

        在每个episode执行后调用，打印耗时。

        参数:
            - **params** (CallbackParam) - 训练参数，用于获取结果。

.. py:class:: mindspore_rl.utils.CheckpointCallback(Callback)

    保存模型的checkpoint文件，保留最新的 `max_ckpt_nums` 个。

    参数：
        - **save_per_episode** (int) - 保存ckpt文件的频率。
        - **directory** (Optional[str]) - 保存ckpt文件的路径。默认当文件夹。
        - **max_ckpt_nums** (Optional[int]) - 最大保留ckpt的个数。默认：5。

    .. py:method:: episode_end()

        在每个episode执行后调用，保存ckpt文件。

        参数:
            - **params** (CallbackParam) - 训练参数，用于获取结果。

.. py:class:: mindspore_rl.utils.EvaluateCallback(Callback)

    推理回调。

    参数：
        - **eval_rate** (int) - 推理的频率。

    .. py:method:: begin()

        在训练开始前保存推理频率。

    .. py:method:: episode_end()

        在每个episode执行后调用，推理并打印结果。
