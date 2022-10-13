# Twin Delayed Deep Deterministic Policy Gradient (TD3)

## 相关论文

1. Scott Fujimoto, Herke van Hoof, et al. ["Addressing Function Approximation Error in Actor-Critic Methods"](https://arxiv.org/pdf/1802.09477.pdf)
1. David Silver, Guy Lever, et al. ["Deterministic Policy Gradient Algorithms"](https://proceedings.mlr.press/v32/silver14.pdf)

## 使用的游戏

与DDPG算法一样，TD3算法使用的是由OpenAI开发的强化学习环境库[Gym](https://github.com/openai/gym)。该环境提供了多种游戏，可用来训练不同的强化学习算法。

同样，TD3解决了[HalfCheetah-v2](https://www.gymlibrary.ml/environments/mujoco/half_cheetah/)游戏。如果要运行这个游戏，训练TD3算法，则必须要安装[MuJoCo](https://github.com/openai/mujoco-py)这个库。游戏界面示意图如下(图源:https://www.gymlibrary.dev/environments/mujoco/half_cheetah/)：

![half_cheetah](./img/half_cheetah.gif)

## 如何运行TD3

在运行TD3前，首先需要安装[MindSpore](https://www.mindspore.cn/install)(>=1.7.0)和[MindSpore-Reinforcement](https://mindspore.cn/reinforcement/docs/zh-CN/r0.5/reinforcement_install.html)。除此之外，还需要安装以下依赖。请根据官网的教程安装。

- [MindInsight](https://mindspore.cn/mindinsight/docs/zh-CN/r1.8/mindinsight_install.html) (版本必须与已经安装的MindSpore的相同，建议使用pip安装)
- numpy >= 1.22.0
- [gym](https://github.com/openai/gym) >= 0.21.0
- [mujoco-py](https://github.com/openai/mujoco-py)<2.2,>=2.1

TD3算法的训练过程目前已支持在[MindInsight](https://mindspore.cn/mindinsight/docs/zh-CN/r1.8/index.html)可视化面板上显示，方便用户实时监控训练过程中返回的结果。

### 训练

```shell
> cd example/td3/scripts
> bash run_standalone_train.sh [EPISODE](可选) [DEVICE_TARGET](可选)
```

#### 参数说明

- `EPISODE`：TD3算法训练的episode总数，即算法训练时需要运行的游戏局数，默认为`2000`
- `DEVICE_TARGET`：指定训练的设备，可选择`Auto`,`CPU`或`GPU`，默认为`GPU`

您会在`example/td3/scripts/td3_train_log.txt`中获得和下面内容相似的输出

```shell
Episode 0 has 1000.0 steps, cost time: 26480.130 ms, per step time: 26.480 ms
Episode 0: loss is -2.387, rewards is -465.396
Episode 1 has 1000.0 steps, cost time: 5216.369 ms, per step time: 5.216 ms
Episode 1: loss is 2.088, rewards is -536.777
Episode 2 has 1000.0 steps, cost time: 4607.308 ms, per step time: 4.607 ms
Episode 2: loss is 1.311, rewards is -509.436
Episode 3 has 1000.0 steps, cost time: 4895.000 ms, per step time: 4.895 ms
Episode 3: loss is -2.143, rewards is -484.398
Episode 4 has 1000.0 steps, cost time: 5132.120 ms, per step time: 5.132 ms
Episode 4: loss is -0.563, rewards is -504.957
Episode 5 has 1000.0 steps, cost time: 5128.416 ms, per step time: 5.128 ms
Episode 5: loss is 0.886, rewards is -535.614
Episode 6 has 1000.0 steps, cost time: 5028.265 ms, per step time: 5.028 ms
Episode 6: loss is 1.338, rewards is -558.457
Episode 7 has 1000.0 steps, cost time: 4774.283 ms, per step time: 4.774 ms
Episode 7: loss is -0.599, rewards is -465.199
Episode 8 has 1000.0 steps, cost time: 4625.283 ms, per step time: 4.625 ms
Episode 8: loss is 7.29, rewards is -318.291
Episode 9 has 1000.0 steps, cost time: 4840.158 ms, per step time: 4.840 ms
Episode 9: loss is 6.913, rewards is -511.637
Episode 10 has 1000.0 steps, cost time: 6550.410 ms, per step time: 6.550 ms
Episode 10: loss is 13.045, rewards is -517.86
-----------------------------------------
Evaluate for episode 10 total rewards is -567.105
```

#### 启动MindInsight训练看板

```python
> mindinsight start --summary-base-dir ./summary
```

本算法已集成MindInsight。无论是否打开训练面板，算法都会在运行训练脚本的目录下，记录训练数据到`summary`文件夹中。

如果您已经安装了MindInsight，一般情况下访问`http://127.0.0.1:8080`，即可打开MindInsight。点击对应目录下的“查看训练面板”，即可将训练输出的数据可视化出来。如下图所示。

![zh_light](./img/example_summary_zh.png)

### 推理

```shell
> cd example/td3/scripts
> bash run_standalone_eval.sh [CKPT_PATH] [DEVICE_TARGET](可选)
```

你会在`example/td3/scripts/td3_eval_log.txt`中获得和下面内容相似的输出

```shell
Load file /path/ckpt/actor_net/actor_net_950.ckpt
-----------------------------------------
Evaluate result is 6000.300, checkpoint file in /path/ckpt/actor_net/actor_net_950.ckpt
-----------------------------------------
```

## 支持平台

TD3算法支持GPU和CPU，且在GPU环境下取得较好的性能。
