# Proximal Policy Optimization (PPO)

## 相关论文

1. Schulman, John, et al. ["Proximal policy optimization algorithm"](https://arxiv.org/pdf/1707.06347.pdf)

## 使用的游戏

PPO Tag使用MindSpore Reinforcement内置的Tag环境作为游戏环境。Tag一个多智能体环境，环境中的predators需要学会捕获prey，prey则需要学会避免被predators捕获；与此同时，所有智能体需要学会避免触碰地图边界，否则它们将得到惩罚。Mindpore Reinforcement内置的Tag环境构建于GPU设备之上，它可以充分利用GPU多线程资源，大幅提升经验收集效率。

目前PPO Tag用于GPU等加速设备上环境研究和性能分析，暂不提供精度等度量数据。

## 如何运行PPO Tag

在运行PPO前，首先需要安装[MindSpore](https://www.mindspore.cn/install)和MindSpore-Reinforcement。除此之外，还需要安装以下依赖。请根据官网的教程安装。

- MindSpore >= 1.6.0

安装成功之后，可以直接通过输入如下指令来运行PPO。

### 训练

```shell
> cd example/ppo_tag/scripts
> bash run_standalone_train.sh [CKPT_PATH]
```

你会在`example/ppo_tag/scripts/ppo_tag_train_log.txt`中获得和下面内容相似的输出

```shell
Episode 0, steps: 100, loss: 0.098, time: 2.832
Episode 1, steps: 100, loss: 0.083, time: 0.096
Episode 2, steps: 100, loss: 0.069, time: 0.095
Episode 3, steps: 100, loss: 0.057, time: 0.097
Episode 4, steps: 100, loss: 0.045, time: 0.093
Episode 5, steps: 100, loss: 0.034, time: 0.102
Episode 6, steps: 100, loss: 0.023, time: 0.103
Episode 7, steps: 100, loss: 0.013, time: 0.095
Episode 8, steps: 100, loss: 0.004, time: 0.094
Episode 9, steps: 100, loss: -0.004, time: 0.097
Episode 10, steps: 100, loss: -0.012, time: 0.095
Episode 11, steps: 100, loss: -0.019, time: 0.097
Episode 12, steps: 100, loss: -0.026, time: 0.095
Episode 13, steps: 100, loss: -0.031, time: 0.096
Episode 14, steps: 100, loss: -0.036, time: 0.095
Episode 15, steps: 100, loss: -0.040, time: 0.095
Episode 16, steps: 100, loss: -0.044, time: 0.097
Episode 17, steps: 100, loss: -0.047, time: 0.095
Episode 18, steps: 100, loss: -0.049, time: 0.097
```

## 支持平台

PPO Tag算法支持GPU。
