# Deep Deterministic Policy Gradient (DDPG)

## 相关论文

1. David Silver, Guy Lever, et al. ["Deterministic Policy Gradient Algorithms"](https://proceedings.mlr.press/v32/silver14.pdf)

## 使用的游戏

DDPG算法使用了OpenAI开发的一个强化学习环境库叫做[Gym](https://github.com/openai/gym)，来作为算法的游戏环境。

在DDPG中，解决了[**HalfCheetah-v2**](https://gym.openai.com/envs/HalfCheetah-v2/)这个来自Gym库的游戏。与于其他游戏如CartPole不同的是，这个游戏还依赖[MuJoCo](https://github.com/openai/mujoco-py)这个库。

## 如何运行DDPG

在运行DDPG前，首先需要安装[MindSpore](https://www.mindspore.cn/install)和MindSpore-Reinforcement。除此之外，还需要安装以下依赖。请根据官网的教程安装。

- MindSpore >= 1.6.0
- numpy >= 1.17.0
- [gym](https://github.com/openai/gym) >= 0.18.3
- [mujoco-py](https://github.com/openai/mujoco-py)<2.1,>=2.0

安装成功之后，可以直接通过输入如下指令来运行DDPG。

在CPU综合性能上的考虑，建议统一配置`OMP_NUM_THREADS` 的值为物理核数的1/4，比如`export OMP_NUM_THREADS=32`(需在`run_standalone_train.sh`中修改)。

### 训练

```shell
> cd example/ddpg/scripts
> bash run_standalone_train.sh [CKPT_PATH] [DEVICE_TARGET](可选)
```

你会在`example/ddpg/scripts/ddpg_train_log.txt`中获得和下面内容相似的输出

```shell
Episode 0: loss is 7.221, -1.886, rewards is -445.258
Episode 1: loss is 23.807, -4.404, rewards is -548.967
Episode 2: loss is 10.704, -2.297, rewards is -501.102
Episode 3: loss is 14.524, -2.905, rewards is -383.301
Episode 4: loss is 16.303, -1.562, rewards is -370.852
Episode 5: loss is 28.705, -7.362, rewards is -467.568
Episode 6: loss is 28.81, -4.244, rewards is -179.526
Episode 7: loss is 28.551, -6.252, rewards is -456.068
Episode 8: loss is 26.221, -4.426, rewards is -446.748
Episode 9: loss is 28.088, -2.146, rewards is -420.563
Episode 10: loss is 49.541, -6.26, rewards is -309.811
-----------------------------------------
Evaluate for episode 10 total rewards is -528.183
-----------------------------------------
```

### 推理

```shell
> cd example/ddpg/scripts
> bash run_standalone_eval.sh [CKPT_PATH] [DEVICE_TARGET](可选)
```

你会在`example/ddpg/scripts/ddpg_eval_log.txt`中获得和下面内容相似的输出

```shell
Load file /path/ckpt/actor_net/actor_net_950.ckpt
-----------------------------------------
Evaluate result is 6000.300, checkpoint file in /path/ckpt/actor_net/actor_net_950.ckpt
-----------------------------------------
```

## 支持平台

DDPG算法支持Ascend，GPU和CPU。
