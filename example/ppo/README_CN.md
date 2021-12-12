# Proximal Policy Optimization (PPO)

## 相关论文

1. Schulman, John, et al. ["Proximal policy optimization algorithm"](https://arxiv.org/pdf/1707.06347.pdf)

## 使用的游戏

PPO算法使用了OpenAI开发的一个强化学习环境库叫做[Gym](https://github.com/openai/gym)，来作为算法的游戏环境。

在PPO中，解决了[**HalfCheetah-v2**](https://gym.openai.com/envs/HalfCheetah-v2/)这个来自Gym库的游戏。与于其他游戏如CartPole不同的是，这个游戏还依赖[MuJoCo](https://github.com/openai/mujoco-py)这个库。

## 如何运行PPO

在运行PPO前，首先需要安装[MindSpore](https://www.mindspore.cn/install)和MindSpore-Reinforcement。除此之外，还需要安装以下依赖。请根据官网的教程安装。

- MindSpore >= 1.6.0

- numpy >= 1.17.0
- matplotlib >=3.1.3
- [gym](https://github.com/openai/gym) >= 0.18.3
- [mujoco-py](https://github.com/openai/mujoco-py)<2.1,>=2.0

安装成功之后，可以直接通过输入如下指令来运行PPO。

在CPU综合性能上的考虑，建议统一配置`OMP_NUM_THREADS` 的值为物理核数的1/4，比如`export OMP_NUM_THREADS=32`(需在`run_standalone_train.sh`中修改)。

### 训练

```shell
> cd example/ppo/scripts
> bash run_standalone_train.sh [CKPT_PATH] [DEVICE_TARGET](可选)
```

你会在`example/ppo/scripts/log.txt`中获得和下面内容相似的输出

```shell
Episode 0, loss is 90.738, reward is -165.2470.
Episode 1, loss is 26.225, reward is -118.340.
Episode 2, loss is 28.407, reward is -49.168.
Episode 3, loss is 46.766, reward is 11.567.
Episode 4, loss is 81.542, reward is 70.142.
Episode 5, loss is 101.262, reward is 131.269.
Episode 6, loss is 95.097, reward is 198.397.
Episode 7, loss is 97.289, reward is 232.347.
Episode 8, loss is 133.97, reward is 260.215.
Episode 9, loss is 112.115, reward is 296.540.
Episode 10, loss is 123.75, reward is 314.946.
Episode 11, loss is 140.75, reward is 315.864.
Episode 12, loss is 157.439, reward is 442.881.
Episode 13, loss is 217.987, reward is 477.359.
Episode 14, loss is 199.457, reward is 461.781.
Episode 15, loss is 194.124, reward is 448.245.
Episode 16, loss is 199.476, reward is 534.768.
Episode 17, loss is 210.211, reward is 535.590.
Episode 18, loss is 207.483, reward is 568.602.
Episode 19, loss is 216,794, reward is 637.380.
-----------------------------------------
Evaluate result for episode 20 total rewards is 782.709
-----------------------------------------
```

### 推理

```shell
> cd example/ppo/scripts
> bash run_standalone_eval.sh [CKPT_PATH] [DEVICE_TARGET](可选)
```

你会在`example/ppo/scripts/ppo_eval_log.txt`中获得和下面内容相似的输出

```shell
Load file /path/ckpt/actor_net/actor_net_950.ckpt
-----------------------------------------
Evaluate result is 6000.300, checkpoint file in /path/ckpt/actor_net/actor_net_950.ckpt
-----------------------------------------
```

## 支持平台

PPO算法支持GPU和CPU。
