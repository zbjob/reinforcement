# Deep Deterministic Policy Graidient (DDPG)

## Related Paper

1. David Silver, Guy Lever, et al. ["Deterministic Policy Gradient Algorithms"](https://proceedings.mlr.press/v32/silver14.pdf)

## Game that this algorithm used

DDPG uses  an open source reinforcement learning environment library called [Gym](https://github.com/openai/gym), which is developed by OpenAI.

The game solved in DDPG is called [**HalfCheetah-v2**](https://gym.openai.com/envs/HalfCheetah-v2/), it is from Gym, but this game depends on an advanced physics simulation called [MuJoCo](https://github.com/openai/mujoco-py).

## How to run DDPG

Before running DDPG, you should first install [MindSpore](https://www.mindspore.cn/install) and MindSpore-Reinforcement. Besides, you should also install following dependencies. Please follow the instruction on the official website.

- MindSpore >= 1.6.0

- numpy >= 1.17.0
- matplotlib >=3.1.3
- [gym](https://github.com/openai/gym) >= 0.18.3
- [mujoco-py](https://github.com/openai/mujoco-py)<2.1,>=2.0

After installation, you can directly use the following command to run the DDPG algorithm.

For comprehensive performance considerations on CPU, it is recommended to set `OMP_NUM_THREADS` and configure a unified configuration to 1/4 of the number of physical cores, such as export `OMP_NUM_THREADS=32`(edit in `run_standalone_train.sh`).

### Train

```shell
> cd example/ddpg/scripts
> bash run_standalone_train.sh [EPISODE](optional) [DEVICE_TARGET](optional)
```

You will obtain outputs which is similar with the things below in `ddpg_train_log.txt`.

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

### Eval

```shell
> cd example/ddpg/scripts
> bash run_standalone_eval.sh [CKPT_FILE_PATH] [DEVICE_TARGET](optional)
```

You will obtain outputs which is similar with the things below in `ddpg_eval_log.txt`.

```shell
Load file /path/ckpt/actor_net/actor_net_950.ckpt
-----------------------------------------
Evaluate result is 6000.300, checkpoint file in /path/ckpt/actor_net/actor_net_950.ckpt
-----------------------------------------
```

## Supported Platform

DDPG algorithm supports Ascend, GPU and CPU platform.
