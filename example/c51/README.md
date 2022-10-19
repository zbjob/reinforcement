# Categorical 51-atom Agent Algorithm (C51)

## Related Paper

1. Marc G. Bellemare, Will Dabney, Rémi Munos, [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)

## Game that this algorithm used

C51 use  an open source reinforcement learning environment library called [Gym](https://github.com/openai/gym) which is developed by OpenAI. Compared with the traditional DQN algorithm, the desired Q is a numerical value, in the series of value distribution reinforcement learning algorithms, the target is changed from a numerical value to a distribution. This change allows you to learn more than just a numerical value, but the complete value distribution.

The game solved in C51 from Gym is [**CartPole-v0**](https://www.gymlibrary.dev/environments/classic_control/cart_pole/). "A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over."[1](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)

## How to run C51

Before running C51, you should first install [MindSpore](https://www.mindspore.cn/install) and MindSpore-Reinforcement. Besides, you should also install following dependencies. Please follow the instruction on the official website.

- MindSpore >= 1.9.0
- Reinforcement >= 0.6
- numpy >= 1.17.0
- [gym](https://github.com/openai/gym) >= 0.18.3

After installation, you can directly use the following command to run the C51 algorithm.

### Train

```shell
> cd example/c51/scripts
> bash run_standalone_train.sh [CKPT_PATH] [DEVICE_TARGET](optional)
```

You will obtain outputs which is similar with the things below in`example/c51/scripts/c51_train_log.txt`

```shell
Episode 0: loss is 3.935, rewards is 8.0
Episode 1: loss is 3.928, rewards is 9.0
Episode 2: loss is 3.916, rewards is 10.0
Episode 3: loss is 3.906, rewards is 9.0
Episode 4: loss is 3.884, rewards is 8.0
Episode 5: loss is 3.874, rewards is 11.0
Episode 6: loss is 3.811, rewards is 9.0
Episode 7: loss is 3.73, rewards is 9.0
Episode 8: loss is 3.637, rewards is 8.0
Episode 9: loss is 3.4, rewards is 9.0
Episode 10: loss is 2.921, rewards is 10.0
-----------------------------------------
Evaluate for episode 10 total rewards is 9.200
```

### Eval

```shell
> cd example/c51/scripts
> bash run_standalone_eval.sh [CKPT_PATH] [DEVICE_TARGET](optional)
```

You will obtain outputs which is similar with the things below in `example/c51/scripts/c51_eval_log.txt`.

```shell
Load file  /ckpt/policy_net/policy_net_100.ckpt
-----------------------------------------
Evaluate result is 200.000, checkpoint file in /ckpt/policy_net/policy_net_100.ckpt
-----------------------------------------
eval end

## Supported Platform

C51 algorithm supports CPU platform.
