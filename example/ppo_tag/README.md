# Proximal Policy Optimization (PPO)

## Related Paper

1. Schulman, John, et al. ["Proximal policy optimization algorithm"](https://arxiv.org/pdf/1707.06347.pdf)

## Game that this algorithm used

PPO uses MindSpore Reinforcement built-in environment called Tag. Tag is a multi-agent environment. The predators should learn to capture prey, and prey should learn to avoid being caught by predators. All agents needs to stay away from the map boundary. Otherwise they will get penalty. Tag environment is built build on GPU. It can greatly improve the efficiency of experience collection through GPU  multi-threading.

PP Tag is used to study environment research on GPU and other accelerated devices. So measurement data such as accuracy will not be provided.

## How to run PPO Tag

Before running PPO Tag, you should first install [MindSpore](https://www.mindspore.cn/install) and MindSpore-Reinforcement. Besides, you should also install following dependencies. Please follow the instruction on the official website.

- MindSpore >= 1.6.0

After installation, you can directly use the following command to run the PPO algorithm.

### Train

```shell
> cd example/ppo_tag/scripts
> bash run_standalone_train.sh [CKPT_PATH]
```

You will obtain outputs which is similar with the things below in `ppo_tag_train_log.txt`.

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

## Supported Platform

PPO Tag algorithm supports GPU platform.
