"""C51 Trainer"""

import mindspore as ms
import mindspore.nn as nn
from mindspore.common.api import ms_function
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P
from mindspore_rl.agent.trainer import Trainer
from mindspore_rl.agent import trainer


class CategoricalDQNTrainer(Trainer):
    """DQN Trainer"""

    def __init__(self, msrl, params):
        nn.Cell.__init__(self, auto_prefix=False)
        self.zero = Tensor(0, ms.float32)
        self.squeeze = P.Squeeze()
        self.less = P.Less()
        self.zero_value = Tensor(0, ms.float32)
        self.fill_value = Tensor(1000, ms.float32)
        self.inited = Parameter(Tensor((False,), ms.bool_), name='init_flag')
        self.mod = P.Mod()
        self.false = Tensor((False,), ms.bool_)
        self.true = Tensor((True,), ms.bool_)
        self.num_evaluate_episode = params['num_evaluate_episode']
        self.print = P.Print()
        super(CategoricalDQNTrainer, self).__init__(msrl)

    def trainable_variables(self):
        """Trainable variables for saving."""
        trainable_variables = {"policy_net": self.msrl.learner.policy_network}
        return trainable_variables

    @ms_function
    def init_training(self):
        """Initialize training"""
        state = self.msrl.collect_environment.reset()
        done = self.false
        i = self.zero_value

        while self.less(i, self.fill_value):
            done, _, new_state, action, my_reward = self.msrl.agent_act(
                trainer.INIT, state)
            self.msrl.replay_buffer_insert(
                [state, action, my_reward, new_state])

            state = new_state
            if done:
                state = self.msrl.collect_environment.reset()
                done = self.false
            i += 1
        return done

    @ms_function
    def train_one_episode(self):
        """Train one episode"""
        if not self.inited:
            self.init_training()
            self.inited = self.true
        state = self.msrl.collect_environment.reset()
        done = self.false
        total_reward = self.zero
        steps = self.zero
        loss = self.zero

        while not done:
            done, r, new_state, action, my_reward = self.msrl.agent_act(
                trainer.COLLECT, state)
            self.msrl.replay_buffer_insert(
                [state, action, my_reward, new_state])

            state = new_state
            r = self.squeeze(r)
            loss = self.msrl.agent_learn(self.msrl.replay_buffer_sample())
            total_reward += r
            steps += 1
            self.msrl.learner.update()
        return loss, total_reward, steps

    @ms_function
    def evaluate(self):
        """Policy evaluate"""
        total_reward = self.zero_value
        eval_iter = self.zero_value
        while self.less(eval_iter, self.num_evaluate_episode):
            episode_reward = self.zero_value
            state = self.msrl.eval_environment.reset()
            done = self.false
            while not done:
                done, r, state = self.msrl.agent_act(trainer.EVAL, state)
                r = self.squeeze(r)
                episode_reward += r
            total_reward += episode_reward
            eval_iter += 1
        avg_reward = total_reward / self.num_evaluate_episode
        return avg_reward
