/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_RL_UTILS_MCTS_MCTS_TREE_H_
#define MINDSPORE_RL_UTILS_MCTS_MCTS_TREE_H_

#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include "utils/mcts/mcts_tree_node.h"

class MonteCarloTree {
 public:
  MonteCarloTree(MonteCarloTreeNodePtr root, float max_utility, int64_t tree_handle, int state_size)
      : root_(root), max_utility_(max_utility), tree_handle_(tree_handle), state_size_(state_size) {}
  ~MonteCarloTree() = default;

  // The Selection phase of monte carlo tree search, it will continue selecting child node based on selection
  // policy (like UCT) until leaf node.
  bool Selection(std::vector<int> *action_list, int max_action);

  // The Expansion phase of monte carlo tree search, it will create the child node based on input action and prior
  // for last node in visited path.
  bool Expansion(std::string node_name, int *action, float *prior, float *init_reward, int num_action, int player,
                 int state_size);

  // The Backpropagation phase of monte carlo tree search, it will update the value in each visited node according to
  // the input returns (obtained in simulation).
  bool Backpropagation(float *returns);

  // Select the best action of root
  int BestAction();

  void UpdateState(float *input_state, int index) { visited_path_[index]->set_state(input_state, state_size_); }
  float *GetState(int index) { return visited_path_[index]->state(); }

  void UpdateOutcome(std::vector<float> input_return, int index) { visited_path_[index]->set_outcome(input_return); }
  void UpdateTerminal(bool is_terminal, int index) { visited_path_[index]->set_terminal(is_terminal); }

  int64_t placeholder_handle() { return placeholder_handle_; }
  std::vector<MonteCarloTreeNodePtr> visited_path() { return visited_path_; }
  int state_size() { return state_size_; }
  MonteCarloTreeNodePtr root() { return root_; }

 private:
  float max_utility_;  // The max utility of game, which is used in backpropagation.

 protected:
  int64_t tree_handle_;                              // The tree handle which is used to create the node.
  int state_size_;                                   // Number of element of state
  int64_t placeholder_handle_ = -1;                  // A dummy handle.
  MonteCarloTreeNodePtr root_;                       // The ptr of root node.
  std::vector<MonteCarloTreeNodePtr> visited_path_;  // The visited path which is obtained in Selection().
};
using MonteCarloTreePtr = std::shared_ptr<MonteCarloTree>;

#endif  // MINDSPORE_RL_UTILS_MCTS_MCTS_TREE_H_
