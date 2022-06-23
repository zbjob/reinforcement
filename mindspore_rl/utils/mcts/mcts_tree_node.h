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

#ifndef MINDSPORE_RL_UTILS_MCTS_MCTS_TREE_NODE_H_
#define MINDSPORE_RL_UTILS_MCTS_MCTS_TREE_NODE_H_

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

class MonteCarloTreeNode {
 public:
  // The base class of MonteCarloTreeNode.
  MonteCarloTreeNode(std::string name, int action, float prior, float *init_reward, int player, int64_t tree_handle,
                     std::shared_ptr<MonteCarloTreeNode> parent_node, int row, int state_size)
      : name_(name),
        action_(action),
        prior_(prior),
        player_(player),
        row_(row),
        total_reward_(0),
        explore_count_(0),
        terminal_(false),
        tree_handle_(tree_handle),
        parent_(parent_node) {
    if (state_size > 0) {
      state_ = reinterpret_cast<float *>(malloc(sizeof(float) * state_size));
    } else {
      std::cout << "The state size is smaller than 0, please check" << std::endl;
    }
  }

  virtual ~MonteCarloTreeNode() = default;

  // It will select the child whose value of SelectionPolicy is the highest.
  std::shared_ptr<MonteCarloTreeNode> SelectChild();

  // The virtual function of SelectionPolicy. In this function, user needs to implement the rule to select
  // child node, such as UCT(UCB) function, RAVE, AMAF, etc.
  virtual bool SelectionPolicy(float *uct_value) const = 0;

  // The virtual function of Update. It is invoked during backpropagation. User needs implement how to update
  // the local value according to the input returns.
  virtual bool Update(float *returns) = 0;

  // After the whole tree finished, use BestAction to obtain the best action for the root.
  std::shared_ptr<MonteCarloTreeNode> BestAction() const;

  // The policy to choose BestAction
  // The default policy is that:
  // 1. First compare the outcome of two nodes
  // 2. If both of them does not have outcome (or same), then compare the explore_count_
  // 3. If they have the same explore_count_, then compare the total_reward_
  virtual bool BestActionPolicy(std::shared_ptr<MonteCarloTreeNode> child_node) const;

  bool IsLeafNode() { return children_.empty(); }
  void AddChild(std::shared_ptr<MonteCarloTreeNode> child) { children_.emplace_back(child); }
  std::vector<std::shared_ptr<MonteCarloTreeNode>> children() { return children_; }

  void set_state(float *input_state, int state_size) {
    for (int i = 0; i < state_size; i++) {
      state_[i] = input_state[i];
    }
  }
  float *state() { return state_; }

  void set_terminal(bool done) { terminal_ = done; }
  bool terminal() { return terminal_; }

  void set_outcome(std::vector<float> new_outcome) { outcome_ = new_outcome; }
  std::vector<float> outcome() { return outcome_; }

  int action() { return action_; }
  int row() { return row_; }
  int player() { return player_; }
  int explore_count() { return explore_count_; }
  float total_reward() { return total_reward_; }

  std::string DebugString() {
    std::ostringstream oss;
    oss << tree_handle_ << "_" << name_ << "_row_" << row_ << "_player_" << player_;
    oss << "_action_" << action_ << "_terminal_" << terminal_;
    return oss.str();
  }

 private:
  std::string name_;     // The name of this node.
  float *state_;         // The state current node states for.
  bool terminal_;        // Whether current node is terminal node.
  int action_;           // The action that transfers from parent node to current node.
  int row_;              // Which row this node belongs to (for DEBUG).

 protected:
  float prior_;          // P(a|s), the probability that choose this node in parent node.
  int player_;           // This node belongs to which player.
  int explore_count_;    // Number of times that current node is visited.
  float total_reward_;   // The total reward of current node.
  int64_t tree_handle_;  // Current node belongs to which tree.

  std::vector<float> outcome_;                                 // The outcome of terminal node.
  std::shared_ptr<MonteCarloTreeNode> parent_;                 // Parent node.
  std::vector<std::shared_ptr<MonteCarloTreeNode>> children_;  // All child node.
};
using MonteCarloTreeNodePtr = std::shared_ptr<MonteCarloTreeNode>;

#endif  // MINDSPORE_RL_UTILS_MCTS_MCTS_TREE_NODE_H_
