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

#include <memory>
#include <string>
#include <vector>

class MonteCarloTreeNode {
 public:
  // The base class of MonteCarloTreeNode.
  MonteCarloTreeNode(std::string name, int action, float prior, int player, int64_t tree_handle,
                     std::shared_ptr<MonteCarloTreeNode> parent_node)
      : name_(name),
        action_(action),
        prior_(prior),
        player_(player),
        explore_count_(0),
        total_reward_(0),
        state_(nullptr),
        tree_handle_(tree_handle),
        parent_(parent_node) {}

  virtual ~MonteCarloTreeNode() = default;

  // The virtual function of SelectChild, it will select the child whose value of SelectionPolicy is the highest.
  virtual std::shared_ptr<MonteCarloTreeNode> SelectChild() = 0;

  // The virtual function of SelectionPolicy. In this function, user needs to implement the rule to select
  // child node, such as UCT(UCB) function, RAVE, AMAF, etc.
  virtual float SelectionPolicy() const = 0;

  // The virtual function of Update. It is invoked during backpropagation. User needs implement how to update
  // the local value according to the input returns.
  virtual bool Update(float *returns) = 0;

  bool IsLeafNode() { return children_.empty(); }

  void set_state(float *input_state) { state_ = input_state; }

  float *state() { return state_; }

 private:
  std::string name_;     // The name of this node.
  float *state_;         // The state current node states for.

 protected:
  int action_;           // The action that transfers from parent node to current node.
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
