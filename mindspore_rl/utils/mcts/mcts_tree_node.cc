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

#include <utils/mcts/mcts_tree_node.h>
#include <algorithm>

MonteCarloTreeNodePtr MonteCarloTreeNode::SelectChild() {
  std::vector<float> selection_value;
  // For each child, use selection policy to calculate corresponding value,
  // then choose the largest one.
  for (auto &child : children_) {
    float uct_value;
    bool ret = child->SelectionPolicy(&uct_value);
    selection_value.emplace_back(uct_value);
    if (!ret) {
      return nullptr;
    }
  }
  int64_t max_position = std::distance(std::begin(selection_value),
                                       std::max_element(std::begin(selection_value), std::end(selection_value)));
  return children_[max_position];
}

MonteCarloTreeNodePtr MonteCarloTreeNode::BestAction() const {
  return *std::max_element(children_.begin(), children_.end(),
                           [](const MonteCarloTreeNodePtr node_a, const MonteCarloTreeNodePtr node_b) {
                             return node_a->BestActionPolicy(node_b);
                           });
}

bool MonteCarloTreeNode::BestActionPolicy(MonteCarloTreeNodePtr node) const {
  float outcome_self = (outcome_.empty() ? 0 : outcome_[player_]);
  float outcome_input = (node->outcome().empty() ? 0 : node->outcome()[node->player()]);
  if (outcome_self != outcome_input) {
    return outcome_self < outcome_input;
  }
  if (explore_count_ != node->explore_count()) {
    return explore_count_ < node->explore_count();
  }
  return total_reward_ < node->total_reward();
}
