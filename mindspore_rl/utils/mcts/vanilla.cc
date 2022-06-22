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

#include <utils/mcts/vanilla.h>
#include <cmath>
#include <iostream>
#include <limits>

bool VanillaTreeNode::SelectionPolicy(float* uct_value) const {
  if (!outcome_.empty()) {
    *uct_value = outcome_[player_];
    return true;
  }
  if (explore_count_ == 0) {
    *uct_value = std::numeric_limits<float>::infinity();
    return true;
  }

  auto global_variable_vector = MonteCarloTreeFactory::GetInstance().GetTreeVariableByHandle(tree_handle_);
  if (global_variable_vector.empty()) {
    std::cout << "[Error]Please input a constant value for UCT calculation" << std::endl;
    return false;
  }
  auto uct_ptr = static_cast<float*>(global_variable_vector[0]);
  *uct_value =
      total_reward_ / explore_count_ + (*uct_ptr) * std::sqrt(std::log(parent_->explore_count()) / explore_count_);
  return true;
}

bool VanillaTreeNode::Update(float* values) {
  total_reward_ += values[player_];
  explore_count_ += 1;
  return true;
}
