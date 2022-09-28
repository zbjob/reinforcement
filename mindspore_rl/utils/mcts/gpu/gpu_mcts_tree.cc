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

#include <utils/mcts/gpu/gpu_mcts_tree.h>

bool GPUMonteCarloTree::Expansion(std::string node_name, int *action, float *prior, float *init_reward, int num_action,
                                  int state_size) {
  // Expand the last node of visited_path.
  auto leaf_node = visited_path_.at(visited_path_.size() - 1);
  if (init_reward != nullptr) {
    leaf_node->SetInitReward(init_reward);
  }
  int player =
    (leaf_node->action() == nullptr) ? (leaf_node->player()) : ((leaf_node->player() + 1) % total_num_player_);
  int *action_host = new int[sizeof(int) * num_action];
  cudaMemcpy(action_host, action, sizeof(int) * num_action, cudaMemcpyDeviceToHost);
  for (int i = 0; i < num_action; i++) {
    if (*(action_host + i) != -1) {
      auto child_node =
        MonteCarloTreeFactory::GetInstance().CreateNode(node_name, action + i, prior + i, init_reward, player,
                                                        tree_handle_, leaf_node, leaf_node->row() + 1, state_size);
      leaf_node->AddChild(child_node);
    }
  }
  return true;
}

void *GPUMonteCarloTree::AllocateMem(size_t size) {
  void *device_state_ptr = nullptr;
  cudaMalloc(&device_state_ptr, size);
  return device_state_ptr;
}

bool GPUMonteCarloTree::Memcpy(void *dst_ptr, void *src_ptr, size_t size) {
  cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToDevice);
  return true;
}

bool GPUMonteCarloTree::Memset(void *dst_ptr, int value, size_t size) {
  cudaMemset(dst_ptr, value, size);
  return true;
}

bool GPUMonteCarloTree::Free(void *ptr) {
  cudaFree(ptr);
  return true;
}
