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

#include <utils/mcts/mcts_factory.h>
#include <utils/mcts/mcts_tree.h>
#include <utils/mcts/mcts_tree_node.h>
#include <cstdint>
#include <iostream>

constexpr int kErrorCode = 2;
constexpr int kInputIndex = 3;

std::map<int, std::string> map_node_enum_to_string = {{0, "Vanilla"}};
std::map<int, std::string> map_tree_enum_to_string = {{0, "Common"}};

extern "C" int MctsCreation(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                            void *extra) {
  // Input value
  // MctsCreation has 5 compulsory input values:
  // 1. The tree enumerate
  // 2. The node enumerate
  // 3. Which player does this root belong to
  // 4. The max utility of this game
  // 5. Number of element of state
  int64_t *tree_enum = static_cast<int64_t *>(params[0]);
  int64_t *node_enum = static_cast<int64_t *>(params[1]);
  int *player = static_cast<int *>(params[2]);
  float *max_utility = static_cast<float *>(params[3]);
  int *state_size = static_cast<int *>(params[4]);
  // The input of MctsCreation which starts from 4th will be treated as the global variable of the monte carlo tree. It
  // is shared by all the node in this monte carlo tree. These variable will be saved in a std::vector with void* type.
  // User can call MonteCarloTreeFactory::GetInstance().GetTreeVariableByHandle(tree_handle_) to obtain the variable
  // vector and select the corresponding variable by index.
  std::vector<void *> input_global_variable;
  for (int i = 5; i < nparam - 1; i++) {
    input_global_variable.push_back(params[i]);
  }
  // Output value
  // The output value of MctsCreation is the unique handle of this new monte carlo tree.
  int64_t *output = static_cast<int64_t *>(params[nparam - 1]);

  auto node_name_iter = map_node_enum_to_string.find(*node_enum);
  if (node_name_iter == map_node_enum_to_string.end()) {
    std::ostringstream oss;
    oss << "[Error]The input enum of node " << *node_enum << " in MctsCreation does not exist.\n";
    oss << "Node register: [";
    for (auto iter = map_node_enum_to_string.begin(); iter != map_node_enum_to_string.end(); iter++) {
      oss << iter->first << " ";
    }
    oss << "]";
    std::cout << oss.str() << std::endl;
    return kErrorCode;
  }
  auto node_name = node_name_iter->second;

  auto tree_name_iter = map_tree_enum_to_string.find(*tree_enum);
  if (tree_name_iter == map_tree_enum_to_string.end()) {
    std::ostringstream oss;
    oss << "[Error]The input enum of tree " << *tree_enum << " in MctsCreation does not exist.\n";
    oss << "Tree register: [";
    for (auto iter = map_tree_enum_to_string.begin(); iter != map_tree_enum_to_string.end(); iter++) {
      oss << iter->first << " ";
    }
    oss << "]";
    std::cout << oss.str() << std::endl;
    return kErrorCode;
  }
  auto tree_name = tree_name_iter->second;
  int64_t tree_handle;
  MonteCarloTreePtr tree;
  std::tie(tree_handle, tree) = MonteCarloTreeFactory::GetInstance().CreateTree(
      tree_name, node_name, *player, *max_utility, *state_size, input_global_variable);
  if (tree == nullptr) {
    return kErrorCode;
  }
  output[0] = tree_handle;
  return 0;
}

extern "C" int MctsSelection(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                             void *extra) {
  // Input value
  // 1. The input value of MctsSelection is the unique tree handle. This function will use the unique handle to
  //    obtain the monte carlo tree that user create in MctsCreation.
  // 2. The max length of action that MctsSelection returns.
  int64_t *tree_handle = static_cast<int64_t *>(params[0]);
  int *max_action = static_cast<int *>(params[1]);
  // Output value
  // It has two output value:
  // 1. The handle of visited path, but its a dummy handle. There is no map to represent the mapping between handle
  //    and visited path. The visited path is saved in tree. This dummy handle is more like an dummy object that
  //    user can operate in python side.
  // 2. If the max_action is given, it will return a Tensor which is combined by actions of each node in visited path.
  //    Moreover, the Tensor will be filled with -1, if its length does not reach the max_action value.
  //    If the max_action is NOT given, it will only return the last action in visited path.
  // 3. the length of visited path
  int64_t *visited_path_handle = static_cast<int64_t *>(params[2]);
  int *out_action = static_cast<int *>(params[3]);
  int *out_path_length = static_cast<int *>(params[4]);

  auto tree = MonteCarloTreeFactory::GetInstance().GetTreeByHandle(*tree_handle);
  if (tree == nullptr) {
    return kErrorCode;
  }
  int size_of_action = *max_action;
  if (*max_action == -1) {
    size_of_action = 1;
  }
  std::vector<int> action_list(size_of_action, -1);
  auto ret = tree->Selection(&action_list, *max_action);
  if (!ret) {
    return kErrorCode;
  }
  visited_path_handle[0] = tree->placeholder_handle();
  out_path_length[0] = tree->visited_path().size();
  for (int i = 0; i < size_of_action; i++) {
    out_action[i] = action_list[i];
  }
  return 0;
}

extern "C" int MctsExpansion(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                             void *extra) {
  // Input value
  // MctsExpansion has 6 input values:
  // 1. tree_handle is the unique tree handle.
  // 2. The node enumerate
  // 3. A dummy handle, it is not used.
  // 4. action is a Tensor that is used to create the node
  // 5. prior is a Tensor that states for probability, which has the same length as action.
  // 6. Which player does these nodes belong to
  int64_t *tree_handle = static_cast<int64_t *>(params[0]);
  int64_t *node_enum = static_cast<int64_t *>(params[1]);
  int *action = static_cast<int *>(params[3]);
  float *init_reward = static_cast<float *>(params[4]);
  float *prior = static_cast<float *>(params[5]);
  int *player = static_cast<int *>(params[6]);
  // Output value
  // Whether expansion executes successfully.
  bool *output = static_cast<bool *>(params[7]);

  auto node_name_iter = map_node_enum_to_string.find(*node_enum);
  if (node_name_iter == map_node_enum_to_string.end()) {
    std::ostringstream oss;
    oss << "[Error]The input enum of node " << *node_enum << " in MctsExpansion does not exist.\n";
    oss << "Node register: [";
    for (auto iter = map_node_enum_to_string.begin(); iter != map_node_enum_to_string.end(); iter++) {
      oss << iter->first << " ";
    }
    oss << "]";
    std::cout << oss.str() << std::endl;
    return kErrorCode;
  }
  auto node_name = node_name_iter->second;
  auto tree = MonteCarloTreeFactory::GetInstance().GetTreeByHandle(*tree_handle);
  if (tree == nullptr) {
    return kErrorCode;
  }
  output[0] =
      tree->Expansion(node_name, action, prior, init_reward, shapes[kInputIndex][0], *player, tree->state_size());
  return 0;
}

extern "C" int MctsBackpropagation(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                   void *stream, void *extra) {
  // Input value
  // MctsBackpropagation has 3 input values:
  // 1. tree_handle is the unique tree handle.
  // 2. A dummy handle, it is not used.
  // 3. Returns that obtains from simulation is used to update all the nodes in visited path.
  int64_t *tree_handle = static_cast<int64_t *>(params[0]);
  float *returns = static_cast<float *>(params[2]);
  // Output value
  // Whether backpropagation executes successfully.
  bool *output = static_cast<bool *>(params[3]);
  auto tree = MonteCarloTreeFactory::GetInstance().GetTreeByHandle(*tree_handle);
  output[0] = tree->Backpropagation(returns);
  return 0;
}

extern "C" int BestAction(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                          void *extra) {
  // Input value
  // Tree_handle is the unique tree handle.
  int64_t *tree_handle = static_cast<int64_t *>(params[0]);
  // Output value
  // Return the best action.
  int *output = static_cast<int *>(params[1]);

  auto tree = MonteCarloTreeFactory::GetInstance().GetTreeByHandle(*tree_handle);
  output[0] = tree->BestAction();
}

extern "C" int UpdateOutcome(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                             void *extra) {
  // Input value
  // UpdateOutcome has 4 input values:
  // 1. tree_handle is the unique tree handle.
  // 2. A dummy handle, it is not used.
  // 3. The outcome of terminal state.
  // 4. Which node in visited path does user need to update.
  int64_t *tree_handle = static_cast<int64_t *>(params[0]);
  int *index_ptr = static_cast<int *>(params[2]);
  float *outcome = static_cast<float *>(params[3]);
  // Output value
  // Whether update executes successfully.
  bool *output = static_cast<bool *>(params[4]);

  int num_element = shapes[kInputIndex][0];
  std::vector<float> return_value;
  for (int i = 0; i < num_element; i++) {
    return_value.emplace_back(outcome[i]);
  }
  auto tree = MonteCarloTreeFactory::GetInstance().GetTreeByHandle(*tree_handle);
  int index = *index_ptr;
  if (index < 0) {
    index += tree->visited_path().size();
  }
  tree->UpdateOutcome(return_value, index);
  output[0] = true;
  return 0;
}

extern "C" int UpdateTerminal(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                              void *stream, void *extra) {
  // Input value
  // UpdateTerminal has 4 input values:
  // 1. tree_handle is the unique tree handle.
  // 2. A dummy handle, it is not used.
  // 3. The terminal state.
  // 4. Which node in visited path does user need to update.
  int64_t *tree_handle = static_cast<int64_t *>(params[0]);
  int *index_ptr = static_cast<int *>(params[2]);
  bool *terminal = static_cast<bool *>(params[3]);
  // Output value
  // Whether update executes successfully.
  bool *output = static_cast<bool *>(params[4]);

  auto tree = MonteCarloTreeFactory::GetInstance().GetTreeByHandle(*tree_handle);
  int index = *index_ptr;
  if (index < 0) {
    index += tree->visited_path().size();
  }
  tree->UpdateTerminal(*terminal, index);
  output[0] = true;
  return 0;
}

extern "C" int UpdateState(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                           void *extra) {
  // Input value
  // UpdateState has 4 input values:
  // 1. tree_handle is the unique tree handle.
  // 2. A dummy handle, it is not used.
  // 3. State of environment
  // 4. Which node in visited path does user need to update.
  int64_t *tree_handle = static_cast<int64_t *>(params[0]);
  int *index_ptr = static_cast<int *>(params[2]);
  float *state = static_cast<float *>(params[3]);
  // Output value
  // Whether update executes successfully.
  bool *output = static_cast<bool *>(params[4]);

  auto tree = MonteCarloTreeFactory::GetInstance().GetTreeByHandle(*tree_handle);
  int index = *index_ptr;
  if (index < 0) {
    index += tree->visited_path().size();
  }
  tree->UpdateState(state, index);
  output[0] = true;
  return 0;
}

extern "C" int UpdateRootState(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                               void *stream, void *extra) {
  // Input value
  // UpdateState has 2 input values:
  // 1. tree_handle is the unique tree handle.
  // 2. State of environment
  int64_t *tree_handle = static_cast<int64_t *>(params[0]);
  float *state = static_cast<float *>(params[1]);
  // Output value
  // Whether update executes successfully.
  bool *output = static_cast<bool *>(params[2]);

  auto tree = MonteCarloTreeFactory::GetInstance().GetTreeByHandle(*tree_handle);
  tree->root()->set_state(state, tree->state_size());
  output[0] = true;
  return 0;
}

extern "C" int GetState(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                        void *extra) {
  // Input value
  // GetState has 3 input values:
  // 1. tree_handle is the unique tree handle.
  // 2. A dummy handle, it is not used.
  // 4. Which node in visited path does user need to get.
  int64_t *tree_handle = static_cast<int64_t *>(params[0]);
  int *index_ptr = static_cast<int *>(params[2]);
  // Output value
  // The state of the node that user specifies
  float *output = static_cast<float *>(params[3]);

  auto tree = MonteCarloTreeFactory::GetInstance().GetTreeByHandle(*tree_handle);
  int index = *index_ptr;
  if (index < 0) {
    index += tree->visited_path().size();
  }
  auto output_state = tree->GetState(index);
  for (int i = 0; i < tree->state_size(); i++) {
    output[i] = output_state[i];
  }
  return 0;
}

extern "C" int DestroyTree(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                           void *extra) {
  // Input value
  // Tree_handle is the unique tree handle.
  int64_t *tree_handle = static_cast<int64_t *>(params[0]);
  // Output value
  // Whether the destroy executes successfully.
  bool *output = static_cast<bool *>(params[1]);

  MonteCarloTreeFactory::GetInstance().DeleteTree(*tree_handle);
  MonteCarloTreeFactory::GetInstance().DeleteTreeVariable(*tree_handle);

  output[0] = true;
  return 0;
}
