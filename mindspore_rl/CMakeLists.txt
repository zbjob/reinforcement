cmake_minimum_required(VERSION 3.14.0)
project(MindSpore_RL)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

file(GLOB MCTS_COMMON_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "utils/mcts/*.cc")

file(GLOB_RECURSE MCTS_SRC_CPU RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "utils/mcts/cpu/*.cc")

list(APPEND MCTS_SRC_CPU ${MCTS_COMMON_SRC})

set(LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
add_library(mcts_cpu SHARED ${MCTS_SRC_CPU})

install(TARGETS mcts_cpu DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/mindspore_rl/utils/mcts")