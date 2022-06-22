cmake_minimum_required(VERSION 3.14.0)
project(MindSpore_RL)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
file(GLOB_RECURSE MCTS_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "utils/mcts/*.cc")
set(LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
add_library(mcts SHARED ${MCTS_SRC})

install(TARGETS mcts DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/mindspore_rl/utils/mcts")