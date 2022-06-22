cmake_minimum_required(VERSION 3.14.0)
project(MindSpore_RL)

set(RL_PATH ${CMAKE_SOURCE_DIR}/mindspore_rl)
include_directories(${RL_PATH})
add_subdirectory(${RL_PATH})