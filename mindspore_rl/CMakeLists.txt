cmake_minimum_required(VERSION 3.14.0)
project(MindSpore_RL)

if(DEBUG_MODE)
    set(CMAKE_BUILD_TYPE "DEBUG")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

file(GLOB MCTS_COMMON_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "utils/mcts/*.cc")

file(GLOB_RECURSE MCTS_SRC_CPU RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "utils/mcts/cpu/*.cc")
list(APPEND MCTS_SRC_CPU ${MCTS_COMMON_SRC})

set(LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
add_library(mcts_cpu SHARED ${MCTS_SRC_CPU})
install(TARGETS mcts_cpu DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/mindspore_rl/utils/mcts")

if(ENABLE_GPU)
    find_package(CUDA REQUIRED)
    include_directories((${CUDA_INCLUDE_DIRS}))
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_53 --expt-relaxed-constexpr")

    if(DEBUG_MODE)
        list(APPEND CUDA_NVCC_FLAGS "-g -G")
    endif()

    file(GLOB_RECURSE MCTS_SRC_GPU RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "utils/mcts/gpu/*.cc" "utils/mcts/gpu/*.cu")
    list(APPEND MCTS_SRC_GPU ${MCTS_COMMON_SRC})
    cuda_add_library(mcts_gpu SHARED ${MCTS_SRC_GPU})
    install(TARGETS mcts_gpu DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/mindspore_rl/utils/mcts")
endif()
