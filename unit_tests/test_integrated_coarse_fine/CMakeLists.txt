cmake_minimum_required(VERSION 3.10)
project(test_integrated_coarse_fine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../cuda)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common)

set(SOURCES
    test_integrated_coarse_fine.cu
    ../common/test_utils.cu
    ../../cuda/l2norm.cu
    ../../cuda/fusion_cos_topk/fusion_cos_topk_warpsort.cu
    ../../cuda/fine_screen_top_n/fine_screen_top_n.cu
    ../../cuda/integrate_screen/integrate_screen.cu
)

add_executable(test_integrated_coarse_fine ${SOURCES})

target_link_libraries(test_integrated_coarse_fine
    ${CUDA_LIBRARIES}
    cublas
    curand
)

set_target_properties(test_integrated_coarse_fine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

enable_testing()
add_test(NAME test_integrated_coarse_fine COMMAND test_integrated_coarse_fine)
