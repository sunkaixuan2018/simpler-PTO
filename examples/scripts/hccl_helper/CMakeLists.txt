# Build libhccl_helper.so — same link as pto-comm-isa (ascendcl, hcomm, runtime).
# Run with CANN env: source /path/to/Ascend/set_env.sh, then:
#   mkdir build && cd build && cmake .. && make

cmake_minimum_required(VERSION 3.16)
project(hccl_helper CXX)

set(CMAKE_CXX_STANDARD 17)

if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
elseif(DEFINED ENV{ASCEND_HOME})
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME})
else()
    message(FATAL_ERROR "ASCEND_HOME_PATH or ASCEND_HOME not set. Run: source /path/to/Ascend/.../set_env.sh")
endif()

add_library(hccl_helper SHARED hccl_helper.cpp)
target_include_directories(hccl_helper PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/hccl
)
target_link_directories(hccl_helper PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(hccl_helper PRIVATE ascendcl hcomm runtime)
