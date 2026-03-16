# Build libsdma_helper.so — SDMA workspace manager for async comm.
# Requires CANN env: source .../setenv.bash, then:
#   mkdir build && cd build && cmake .. && make

cmake_minimum_required(VERSION 3.16)
project(sdma_helper CXX)

set(CMAKE_CXX_STANDARD 17)

if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
elseif(DEFINED ENV{ASCEND_HOME})
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME})
else()
    message(FATAL_ERROR "ASCEND_HOME_PATH or ASCEND_HOME not set. Run: source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash")
endif()

# Resolve pto-comm-isa root to absolute path.
# Allow PTO_COMM_ISA_ROOT env override (same convention as kernel compiler).
if(DEFINED ENV{PTO_COMM_ISA_ROOT})
    get_filename_component(PTO_COMM_ISA_ROOT "$ENV{PTO_COMM_ISA_ROOT}" ABSOLUTE)
else()
    get_filename_component(PTO_COMM_ISA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../_deps/pto-comm-isa" ABSOLUTE)
endif()
if(NOT EXISTS "${PTO_COMM_ISA_ROOT}/include")
    message(FATAL_ERROR "pto-comm-isa include dir not found at ${PTO_COMM_ISA_ROOT}/include. "
        "Set PTO_COMM_ISA_ROOT env or ensure examples/scripts/_deps/pto-comm-isa is present.")
endif()
message(STATUS "PTO_COMM_ISA_ROOT: ${PTO_COMM_ISA_ROOT}")

add_library(sdma_helper SHARED sdma_helper.cpp)
target_include_directories(sdma_helper PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${PTO_COMM_ISA_ROOT}/include
)
target_link_directories(sdma_helper PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(sdma_helper PRIVATE ascendcl)
