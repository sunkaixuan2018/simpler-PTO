# comm_gather_runner: host + device (__global__ AICORE) in one TU.
# Must be built with CANN toolchain on a2a3 (compiler that handles device code).
# Set PTO_ISA_ROOT and ASCEND_HOME (or CANN install path) before make.
#
# Example (on a2a3 with CANN):
#   export PTO_ISA_ROOT=/path/to/pto-isa
#   export ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest
#   make -C examples/host_build_graph/comm_gather/kernels/host

CXX ?= g++
TARGET = comm_gather_runner
SRCS = main_comm_gather.cpp

# PTO_ISA_ROOT and ASCEND_HOME should be set by caller
PTO_ISA_ROOT ?= $(abspath ../../../../../scripts/_deps/pto-isa)
ASCEND_HOME ?= /usr/local/Ascend/ascend-toolkit/latest

INC = -I$(PTO_ISA_ROOT)/include -I$(ASCEND_HOME)/include
LDFLAGS = -L$(ASCEND_HOME)/lib64 -lacl -lhccl -lpthread

# Note: main_comm_gather.cpp includes run_gather_kernel.cpp which contains
# __global__ AICORE device code. g++ cannot compile it. Use CANN mixed compiler
# if available (e.g. set CXX to CANN wrapper). Otherwise build on device with CANN.
$(TARGET): $(SRCS) comm_common.hpp run_gather_kernel.cpp
	$(CXX) $(INC) -o $@ main_comm_gather.cpp $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(TARGET)
