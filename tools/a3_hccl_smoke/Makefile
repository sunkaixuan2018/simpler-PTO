CONFIG ?=
ROLE ?=

ifneq ($(strip $(CONFIG)),)
include $(CONFIG)
ifeq ($(ROLE),master)
MPI_CXX := $(MASTER_MPI_CXX)
ASCEND_HOME_PATH := $(MASTER_ASCEND_HOME_PATH)
PTO_ISA_ROOT := $(MASTER_PTO_ISA_ROOT)
DRIVER_LIB := $(MASTER_DRIVER_LIB)
PTO_ENABLE_FLAG := $(MASTER_PTO_ENABLE_FLAG)
CCE_AICORE_ARCH := $(MASTER_CCE_AICORE_ARCH)
else ifeq ($(ROLE),slave)
MPI_CXX := $(SLAVE_MPI_CXX)
ASCEND_HOME_PATH := $(SLAVE_ASCEND_HOME_PATH)
PTO_ISA_ROOT := $(SLAVE_PTO_ISA_ROOT)
DRIVER_LIB := $(SLAVE_DRIVER_LIB)
PTO_ENABLE_FLAG := $(SLAVE_PTO_ENABLE_FLAG)
CCE_AICORE_ARCH := $(SLAVE_CCE_AICORE_ARCH)
else
$(error ROLE must be master or slave when CONFIG is set)
endif
endif

MPI_CXX ?= mpic++
ASCEND_HOME_PATH ?=
PTO_ISA_ROOT ?=
DRIVER_LIB ?=

CCEC := $(ASCEND_HOME_PATH)/bin/bisheng
CXXFLAGS ?= -std=c++17 -O2 -g -Wall -Wextra
PTO_ENABLE_FLAG ?=
CCE_AICORE_ARCH ?=
ACL_LIB ?= acl_rt
FABRIC_ACL_LIB ?= ascendcl

HOST_ARCH := $(shell uname -m)
CANN_PKG_INC := $(ASCEND_HOME_PATH)/$(HOST_ARCH)-linux/pkg_inc
CANN_INCLUDES := \
	-I$(ASCEND_HOME_PATH)/include \
	-I$(ASCEND_HOME_PATH)/include/external \
	-I$(ASCEND_HOME_PATH)/include/c_api \
	-I$(ASCEND_HOME_PATH)/include/c_api/internal \
	-I$(ASCEND_HOME_PATH)/pkg_inc \
	-I$(ASCEND_HOME_PATH)/pkg_inc/runtime \
	-I$(ASCEND_HOME_PATH)/pkg_inc/runtime/runtime \
	-I$(CANN_PKG_INC) \
	-I$(CANN_PKG_INC)/runtime \
	-I$(CANN_PKG_INC)/runtime/runtime

HCCL_TARGET := hccl_allgather_smoke
ROOTINFO_TARGET := hccl_rootinfo_smoke
FABRIC_TARGET := fabric_tload_smoke
KERNEL_LIB := libfabric_tload_kernel.so

DRIVER_RPATH_LINK := \
	-Wl,-rpath-link,$(DRIVER_LIB) \
	-Wl,-rpath-link,$(DRIVER_LIB)/common \
	-Wl,-rpath-link,$(DRIVER_LIB)/driver
HCCL_LIBS := -L$(ASCEND_HOME_PATH)/lib64 -lhccl -l$(ACL_LIB) -pthread $(DRIVER_RPATH_LINK)
FABRIC_LIBS := \
	-L$(ASCEND_HOME_PATH)/lib64 -L$(ASCEND_HOME_PATH)/runtime/lib64 \
	-Wl,-rpath,$(ASCEND_HOME_PATH)/lib64 -Wl,-rpath,$(ASCEND_HOME_PATH)/runtime/lib64 \
	-l$(FABRIC_ACL_LIB) -lruntime -pthread -ldl $(DRIVER_RPATH_LINK)

ASC_ARCH := $(ASCEND_HOME_PATH)/$(HOST_ARCH)-linux
KERNEL_INCLUDES := \
	-I$(PTO_ISA_ROOT)/include \
	-I$(PTO_ISA_ROOT)/include/pto \
	-I$(ASCEND_HOME_PATH)/include \
	-I$(ASCEND_HOME_PATH)/include/hccl \
	-I$(ASC_ARCH)/asc \
	-I$(ASC_ARCH)/asc/include \
	-I$(ASC_ARCH)/ascendc/include/basic_api \
	-I$(ASC_ARCH)/include/ascendc/basic_api \
	-I$(ASC_ARCH)/asc/impl/basic_api \
	-I$(ASC_ARCH)/ascendc/include/basic_api/impl \
	-I$(ASC_ARCH)/asc/include/interface \
	-I$(ASC_ARCH)/asc/include/utils
KERNEL_FLAGS := \
	-O3 -g -xcce -Wall -std=c++17 \
	-Xhost-start -Xhost-end \
	--cce-aicore-arch=$(CCE_AICORE_ARCH) \
	$(PTO_ENABLE_FLAG) \
	-mllvm -cce-aicore-stack-size=0x8000 \
	-mllvm -cce-aicore-function-stack-size=0x8000 \
	-mllvm -cce-aicore-record-overflow=false \
	-mllvm -cce-aicore-addr-transform \
	-mllvm -cce-aicore-dcci-insert-for-scalar=false \
	-DMEMORY_BASE

.PHONY: all baseline rootinfo fabric check-common-deps check-hccl-deps check-fabric-deps clean run run-rootinfo run-hccl

all: baseline rootinfo fabric

baseline: $(HCCL_TARGET)

rootinfo: $(ROOTINFO_TARGET)

fabric: check-fabric-deps $(FABRIC_TARGET) $(KERNEL_LIB)

$(FABRIC_TARGET) $(KERNEL_LIB): | check-fabric-deps

$(HCCL_TARGET) $(ROOTINFO_TARGET): | check-hccl-deps

$(HCCL_TARGET): hccl_allgather_smoke.cc
	$(MPI_CXX) $(CXXFLAGS) -I$(ASCEND_HOME_PATH)/include -o $@ $< $(HCCL_LIBS)

$(ROOTINFO_TARGET): hccl_rootinfo_smoke.cc
	$(MPI_CXX) $(CXXFLAGS) -I$(ASCEND_HOME_PATH)/include -o $@ $< $(HCCL_LIBS)

$(FABRIC_TARGET): fabric_tload_smoke.cc a3_fabric_window.cc a3_fabric_window.h $(KERNEL_LIB)
	$(MPI_CXX) $(CXXFLAGS) $(CANN_INCLUDES) -o $@ fabric_tload_smoke.cc a3_fabric_window.cc $(FABRIC_LIBS) \
		-L. -lfabric_tload_kernel -Wl,-rpath,'$$ORIGIN'

$(KERNEL_LIB): fabric_tload_kernel.cce
	$(CCEC) $(KERNEL_FLAGS) $(KERNEL_INCLUDES) -fPIC -shared --cce-fatobj-link -o $@ $<

check-common-deps:
	test -n "$(ASCEND_HOME_PATH)" || { echo "ASCEND_HOME_PATH is required"; exit 1; }
	test -n "$(DRIVER_LIB)" || { echo "DRIVER_LIB is required"; exit 1; }
	test -x "$(MPI_CXX)" || command -v "$(MPI_CXX)" >/dev/null 2>&1 || { \
		echo "MPI C++ compiler not found: $(MPI_CXX)"; exit 1; \
	}
	test -d "$(DRIVER_LIB)" || { echo "driver libraries not found: $(DRIVER_LIB)"; exit 1; }

check-hccl-deps: check-common-deps
	test -f "$(ASCEND_HOME_PATH)/include/hccl/hccl.h" || { \
		echo "CANN HCCL headers not found under ASCEND_HOME_PATH=$(ASCEND_HOME_PATH)"; exit 1; \
	}

check-fabric-deps: check-common-deps
	test -x "$(CCEC)" || { echo "bisheng not found: $(CCEC)"; exit 1; }
	test -n "$(PTO_ISA_ROOT)" || { echo "PTO_ISA_ROOT is required"; exit 1; }
	test -n "$(PTO_ENABLE_FLAG)" || { echo "PTO_ENABLE_FLAG is required"; exit 1; }
	test -n "$(CCE_AICORE_ARCH)" || { echo "CCE_AICORE_ARCH is required"; exit 1; }
	test -f "$(ASCEND_HOME_PATH)/include/acl/acl_rt.h" || { \
		echo "CANN Runtime headers not found under ASCEND_HOME_PATH=$(ASCEND_HOME_PATH)"; exit 1; \
	}
	test -f "$(PTO_ISA_ROOT)/include/pto/pto-inst.hpp" || { \
		echo "PTO headers not found under PTO_ISA_ROOT=$(PTO_ISA_ROOT)"; exit 1; \
	}

clean:
	rm -f $(HCCL_TARGET) $(ROOTINFO_TARGET) $(FABRIC_TARGET) $(KERNEL_LIB) \
		shmem_tload_smoke libshmem_tload_kernel.so

run: fabric
	test -n "$(CONFIG)" || { echo "CONFIG is required for two-host runs"; exit 1; }
	bash run_2host_32rank.sh "$(CONFIG)" fabric

run-rootinfo: $(ROOTINFO_TARGET)
	test -n "$(CONFIG)" || { echo "CONFIG is required for two-host runs"; exit 1; }
	bash run_2host_32rank.sh "$(CONFIG)" rootinfo

run-hccl: $(HCCL_TARGET) ranktable_a3_2host_32rank.json
	LD_LIBRARY_PATH=$(ASCEND_HOME_PATH)/lib64:$$LD_LIBRARY_PATH \
	mpirun --hostfile hostfile.mpi -np 32 ./$(HCCL_TARGET) ranktable_a3_2host_32rank.json 16

ranktable_a3_2host_32rank.json:
	python3 gen_ranktable.py -o $@
