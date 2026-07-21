# A3 Two-Host Communication Smoke

This directory contains three validation paths:

1. `hccl_allgather_smoke` is the existing rank-table HCCL baseline.
2. `hccl_rootinfo_smoke` validates HCCL RootInfo bootstrap and AllGather.
3. `fabric_tload_smoke` maps cross-server memory with CANN Fabric APIs and
   validates a PTO `TLOAD/TSTORE` data path.

The Fabric smoke does not depend on ACLSHMEM. It uses public CANN Runtime VMM
APIs to allocate a 2 MiB huge-page HBM window, exchange Fabric handles through
MPI, and map the matching peer rank's HBM into the local device address space.

## Fabric Data Path

```text
MPI starts equal ranks on two hosts
  -> bind each rank to the device matching its local MPI rank
  -> allocate and map one local CANN VMM window
  -> export an ACL_MEM_SHARE_HANDLE_TYPE_FABRIC handle
  -> exchange all 128-byte Fabric handles with MPI_Allgather
  -> import and map the matching rank on the other host
  -> use PTO TLOAD to read the peer GVA
  -> use PTO TSTORE to write a local destination
  -> copy the result to the host and verify the peer-rank pattern
```

The Host-side CANN sequence is:

```text
aclrtMemGetAllocationGranularity
  -> aclrtMallocPhysical
  -> aclrtReserveMemAddress
  -> aclrtMapMem
  -> aclrtMemSetAccess
  -> aclrtMemExportToShareableHandleV2(FABRIC)
  -> MPI handle exchange
  -> aclrtMemImportFromShareableHandleV2(FABRIC)
  -> aclrtReserveMemAddress
  -> aclrtMapMem
  -> aclrtMemSetAccess
```

The device kernel is built as `libfabric_tload_kernel.so`, a fat shared
object containing the Host launch stub and the device kernel. The kernel only
receives ordinary device pointers and does not include SHMEM headers or call
SHMEM functions.

## Files

- `a3_fabric_window.h/.cc`: CANN Fabric VMM window lifecycle.
- `fabric_tload_smoke.cc`: MPI handle exchange, launch, and verification.
- `fabric_tload_kernel.cce`: one 64-float remote load and local store.
- `hccl_rootinfo_smoke.cc`: RootInfo bootstrap and HCCL AllGather.
- `hccl_allgather_smoke.cc`: existing rank-table HCCL baseline.
- `run_2host_32rank.sh`: configuration-driven two-host launcher.
- `run_local.sh`: loads one host's CANN environment and runs one binary.
- `smoke_config.example`: tracked configuration template.
- `smoke_config.conf`: machine-local configuration ignored by Git.

## Configuration

Create the machine-local configuration next to the launcher and replace all
placeholders:

```bash
cd tools/a3_hccl_smoke
cp smoke_config.example smoke_config.conf
```

`smoke_config.conf` is ignored by Git. Keep one private copy for each OpenMPI
or MPICH machine pair, and do not commit real management IPs or
machine-specific paths. The file uses plain `KEY=VALUE` lines. It is parsed as
data and is not sourced as a shell script. Do not add shell commands or quote
the values. Paths must not contain whitespace because GNU Make reads the same
file.

Important fields include:

- `MPI_IMPLEMENTATION`: exactly `openmpi` or `mpich`.
- `MPI_RUN`: full path to the master host's `mpirun`.
- `MASTER_HOST` and `SLAVE_HOST`: management IPs used by MPI over SSH.
- `MASTER_*` and `SLAVE_*`: each host's code, MPI compiler, CANN, PTO ISA,
  driver, and log paths.
- `*_PTO_ENABLE_FLAG`: the PTO flag shown by that host's `bisheng --help`.
- `*_CCE_AICORE_ARCH`: that host's AICore architecture passed to `bisheng`.
- `RANKS_PER_HOST`: normally `16` for the 32-rank A3 smoke.
- `MPI_TIMEOUT_SECONDS`: launcher timeout.
- `OPENMPI_OVERSUBSCRIBE`: set to `1` only when OpenMPI requires
  `--oversubscribe`; the default is `0`.
- `OPENMPI_TCP_IF_INCLUDE`: optional OpenMPI TCP interface passed to both the
  OOB and BTL include parameters.

The launcher verifies that `MPI_RUN --version` matches
`MPI_IMPLEMENTATION`. The same configuration file is accepted by the
Makefile, keeping build and runtime paths in one place.

Configurations created for the older ACLSHMEM smoke must remove
`MASTER_SHMEM_HOME` and `SLAVE_SHMEM_HOME`; those fields are no longer used.

## Dependencies

Both hosts need:

- Atlas A3 hardware and a CANN package exposing the V2 Fabric memory APIs.
- The same MPI implementation and ABI: OpenMPI or MPICH.
- PTO ISA headers.

No ACLSHMEM checkout, headers, or shared libraries are required.

Use the Simpler-pinned PTO ISA revision on both hosts:

| Dependency | Git repository | Version |
| ---------- | -------------- | ------- |
| PTO ISA | [PTO ISA repository] | `pto_isa.pin` |
| CANN Runtime reference | [CANN Runtime repository] | Match installed CANN |

[PTO ISA repository]: https://github.com/hw-native-sys/pto-isa.git
[CANN Runtime repository]: https://gitcode.com/cann/runtime.git

Clone PTO ISA from the Simpler repository root and check out the pinned
revision:

```bash
git clone https://github.com/hw-native-sys/pto-isa.git build/pto-isa
git -C build/pto-isa checkout "$(tr -d '[:space:]' < pto_isa.pin)"
```

The CANN Runtime source is optional and is only a reference for the public
cross-server sample. Building this smoke uses the CANN package installed on
each server.

## Build

Build separately on both hosts because CANN, MPI, driver, and repository paths
may differ.

On the master:

```bash
cd <master-smoke-directory>
make clean
make rootinfo fabric CONFIG=smoke_config.conf ROLE=master
```

On the slave:

```bash
cd <slave-smoke-directory>
make clean
make rootinfo fabric CONFIG=smoke_config.conf ROLE=slave
```

The Fabric outputs are:

```text
fabric_tload_smoke
libfabric_tload_kernel.so
```

Before running, confirm that both hosts have all outputs and that the fat
shared object exports the launch symbol:

```bash
test -x hccl_rootinfo_smoke
test -x fabric_tload_smoke
test -f libfabric_tload_kernel.so
nm -D libfabric_tload_kernel.so | grep LaunchFabricTload
```

The Host executable records `$ORIGIN` for the kernel library, so master and
slave do not need to use the same absolute repository path.

## Run

Run from the master and pass the configuration file explicitly:

```bash
cd <master-smoke-directory>
bash run_2host_32rank.sh smoke_config.conf rootinfo
bash run_2host_32rank.sh smoke_config.conf fabric
```

The launcher generates a temporary hostfile from `MASTER_HOST`, `SLAVE_HOST`,
and `RANKS_PER_HOST`:

- OpenMPI uses two app contexts with explicit per-host slot counts. Optional
  oversubscription and TCP-interface settings come from the configuration.
- MPICH uses `-f`, `-ppn`, and `-np`. Each rank selects the configured master
  or slave paths before invoking `run_local.sh`.

`run_local.sh` sources each host's configured CANN `set_env.sh`, adds the local
smoke and CANN directories to `LD_LIBRARY_PATH`, and creates the configured
device log directory before starting the binary.

## Pass Criteria

RootInfo succeeds when rank 0 prints:

```text
A3 HCCL RootInfo AllGather smoke PASS
```

Fabric succeeds when all ranks print:

```text
cross-host CANN Fabric + PTO TLOAD/TSTORE verify OK
```

and rank 0 prints:

```text
A3 CANN Fabric cross-host PTO TLOAD smoke PASS
```

## Existing Rank-Table Baseline

The old OpenMPI rank-table baseline remains available separately:

```bash
python3 gen_ranktable.py
make baseline CONFIG=smoke_config.conf ROLE=master
make run-hccl CONFIG=smoke_config.conf ROLE=master
```

It is not used by the configuration-driven Fabric or RootInfo launcher.

## Failure Classification

- MPI version mismatch: correct `MPI_IMPLEMENTATION` or `MPI_RUN`.
- MPI does not start: check SSH and the configured management IPs.
- CANN V2 Fabric export returns `207000`: the CANN/driver/hardware combination
  does not expose cross-server Fabric handles.
- Fabric import or mapping fails: verify that both hosts use compatible CANN
  and driver versions and that the A3 Fabric path is available.
- Kernel launch or stream sync fails: inspect the configured device log path.
- Verification fails: inspect peer-rank pairing, mapping size, and memory
  visibility.
