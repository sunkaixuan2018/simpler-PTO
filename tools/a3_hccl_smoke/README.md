# A3 Two-Host Communication Smoke

This directory contains three validation paths:

1. `hccl_allgather_smoke` is the existing rank-table HCCL baseline.
2. `hccl_rootinfo_smoke` validates HCCL RootInfo bootstrap and AllGather.
3. `shmem_tload_smoke` validates ACLSHMEM bootstrap, symmetric memory, and
   a PTO `TLOAD/TSTORE` data path.

The SHMEM smoke does not change the Simpler production communication layer.
It verifies that the remote GVA returned by `aclshmem_ptr` can be consumed by
the same PTO load/store path used by L3.

## SHMEM Data Path

```text
MPI starts equal ranks on two hosts
  -> bind each rank to the device matching its local MPI rank
  -> rank 0 creates an ACLSHMEM unique ID
  -> broadcast the unique ID with MPI
  -> initialize ACLSHMEM and allocate a symmetric window
  -> resolve the matching rank's remote GVA with aclshmem_ptr
  -> use PTO TLOAD to read the remote GVA
  -> use PTO TSTORE to write a local destination
  -> copy the result to the host and verify the peer-rank pattern
```

The device kernel is built as `libshmem_tload_kernel.so`, a fat shared object
containing both the host launch stub and the device kernel. The host calls
`LaunchShmemTload` directly; it does not register a raw AICore object through
`rtRegisterAllKernel`.

## Files

- `shmem_tload_smoke.cc`: ACLSHMEM bootstrap, launch, and verification.
- `shmem_tload_kernel.cce`: one 64-float remote load and local store.
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
machine-specific paths. When updating the repository, preserve the local file
and compare it with `smoke_config.example` for newly added fields.

The file uses plain `KEY=VALUE` lines. It is parsed as data and is not sourced
as a shell script. Do not add shell commands or quote the values. Paths must
not contain whitespace because GNU Make reads the same file.

Important fields include:

- `MPI_IMPLEMENTATION`: exactly `openmpi` or `mpich`.
- `MPI_RUN`: full path to the master host's `mpirun`.
- `MASTER_HOST` and `SLAVE_HOST`: management IPs used by MPI over SSH.
- `MASTER_*` and `SLAVE_*`: each host's code, MPI compiler, CANN, ACLSHMEM,
  PTO ISA, driver, and log paths.
- `*_PTO_ENABLE_FLAG`: the PTO flag shown by that host's `bisheng --help`,
  such as the flag used by its installed CANN version.
- `*_CCE_AICORE_ARCH`: that host's AICore architecture passed to `bisheng`.
- `RANKS_PER_HOST`: normally `16` for the 32-rank A3 smoke.
- `MPI_TIMEOUT_SECONDS`: launcher timeout.
- `OPENMPI_OVERSUBSCRIBE`: set to `1` only when the OpenMPI environment
  requires `--oversubscribe`; the default is `0`.
- `OPENMPI_TCP_IF_INCLUDE`: optional OpenMPI TCP interface passed to both the
  OOB and BTL include parameters. Leave it empty unless OpenMPI must be pinned
  to a specific communication interface.

The launcher checks that `MPI_RUN --version` matches
`MPI_IMPLEMENTATION`. This prevents an OpenMPI configuration from silently
calling an MPICH installation, or the reverse.

The same configuration file is also accepted by the Makefile. This keeps the
build and runtime paths in one place.

## Dependencies

Both hosts need:

- CANN with `bisheng`.
- The same MPI implementation and ABI: OpenMPI or MPICH.
- PTO ISA headers.
- A built ACLSHMEM installation.

Use these source revisions on both hosts:

| Dependency | Git repository | Version |
| ---------- | -------------- | ------- |
| PTO ISA | <https://github.com/hw-native-sys/pto-isa.git> | `pto_isa.pin` (`83d01313d9bfc247c4b7c8bcf969d1019f0d106f`) |
| ACLSHMEM | <https://gitcode.com/cann/shmem.git> | `v1.3.0` |

Clone PTO ISA from the Simpler repository root and check out the pinned
revision:

```bash
git clone https://github.com/hw-native-sys/pto-isa.git build/pto-isa
git -C build/pto-isa checkout "$(tr -d '[:space:]' < pto_isa.pin)"
```

Clone and build ACLSHMEM separately on both hosts:

```bash
git clone --branch v1.3.0 --depth 1 https://gitcode.com/cann/shmem.git
cd shmem
bash scripts/build.sh
```

The default ACLSHMEM source build installs under `install/shmem`. Set each
host's `*_SHMEM_HOME` to that directory and set `*_PTO_ISA_ROOT` to the PTO ISA
checkout. Keep the actual installation paths in `smoke_config.conf`.

## Build

Build separately on both hosts because CANN, MPI, driver, and repository paths
may differ.

On the master:

```bash
cd <master-smoke-directory>
make clean
make rootinfo shmem CONFIG=smoke_config.conf ROLE=master
```

On the slave:

```bash
cd <slave-smoke-directory>
make clean
make rootinfo shmem CONFIG=smoke_config.conf ROLE=slave
```

The SHMEM outputs are:

```text
shmem_tload_smoke
libshmem_tload_kernel.so
```

Before running, confirm that both hosts have all three outputs and that the
fat shared object exports the launch symbol:

```bash
test -x hccl_rootinfo_smoke
test -x shmem_tload_smoke
test -f libshmem_tload_kernel.so
nm -D libshmem_tload_kernel.so | grep LaunchShmemTload
```

The host executable records `$ORIGIN` for the kernel library, so master and
slave do not need to use the same absolute repository path.

## Run

Run from the master and pass the configuration file explicitly:

```bash
cd <master-smoke-directory>
bash run_2host_32rank.sh smoke_config.conf rootinfo
bash run_2host_32rank.sh smoke_config.conf shmem
```

The launcher does not read the legacy tracked `hostfile.mpi`. It generates a
temporary hostfile from `MASTER_HOST`, `SLAVE_HOST`, and `RANKS_PER_HOST`:

- OpenMPI gets `host slots=N` entries and uses two app contexts with explicit
  per-host slot counts so each host can use different code and CANN paths.
  Optional oversubscription and TCP-interface settings come from the
  configuration file.
- MPICH gets a pure host list and uses `-f`, `-ppn`, and `-np`. The MPICH rank
  selects the configured master or slave paths before invoking `run_local.sh`.

`run_local.sh` sources each host's configured CANN `set_env.sh`, adds the local
ACLSHMEM directory to `LD_LIBRARY_PATH`, and creates the configured device log
directory before starting the smoke binary.

## Pass Criteria

RootInfo succeeds when rank 0 prints:

```text
A3 HCCL RootInfo AllGather smoke PASS
```

SHMEM succeeds when all ranks print:

```text
cross-host aclshmem_ptr + PTO TLOAD/TSTORE verify OK
```

and rank 0 prints:

```text
A3 ACLSHMEM cross-host PTO TLOAD smoke PASS
```

## Existing Rank-Table Baseline

The old OpenMPI rank-table baseline and its environment-specific files remain
available separately. It is not the dual-MPI launcher described above:

```bash
python3 gen_ranktable.py
make baseline CONFIG=smoke_config.conf ROLE=master
make run-hccl CONFIG=smoke_config.conf ROLE=master
```

They are not used by the configuration-driven SHMEM or RootInfo launcher.

## Failure Classification

- MPI version mismatch: correct `MPI_IMPLEMENTATION` or `MPI_RUN`.
- MPI does not start: check SSH and the two configured management IPs.
- `run_local.sh` is missing: synchronize the smoke directory to that host.
- CANN loading fails: check that host's `*_ASCEND_HOME_PATH`.
- `aclshmemx_init_attr` fails: check ACLSHMEM build and heap mapping.
- `aclshmem_ptr` returns null: the remote symmetric window was not mapped.
- Kernel launch or stream sync fails: inspect the configured device log path.
- Verification fails: inspect address offsets and memory visibility.
