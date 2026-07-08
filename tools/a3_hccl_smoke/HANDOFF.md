# A3 HCCL Two-Host Validation Runbook for Master `.37`

## Scope and Fixed Topology

This task validates a 32-rank HCCL AllGather across two A3 servers. The smoke
test is independent of Simpler. Validate each boundary in this order:

```text
SSH -> OpenMPI launch -> ACL device setup -> HCCL communicator setup
    -> HcclAllGather -> payload verification
```

Run this document directly from a shell on master `.37`. Commands run locally
on `.37` unless they explicitly start with `ssh 120.9.10.35`. Do not run this
procedure from the external Windows workspace.

**Master `.37`**

- Management IP: `120.9.10.37`
- Current hostname: `localhost.localdomain`
- Required hostname: `master37`
- User: `s00868307`
- Global ranks: `0..15`
- `ASCEND_HOME_PATH`:
  `/home/s00868307/Ascend/ascend-toolkit/latest`

**Slave `.35`**

- Management IP: `120.9.10.35`
- Hostname: `slave35`
- User: `s00868307`
- Global ranks: `16..31`
- `ASCEND_HOME_PATH`:
  `/usr/local/Ascend/ascend-toolkit/latest`

Each host runs 16 local ranks. Local ranks `0..15` map to devices `0..15`.
Both management addresses use interface `enp196s0f0`:

```text
master: 120.9.10.37/8
slave:  120.9.10.35/8
```

Keep the agent session on master `.37` for the entire procedure. The smoke
directory must have the same absolute path on both hosts.

## Current State

Confirmed facts:

- Passwordless SSH works from `.37` to both `.37` and `.35`.
- Both routes use `enp196s0f0` with the expected source IP.
- OpenMPI is 4.1.5 and `mpirun` is `/usr/bin/mpirun`.
- The code consistently reads `ASCEND_HOME_PATH` and falls back to
  `/usr/local/Ascend/ascend-toolkit/latest` when it is unset.

The first failing boundary is OpenMPI daemon startup:

```text
A process or daemon was unable to complete a TCP connection
Local host:  slave35
Remote host: localhost
ORTE was unable to reliably start one or more daemons
```

Master currently reports:

```text
hostname    -> localhost.localdomain
hostname -f -> localhost
getent ahostsv4 localhost.localdomain -> 127.0.0.1
```

The failure remains when OpenMPI is given these subnet filters:

```bash
--mca oob_tcp_if_include 120.9.10.0/24
--mca btl_tcp_if_include 120.9.10.0/24
```

No rank has started, so ACL and HCCL have not been reached. Do not change the
rank table or diagnose HCCL until the two-rank MPI test passes.

## Validation Procedure

### 1. Confirm `.37` and create a run log directory

```bash
hostname
hostname -I
ip route get 120.9.10.35

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    echo "Start the agent inside the simpler repository"
    exit 1
}
cd "$REPO_ROOT/tools/a3_hccl_smoke"
SMOKE_DIR="$PWD"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="$PWD/handoff-logs/$RUN_ID"
mkdir -p "$LOG_DIR"

test -f ./hostfile.mpi
test -f ./ranktable_a3_2host_32rank.json
ssh 120.9.10.35 "test -d '$SMOKE_DIR'"

{
    date
    hostname
    hostname -f
    hostname -I
    id
    command -v mpirun
    mpirun --version | head -n 1
    ip -o -4 addr show
    ip route get 120.9.10.35
} 2>&1 | tee "$LOG_DIR/master-baseline.log"

ssh 120.9.10.35 '
    date
    hostname
    hostname -f
    hostname -I
    id
    command -v mpirun
    mpirun --version | head -n 1
    ip -o -4 addr show
    ip route get 120.9.10.37
' 2>&1 | tee "$LOG_DIR/slave-baseline.log"
```

Keep all following logs under this `LOG_DIR`.

### 2. Validate CANN independently on each host

The CANN roots are different. Check each one explicitly:

```bash
export ASCEND_HOME_PATH=/home/s00868307/Ascend/ascend-toolkit/latest
echo "host=$(hostname) ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
test -f "$ASCEND_HOME_PATH/include/hccl/hccl.h"
test -f "$ASCEND_HOME_PATH/lib64/libhccl.so"
ls -l "$ASCEND_HOME_PATH/include/hccl/hccl.h"
readlink -f "$ASCEND_HOME_PATH/lib64/libhccl.so"

ssh 120.9.10.35 '
    export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
    echo "host=$(hostname) ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
    test -f "$ASCEND_HOME_PATH/include/hccl/hccl.h"
    test -f "$ASCEND_HOME_PATH/lib64/libhccl.so"
    ls -l "$ASCEND_HOME_PATH/include/hccl/hccl.h"
    readlink -f "$ASCEND_HOME_PATH/lib64/libhccl.so"
'
```

Stop if any `test` command fails.

Master `~/.bashrc` should export:

```bash
export ASCEND_HOME_PATH=/home/s00868307/Ascend/ascend-toolkit/latest
```

Slave may use the code default, but an explicit setting is clearer:

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
```

OpenMPI remote processes do not reliably read an interactive `.bashrc`.
The final launch must therefore provide the correct library path separately
for each host.

### 3. Verify identical inputs

```bash
{
    echo "===== master 120.9.10.37 ====="
    sha256sum \
      hccl_allgather_smoke.cc \
      ranktable_a3_2host_32rank.json \
      hostfile.mpi

    echo "===== slave 120.9.10.35 ====="
    ssh 120.9.10.35 "
        cd '$SMOKE_DIR' &&
        sha256sum \
          hccl_allgather_smoke.cc \
          ranktable_a3_2host_32rank.json \
          hostfile.mpi
    "
} | tee "$LOG_DIR/input-checksums.log"
```

Each file must have the same SHA-256 value on both hosts.

### 4. Repair the MPI return path

#### 4.1 Correct the master hostname

`localhost.localdomain` resolves to `127.0.0.1` and is not valid as the
identity of a multi-host MPI node. Run this locally if the account has sudo;
otherwise ask the server administrator to apply it:

```bash
sudo hostnamectl set-hostname master37
```

Both hosts should contain these entries in `/etc/hosts`:

```text
127.0.0.1 localhost localhost.localdomain
120.9.10.37 master37
120.9.10.35 slave35
```

Do not map `master37` to `127.0.0.1`. Continue in the current `.37` agent
session and verify from both hosts:

```bash
hostname
hostname -f
getent ahostsv4 master37
ssh 120.9.10.35 'getent ahostsv4 master37'
```

Both lookups must resolve `master37` to `120.9.10.37`.
If the system requires a new login before the hostname changes, reconnect the
agent and repeat section 1 to restore `SMOKE_DIR` and `LOG_DIR`.

#### 4.2 Test reverse TCP connectivity

Open a second `.37` shell. In the first `.37` shell, start a temporary server:

```bash
python3 -m http.server 25000 --bind 120.9.10.37
```

Keep it running. In the second `.37` shell, run:

```bash
ssh 120.9.10.35 \
    'timeout 5 bash -c "</dev/tcp/120.9.10.37/25000"' \
    && echo TCP_CONNECT_OK \
    || echo TCP_CONNECT_FAILED
```

`TCP_CONNECT_FAILED` means firewall or security policy is blocking the return
path from `.35` to `.37`. Stop the temporary server with `Ctrl+C`.

#### 4.3 Run a two-rank MPI-only test

```bash
timeout 30 mpirun --tag-output \
    --hostfile hostfile.mpi \
    --map-by ppr:1:node \
    -np 2 \
    --wdir "$PWD" \
    --mca oob_tcp_if_include enp196s0f0 \
    --mca btl_tcp_if_include enp196s0f0 \
    /bin/hostname 2>&1 | tee "$LOG_DIR/mpi-hostname.log"
```

Pass criteria: output contains both `master37` and `slave35`, and the command
returns zero.

On failure, collect the actual OpenMPI endpoint selection:

```bash
timeout 30 mpirun --tag-output \
    --hostfile hostfile.mpi \
    --map-by ppr:1:node \
    -np 2 \
    --wdir "$PWD" \
    --mca oob_tcp_if_include enp196s0f0 \
    --mca btl_tcp_if_include enp196s0f0 \
    --mca oob_base_verbose 100 \
    --mca plm_base_verbose 100 \
    /bin/hostname 2>&1 | tee "$LOG_DIR/mpi-oob.log"
```

Use this log to identify the selected IP and port before blaming libraries or
HCCL. The standard ORTE error text lists several unrelated possibilities.

### 5. Build separately on both hosts

Build locally on `.37` with its CANN path:

```bash
export ASCEND_HOME_PATH=/home/s00868307/Ascend/ascend-toolkit/latest
make clean
make -Bn 2>&1 | tee "$LOG_DIR/master-make-dry-run.log"
make 2>&1 | tee "$LOG_DIR/master-make.log"
```

The dry run must contain:

```text
-I/home/s00868307/Ascend/ascend-toolkit/latest/include
-L/home/s00868307/Ascend/ascend-toolkit/latest/lib64
```

Build on slave with its local path:

```bash
ssh 120.9.10.35 "
    cd '$SMOKE_DIR' &&
    export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest &&
    make clean &&
    make -Bn &&
    make
" 2>&1 | tee "$LOG_DIR/slave-make.log"
```

The slave build must contain:

```text
-I/usr/local/Ascend/ascend-toolkit/latest/include
-L/usr/local/Ascend/ascend-toolkit/latest/lib64
```

Check the local binary on `.37`, then the remote binary on `.35`:

```bash
LD_LIBRARY_PATH=/home/s00868307/Ascend/ascend-toolkit/latest/lib64:\
${LD_LIBRARY_PATH:-} ldd ./hccl_allgather_smoke

ssh 120.9.10.35 "
    cd '$SMOKE_DIR' &&
    LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:\
\${LD_LIBRARY_PATH:-} ldd ./hccl_allgather_smoke
"
```

Neither output may contain `not found`.

### 6. Validate devices and the rank table

Capture device health and SDIDs from `.37` and `.35`:

```bash
{
    echo "===== master 120.9.10.37 ====="
    npu-smi info
    npu-smi info -t spod-info

    echo "===== slave 120.9.10.35 ====="
    ssh 120.9.10.35 'npu-smi info; npu-smi info -t spod-info'
} 2>&1 | tee "$LOG_DIR/device-info.log"
```

Capture all device IPs:

```bash
{
    echo "===== master 120.9.10.37 ====="
    for i in $(seq 0 15); do
        echo "device=$i"
        hccn_tool -i "$i" -ip -g
    done

    echo "===== slave 120.9.10.35 ====="
    ssh 120.9.10.35 \
      'for i in $(seq 0 15); do
           echo "device=$i"
           hccn_tool -i "$i" -ip -g
       done'
} 2>&1 | tee "$LOG_DIR/device-ips.log"
```

Confirm all of the following:

- Devices `0..15` exist and are healthy on each host.
- Every rank-table device IP matches `hccn_tool -i <id> -ip -g`.
- Every `super_device_id` matches the physical SDID reported by
  `npu-smi info -t spod-info`.
- Both hosts use an identical `ranktable_a3_2host_32rank.json`.

The current generator writes the global rank as `super_device_id`. If physical
SDIDs differ, fix the generator and regenerate the JSON. Do not hand-edit only
one host's copy.

### 7. Run the 32-rank HCCL smoke

The two CANN roots differ. Do not export the master's `LD_LIBRARY_PATH` to all
ranks. Use two OpenMPI application contexts with host-specific environments:

```bash
timeout 300 mpirun --tag-output \
    --hostfile hostfile.mpi \
    --mca oob_tcp_if_include enp196s0f0 \
    --mca btl_tcp_if_include enp196s0f0 \
    -np 16 --host 120.9.10.37 --wdir "$PWD" \
    -x ASCEND_HOME_PATH=/home/s00868307/Ascend/ascend-toolkit/latest \
    -x LD_LIBRARY_PATH=/home/s00868307/Ascend/ascend-toolkit/latest/lib64 \
    ./hccl_allgather_smoke ranktable_a3_2host_32rank.json 16 \
    : \
    -np 16 --host 120.9.10.35 --wdir "$PWD" \
    -x ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest \
    -x LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64 \
    ./hccl_allgather_smoke ranktable_a3_2host_32rank.json 16 \
    2>&1 | tee "$LOG_DIR/hccl-32rank.log"
```

If `set_env.sh` adds other required library paths, capture the complete
`LD_LIBRARY_PATH` after sourcing each host's own `set_env.sh`, then pass the
two complete values to their respective application contexts.

The success marker is:

```text
A3 HCCL AllGather smoke PASS
```

All 32 ranks must also print:

```text
AllGather verify OK
MPI_Barrier OK
```

## Failure Classification

- No `[pre-mpi] MPI_Init begin`: ranks did not start. Check SSH, hostnames,
  ORTE return connectivity, and firewalls.
- Last marker is `MPI_Init begin`: MPI rendezvous failed. Check interface
  selection and bidirectional TCP.
- Last marker is `aclInit begin`: ACL setup failed. Check the local CANN root
  and dynamic libraries.
- Last marker is `aclrtSetDevice begin`: device access failed. Check health,
  permissions, and device ownership.
- Last marker is `HcclCommInitClusterInfo begin`: communicator setup failed.
  Check rank-table fields, SDIDs, HCCS IPs, and ports.
- Last marker is `aclrtSynchronizeStream begin`: the collective did not
  complete. Check HCCL/device logs and the cross-host device link.

Change one layer at a time. Do not modify MPI, rank-table, and HCCL settings in
the same experiment.

## Evidence to Keep and Report

Keep the following files and include the relevant output in the final report:

- Both baseline logs and hostname lookup results.
- Reverse TCP result.
- `mpi-hostname.log`, plus `mpi-oob.log` when MPI fails.
- Actual `ASCEND_HOME_PATH`, CANN version, and build output for both hosts.
- Input SHA-256 values from both hosts.
- `npu-smi info`, SPOD/SDID, and device-IP evidence from both hosts.
- Complete `hccl-32rank.log`, not only its final line.
- A clear statement of the first failing step, its evidence, and whether the
  next boundary was reached.

Do not change production code or claim HCCL success without this evidence.
