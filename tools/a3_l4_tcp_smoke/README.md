# A3 L4 MPI mailbox smoke

`mpirun_compute_then_tload_2x2_smoke.py` validates the current two-server
topology:

- server 37: L4 plus MPI rank 0/L3, managing two L2 devices;
- server 35: MPI rank 1/L3, managing two L2 devices;
- four global device ranks perform local addition, then peer TLOAD.

The L4 process creates one named shared-memory mailbox. Full-group submissions
use `submit_next_level_group`, so one `PER_RANK` request carries a distinct
payload for each MPI rank. No Simpler command or health TCP endpoint is
created. OpenMPI, SSH, or the MPI implementation may still use TCP internally.

Example from server 37:

```bash
python tools/a3_l4_tcp_smoke/mpirun_compute_then_tload_2x2_smoke.py \
  --host-37 120.9.10.37 \
  --host-35 120.9.10.35 \
  --devices-37 0,1 \
  --devices-35 0,1
```

Success requires all four compute and TLOAD `max_diff` values to stay within
their tolerances. Shutdown also checks that `mpirun`, the named shared-memory
object, and the temporary manifest are gone.
