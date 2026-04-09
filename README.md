# PTO Runtime - Task Runtime Execution Framework

Modular runtime for building and executing task dependency graphs on Ascend devices with coordinated AICPU and AICore execution. Three independently compiled programs (Host `.so`, AICPU `.so`, AICore `.o`) work together through clearly defined APIs.

## Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd simpler

# Run the vector example (simulation, no hardware required)
python examples/scripts/run_example.py \
  -k examples/a2a3/host_build_graph/vector_example/kernels \
  -g examples/a2a3/host_build_graph/vector_example/golden.py \
  -p a2a3sim
```

PTO ISA headers are automatically cloned on first run. See [Getting Started](docs/getting-started.md) for manual setup and troubleshooting.

## Platforms

| Platform | Description | Requirements |
|----------|-------------|--------------|
| `a2a3` | Real Ascend hardware | CANN toolkit (ccec, aarch64 cross-compiler) |
| `a2a3sim` | Thread-based host simulation | gcc/g++ only (no Ascend SDK needed) |

## Runtime Variants

Three runtimes under `src/{arch}/runtime/`, each with a different graph-building strategy:

| Runtime | Graph built on | Use case |
|---------|---------------|----------|
| `host_build_graph` | Host CPU | Development, debugging |
| `aicpu_build_graph` | AICPU (device) | Reduced host-device transfer |
| `tensormap_and_ringbuffer` | AICPU (device) | Production workloads |

See runtime docs per arch: [a2a3](src/a2a3/docs/runtimes.md), [a5](src/a5/docs/runtimes.md).

## Testing

```bash
# Simulation tests (no hardware)
./ci.sh -p a2a3sim

# Hardware tests (requires Ascend device)
./ci.sh -p a2a3 -d 4-7 --parallel

# Python unit tests
pytest tests -m "not requires_hardware" -v

# C++ unit tests
cmake -B tests/cpp/build -S tests/cpp && cmake --build tests/cpp/build && ctest --test-dir tests/cpp/build --output-on-failure
```

See [Testing Guide](docs/testing.md) for details.

## Environment Setup

```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
```

## SDMA Prefetch (Optional, a2a3 Hardware)

The a2a3 onboard runtime supports an experimental "SDMA prefetch" path (AICPU issues STARS SDMA CMO PREFETCH SQEs)
to warm AICore L2 before compute. This requires an additional minimal provider bundle containing:

- `x86_64-linux/lib64/libopapi.so` (exports `aclnnShmemSdmaStarsQuery*`)
- `opp/` (OPP metadata used by the provider)

### Install Provider Bundle

If you received a `sdma_legacy_provider.tar.gz`, unpack it into the repo root:

```bash
mkdir -p _deps
tar -C _deps -xzf /path/to/sdma_legacy_provider.tar.gz
```

After unpacking, the layout should be:

```text
_deps/sdma_legacy_provider/x86_64-linux/lib64/libopapi.so
_deps/sdma_legacy_provider/opp/...
```

Alternatively, point to an existing provider directory:

```bash
export PTO_SDMA_PROVIDER_ROOT=/abs/path/to/sdma_legacy_provider
export ASCEND_OPP_PATH=/abs/path/to/sdma_legacy_provider/opp
```

### Run Tests (Different Prefetch Modes)

All modes are controlled by `PTO_SDMA_PREFETCH_MODE`:

- `baseline`: disable SDMA prefetch (no SDMA channel setup)
- `twoslot`: disable SDMA prefetch (two-slot scheduler mode)
- `sdma`: enable SDMA prefetch (requires provider bundle above)
- `sdma_fake`: disables real SDMA, keeps the control-path for A/B (no provider required)

Example: run a small workload on real hardware (skip golden for benchmarking):

```bash
# Baseline
PTO_SDMA_PREFETCH_MODE=baseline \
python examples/scripts/run_example.py \
  -k examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels \
  -g examples/a2a3/tensormap_and_ringbuffer/vector_example/golden.py \
  -p a2a3 -d 0 --skip-golden

# SDMA prefetch
PTO_SDMA_PREFETCH_MODE=sdma \
python examples/scripts/run_example.py \
  -k examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels \
  -g examples/a2a3/tensormap_and_ringbuffer/vector_example/golden.py \
  -p a2a3 -d 0 --skip-golden
```

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | Three-program model, API layers, execution flow, handshake protocol |
| [Getting Started](docs/getting-started.md) | Setup, prerequisites, build process, configuration |
| [Developer Guide](docs/developer-guide.md) | Directory structure, role ownership, conventions |
| [Testing Guide](docs/testing.md) | CI pipeline, test types, writing new tests |
| **Per-arch (a2a3):** | |
| [Runtimes](src/a2a3/docs/runtimes.md) | Runtime comparison and links to per-runtime docs |
| [Platform](src/a2a3/docs/platform.md) | onboard vs sim, hardware requirements |
| **Per-arch (a5):** | |
| [Runtimes](src/a5/docs/runtimes.md) | Runtime comparison and links to per-runtime docs |
| [Platform](src/a5/docs/platform.md) | onboard vs sim, hardware requirements |

## License

This project is licensed under the **CANN Open Software License Agreement Version 2.0**. See the [LICENSE](LICENSE) file for the full license text.

## References

- [src/a2a3/platform/](src/a2a3/platform/) - Platform implementations
- [src/a2a3/runtime/](src/a2a3/runtime/) - Runtime implementations
- [examples/a2a3/](examples/a2a3/) - Examples organized by runtime
- [examples/scripts/](examples/scripts/) - Test framework
- [python/](python/) - Python bindings and compiler
