#!/bin/bash
# Apply Stage 2 patches to pto-comm-isa (SdmaWaitEventCounted + AsyncEvent::WaitCounted)
#
# Usage: bash patches/apply_pto_comm_isa_patch.sh
#
# This copies 3 patched header files into the pto-comm-isa dependency:
#   1. sdma_async_intrin.hpp  - adds SdmaWaitEventCounted() (returns poll count)
#   2. comm_types.hpp         - adds WaitCounted() declaration to AsyncEvent
#   3. async_event_impl.hpp   - adds AsyncEvent::WaitCounted() implementation

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TARGET="$REPO_ROOT/examples/scripts/_deps/pto-comm-isa/include"
PATCH="$SCRIPT_DIR/pto-comm-isa/include"

echo "Applying pto-comm-isa patches..."

cp "$PATCH/pto/npu/comm/async/sdma/sdma_async_intrin.hpp" \
   "$TARGET/pto/npu/comm/async/sdma/sdma_async_intrin.hpp"
echo "  [ok] sdma_async_intrin.hpp"

cp "$PATCH/pto/comm/comm_types.hpp" \
   "$TARGET/pto/comm/comm_types.hpp"
echo "  [ok] comm_types.hpp"

cp "$PATCH/pto/comm/async/async_event_impl.hpp" \
   "$TARGET/pto/comm/async/async_event_impl.hpp"
echo "  [ok] async_event_impl.hpp"

echo "Done. Patches applied to: $TARGET"
