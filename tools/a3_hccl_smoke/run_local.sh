#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Load one host's configured runtime and execute the selected smoke binary.
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <fabric|rootinfo> <cann-root> <log-dir> [program-args...]" >&2
    exit 2
fi

MODE="$1"
CANN_ROOT="$2"
LOG_DIR="$3"
shift 3

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "${CANN_ROOT}/set_env.sh" ]; then
    echo "[run_local] CANN set_env.sh not found: ${CANN_ROOT}/set_env.sh" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"
export ASCEND_PROCESS_LOG_PATH="${LOG_DIR}"

# CANN's set_env.sh may read unset variables. Source it without nounset, then
# restore this wrapper's strict mode.
set +u
# shellcheck disable=SC1090
source "${CANN_ROOT}/set_env.sh"
set -euo pipefail

export LD_LIBRARY_PATH="${SCRIPT_DIR}:${CANN_ROOT}/lib64:${CANN_ROOT}/runtime/lib64:${LD_LIBRARY_PATH:-}"

case "${MODE}" in
    fabric)
        if [ ! -x "${SCRIPT_DIR}/fabric_tload_smoke" ]; then
            echo "[run_local] fabric_tload_smoke is missing under ${SCRIPT_DIR}" >&2
            exit 1
        fi
        exec "${SCRIPT_DIR}/fabric_tload_smoke" "$@"
        ;;
    rootinfo)
        if [ ! -x "${SCRIPT_DIR}/hccl_rootinfo_smoke" ]; then
            echo "[run_local] hccl_rootinfo_smoke is missing under ${SCRIPT_DIR}" >&2
            exit 1
        fi
        exec "${SCRIPT_DIR}/hccl_rootinfo_smoke" "$@"
        ;;
    *)
        echo "[run_local] unknown mode: ${MODE}" >&2
        exit 2
        ;;
esac
