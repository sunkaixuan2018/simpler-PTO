#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <config-file> [shmem|rootinfo]" >&2
    exit 2
fi

CONFIG_FILE="$1"
MODE="${2:-shmem}"

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "[a3-smoke] config file not found: ${CONFIG_FILE}" >&2
    exit 1
fi

MPI_IMPLEMENTATION=
MPI_RUN=
MPI_TIMEOUT_SECONDS=
RANKS_PER_HOST=
ROOTINFO_COUNT=
OPENMPI_OVERSUBSCRIBE=0
OPENMPI_TCP_IF_INCLUDE=
MASTER_HOST=
MASTER_SCRIPT_DIR=
MASTER_MPI_CXX=
MASTER_ASCEND_HOME_PATH=
MASTER_SHMEM_HOME=
MASTER_PTO_ISA_ROOT=
MASTER_DRIVER_LIB=
MASTER_PTO_ENABLE_FLAG=
MASTER_CCE_AICORE_ARCH=
MASTER_LOG_DIR=
SLAVE_HOST=
SLAVE_SCRIPT_DIR=
SLAVE_MPI_CXX=
SLAVE_ASCEND_HOME_PATH=
SLAVE_SHMEM_HOME=
SLAVE_PTO_ISA_ROOT=
SLAVE_DRIVER_LIB=
SLAVE_PTO_ENABLE_FLAG=
SLAVE_CCE_AICORE_ARCH=
SLAVE_LOG_DIR=

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "${value}"
}

declare -A seen_keys=()
line_number=0
while IFS= read -r raw_line || [ -n "${raw_line}" ]; do
    line_number=$((line_number + 1))
    raw_line="${raw_line%$'\r'}"
    line="$(trim "${raw_line}")"
    if [ -z "${line}" ] || [[ "${line}" == \#* ]]; then
        continue
    fi
    if [[ "${line}" != *=* ]]; then
        echo "[a3-smoke] invalid config line ${line_number}: expected KEY=VALUE" >&2
        exit 1
    fi

    key="$(trim "${line%%=*}")"
    value="$(trim "${line#*=}")"
    case "${key}" in
        MPI_IMPLEMENTATION|MPI_RUN|MPI_TIMEOUT_SECONDS|RANKS_PER_HOST|ROOTINFO_COUNT|\
        OPENMPI_OVERSUBSCRIBE|OPENMPI_TCP_IF_INCLUDE|\
        MASTER_HOST|MASTER_SCRIPT_DIR|MASTER_MPI_CXX|MASTER_ASCEND_HOME_PATH|MASTER_SHMEM_HOME|\
        MASTER_PTO_ISA_ROOT|MASTER_DRIVER_LIB|MASTER_PTO_ENABLE_FLAG|MASTER_CCE_AICORE_ARCH|\
        MASTER_LOG_DIR|SLAVE_HOST|SLAVE_SCRIPT_DIR|SLAVE_MPI_CXX|SLAVE_ASCEND_HOME_PATH|\
        SLAVE_SHMEM_HOME|SLAVE_PTO_ISA_ROOT|SLAVE_DRIVER_LIB|SLAVE_PTO_ENABLE_FLAG|\
        SLAVE_CCE_AICORE_ARCH|SLAVE_LOG_DIR)
            ;;
        *)
            echo "[a3-smoke] unknown config key on line ${line_number}: ${key}" >&2
            exit 1
            ;;
    esac
    if [ -n "${seen_keys[${key}]+x}" ]; then
        echo "[a3-smoke] duplicate config key on line ${line_number}: ${key}" >&2
        exit 1
    fi
    seen_keys["${key}"]=1
    printf -v "${key}" '%s' "${value}"
done < "${CONFIG_FILE}"

require_config() {
    local key="$1"
    local value="${!key}"
    if [ -z "${value}" ]; then
        echo "[a3-smoke] required config value is empty: ${key}" >&2
        exit 1
    fi
    if [[ "${value}" == *'<'* || "${value}" == *'>'* ]]; then
        echo "[a3-smoke] replace the placeholder value for ${key}" >&2
        exit 1
    fi
    if [[ "${value}" =~ [[:space:]] ]]; then
        echo "[a3-smoke] config values may not contain whitespace: ${key}" >&2
        exit 1
    fi
}

for key in MPI_IMPLEMENTATION MPI_RUN MPI_TIMEOUT_SECONDS RANKS_PER_HOST ROOTINFO_COUNT \
    MASTER_HOST MASTER_SCRIPT_DIR MASTER_MPI_CXX MASTER_ASCEND_HOME_PATH MASTER_SHMEM_HOME \
    MASTER_PTO_ISA_ROOT MASTER_DRIVER_LIB MASTER_PTO_ENABLE_FLAG MASTER_CCE_AICORE_ARCH \
    MASTER_LOG_DIR SLAVE_HOST SLAVE_SCRIPT_DIR SLAVE_MPI_CXX SLAVE_ASCEND_HOME_PATH \
    SLAVE_SHMEM_HOME SLAVE_PTO_ISA_ROOT SLAVE_DRIVER_LIB SLAVE_PTO_ENABLE_FLAG \
    SLAVE_CCE_AICORE_ARCH SLAVE_LOG_DIR; do
    require_config "${key}"
done

case "${MODE}" in
    shmem)
        MODE_ARGS=()
        ;;
    rootinfo)
        MODE_ARGS=("${ROOTINFO_COUNT}")
        ;;
    *)
        echo "Usage: $0 <config-file> [shmem|rootinfo]" >&2
        exit 2
        ;;
esac

case "${MPI_IMPLEMENTATION}" in
    openmpi|mpich)
        ;;
    *)
        echo "[a3-smoke] MPI_IMPLEMENTATION must be openmpi or mpich" >&2
        exit 1
        ;;
esac

for numeric_key in MPI_TIMEOUT_SECONDS RANKS_PER_HOST ROOTINFO_COUNT; do
    numeric_value="${!numeric_key}"
    if ! [[ "${numeric_value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[a3-smoke] ${numeric_key} must be a positive integer" >&2
        exit 1
    fi
done

case "${OPENMPI_OVERSUBSCRIBE}" in
    0|1)
        ;;
    *)
        echo "[a3-smoke] OPENMPI_OVERSUBSCRIBE must be 0 or 1" >&2
        exit 1
        ;;
esac

if [[ "${OPENMPI_TCP_IF_INCLUDE}" =~ [[:space:]] ]]; then
    echo "[a3-smoke] OPENMPI_TCP_IF_INCLUDE may not contain whitespace" >&2
    exit 1
fi
if [[ "${OPENMPI_TCP_IF_INCLUDE}" == *'<'* || "${OPENMPI_TCP_IF_INCLUDE}" == *'>'* ]]; then
    echo "[a3-smoke] replace the placeholder value for OPENMPI_TCP_IF_INCLUDE" >&2
    exit 1
fi

if [ "${MASTER_HOST}" = "${SLAVE_HOST}" ]; then
    echo "[a3-smoke] MASTER_HOST and SLAVE_HOST must be different" >&2
    exit 1
fi

if [ ! -x "${MPI_RUN}" ]; then
    echo "[a3-smoke] configured MPI_RUN is not executable: ${MPI_RUN}" >&2
    exit 1
fi
if [ ! -f "${MASTER_SCRIPT_DIR}/run_local.sh" ]; then
    echo "[a3-smoke] master run_local.sh not found under ${MASTER_SCRIPT_DIR}" >&2
    exit 1
fi

MPI_VERSION="$("${MPI_RUN}" --version 2>&1 || true)"
case "${MPI_IMPLEMENTATION}" in
    openmpi)
        if ! grep -Eqi 'Open MPI|OpenRTE' <<< "${MPI_VERSION}"; then
            echo "[a3-smoke] MPI_RUN does not look like OpenMPI: ${MPI_RUN}" >&2
            exit 1
        fi
        ;;
    mpich)
        if ! grep -Eqi 'HYDRA|MPICH' <<< "${MPI_VERSION}"; then
            echo "[a3-smoke] MPI_RUN does not look like MPICH/Hydra: ${MPI_RUN}" >&2
            exit 1
        fi
        ;;
esac

TOTAL_RANKS=$((RANKS_PER_HOST * 2))
HOSTFILE="$(mktemp "${TMPDIR:-/tmp}/a3-smoke-hostfile.XXXXXX")"
trap 'rm -f "${HOSTFILE}"' EXIT

if [ "${MPI_IMPLEMENTATION}" = "openmpi" ]; then
    printf '%s slots=%s\n%s slots=%s\n' \
        "${MASTER_HOST}" "${RANKS_PER_HOST}" "${SLAVE_HOST}" "${RANKS_PER_HOST}" > "${HOSTFILE}"
else
    printf '%s\n%s\n' "${MASTER_HOST}" "${SLAVE_HOST}" > "${HOSTFILE}"
fi

echo "[a3-smoke] mpi=${MPI_IMPLEMENTATION} launcher=${MPI_RUN}"
echo "[a3-smoke] master=${MASTER_HOST} script=${MASTER_SCRIPT_DIR}"
echo "[a3-smoke] slave=${SLAVE_HOST} script=${SLAVE_SCRIPT_DIR}"
echo "[a3-smoke] mode=${MODE} ranks=${RANKS_PER_HOST}+${RANKS_PER_HOST}"

if [ "${MPI_IMPLEMENTATION}" = "openmpi" ]; then
    MPI_ROOT_ARGS=()
    if [ "$(id -u)" -eq 0 ] && "${MPI_RUN}" --allow-run-as-root --version >/dev/null 2>&1; then
        MPI_ROOT_ARGS=(--allow-run-as-root)
    fi
    MPI_TAG_ARGS=()
    if "${MPI_RUN}" --tag-output --version >/dev/null 2>&1; then
        MPI_TAG_ARGS=(--tag-output)
    fi
    MPI_OVERSUBSCRIBE_ARGS=()
    if [ "${OPENMPI_OVERSUBSCRIBE}" = "1" ]; then
        MPI_OVERSUBSCRIBE_ARGS=(--oversubscribe)
    fi
    MPI_TCP_IF_ARGS=()
    if [ -n "${OPENMPI_TCP_IF_INCLUDE}" ]; then
        MPI_TCP_IF_ARGS=(
            --mca oob_tcp_if_include "${OPENMPI_TCP_IF_INCLUDE}"
            --mca btl_tcp_if_include "${OPENMPI_TCP_IF_INCLUDE}"
        )
    fi

    timeout "${MPI_TIMEOUT_SECONDS}" "${MPI_RUN}" \
        "${MPI_ROOT_ARGS[@]}" "${MPI_TAG_ARGS[@]}" "${MPI_OVERSUBSCRIBE_ARGS[@]}" \
        "${MPI_TCP_IF_ARGS[@]}" --hostfile "${HOSTFILE}" \
        -np "${RANKS_PER_HOST}" --host "${MASTER_HOST}:${RANKS_PER_HOST}" \
        --wdir "${MASTER_SCRIPT_DIR}" \
        bash "${MASTER_SCRIPT_DIR}/run_local.sh" "${MODE}" "${MASTER_ASCEND_HOME_PATH}" \
        "${MASTER_SHMEM_HOME}" "${MASTER_LOG_DIR}" "${MODE_ARGS[@]}" \
        : \
        -np "${RANKS_PER_HOST}" --host "${SLAVE_HOST}:${RANKS_PER_HOST}" \
        --wdir "${SLAVE_SCRIPT_DIR}" \
        bash "${SLAVE_SCRIPT_DIR}/run_local.sh" "${MODE}" "${SLAVE_ASCEND_HOME_PATH}" \
        "${SLAVE_SHMEM_HOME}" "${SLAVE_LOG_DIR}" "${MODE_ARGS[@]}"
else
    MPICH_DISPATCH='rank="${PMI_RANK:-${PMIX_RANK:-}}"
if ! [[ "${rank}" =~ ^[0-9]+$ ]]; then
    echo "[a3-smoke] MPICH rank environment is unavailable" >&2
    exit 1
fi
ranks_per_host="$1"
master_dir="$2"
slave_dir="$3"
mode="$4"
master_cann="$5"
slave_cann="$6"
master_shmem="$7"
slave_shmem="$8"
master_log="$9"
slave_log="${10}"
shift 10
if [ "${rank}" -lt "${ranks_per_host}" ]; then
    exec bash "${master_dir}/run_local.sh" "${mode}" "${master_cann}" "${master_shmem}" "${master_log}" "$@"
fi
exec bash "${slave_dir}/run_local.sh" "${mode}" "${slave_cann}" "${slave_shmem}" "${slave_log}" "$@"'

    timeout "${MPI_TIMEOUT_SECONDS}" "${MPI_RUN}" -f "${HOSTFILE}" \
        -ppn "${RANKS_PER_HOST}" -np "${TOTAL_RANKS}" \
        bash -c "${MPICH_DISPATCH}" _ "${RANKS_PER_HOST}" \
        "${MASTER_SCRIPT_DIR}" "${SLAVE_SCRIPT_DIR}" "${MODE}" \
        "${MASTER_ASCEND_HOME_PATH}" "${SLAVE_ASCEND_HOME_PATH}" \
        "${MASTER_SHMEM_HOME}" "${SLAVE_SHMEM_HOME}" \
        "${MASTER_LOG_DIR}" "${SLAVE_LOG_DIR}" "${MODE_ARGS[@]}"
fi
