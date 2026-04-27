#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  tools/push_changes.sh -m "<commit title>" [options] -- <file1> [file2 ...]

Options:
  -m, --message   Commit title (required)
  -d, --detail    Commit body/detail (optional)
  -r, --remote    Push remote URL (default: git@github.com:sunkaixuan2018/simpler-PTO.git)
  -b, --branch    Remote branch (default: feat/sdma-prefetch-sync-20260422)
  -h, --help      Show this help

Examples:
  tools/push_changes.sh -m "Update benchmark parser" -- tools/benchmark_rounds.sh

  tools/push_changes.sh \
    -m "Refine prefetch logic" \
    -d "Adjust payload metadata and reserve/issue split." \
    -- src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp
USAGE
}

REMOTE_URL="git@github.com:sunkaixuan2018/simpler-PTO.git"
REMOTE_BRANCH="feat/sdma-prefetch-sync-20260422"
COMMIT_TITLE=""
COMMIT_DETAIL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--message)
            COMMIT_TITLE="${2:-}"
            shift 2
            ;;
        -d|--detail)
            COMMIT_DETAIL="${2:-}"
            shift 2
            ;;
        -r|--remote)
            REMOTE_URL="${2:-}"
            shift 2
            ;;
        -b|--branch)
            REMOTE_BRANCH="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

if [[ -z "$COMMIT_TITLE" ]]; then
    echo "ERROR: commit title is required."
    usage
    exit 1
fi

if [[ $# -eq 0 ]]; then
    echo "ERROR: please provide at least one file to upload."
    usage
    exit 1
fi

if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "ERROR: current directory is not a git repo."
    exit 1
fi

for f in "$@"; do
    if [[ ! -e "$f" ]]; then
        echo "ERROR: file not found: $f"
        exit 1
    fi
done

git add -- "$@"

TREE_HASH="$(git write-tree)"
PARENT_HASH="$(git rev-parse HEAD)"

if [[ -n "$COMMIT_DETAIL" ]]; then
    COMMIT_HASH="$(
        git commit-tree "$TREE_HASH" -p "$PARENT_HASH" \
            -m "$COMMIT_TITLE" \
            -m "$COMMIT_DETAIL"
    )"
else
    COMMIT_HASH="$(
        git commit-tree "$TREE_HASH" -p "$PARENT_HASH" \
            -m "$COMMIT_TITLE"
    )"
fi

git update-ref HEAD "$COMMIT_HASH"
git push "$REMOTE_URL" HEAD:"$REMOTE_BRANCH"

echo "Done."
echo "  commit : $COMMIT_HASH"
echo "  remote : $REMOTE_URL"
echo "  branch : $REMOTE_BRANCH"
