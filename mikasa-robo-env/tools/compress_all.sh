#!/bin/bash
# Compress all task zarr directories to .tar.lz4 in parallel.
# Usage: bash tools/compress_all.sh /path/to/zarr_dir
set -e

DIR="${1:?Usage: compress_all.sh <zarr_root_dir>}"

pids=()
for task_dir in "$DIR"/*/; do
    [ -d "$task_dir" ] || continue
    task_name=$(basename "$task_dir")
    echo "Compressing: $task_name"
    (cd "$DIR" && tar cf - "$task_name" | lz4 -c > "$DIR/${task_name}.tar.lz4") &
    pids+=($!)
done

echo "Launched ${#pids[@]} compression jobs, waiting..."
for pid in "${pids[@]}"; do
    wait "$pid"
done
echo "All done."
