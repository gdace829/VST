#!/usr/bin/env bash

# ---------------------------------------------------------------------------
# Script: run.sh
# Description: Pipeline script to execute a predefined list of benchmarks.
# ---------------------------------------------------------------------------

# Exit immediately if a command exits with a non-zero status,
# treat unset variables as an error, and fail on pipeline errors.
set -euo pipefail

# Ensure the script runs in its own directory regardless of where it's called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Resolve repository root for stable relative-path handling
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 1. Source configuration if available
CONFIG_FILE="config.sh"
if [[ -f "${CONFIG_FILE}" ]]; then
    echo "[INFO] Sourcing configuration from ${CONFIG_FILE}..."
    source "${CONFIG_FILE}"
else
    echo "[WARN] Configuration file ${CONFIG_FILE} not found. Proceeding without it."
fi

# 1.5 Resolve log directory (relative paths are interpreted from repo root)
LOG_DIR="${LOG_DIR:-./eval/result}"
if [[ "${LOG_DIR}" != /* ]]; then
    LOG_DIR="${REPO_ROOT}/${LOG_DIR#./}"
fi

# Ensure the log directory exists before running any benchmark
mkdir -p "${LOG_DIR}"

# 2. Define the list of benchmarks to run
# Note: Comment out any benchmark name to exclude it from the run
BENCHMARKS=(
    "longvideobench"
    "ovobench"
    "streamingbench"
    "videoholmes"
    "videomme"
)

echo "[INFO] Starting benchmark pipeline..."
echo "============================================================"

# 3. Iterate over the list and execute each benchmark script
for benchmark in "${BENCHMARKS[@]}"; do
    script_name="${benchmark}.sh"
    timestamp="$(date +'%Y%m%d_%H%M%S')"
    log_file="${LOG_DIR}/${benchmark}_${timestamp}.log"
    
    if [[ -f "${script_name}" ]]; then
        echo "[INFO] [$(date +'%Y-%m-%d %H:%M:%S')] Running: ${script_name}"
        echo "[INFO] Logging output to: ${log_file}"
        
        # Ensure the script has execution permissions
        chmod +x "${script_name}"
        
        # Execute benchmark script, keep terminal output visible, and persist logs.
        bash "${script_name}" 2>&1 | tee "${log_file}"
        
        echo "[INFO] Successfully completed: ${script_name}"
    else
        echo "[ERROR] Script not found: ${script_name}. Stopping pipeline."
        exit 1
    fi
    echo "------------------------------------------------------------"
done

echo "[INFO] All scheduled benchmarks finished successfully."
