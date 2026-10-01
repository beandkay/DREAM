#!/bin/bash
#SBATCH --job-name=M3MAD_api_array
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1           # total number of tasks per node
#SBATCH --partition=gpu               # no CPU-only partition on this cluster; no --gpus requested below, so no GPU is allocated
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=5-00:00:00
#SBATCH --qos=batch-short
#SBATCH --array=0-1                   # NUM_MODELS * NUM_CONFIGS - 1 -- keep in sync with the arrays below
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err

module purge
module load Anaconda3
source activate
conda activate m3mad

# 1. Define model keys for the API-based models. No MODEL_PATHS/vLLM needed --
#    these are remote APIs (Claude via proxy, GPT-4o-mini via OpenRouter).
MODEL_KEYS=(
  "claude-sonnet"
  # "gpt-4o-mini"
)

# 2. Define the method-config sweep. Kept in sync with run_mm.sl's CONFIG_NAMES
#    so results are directly comparable.
CONFIG_NAMES=(
  ""
  # "config_main_gated"
  # "config_main_consensus"
  "config_main_gated_adaptive"
  # "config_ablation_gated_adaptive_no_history"
)

NUM_MODELS=${#MODEL_KEYS[@]}
NUM_CONFIGS=${#CONFIG_NAMES[@]}
TOTAL_TASKS=$(( NUM_MODELS * NUM_CONFIGS ))

if [ "$SLURM_ARRAY_TASK_COUNT" != "$TOTAL_TASKS" ]; then
  echo "WARNING: #SBATCH --array size ($SLURM_ARRAY_TASK_COUNT) does not match" \
       "NUM_MODELS * NUM_CONFIGS ($TOTAL_TASKS = $NUM_MODELS models x $NUM_CONFIGS configs)." \
       "Update the --array directive to match."
fi

# 3. Map this array task to one (model, config) pair. No GPU/vLLM needed --
#    these tasks just make outbound API calls, so no per-task port config either.
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_CONFIGS ))
CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_CONFIGS ))

MODEL_KEY="${MODEL_KEYS[$MODEL_IDX]}"
CONFIG_NAME="${CONFIG_NAMES[$CONFIG_IDX]}"

echo "======================================================================"
echo "Starting Slurm Array Task ID : $SLURM_ARRAY_TASK_ID / $TOTAL_TASKS"
echo "Model Key                    : $MODEL_KEY"
echo "Method Config                : ${CONFIG_NAME:-<default>}"
echo "======================================================================"

# 4. Run the inference benchmark for this task's (model, config) pair only.
#    No vLLM server, no GPU allocation, no TASK_API_CONFIG copy -- the shared
#    model_api_config.json is safe to read concurrently since nothing writes
#    to it for these API-backed model keys.
bash scripts/run_mm.sh "$MODEL_KEY" "$CONFIG_NAME"

echo "Slurm Array Task ID $SLURM_ARRAY_TASK_ID ($MODEL_KEY, ${CONFIG_NAME:-<default>}) completed successfully."
