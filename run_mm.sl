#!/bin/bash
#SBATCH --job-name=M3MAD_array
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1           # total number of tasks per node
#SBATCH --partition=gpu-large
#SBATCH --gpus=h100:1                 # 1 GPU per (model, config) task
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --qos=batch-short
#SBATCH --array=0-23                  # NUM_MODELS * NUM_CONFIGS - 1 -- keep in sync with the arrays below
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err

module purge
module load Anaconda3
source activate
conda activate m3mad

# 1. Define model keys and paths arrays
MODEL_KEYS=(
  "qwen2.5-vl-7b-local"
  "internvl3-8b-local"
  "llavanext-7b-local"
  "gemma3-4b-local"
  "gemma3-12b-local"
  "internvl3-14b-local"
  # "gemma3-27b-local"
  # "qwen2.5-vl-32b-local"
  # "claude-sonnet"
  # "gpt-4o-mini"
)

MODEL_PATHS=(
  "Qwen/Qwen2.5-VL-7B-Instruct"
  "OpenGVLab/InternVL3-8B-Instruct"
  "llava-hf/llava-v1.6-mistral-7b-hf"
  "google/gemma-3-4b-it"
  "google/gemma-3-12b-it"
  "OpenGVLab/InternVL3-14B-Instruct"
  # "google/gemma-3-27b-it"
  # "Qwen/Qwen2.5-VL-32B-Instruct"
  # "https://llmproxy.deakin.edu.au/"
  # "https://openrouter.ai/api/v1/"
)

# 2. Define the method-config sweep. "" runs the method's default config_main.
CONFIG_NAMES=(
  # ""
  "config_main_gated"
  "config_main_consensus"
  "config_main_gated_adaptive"
  "config_ablation_gated_adaptive_no_history"
)

NUM_MODELS=${#MODEL_KEYS[@]}
NUM_CONFIGS=${#CONFIG_NAMES[@]}
TOTAL_TASKS=$(( NUM_MODELS * NUM_CONFIGS ))

if [ "$SLURM_ARRAY_TASK_COUNT" != "$TOTAL_TASKS" ]; then
  echo "WARNING: #SBATCH --array size ($SLURM_ARRAY_TASK_COUNT) does not match" \
       "NUM_MODELS * NUM_CONFIGS ($TOTAL_TASKS = $NUM_MODELS models x $NUM_CONFIGS configs)." \
       "Update the --array directive to match."
fi

# 3. Map this array task to one (model, config) pair -> one GPU each.
#    Task IDs are laid out as: model_idx * NUM_CONFIGS + config_idx
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_CONFIGS ))
CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_CONFIGS ))

MODEL_KEY="${MODEL_KEYS[$MODEL_IDX]}"
MODEL_PATH="${MODEL_PATHS[$MODEL_IDX]}"
CONFIG_NAME="${CONFIG_NAMES[$CONFIG_IDX]}"

# Each task gets its own copy of the API config so concurrent tasks serving
# the same model_key (different config, or a rerun) don't stomp on each
# other's vLLM port in the shared model_api_config.json.
TASK_API_CONFIG="model_api_configs/.task_config_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
cp model_api_configs/model_api_config.json "$TASK_API_CONFIG"

echo "======================================================================"
echo "Starting Slurm Array Task ID : $SLURM_ARRAY_TASK_ID / $TOTAL_TASKS"
echo "Model Key                    : $MODEL_KEY"
echo "Model Path                   : $MODEL_PATH"
echo "Method Config                : ${CONFIG_NAME:-<default>}"
echo "Task API Config              : $TASK_API_CONFIG"
echo "======================================================================"

# 4. Clean up background vLLM process and the per-task config copy on exit/cancel
cleanup() {
  if [ -n "$VLLM_PID" ]; then
    echo "Cleaning up vLLM process (PID: $VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null || true
  fi
  rm -f "$TASK_API_CONFIG"
}

trap cleanup EXIT

# 5. Start vLLM background server (bound to this task's single allocated GPU)
python ./scripts/start_vllm.py \
  --model_key "$MODEL_KEY" \
  --model_path "$MODEL_PATH" \
  --config_path "$TASK_API_CONFIG" &

VLLM_PID=$!

# Wait for vLLM server initialization
sleep 240

# 6. Run the inference benchmark for this task's (model, config) pair only
bash scripts/run_mm.sh "$MODEL_KEY" "$CONFIG_NAME" "$TASK_API_CONFIG"

echo "Slurm Array Task ID $SLURM_ARRAY_TASK_ID ($MODEL_KEY, ${CONFIG_NAME:-<default>}) completed successfully."
