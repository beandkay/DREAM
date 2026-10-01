# for single modal inference

# METHOD_NAMES=("LLM_Debate" "DMAD" "Div_MAD")
METHOD_NAMES=("LLM_Debate")
DATASETS=("MMLU" "MMLU-Pro" "MATH" "GSM-Hard" "MedMCQA" "MedQA" "GPQA")
# DATASETS=("MMLU")
MODEL_NAME=$1
METHOD_CONFIG_NAME=${2:-}   # empty if not passed

for seed in {1..3}; do
  for METHOD in "${METHOD_NAMES[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
      echo "Running $METHOD on $DATASET (model: $MODEL_NAME, seed: $seed, config: $METHOD_CONFIG_NAME)..."
      cmd=(
        python inference.py
        --method_name "$METHOD"
        --test_dataset_name "$DATASET"
        --model_name "$MODEL_NAME"
        --seed "$seed"
      )
      if [[ -n "$METHOD_CONFIG_NAME" ]]; then
        cmd+=(--method_config_name "$METHOD_CONFIG_NAME")
      fi
      "${cmd[@]}"
    done
  done
done

echo "All inference jobs finished."