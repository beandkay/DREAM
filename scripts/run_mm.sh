# for multi modal inference

METHOD_NAMES=("LLM_Debate" "DMAD" "Div_MAD")
# METHOD_NAMES=("io" "self_consistency" "cot" "debate_with_images" "LLM_Debate" "DMAD" "Div_MAD")
DATASETS=("MathVision" "MathVista" "VisualPuzzles" "PathVQA" "MME" "MME-reasoning")
# DATASETS=("MathVision" "PathVQA" "MME")
MODEL_NAME=$1
METHOD_CONFIG_NAME=${2:-}   # empty if not passed
MODEL_API_CONFIG=${3:-}     # empty -> inference.py uses its default shared config path

if [[ "$MODEL_NAME" == *claude* ]]; then
  SEEDS=(1)   # claude API calls are costly/rate-limited -- run a single seed only
else
  SEEDS=(1 2 3)
fi

for seed in "${SEEDS[@]}"; do
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
      if [[ -n "$MODEL_API_CONFIG" ]]; then
        cmd+=(--model_api_config "$MODEL_API_CONFIG")
      fi
      "${cmd[@]}"
    done
  done
done

echo "All inference jobs finished."   