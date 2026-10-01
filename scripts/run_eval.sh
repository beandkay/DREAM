SINGLE=$1

if [ "$SINGLE" == "single" ]; then
  # for single modal inference
  # METHOD_NAMES=("IO" "CoT" "LLM_Debate" "MAD" "ChatEval" "DMAD")
  METHOD_NAMES=("reld" "DMAD")
  DATASETS=("MMLU" "MMLU-Pro" "MATH" "GSM-Hard" "MedMCQA" "MedQA" "GPQA")
  MODEL_NAME=$2

  for SEED in {1..3}; do
    for METHOD in "${METHOD_NAMES[@]}"; do
      for DATASET in "${DATASETS[@]}"; do
        echo "Evaluating $METHOD on $DATASET ..."
        python evaluate.py --eval_protocol xverify --tested_dataset_name "$DATASET" --tested_method_name "$METHOD" --tested_mas_model_name "$MODEL_NAME" --seed "$SEED"
      done
    done
  done
else
  # METHOD_NAMES=("LLM_Debate" "refllm_debate" "swarm_describe" "vcot_debate" "crossmodal_debate" "saliency_debate")
  METHOD_NAMES=("LLM_Debate" "DMAD" "Div_MAD" "io" "cot" "self_consistency")
  DATASETS=("MathVision" "MathVista" "VisualPuzzles" "PathVQA" "MME" "MME-reasoning")
  MODEL_NAME=$2

  if [ -n "$3" ]; then
    TESTED_METHOD_CONFIG_NAME=$3
    for SEED in {1..3}; do
      for METHOD in "${METHOD_NAMES[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
          echo "Evaluating $METHOD on $DATASET with $MODEL_NAME as MAS model and $PERCEIVER_MODEL_NAME as Perceiver model ..."
          python evaluate.py --eval_protocol xverify --tested_dataset_name "$DATASET" --tested_method_name "$METHOD" --tested_mas_model_name "$MODEL_NAME" --tested_method_config_name "$TESTED_METHOD_CONFIG_NAME" --seed "$SEED"
        done
      done
    done
  else
    for SEED in {1..3}; do
      for METHOD in "${METHOD_NAMES[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
          echo "Evaluating $METHOD on $DATASET with $MODEL_NAME as MAS model and no Perceiver model ..."
          python evaluate.py --eval_protocol xverify --tested_dataset_name "$DATASET" --tested_method_name "$METHOD" --tested_mas_model_name "$MODEL_NAME" --seed "$SEED"
        done
      done
    done
  fi
fi