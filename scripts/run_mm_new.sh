# for multi modal inference (Slurm colocated templates: slurm_scripts/run_mm_new_*.sl)

DATASETS=("MathVision" "MathVista" "VisualPuzzles" "PathVQA" "MME" "MME-reasoning")
# METHOD_NAMES=("LLM_Debate" "DMAD" "Div_MAD")
METHOD_NAMES=("DMAD" "LLM_Debate")

PERCEIVER_MODEL_NAME=$2
MODEL_NAME=$1

for seed in {1..3}; do
  for METHOD in "${METHOD_NAMES[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
      echo "Running $METHOD on $DATASET ..."
      python inference.py --method_name "$METHOD" --test_dataset_name "$DATASET" --model_name "$MODEL_NAME" --perceiver_model_name "$PERCEIVER_MODEL_NAME" --seed "$seed"
        #  --sequential
    done
  done
done

echo "All inference jobs finished."