#!/bin/bash

datasets=(MMLU MMLU-Pro MATH GSM-Hard MedMCQA MedQA GPQA MME-Reasoning MathVista MathVision PathVQA VisualPuzzles)

for dataset in "${datasets[@]}"; do
    if [ "$dataset" == "MedQA" ]; then
        python datasets/build_test_dataset.py --dataset_name "$dataset" --dataset_path datasets/data/RawMedQA
    else
        python datasets/build_test_dataset.py --dataset_name "$dataset"
    fi
done