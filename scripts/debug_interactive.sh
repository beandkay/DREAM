#!/bin/bash

# Default values
DEFAULT_METHOD="llm_debate"
DEFAULT_DATASET="MME"
DEFAULT_MODEL="qwen2.5-vl-7b-local"
DEFAULT_METHOD_CONFIG="config_main_gated"

echo "=== Interactive Single Sample Debugging ==="

read -p "Enter method name [$DEFAULT_METHOD]: " METHOD
METHOD=${METHOD:-$DEFAULT_METHOD}

read -p "Enter dataset name [$DEFAULT_DATASET]: " DATASET
DATASET=${DATASET:-$DEFAULT_DATASET}

read -p "Enter method config name [$DEFAULT_METHOD_CONFIG]: " METHOD_CONFIG
METHOD_CONFIG=${METHOD_CONFIG:-$DEFAULT_METHOD_CONFIG}

read -p "Enter model name [$DEFAULT_MODEL]: " MODEL
MODEL=${MODEL:-$DEFAULT_MODEL}

echo "Choose debug mode:"
echo "1) Enter a custom query"
echo "2) Debug a specific sample index from the dataset"
read -p "Mode (1 or 2) [2]: " MODE
MODE=${MODE:-2}

if [ "$MODE" = "1" ]; then
    read -p "Enter your custom query: " CUSTOM_QUERY
    # Go back to the parent directory to run inference.py correctly
    cd "$(dirname "$0")/.." || exit 1
    CMD="python inference.py --method_name \"$METHOD\" --test_dataset_name \"$DATASET\" --model_name \"$MODEL\" --debug_query \"$CUSTOM_QUERY\" --method_config_name \"$METHOD_CONFIG\"" 
    echo $CMD
    eval $CMD
else
    read -p "Enter sample index to debug [0]: " SAMPLE_IDX
    SAMPLE_IDX=${SAMPLE_IDX:-0}
    cd "$(dirname "$0")/.." || exit 1
    CMD="python inference.py --method_name \"$METHOD\" --test_dataset_name \"$DATASET\" --model_name \"$MODEL\" --sample_idx \"$SAMPLE_IDX\" --method_config_name \"$METHOD_CONFIG\""
    echo $CMD
    eval $CMD
fi
