#!/bin/bash
#SBATCH --job-name=M3MAD-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1           # total number of tasks per node
#SBATCH --partition=gpu-large
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=5-00:00:00 
#SBATCH --qos=batch-short

module purge

module load Anaconda3

source activate

conda activate m3mad

# API_PORT=8001 srun nohup llamafactory-cli api local_api_server/xverify_9b_c.yaml trust_remote_code=True &
srun nohup vllm serve IAAR-Shanghai/xVerify-9B-C \
--port 8010 \
--host 0.0.0.0 \
--dtype bfloat16 \
--limit-mm-per-prompt.image=1 \
--limit-mm-per-prompt.video=0 \
--served-model-name xverify-9b-c \
--gpu-memory-utilization 0.9 --trust-remote-code &

sleep 90

# bash scripts/run_eval.sh single llama3.1-8b-local
# bash scripts/run_eval.sh single qwen2.5-7b-local
# bash scripts/run_eval.sh multiple qwen2.5-7b-local
# bash scripts/run_eval.sh multiple qwen2.5-7b-local qwen2.5-vl-7b-local
# bash scripts/run_eval.sh multiple qwen3-8b-local qwen2.5-vl-7b-local
# bash scripts/run_eval.sh multiple qwen2.5-vl-7b-local qwen2.5-vl-7b-local
bash scripts/run_eval.sh multiple qwen2.5-vl-7b-local
bash scripts/run_eval.sh multiple qwen2.5-vl-32b-local
bash scripts/run_eval.sh multiple llavanext-7b-local
bash scripts/run_eval.sh multiple gemma3-4b-local
bash scripts/run_eval.sh multiple gemma3-12b-local
bash scripts/run_eval.sh multiple gemma3-27b-local
# bash scripts/run_eval.sh multiple qwen2.5-vl-7b-local config_main_flip
# bash scripts/run_eval.sh multiple qwen2.5-vl-7b-local config_main_scale_up
# bash scripts/run_eval.sh multiple qwen2.5-vl-7b-local config_main_scale_down
# bash scripts/run_eval.sh multiple qwen2.5-vl-7b-local config_main_mix_res
# bash scripts/run_eval.sh multiple qwen2.5-vl-7b-local config_main_mix_all
# bash scripts/run_eval.sh multiple qwen2.5-vl-7b-local
# bash scripts/run_eval.sh multiple qwen2.5-vl-7b-local config_main_7b
# bash scripts/run_eval.sh multiple qwen2.5-vl-7b-local config_main_intern_14b
# bash scripts/run_eval.sh multiple qwen2.5-vl-7b-local config_mix_1v2l
# bash scripts/run_eval.sh multiple qwen2.5-vl-7b-local config_mix_2v1l
# bash scripts/run_eval.sh multiple qwen2.5-vl-14b-local
# bash scripts/run_eval.sh multiple qwen2.5-14b-local qwen2.5-vl-7b-local
# bash scripts/run_eval.sh multiple internlm3-8b-local qwen2.5-vl-7b-local
# bash scripts/run_eval.sh multiple lvr-7b-local
# bash scripts/run_eval.sh multiple capimagine-7b-local
bash scripts/run_eval.sh multiple internvl3-8b-local
bash scripts/run_eval.sh multiple internvl3-14b-local
bash scripts/run_eval.sh multiple gpt-4o-mini

python parse_results.py