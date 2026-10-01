import os
import sys
import json
import argparse
import threading
import concurrent.futures
from tqdm import tqdm
import traceback
import time
from collections import defaultdict

from methods import get_method_class
from utils import reserve_unprocessed_queries, load_model_api_config, write_to_jsonl

def process_sample(args, general_config, sample, output_path, lock, global_token_stats=None, global_time_stats=None):
    MAD_METHOD = get_method_class(args.method_name, args.test_dataset_name)
    mad = MAD_METHOD(general_config, method_config_name=args.method_config_name)
    save_data = sample.copy()
    
    # Do not save the image content to the output file to save space
    for key in ["image", "image_url", "image_path"]:
        save_data.pop(key, None)
    
    # Start timing
    start_time = time.time()
    start_cpu_time = time.process_time()
    
    try:
        mad_output = mad.inference(sample)
        if "response" not in mad_output:    # ensure that there is a key named "response"
            raise ValueError(f"The key 'response' is not found in the MAD output: {mad_output}")
        save_data.update(mad_output)
    except Exception as e:
        save_data["error"] = f"Inference Error: {traceback.format_exc()}"
    
    # End timing
    end_time = time.time()
    inference_time = end_time - start_time
    end_cpu_time = time.process_time()
    cpu_time = end_cpu_time - start_cpu_time
    
    token_stats = mad.get_token_stats()
    save_data.update({
        "token_stats": token_stats,
        "inference_time": inference_time,
        "cpu_time": cpu_time
    })
    write_to_jsonl(lock, output_path, save_data)
    
    # Update global token stats
    if global_token_stats is not None:
        for model_name, stats in token_stats.items():
            global_token_stats[model_name]["num_llm_calls"] += stats["num_llm_calls"]
            global_token_stats[model_name]["prompt_tokens"] += stats["prompt_tokens"]
            global_token_stats[model_name]["completion_tokens"] += stats["completion_tokens"]
    
    # Update global time stats
    if global_time_stats is not None:
        global_time_stats["total_inference_time"] += inference_time
        global_time_stats["total_cpu_time"] += cpu_time
        global_time_stats["num_samples"] += 1
        global_time_stats["min_time"] = min(global_time_stats.get("min_time", float('inf')), inference_time)
        global_time_stats["max_time"] = max(global_time_stats.get("max_time", 0), inference_time)
        global_time_stats["min_cpu_time"] = min(global_time_stats.get("min_cpu_time", float('inf')), cpu_time)
        global_time_stats["max_cpu_time"] = max(global_time_stats.get("max_cpu_time", 0), cpu_time)
    
    return token_stats, inference_time

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # args related to the method
    parser.add_argument("--method_name", type=str, default="vanilla", help="MAD name.")
    parser.add_argument("--method_config_name", type=str, default=None, help="The config file name. If None, the default config file will be used.")

    # args related to the model
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini-2024-07-18", help="The agent backend to be used for inference.")
    parser.add_argument("--model_api_config", type=str, default="model_api_configs/model_api_config.json")
    parser.add_argument("--model_temperature", type=float, default=0.5, help="Temperature for sampling.")
    parser.add_argument("--model_max_tokens", type=int, default=2048, help="Maximum tokens for sampling.")
    parser.add_argument("--model_timeout", type=int, default=600, help="Timeout for sampling.")
    
    # args related to dataset
    parser.add_argument("--test_dataset_name", type=str, default="example_math", help="The dataset to be used for testing.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to the output file.")
    parser.add_argument("--require_val", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the LLM API.")
    
    # args related to teacher distillation (Approach A)
    parser.add_argument("--teacher_ref_db_path", type=str, default=None,
                        help="Path to teacher reference DB JSON (overrides method config).")
    
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_query", type=str, default=None, help="Custom query to debug.")
    parser.add_argument("--sample_idx", type=int, default=None, help="Index of sample to debug in the dataset.")
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()
    
    general_config = vars(args)
    
    # Load model config
    model_api_config = load_model_api_config(args.model_api_config, args.model_name)
    general_config.update({"model_api_config": model_api_config})
    general_config["test_dataset_name"] = args.test_dataset_name
    print("-"*50, f"\n>> Model API config: {model_api_config[args.model_name]}")
    
    if args.debug or args.debug_query is not None or args.sample_idx is not None:
        general_config["debug"] = True
        if args.debug_query is not None:
            sample = {"query": args.debug_query}
        elif args.sample_idx is not None:
            # load dataset
            with open(f"./datasets/data/{args.test_dataset_name}.json", "r") as f:
                test_dataset = json.load(f)
            sample = test_dataset[args.sample_idx]
            print(f">> Debugging sample index {args.sample_idx}:")
            print(json.dumps(sample, indent=4))
        else:
            sample = {"query": "If $|x+5|-|3x-6|=0$, find the largest possible value of $x$. Express your answer as an improper fraction."}
            
        # MAD inference
        MAD_METHOD = get_method_class(args.method_name, args.test_dataset_name)
        mad = MAD_METHOD(general_config, method_config_name=args.method_config_name)

        response = mad.inference(sample)
        
        print(json.dumps(response, indent=4))
        print(f"\n>> Token stats: {json.dumps(mad.get_token_stats(), indent=4)}")
        
        # Save to debug json file
        config_suffix = f"_{args.method_config_name}" if args.method_config_name else ""
        debug_output_path = f"debug_results{config_suffix}.json"
        
        # Create a unified debug object containing the original query, the debate response, and token stats
        debug_output = {
            "sample": sample,
            "response": response,
            "token_stats": mad.get_token_stats()
        }
        
        with open(debug_output_path, "w") as f:
            json.dump(debug_output, f, indent=4)
        print(f"\n>> Saved debug results to {debug_output_path}")
    
    else:
        print(f">> Method: {args.method_name} | Dataset: {args.test_dataset_name}")

        # load dataset
        with open(f"./datasets/data/{args.test_dataset_name}.json", "r") as f:
            test_dataset = json.load(f)
        
        if args.require_val:
            val_dataset_path = f"./datasets/data/{args.test_dataset_name}_val.json"
            if not os.path.exists(val_dataset_path):
                raise FileNotFoundError(f"Validation dataset not found at {val_dataset_path}. Please provide a valid path.")
            with open(val_dataset_path, "r") as f:
                val_dataset = json.load(f)
        
        # get output path
        if args.output_path is not None:
            output_path = args.output_path
        else:
            # If a custom config is used, append its name to differentiate outputs automatically
            config_suffix = ""
            if args.method_config_name and args.method_config_name != "config_main":
                config_suffix = f"_{args.method_config_name}"
            
            seed_suffix = f"_{args.seed}" if args.seed is not None else ""
            
            env_suffix = ""
            if os.environ.get("VIF_YARD_SINK") == "1":
                env_suffix += f"_yard-{os.environ.get('VIF_AGENT_ROLE', 'bal')}"
            if os.environ.get("VIF_HIVE_MIND") == "1":
                env_suffix += f"_hive-{os.environ.get('VIF_CROSS_ALPHA', '0.15')}"
            if os.environ.get("GLOBAL_RESIDUAL_ALPHA") and float(os.environ.get("GLOBAL_RESIDUAL_ALPHA")) > 0:
                env_suffix += f"_res-{os.environ.get('GLOBAL_RESIDUAL_ALPHA')}"

            output_path = f"./results/{args.test_dataset_name}/{args.model_name}{seed_suffix}/{args.method_name}{config_suffix}{env_suffix}_infer.jsonl"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # reserve unprocessed samples
        test_dataset = reserve_unprocessed_queries(output_path, test_dataset)
        remaining_samples = len(test_dataset)
        print(f">> After filtering: {remaining_samples} samples")
        if remaining_samples == 0:
            print(">> All queries have already been processed previously. Existing results and statistics are preserved.")
            print(f">> Output path: {output_path}")
            stats_output_path = output_path.replace('.jsonl', '_stats.json')
            if os.path.exists(stats_output_path):
                print(f">> Existing stats file: {stats_output_path}")
            sys.exit(0)
        
        # optimize MAD if required (e.g., GPTSwarm, ADAS, and AFlow)
        if args.require_val:
            # get MAD instance
            MAD_METHOD = get_method_class(args.method_name, args.test_dataset_name)
            mad = MAD_METHOD(general_config, method_config_name=args.method_config_name)
            mad.optimizing(val_dataset)
        
        # Initialize global token stats and time stats
        global_token_stats = defaultdict(lambda: {"num_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        global_time_stats = {
            "total_inference_time": 0.0,
            "total_cpu_time": 0.0,
            "num_samples": 0,
            "min_time": float('inf'),
            "max_time": 0.0,
            "min_cpu_time": float('inf'),
            "max_cpu_time": 0.0
        }
        
        # Start overall timing
        overall_start_time = time.time()
        overall_start_cpu_time = time.process_time()
        
        # inference the MAD method
        lock = threading.Lock()
        if args.sequential:
            for sample in tqdm(test_dataset, desc="Processing queries"):
                process_sample(args, general_config, sample, output_path, lock, global_token_stats, global_time_stats)
        else:
            max_workers = model_api_config[args.model_name]["max_workers"]
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for _ in tqdm(executor.map(lambda sample: process_sample(args, general_config, sample, output_path, lock, global_token_stats, global_time_stats), test_dataset), total=len(test_dataset), desc="Processing queries"):
                    pass
        
        # End overall timing
        overall_end_time = time.time()
        overall_end_cpu_time = time.process_time()
        global_time_stats["total_elapsed_time"] = overall_end_time - overall_start_time
        global_time_stats["total_elapsed_cpu_time"] = overall_end_cpu_time - overall_start_cpu_time
        
        # Calculate average inference time / cpu time
        if global_time_stats["num_samples"] > 0:
            global_time_stats["avg_time"] = global_time_stats["total_inference_time"] / global_time_stats["num_samples"]
            global_time_stats["avg_cpu_time"] = global_time_stats["total_cpu_time"] / global_time_stats["num_samples"]
        
        # Output final statistics
        print(f"\n>> Final Token stats: {json.dumps(dict(global_token_stats), indent=4)}")
        print(f"\n>> Time stats:")
        print(f"    Total elapsed time (wall): {global_time_stats['total_elapsed_time']:.2f} seconds")
        print(f"    Total elapsed CPU time: {global_time_stats['total_elapsed_cpu_time']:.2f} seconds")
        print(f"    Total inference time: {global_time_stats['total_inference_time']:.2f} seconds")
        print(f"    Number of samples: {global_time_stats['num_samples']}")
        if global_time_stats["num_samples"] > 0:
            print(f"    Average inference time: {global_time_stats['avg_time']:.2f} seconds")
            print(f"    Min inference time: {global_time_stats['min_time']:.2f} seconds")
            print(f"    Max inference time: {global_time_stats['max_time']:.2f} seconds")
            print(f"    Total CPU time: {global_time_stats['total_cpu_time']:.2f} seconds")
            print(f"    Average CPU time: {global_time_stats['avg_cpu_time']:.2f} seconds")
            print(f"    Min CPU time: {global_time_stats['min_cpu_time']:.2f} seconds")
            print(f"    Max CPU time: {global_time_stats['max_cpu_time']:.2f} seconds")
        
        # Save combined stats to a separate file
        combined_stats = {
            "token_stats": dict(global_token_stats),
            "time_stats": global_time_stats
        }
        stats_output_path = output_path.replace('.jsonl', '_stats.json')
        with open(stats_output_path, 'w') as f:
            json.dump(combined_stats, f, indent=4)
        print(f">> Combined stats saved to: {stats_output_path}")
