#!/usr/bin/env python3
import socket
import json
import os
import subprocess
import argparse

CONFIG_PATH = "model_api_configs/model_api_config.json"

def get_free_port():
    """Find an available port on the local system."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def is_port_free(port):
    """Check if a specific port is free."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('', port))
            return True
        except OSError:
            return False
def get_configured_port(model_key, config_path=CONFIG_PATH):
    """Get the currently configured port for the given model_key."""
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r") as f:
        config = json.load(f)
    if model_key in config and "model_list" in config[model_key]:
        old_url = config[model_key]["model_list"][0]["model_url"]
        base = old_url.split(":", 2)
        if len(base) >= 3:
            port_str = base[2].split("/", 1)[0]
            if port_str.isdigit():
                return int(port_str)
    return None

def update_config(port, model_key, config_path=CONFIG_PATH):
    """Update the JSON config to point to the new port."""
    if not os.path.exists(config_path):
        print(f"Error: Could not find {config_path}")
        return False

    with open(config_path, "r") as f:
        config = json.load(f)
        
    if model_key in config and "model_list" in config[model_key]:
        # Update the port in the model_url
        old_url = config[model_key]["model_list"][0]["model_url"]
        # Expected format: http://localhost:8000/v1
        base = old_url.split(":", 2) # ['http', '//localhost', '8000/v1']
        if len(base) >= 3:
            suffix = base[2].split("/", 1)[1] if "/" in base[2] else "v1"
            new_url = f"{base[0]}:{base[1]}:{port}/{suffix}"
            config[model_key]["model_list"][0]["model_url"] = new_url

            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)
            print(f"Updated {config_path} [{model_key}] -> {new_url}")
            return True
    
    print(f"Error: Could not find '{model_key}' in config to update.")
    return False

def main():
    parser = argparse.ArgumentParser(description="Start vllm API server on a dynamic port.")
    parser.add_argument("--model_key", type=str, default="qwen2.5-vl-7b-local",
                        help="The model key in model_api_config.json to update.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="The model path or identifier to serve via vllm.")
    parser.add_argument("--config_path", type=str, default=CONFIG_PATH,
                        help="Path to the model_api_config.json to update. Use a per-job copy "
                             "when running multiple servers for the same model_key in parallel, "
                             "so concurrent jobs don't overwrite each other's port.")

    # We use parse_known_args to capture all extra arguments and pass them to vllm
    args, extra_args = parser.parse_known_args()

    configured_port = get_configured_port(args.model_key, args.config_path)
    if configured_port and is_port_free(configured_port):
        port = configured_port
        print(f"Configured port {port} is free. Using it.")
    else:
        port = get_free_port()
        print(f"Configured port busy or not found. Found unused port: {port}")

    if update_config(port, args.model_key, args.config_path):
        print(f"Starting vllm on port {port}...")
        
        # Build the vllm command
        # "python -m vllm.entrypoints.openai.api_server" is widely compatible across vllm versions
        cmd = [
            "nohup", "vllm", "serve",
            "--host", "0.0.0.0",
            "--model", args.model_path,
            "--port", str(port),
            "--dtype", "bfloat16",
            "--served-model-name", f"{args.model_key}",
            "--gpu-memory-utilization", "0.9",
            "--trust_remote_code",
            # "--chat-template-content-format", "string",
            "--limit-mm-per-prompt.image", "10",
        ] + extra_args
        
        with open(f"vllm_server_{port}.log", "w") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True # Detach from parent
            )
            
        print(f"Server started in background (PID: {process.pid}). Logs are being written to vllm_server.log")
        print("Command executed:", " ".join(cmd))

if __name__ == "__main__":
    main()
