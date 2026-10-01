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

def update_config(port, model_key):
    """Update the JSON config to point to the new port."""
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: Could not find {CONFIG_PATH}")
        return False
        
    with open(CONFIG_PATH, "r") as f:
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
            
            with open(CONFIG_PATH, "w") as f:
                json.dump(config, f, indent=4)
            print(f"Updated {CONFIG_PATH} [{model_key}] -> {new_url}")
            return True
    
    print(f"Error: Could not find '{model_key}' in config to update.")
    return False

def main():
    parser = argparse.ArgumentParser(description="Start llamafactory-cli API server on a dynamic port.")
    parser.add_argument("--model_key", type=str, default="qwen2.5-vl-7b-local",
                        help="The model key in model_api_config.json to update.")
    parser.add_argument("--yaml_path", type=str, default="local_api_server/qwen2.5_vl_7b.yaml",
                        help="Path to the llamafactory-cli yaml configuration file.")
    args = parser.parse_args()

    port = get_free_port()
    print(f"Found unused port: {port}")
    
    if update_config(port, args.model_key):
        print(f"Starting llamafactory-cli on port {port}...")
        env = os.environ.copy()
        env["API_PORT"] = str(port)
        env["API_VERBOSE"] = "0"
        
        # Start the process in the background
        cmd = ["nohup", "llamafactory-cli", "api", args.yaml_path]
        
        with open(f"api_server_{env["API_PORT"]}.log", "w") as log_file:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True # Detach from parent
            )
            
        print(f"Server started in background (PID: {process.pid}). Logs are being written to api_server.log")

if __name__ == "__main__":
    main()
