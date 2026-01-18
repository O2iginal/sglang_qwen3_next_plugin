#!/usr/bin/env python3
"""
SGLang launch_server wrapper that ensures plugin is loaded before SGLang initializes.

This wrapper ensures the plugin model is registered before SGLang's ModelRegistry
initializes, allowing the plugin to override the default implementation.

Usage:
    python -m sglang_qwen3_next_plugin.launch_server --model-path <path> --port 30110
"""

import os
import sys

# Set environment variable before importing SGLang
os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "sglang_qwen3_next_plugin"

# Import and register plugin BEFORE importing SGLang
# This ensures plugin model overrides the default implementation
try:
    import sglang_qwen3_next_plugin
    from sglang_qwen3_next_plugin import register
    register()
except Exception as e:
    print(f"Warning: Failed to register plugin: {e}", file=sys.stderr)

# Now import and run SGLang's launch_server
if __name__ == "__main__":
    from sglang.launch_server import launch_server, prepare_server_args, kill_process_tree
    
    # Parse arguments and launch server
    server_args = prepare_server_args(sys.argv[1:])
    
    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
