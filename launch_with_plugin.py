#!/usr/bin/env python3
"""
Launch script to ensure the plugin is loaded before starting SGLang server.
This script registers the plugin model before launching the server.

Usage:
    python launch_with_plugin.py --model-path <path> --port 30110 --tp 1
"""

import os
import sys

# Register the plugin before importing sglang
from sglang.srt.models.registry import ModelRegistry
from sglang_qwen3_next_plugin import Qwen3NextForCausalLM

# Register the model class
ModelRegistry.models["Qwen3NextForCausalLM"] = Qwen3NextForCausalLM

print("=" * 80)
print("  ✓ Plugin model registered: Qwen3NextForCausalLM")
print("=" * 80)

# Now launch the server by calling the original launch_server module
# This passes all arguments through to sglang.launch_server
if __name__ == "__main__":
    # Import the launch_server module and run it
    # This will parse arguments and launch the server
    import runpy
    # Replace sys.argv[0] to make it look like we're calling sglang.launch_server
    sys.argv[0] = "-m"
    sys.argv.insert(0, "python")
    sys.argv.insert(1, "sglang.launch_server")
    runpy.run_module("sglang.launch_server", run_name="__main__")
