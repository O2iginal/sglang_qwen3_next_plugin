#!/usr/bin/env python3
"""
Quick test script to verify plugin structure and imports.
This script checks if the plugin can be imported correctly.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_import():
    """Test if plugin can be imported"""
    try:
        from sglang_qwen3_next_plugin import Qwen3NextForCausalLM, EntryClass
        print("✓ Plugin import successful!")
        print(f"✓ EntryClass: {EntryClass}")
        print(f"✓ Qwen3NextForCausalLM: {Qwen3NextForCausalLM}")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("Testing SGLang Qwen3 Next Plugin...")
    print("=" * 60)
    success = test_import()
    print("=" * 60)
    if success:
        print("Plugin structure is correct!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -e .")
        print("2. Set environment variable: export SGLANG_EXTERNAL_MODEL_PACKAGE='sglang_qwen3_next_plugin'")
        print("3. Launch SGLang server with your Qwen3 Next model")
    else:
        print("Please check the plugin structure and dependencies.")
    sys.exit(0 if success else 1)
