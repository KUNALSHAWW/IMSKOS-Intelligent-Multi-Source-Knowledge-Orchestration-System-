#!/usr/bin/env python
"""
IMSKOS Launcher Script
======================
This script launches the Streamlit app with proper environment configuration
to suppress TensorFlow warnings and ensure smooth operation.
"""

import os
import sys
import signal
import subprocess

# Global process reference for signal handling
_process = None


def signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    global _process
    if _process is not None:
        print(f"\n⏹️  Received signal {signum}, shutting down IMSKOS...")
        _process.terminate()
        try:
            _process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("⚠️  Process did not terminate gracefully, forcing kill...")
            _process.kill()


def main():
    global _process
    
    # Set environment variables to suppress TensorFlow warnings
    env = os.environ.copy()
    env.update({
        "TF_ENABLE_ONEDNN_OPTS": "0",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "TF_SILENCE_DEPRECATION_WARNINGS": "1",
        "PYTHONWARNINGS": "ignore::DeprecationWarning:tensorflow",
        "TOKENIZERS_PARALLELISM": "false",
        "GRPC_VERBOSITY": "ERROR",
        "GLOG_minloglevel": "3",
        # Disable TensorFlow entirely for transformers - use PyTorch only
        "USE_TF": "0",
        "USE_TORCH": "1",
    })
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "app.py")
    
    # Build the command
    cmd = [
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.port", "8503",
        "--server.address", "0.0.0.0",
    ]
    
    print("🚀 Starting IMSKOS - Intelligent Multi-Source Knowledge Orchestration System")
    print(f"📂 Working directory: {script_dir}")
    print(f"🌐 App will be available at: http://localhost:8503")
    print("-" * 60)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run Streamlit with Popen for proper signal handling
    try:
        _process = subprocess.Popen(cmd, env=env, cwd=script_dir)
        return _process.wait()
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down IMSKOS...")
        if _process is not None:
            _process.terminate()
            try:
                _process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _process.kill()
        return 0

if __name__ == "__main__":
    sys.exit(main() or 0)
