"""Isolated real simulator entry point for the explicit XVLA Mem derivative."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.libero_mem_sim_worker import main
from src.simulator.xvla_mem_env import XVLAMemEnv

if __name__ == "__main__":
    sys.exit(main(env_type=XVLAMemEnv))
