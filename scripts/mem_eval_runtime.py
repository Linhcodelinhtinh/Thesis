#!/usr/bin/env python
"""Emit environment identity without importing simulator, policy or perception."""
import importlib.metadata
import json
import os
import platform
import sys


def main():
    versions = {}
    for name in ("numpy", "torch", "torchvision", "lerobot", "transformers", "peft", "tokenizers",
                 "robosuite", "bddl", "mujoco", "robomimic", "imageio", "imageio-ffmpeg"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    print(json.dumps(dict(prefix=os.path.abspath(sys.prefix), executable=sys.executable,
                         python=platform.python_version(), platform=platform.platform(), packages=versions)))


if __name__ == "__main__":
    main()
