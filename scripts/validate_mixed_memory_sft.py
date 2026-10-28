#!/usr/bin/env python
"""Validate a completed mixed dataset and optionally compare F0/F1 selections."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.training.mixed_sft import validate_mixed_dataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--compare',type=Path)
    a=p.parse_args()
    m=validate_mixed_dataset(a.data_dir)
    if a.compare:
        other=validate_mixed_dataset(a.compare)
        for key in ('sample_selection_sha256','split_lock_sha256','sources','counts','tokenizer_files_sha256'):
            if m[key]!=other[key]:raise ValueError('F0/F1 dataset mismatch: '+key)
    print(json.dumps(dict(status='passed',split_lock_sha256=m['split_lock_sha256'],
                         memory_source=m['memory_source'],counts=m['counts'],matched_comparison=bool(a.compare)),indent=2))


if __name__=='__main__':main()
