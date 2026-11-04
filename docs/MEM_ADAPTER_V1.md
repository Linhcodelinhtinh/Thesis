# V2 isolated Mem adapter, expanded data

This is a custom V2 training system, not an official LIBERO score. V1 baseline
execution still loads the original checkpoint without an adapter. No training
or rollout improvement is claimed by the implementation alone.

## Scope and source coverage

Expand replay from seven to all ten official LIBERO-10 tasks. Keep all author
T1/T3 demonstrations, with train/validation/test initial-state separation.
The old split assignments are preserved when extending the lock. The RGB v3
writer supports bowl/original-plate only; T2/T4 and other object profiles are
not enabled by this version. Never substitute demonstrations for missing files.

Preparation produces predicted memory for every locked Mem frame using the
same visual_writer_factory and LiberoMemTextRenderer as evaluation. Every
prompt uses only current/past observations and the last executed action.
Source hashes, detector settings, writer code/config, tokenizer and index hashes
are verified. Empty/uncertain memory is retained; coverage alone is not accuracy.

```powershell
.venv_memory/Scripts/python.exe scripts/prepare_mem_adapter_data.py --download --output-root experiments/training/mem_adapter_full10_t1t3_v1 --perception-python .venv_memory/Scripts/python.exe --tokenizer-path experiments/training/results/no_memory-20261029-175154/pretrained_model/backbone
```

This requires space for downloaded sources and a new immutable dataset; it does
not overwrite the previous F0 dataset. Preparation refuses incomplete existing
stage directories; inspect a failed stage before restarting with a new root.

After preparation completes, package the verified data with:

```powershell
.venv_memory/Scripts/python.exe scripts/package_mem_adapter_data.py --data-dir experiments/training/mem_adapter_full10_t1t3_v1/dataset --output experiments/training/mem_adapter_v1_data.zip
python scripts/package_memory_sft.py --output experiments/training/mem_adapter_v1_code.zip
```

For Kaggle, attach the code ZIP and completed data ZIP plus its JSON checksum
sidecar (or the extracted dataset directory). Import
notebooks/kaggle_mem_adapter_v1.ipynb, enable Internet and select two T4 GPUs.
The notebook validates the expanded memory dataset before starting training.
Do not upload a `.partial` archive or an unfinished memory index.

## Training

Use the isolated training environment / Kaggle runtime with the existing pinned
training dependencies. Do not install training packages in the perception or
simulator environment. Fetch pinned assets with memory_sft_assets.py first.
The primary package pins are in envs/mem_adapter_training.txt, taken from the
previous successful training run; actual resolved dependencies are recorded
again by the trainer. Use a separate virtual environment for a local run.

```bash
python scripts/memory_sft_assets.py fetch /kaggle/temp/sft-assets
python scripts/train_mem_adapter.py --data-dir DATASET --assets-dir /kaggle/temp/sft-assets --output-dir OUTPUT --gpus 2
```

The launcher consumes configs/training/mem_adapter_v1.json: memory enabled,
attention-only expert LoRA rank 16, LR 2e-5, batch 32, 3,000 updates, 50/50
replay, checkpoints every 500. One GPU uses accumulation 16; two use 8.
These are pilot settings, not a validated optimum. Select later deployment
checkpoints using held-out Mem success/progress/overshoot, not flow loss alone.

The final pretrained_model directory contains an unmerged PEFT adapter and
mem_adapter_manifest.json, not a replacement base model. Base weights and
normalizer hashes bind it to the original checkpoint. Source training
checkpoints remain available; the final export is not automatically best.

## Inference

SmolVLAAdapter.load(adapter_directory, base_checkpoint=original_checkpoint)
loads the original policy/processors and attaches the frozen adapter. The
default base is resources/checkpoints/smolvla_libero. LiberoMemRolloutRunner
condition F1 binds the online writer and strict renderer to the training
contract before inference. Use the adapter's pinned tokenizer for the renderer.
The runtime rejects different detector/config/code and unverified memory paths.
Image conventions differ between refined offline HDF5 and raw simulator RGB;
use the explicit source-specific conventions in the shared factory.

For baseline tasks, load the original checkpoint, not the adapter directory.
No routing decision reads success predicates, hidden progress, or future state.
This version has explicit caller-selected modes; it does not claim learned routing.
