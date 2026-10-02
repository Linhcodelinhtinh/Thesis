---
library_name: transformers
tags:
- robotics
- vla
- image-text-to-text
- multimodal
- pretraining
license: mit
language:
- en
pipeline_tag: image-text-to-text
---

# MiniVLA 1B (Prismatic-Compatible Version)

<b>This checkpoint is in a format that is compatible with the training script from the original [Prismatic VLMs project codebase](https://github.com/TRI-ML/prismatic-vlms), which the OpenVLA
team built on top of to develop the OpenVLA model.</b>

This Prismatic-compatible checkpoint may be useful if you wish to <b>fully fine-tune</b> MiniVLA (all 1 billion parameters) via native PyTorch Fully
Sharded Data Parallel (FSDP) using the Prismatic VLMs training script. If you instead wish to do Parameter-Efficient Fine-Tuning via LoRA, you
can use the MiniVLA checkpoint linked above, which is compatible with the Hugging Face `transformers` library. We recommend fine-tuning via LoRA if
you do not have sufficient compute to fully fine-tune a 1B-parameter model (e.g., multiple A100/H100 GPUs).

## Usage Instructions

See the [MiniVLA GitHub README](https://github.com/Stanford-ILIAD/openvla-mini/blob/main/README.md) for instructions on how to use this checkpoint for full fine-tuning.

## Citation

**BibTeX:**

```bibtex
@article{belkhale24minivla,
    title={MiniVLA: A Better VLA with a Smaller Footprint},
    author={Suneel Belkhale and Dorsa Sadigh},
    url={https://github.com/Stanford-ILIAD/openvla-mini}
    year={2024}
} 
```