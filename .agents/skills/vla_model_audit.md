When a new VLA model is proposed:

1. Identify canonical repository.
2. Verify exact checkpoint.
3. Inspect model config.
4. Determine:
   - input images
   - image resolution
   - camera ordering
   - state/proprioception
   - action dimensionality
   - action ordering
   - normalization
   - action tokenizer
   - VQ decoder
   - chunk size
   - action execution horizon
   - control frequency
5. Locate official inference code.
6. Identify official LIBERO evaluation code if available.
7. Record all information in model_manifest.yaml.
8. Only then implement adapter.

MODEL AUDIT
-----------
Model:
Repository:
Revision:
Checkpoint:
Training distribution:
LIBERO adaptation:
Image inputs:
State inputs:
Action:
Action decoder:
Chunk size:
Normalization:
Official inference:
Official LIBERO evaluation:
Hardware requirement:
Known incompatibilities: