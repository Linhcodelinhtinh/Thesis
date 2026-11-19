"""Runtime compatibility shims for official Prismatic codebase.

Enables Stanford-ILIAD openvla-mini to execute cleanly under modern
environments without altering any pinned upstream source files.
"""

import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def apply_prismatic_shims() -> None:
    """Apply compatibility shims before importing Prismatic modules."""
    # 0. Ensure upstream openvla-mini package root is on sys.path
    repo_root = Path(__file__).resolve().parents[3]
    upstream_root = repo_root / "external" / "openvla-mini"
    if upstream_root.exists() and str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))

    # 1. PRISMATIC_DATA_ROOT environment variable
    if "PRISMATIC_DATA_ROOT" not in os.environ:
        data_root = Path.cwd() / "resources" / "data"
        data_root.mkdir(parents=True, exist_ok=True)
        os.environ["PRISMATIC_DATA_ROOT"] = str(data_root)

    # 2. Qwen2TokenizerFast import path shim for modern transformers
    try:
        import transformers
        if hasattr(transformers, "Qwen2TokenizerFast"):
            submod_name = "transformers.models.qwen2.tokenization_qwen2_fast"
            if submod_name not in sys.modules:
                mod = types.ModuleType(submod_name)
                mod.Qwen2TokenizerFast = transformers.Qwen2TokenizerFast
                sys.modules[submod_name] = mod
    except ImportError:
        pass

    # 3. Decouple inference from RLDS training dataset dependencies (materialize, dlimp, tensorflow)
    if "prismatic.vla.materialize" not in sys.modules:
        mat_mod = types.ModuleType("prismatic.vla.materialize")
        mat_mod.get_vla_dataset_and_collator = None
        sys.modules["prismatic.vla.materialize"] = mat_mod

    if "dlimp" not in sys.modules:
        dlimp_mod = types.ModuleType("dlimp")
        dlimp_trans = types.ModuleType("dlimp.transforms")
        dlimp_mod.transforms = dlimp_trans
        sys.modules["dlimp"] = dlimp_mod
        sys.modules["dlimp.transforms"] = dlimp_trans

    # 4. TIMM Vision Featurizer list/tuple unpacking compatibility shim
    # In newer versions of timm (0.9+), get_intermediate_layers returns List[torch.Tensor]
    # rather than Tuple[torch.Tensor]. Prismatic's unpack_tuple only checked isinstance(..., tuple),
    # causing featurizer output to remain a list and triggering TypeError in torch.cat.
    def safe_unpack_tuple(fn: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            res = fn(*args, **kwargs)
            return res[0] if isinstance(res, (tuple, list)) else res
        return wrapper

    try:
        import prismatic.models.backbones.vision.base_vision as base_vision
        base_vision.unpack_tuple = safe_unpack_tuple
        try:
            import prismatic.models.backbones.vision.dinosiglip_vit as dinosiglip_vit
            dinosiglip_vit.unpack_tuple = safe_unpack_tuple
        except ImportError:
            pass
    except ImportError:
        pass

    # 5. Direct-to-Device / VRAM Checkpoint Streaming Loader Shim
    # Prevents System RAM exhaustion and Windows crashes by loading checkpoint via
    # OS mmap and transferring submodule parameters directly to CUDA in bfloat16.
    try:
        import torch
        from prismatic.models.vlms.prismatic import PrismaticVLM
        if not hasattr(PrismaticVLM, "_prismatic_cuda_shim_applied"):
            def direct_device_from_pretrained(
                cls,
                pretrained_checkpoint,
                model_id,
                vision_backbone,
                llm_backbone,
                enable_mixed_precision_training=True,
                arch_specifier="gelu-mlp",
                freeze_weights=True,
                **kwargs,
            ):
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                vlm = cls(
                    model_id,
                    vision_backbone,
                    llm_backbone,
                    enable_mixed_precision_training=enable_mixed_precision_training,
                    arch_specifier=arch_specifier,
                    **kwargs,
                )

                # Ensure vision backbone featurizers unpack list returns correctly
                if hasattr(vlm, "vision_backbone"):
                    for feat_name in ["dino_featurizer", "siglip_featurizer", "featurizer"]:
                        feat = getattr(vlm.vision_backbone, feat_name, None)
                        if feat is not None and hasattr(feat, "forward"):
                            feat.forward = safe_unpack_tuple(feat.forward)

                # Move model to target device and half precision before loading weights
                dtype = vlm.llm_backbone.half_precision_dtype if device.type == "cuda" else torch.float32
                vlm.to(dtype=dtype, device=device)

                # Mmap checkpoint with zero system RAM inflation
                ckpt = torch.load(pretrained_checkpoint, map_location="cpu", mmap=True, weights_only=False)
                model_state_dict = ckpt["model"]

                # Load weights submodule-by-submodule to cap peak VRAM during load
                for module_name, module in [
                    ("projector", vlm.projector),
                    ("llm_backbone", vlm.llm_backbone),
                    ("vision_backbone", getattr(vlm, "vision_backbone", None)),
                ]:
                    if module is None or module_name not in model_state_dict:
                        continue
                    sub_dict = {k: v.to(device=device, dtype=dtype) for k, v in model_state_dict[module_name].items()}
                    module.load_state_dict(sub_dict)
                    del sub_dict
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                del model_state_dict, ckpt
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                if freeze_weights:
                    vlm.requires_grad_(False)
                    vlm.eval()

                return vlm

            PrismaticVLM.from_pretrained = classmethod(direct_device_from_pretrained)
            PrismaticVLM._prismatic_cuda_shim_applied = True
    except Exception:
        pass

    # 6. DynamicCache & Modern Transformers Generation Compatibility Shim
    # In transformers >= 4.40, generate() initializes an empty DynamicCache with len(pkv) > 0.
    # In upstream PrismaticVLM:
    #   if past_key_values: input_ids = input_ids[:, -1:]
    #   if input_ids.shape[1] == 1 and past_key_values is not None: return self.llm_backbone(...)
    # Because len(DynamicCache) > 0 on iteration 0, input_ids was truncated to 1 token and
    # forward() short-circuited to llm_backbone without ever processing vision patches or prompt.
    # This shim verifies whether cached tokens actually exist before truncating or short-circuiting.
    try:
        import torch
        from prismatic.models.vlms.prismatic import PrismaticVLM
        if not hasattr(PrismaticVLM, "_prismatic_dynamic_cache_shim_applied"):
            orig_forward = PrismaticVLM.forward

            def _has_cached_tokens(past_key_values: Any) -> bool:
                if past_key_values is None:
                    return False
                if hasattr(past_key_values, "get_seq_length"):
                    return past_key_values.get_seq_length() > 0
                if isinstance(past_key_values, tuple) and len(past_key_values) > 0:
                    return past_key_values[0] is not None
                return False

            def fixed_prepare_inputs_for_generation(
                self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.FloatTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                use_cache: Optional[bool] = None,
                **kwargs: torch.Tensor,
            ) -> Dict[str, torch.Tensor]:
                has_cached = _has_cached_tokens(past_key_values)
                if has_cached and input_ids is not None:
                    input_ids = input_ids[:, -1:]

                if inputs_embeds is not None and not has_cached:
                    model_inputs = {"inputs_embeds": inputs_embeds}
                else:
                    model_inputs = {"input_ids": input_ids}

                model_inputs.update(
                    {
                        "attention_mask": attention_mask,
                        "pixel_values": pixel_values,
                        "past_key_values": past_key_values,
                        "use_cache": use_cache,
                    }
                )
                return model_inputs

            def fixed_forward(
                self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                multimodal_indices: Optional[torch.LongTensor] = None,
            ):
                has_cached = _has_cached_tokens(past_key_values)
                if input_ids is not None and input_ids.shape[1] == 1 and has_cached:
                    return self.llm_backbone(
                        input_ids=input_ids,
                        attention_mask=None,
                        position_ids=None,
                        past_key_values=past_key_values,
                        inputs_embeds=None,
                        labels=None,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                    )
                return orig_forward(
                    self,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                    inputs_embeds=inputs_embeds,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    multimodal_indices=multimodal_indices,
                )

            PrismaticVLM.prepare_inputs_for_generation = fixed_prepare_inputs_for_generation
            PrismaticVLM.forward = fixed_forward
            PrismaticVLM._prismatic_dynamic_cache_shim_applied = True
    except Exception:
        pass
