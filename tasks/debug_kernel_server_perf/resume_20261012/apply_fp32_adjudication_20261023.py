import argparse
import ast
from pathlib import Path


def patch_source(source):
    marker = "fp32_attention_adjudication_20261023 import reference_attention"
    if marker in source:
        return source
    anchor = '''            elif (
                getattr(metadata, "native_pdl_plan", None) is not None'''
    replacement = '''            elif (
                os.environ.get("SGLANG_NATIVE_PDL_FP32") == "1"
                and causal and not use_cascade_attn and not return_lse
                and not use_local_attn and not is_swa_layer
                and not layer.is_cross_attention and not layer.logit_cap
                and not kwargs and window_size == (-1, -1)
                and layer.head_dim == layer.v_head_dim == 128
                and q.dtype == key_cache.dtype == value_cache.dtype == torch.bfloat16
            ):
                from fp32_attention_adjudication_20261023 import reference_attention

                result = reference_attention(
                    q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    key_cache, value_cache, metadata, forward_batch, layer, output=_fa_out,
                )
''' + anchor
    if source.count(anchor) != 1:
        raise ValueError("Expected exactly one installed native PDL dispatch seam")
    result = source.replace(anchor, replacement, 1)
    ast.parse(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", type=Path)
    arguments = parser.parse_args()
    original = arguments.backend.read_text()
    patched = patch_source(original)
    if patched != original:
        arguments.backend.write_text(patched)
