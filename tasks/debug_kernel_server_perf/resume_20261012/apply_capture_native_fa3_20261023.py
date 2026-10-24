import argparse
import ast
from pathlib import Path


def patch_source(source):
    marker = "capture_mixed_attention_20261023 import capture_mixed_attention"
    if marker in source:
        return source
    anchor = '''            else:
                result = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=False if use_cascade_attn else causal,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    return_softmax_lse=use_cascade_attn or return_lse,'''
    inserted = '''            else:
                if (
                    causal and not use_cascade_attn and not use_local_attn
                    and not is_swa_layer and not layer.is_cross_attention
                    and not layer.logit_cap and not kwargs and window_size == (-1, -1)
                ):
                    from capture_mixed_attention_20261023 import capture_mixed_attention

                    capture_mixed_attention(
                        q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                        key_cache, value_cache, metadata, forward_batch, layer, self,
                    )
                result = flash_attn_with_kvcache('''
    if source.count(anchor) != 1:
        raise ValueError("Unexpected native FA3 capture seam")
    result = source.replace(anchor, anchor.replace(
        "            else:\n                result = flash_attn_with_kvcache(", inserted, 1
    ), 1)
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
