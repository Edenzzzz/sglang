"""Install the opt-in PDL seam into the frozen native FA3 backend."""

import argparse
import ast
from pathlib import Path


def patch_source(source):
    marker = 'os.environ.get("SGLANG_NATIVE_PDL") == "1"'
    if marker in source:
        if (
            "prepare_native_pdl(self, forward_batch, metadata)" not in source
            or "metadata.native_pdl_plan.run(" not in source
            or "forward_batch.forward_mode in (ForwardMode.EXTEND, ForwardMode.MIXED)" not in source
            or 'os.environ.get("SGLANG_NATIVE_PDL_SERIAL")' not in source
            or 'os.environ.get("SGLANG_NATIVE_PDL_ORIGINAL_ROW_TILE")' not in source
        ):
            raise ValueError("Partially installed native PDL seam")
        return source
    imports = "from dataclasses import dataclass\n"
    constructor_anchor = "        self.has_softcap = _softcapping is not None and _softcapping > 0.0\n"
    constructor_replacement = constructor_anchor + '''        if os.environ.get("SGLANG_NATIVE_PDL") == "1":
            from native_fa3_fa2_pdl_20261023 import announce_backend

            announce_backend(self, __file__)
'''
    entry_anchor = "        metadata = FlashAttentionMetadata()\n        seqlens_in_batch = forward_batch.seq_lens\n"
    entry_replacement = '''        if os.environ.get("SGLANG_NATIVE_PDL") == "1":
            from native_fa3_fa2_pdl_20261023 import announce_metadata

            announce_metadata(self, forward_batch)
''' + entry_anchor
    metadata_anchor = "        self.forward_metadata = metadata\n\n    def get_paged_mha_kv_cache("
    metadata_replacement = '''        if (
            os.environ.get("SGLANG_NATIVE_PDL") == "1"
            and forward_batch.forward_mode in (ForwardMode.EXTEND, ForwardMode.MIXED)
            and not self.is_draft_runner
        ):
            from native_fa3_fa2_pdl_20261023 import prepare_native_pdl

            metadata.native_pdl_plan = prepare_native_pdl(self, forward_batch, metadata)
        self.forward_metadata = metadata

    def get_paged_mha_kv_cache('''
    dispatch_anchor = '''            else:
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
    dispatch_replacement = '''            elif (
                getattr(metadata, "native_pdl_plan", None) is not None
                and causal
                and not use_cascade_attn
                and not return_lse
                and not use_local_attn
                and not is_swa_layer
                and not layer.is_cross_attention
                and not layer.logit_cap
                and not kwargs
                and window_size == (-1, -1)
                and layer.head_dim == layer.v_head_dim == 128
                and q.dtype == key_cache.dtype == value_cache.dtype == torch.bfloat16
            ):
                result = metadata.native_pdl_plan.run(
                    query=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    key=key_cache, value=value_cache, scale=layer.scaling, output=_fa_out,
                    pdl=os.environ.get("SGLANG_NATIVE_PDL_SERIAL") != "1",
                    query_tile=0 if os.environ.get("SGLANG_NATIVE_PDL_ORIGINAL_ROW_TILE") == "1" else None,
                )
''' + dispatch_anchor
    for anchor in (imports, constructor_anchor, entry_anchor, metadata_anchor, dispatch_anchor):
        if source.count(anchor) != 1:
            raise ValueError(f"Unexpected native FA3 source anchor count: {source.count(anchor)}")
    result = source.replace(imports, imports + "import os\n", 1)
    result = result.replace(constructor_anchor, constructor_replacement, 1)
    result = result.replace(entry_anchor, entry_replacement, 1)
    result = result.replace(metadata_anchor, metadata_replacement, 1)
    result = result.replace(dispatch_anchor, dispatch_replacement, 1)
    ast.parse(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", type=Path)
    arguments = parser.parse_args()
    original = arguments.backend.read_text()
    result = patch_source(original)
    if result != original:
        arguments.backend.write_text(result)


if __name__ == "__main__":
    main()
