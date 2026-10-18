"""Let the FA3/FA2 split serve layers whose KV scales are the identity (FP8 linear layers, BF16 KV cache).

Under --quantization fp8 the attention layers carry k_scale_float = v_scale_float = 1.0 while the KV cache stays bf16, and
_fused_split_dispatch refused any layer with a scale set (its TODO(fp8kv) guard), so every layer took the single-wrapper
path and the 4-mode serving arm ran without the split (no FUSED_SPLIT_CALLS; 2026-10-18 FP8 A/B). An identity scale on a
bf16 cache is a no-op for both kernels; only a non-unit scale or an fp8 KV cache still needs the unverified descale
convention, and those stay refused.
  python apply_split_unit_kv_scale.py <tree>/python/sglang/srt
"""
import sys

p = f"{sys.argv[1]}/layers/attention/flashinfer_backend.py"
s = open(p).read()
old = '''        if (
            layer.sliding_window_size is not None and layer.sliding_window_size > -1
        ) or layer.k_scale_float is not None or layer.v_scale_float is not None:
            return None'''
new = '''        kv_scaled = any(
            scale is not None and scale != 1.0
            for scale in (layer.k_scale_float, layer.v_scale_float)
        ) or self.flashinfer_kv_cache_dtype not in (torch.bfloat16, torch.float16)
        if (
            layer.sliding_window_size is not None and layer.sliding_window_size > -1
        ) or kv_scaled:
            return None'''
if new in s:
    print("already")
else:
    assert s.count(old) == 1, s.count(old)
    open(p, "w").write(s.replace(old, new, 1))
    print("patched")
