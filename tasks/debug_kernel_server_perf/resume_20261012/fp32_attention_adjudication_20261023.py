"""High-precision attention for same-input target-forward adjudication only."""

import copy
import os

import torch


_layers = []


def reset_layer_trace():
    _layers.clear()


def layer_trace():
    return list(_layers)


def causal_fp32(query, key, value, page_table, query_lengths, kv_lengths, scale, query_chunk=32):
    if len(query_lengths) != len(kv_lengths) or len(query_lengths) != page_table.shape[0]:
        raise ValueError("Reference batch lengths disagree")
    if sum(query_lengths) != query.shape[0] or any(query_length <= 0 or kv_length < query_length for query_length, kv_length in zip(query_lengths, kv_lengths)):
        raise ValueError("Invalid causal reference lengths")
    if key.ndim == 3:
        key, value = key.unsqueeze(1), value.unsqueeze(1)
    if key.shape[1] != 1 or key.shape != value.shape or query.shape[1] % key.shape[2]:
        raise ValueError("Reference requires page-one GQA caches")
    expected = torch.empty(query.shape, device=query.device, dtype=torch.float32)
    group = query.shape[1] // key.shape[2]
    query_begin = 0
    for request, (query_length, kv_length) in enumerate(zip(query_lengths, kv_lengths)):
        pages = page_table[request, :kv_length].long()
        for kv_head in range(key.shape[2]):
            request_key = key[pages, 0, kv_head].float()
            request_value = value[pages, 0, kv_head].float()
            head_begin, head_end = kv_head * group, (kv_head + 1) * group
            for begin in range(0, query_length, query_chunk):
                end = min(begin + query_chunk, query_length)
                request_query = query[query_begin + begin:query_begin + end, head_begin:head_end].float()
                logits = torch.einsum("thd,kd->htk", request_query, request_key) * scale
                mask = torch.arange(kv_length, device=query.device)[None, :] > (
                    kv_length - query_length + torch.arange(begin, end, device=query.device)[:, None]
                )
                logits.masked_fill_(mask, -float("inf"))
                expected[query_begin + begin:query_begin + end, head_begin:head_end] = torch.einsum(
                    "htk,kd->thd", logits.softmax(dim=-1), request_value
                )
        query_begin += query_length
    return expected


def reference_attention(query, key, value, metadata, forward_batch, layer, output=None):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("FP32 adjudication requires eager attention")
    queries = forward_batch.extend_seq_lens_cpu
    prefixes = forward_batch.extend_prefix_lens_cpu
    if queries is None or prefixes is None or len(queries) != len(prefixes):
        raise RuntimeError("FP32 adjudication needs exact host lengths")
    lengths = [prefix + query_length for prefix, query_length in zip(prefixes, queries)]
    if not _layers:
        if metadata.cache_seqlens_int32.cpu().tolist() != lengths:
            raise RuntimeError("FP32 reference host/device KV mismatch")
        boundaries = [0]
        for length in queries:
            boundaries.append(boundaries[-1] + length)
        if metadata.cu_seqlens_q.cpu().tolist() != boundaries:
            raise RuntimeError("FP32 reference host/device query mismatch")
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        expected = causal_fp32(query, key, value, metadata.page_table, queries, lengths, layer.scaling)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    _layers.append(int(layer.layer_id))
    if output is None:
        return expected.to(query.dtype)
    output.copy_(expected)
    return output


def target_reference(worker, forward_batch):
    reset_layer_trace()
    result = worker.forward_batch_generation(batch=None, forward_batch=copy.copy(forward_batch), is_verify=True)
    expected_layers = int(worker.model_runner.model_config.num_hidden_layers)
    if layer_trace() != list(range(expected_layers)):
        raise RuntimeError(f"Incomplete FP32 target forward: {layer_trace()}")
    logits = result.logits_output.next_token_logits.detach().clone()
    hidden = result.logits_output.hidden_states
    if hidden is None:
        raise RuntimeError("FP32 target reference did not expose hidden states")
    return logits, hidden.detach().clone(), layer_trace()


def golden_and_restore(worker, forward_batch, candidate):
    if os.environ.get("SGLANG_NATIVE_PDL") != "1":
        raise RuntimeError("Adjudication requires the native candidate to remain enabled")
    original_logits = candidate.logits_output.next_token_logits.detach().clone()
    original_hidden = candidate.logits_output.hidden_states.detach().clone()
    prior = os.environ.get("SGLANG_NATIVE_PDL_FP32")
    try:
        os.environ["SGLANG_NATIVE_PDL_FP32"] = "1"
        golden, golden_hidden, layers = target_reference(worker, forward_batch)
        repeated, repeated_hidden, repeated_layers = target_reference(worker, forward_batch)
        if not torch.equal(golden, repeated) or not torch.equal(golden_hidden, repeated_hidden):
            raise RuntimeError("FP32 target references are not repeatable")
    finally:
        if prior is None:
            os.environ.pop("SGLANG_NATIVE_PDL_FP32", None)
        else:
            os.environ["SGLANG_NATIVE_PDL_FP32"] = prior
    restored = worker.forward_batch_generation(batch=None, forward_batch=copy.copy(forward_batch), is_verify=True)
    if not torch.equal(original_logits, restored.logits_output.next_token_logits):
        raise RuntimeError("Native logits changed after FP32 KV restoration")
    if not torch.equal(original_hidden, restored.logits_output.hidden_states):
        raise RuntimeError("Native hidden states changed after FP32 KV restoration")
    if not torch.equal(original_logits, candidate.logits_output.next_token_logits) or not torch.equal(original_hidden, candidate.logits_output.hidden_states):
        raise RuntimeError("FP32 probe changed retained production outputs")
    return golden, dict(
        scope="FP32 attention with BF16 output boundaries and unchanged BF16 GEMMs, using the same live prefix KV",
        fp32_layers=layers, repeated_fp32_layers=repeated_layers,
        fp32_logits_repeat_exact=True, fp32_hidden_repeat_exact=True,
        native_logits_restored_exact=True, native_hidden_restored_exact=True,
    )
