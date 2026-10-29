import torch


def verify_disputed_heads(
    query, keys, values, output, reference, query_offsets, page_offsets, kv_lengths,
    page_indices, disputed, scale, tolerance, causal=True, logits_soft_cap=0.0,
    key_scale=1.0, value_scale=1.0,
):
    """Adjudicate approximate-kernel disagreements using fp32 arithmetic without TF32."""
    query_offsets = query_offsets.cpu().tolist()
    page_offsets = page_offsets.cpu().tolist()
    kv_lengths = kv_lengths.cpu().tolist()
    group_size = query.shape[1] // keys.shape[-2]
    results = []
    for query_row, query_head in disputed.cpu().tolist():
        request = next(
            index for index in range(len(query_offsets) - 1)
            if query_offsets[index] <= query_row < query_offsets[index + 1]
        )
        query_length = query_offsets[request + 1] - query_offsets[request]
        causal_end = (
            kv_lengths[request] - query_length + query_row - query_offsets[request] + 1
        ) if causal else kv_lengths[request]
        causal_end = max(0, causal_end)
        indices = page_indices[page_offsets[request] : page_offsets[request + 1]].long()
        kv_head = query_head // group_size
        key_head = keys.select(-2, kv_head).index_select(0, indices).flatten(0, 1)[:causal_end].float()
        value_head = values.select(-2, kv_head).index_select(0, indices).flatten(0, 1)[:causal_end].float()
        scores = (key_head * query[query_row, query_head].float()).sum(dim=-1) * scale * key_scale
        if logits_soft_cap:
            scores = logits_soft_cap * torch.tanh(scores / logits_soft_cap)
        expected = (scores.softmax(0)[:, None] * value_head).sum(dim=0)
        expected = expected * value_scale
        denominator = expected.norm().clamp_min(1e-6)
        fused_error = float((output[query_row, query_head].float() - expected).norm() / denominator)
        stock_error = float((reference[query_row, query_head].float() - expected).norm() / denominator)
        results.append({
            "query_row": query_row,
            "query_head": query_head,
            "fused_relative_error": fused_error,
            "stock_relative_error": stock_error,
            "passed": bool(torch.isfinite(expected).all()) and fused_error <= tolerance,
        })
        if not results[-1]["passed"]:
            break
    return results
