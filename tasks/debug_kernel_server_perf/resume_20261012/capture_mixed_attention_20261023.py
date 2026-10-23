"""Capture real mixed-forward attention inputs without changing the attention result."""

import hashlib
import json
import os
from pathlib import Path
import time

import torch

from native_fa3_fa2_pdl_20261023 import partition_lengths


_states = {}
_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def verify_tail_rows(query_lengths, verify_tokens, width=4):
    if verify_tokens <= 0 or verify_tokens % width:
        raise ValueError("Invalid captured verify width")
    rows = verify_tokens // width
    if rows >= len(query_lengths) or query_lengths[-rows:] != [width] * rows:
        raise ValueError("Captured batch does not end in the actual verify rows")
    return rows


def capture_mixed_attention(query, key, value, metadata, forward_batch, layer, backend):
    root_name = os.environ.get("SGLANG_CAPTURE_MIXED_ATTN")
    if not root_name or layer.layer_id != int(os.environ.get("CAPTURE_LAYER", "14")):
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank != int(os.environ.get("CAPTURE_RANK", "0")):
        return
    barrier = Path(root_name).parent / "requests/barrier/go"
    if not barrier.exists():
        return
    if backend.is_draft_runner or backend.page_size != 1 or query.dtype != torch.bfloat16:
        return
    queries = forward_batch.extend_seq_lens_cpu
    prefixes = forward_batch.extend_prefix_lens_cpu
    if not queries or prefixes is None:
        return
    mode = os.environ["CAPTURE_MODE"]
    verify_tokens = int(getattr(forward_batch, "mixed_verify_tokens", 0))
    if mode == "four" and verify_tokens <= 0:
        return
    if mode == "two" and verify_tokens:
        raise RuntimeError("Speculative tokens appeared in the two-mode capture")
    packed_query_limit = int(os.environ.get("CAPTURE_PACKED_QUERY_LIMIT", "32"))
    shape = partition_lengths(queries, prefixes, layer.tp_q_head_num, layer.tp_k_head_num, packed_query_limit)
    if shape is None:
        return
    row_queries = queries[shape.prefix_rows:]
    if mode == "two" and any(length != 1 for length in row_queries):
        return
    verify_rows = verify_tail_rows(queries, verify_tokens) if mode == "four" else 0
    if verify_rows > len(row_queries):
        raise RuntimeError("Kernel row partition omits actual verify rows")
    row_context = max(shape.kv_lengths[shape.prefix_rows:])
    if row_context < int(os.environ["CAPTURE_MIN_ROW_CONTEXT"]):
        return
    root = Path(root_name)
    state = _states.setdefault(root_name, {"index": 0, "saved": set()})
    state["index"] += 1
    prefix_class = "fresh" if max(prefixes[:shape.prefix_rows]) < 4096 else "cached"
    query_class = "small" if shape.prefix_tokens <= 512 else "large"
    category = prefix_class + "_" + query_class
    geometry = dict(
        ordinal=state["index"], mode=mode, category=category, layer=layer.layer_id,
        query_lengths=list(queries), prefix_lengths=list(prefixes),
        kv_lengths=shape.kv_lengths, prefix_rows=shape.prefix_rows,
        prefix_tokens=shape.prefix_tokens, row_query_max=shape.maximum_row_query,
        row_context_max=row_context, verify_tokens=verify_tokens,
        verify_rows=verify_rows,
        small_prefill_rows_in_fa2=len(row_queries) - verify_rows if mode == "four" else None,
        heads=layer.tp_q_head_num, kv_heads=layer.tp_k_head_num,
        rank=rank, packed_query_limit=packed_query_limit,
        scale=float(layer.scaling), timestamp=time.time(),
    )
    root.mkdir(parents=True, exist_ok=True)
    with (root / "census.jsonl").open("a") as census:
        census.write(json.dumps(geometry) + "\n")
    # CAPTURE_EVERY=N (trace-proportional mode, 2026-10-23): save every Nth census step whatever its bucket, so the
    # saved set reproduces the trace's own shape distribution instead of one first-eligible step per bucket. Capped at
    # CAPTURE_MAX files. Unset, the original first-per-bucket behaviour is unchanged.
    every = int(os.environ.get("CAPTURE_EVERY", "0"))
    if every:
        if state["index"] % every != 0 or len(state["saved"]) >= int(os.environ.get("CAPTURE_MAX", "24")):
            return
        save_key = f"s{state['index']:05d}_{category}"
    else:
        if category in state["saved"]:
            return
        save_key = category
    cache_lengths = metadata.cache_seqlens_int32.cpu()
    if cache_lengths.tolist() != shape.kv_lengths:
        raise RuntimeError("Capture host and device KV lengths disagree")
    cumulative_query = metadata.cu_seqlens_q.cpu()
    if cumulative_query[-1].item() != query.shape[0]:
        raise RuntimeError("Capture contains padded or unaccounted query rows")
    page_table = metadata.page_table[:, :max(shape.kv_lengths)].clone()
    live = []
    for index, length in enumerate(shape.kv_lengths):
        live.append(page_table[index, :length])
        page_table[index, length:] = 0
    physical_ids = torch.unique(torch.cat(live), sorted=True).long()
    geometry.update(
        key_shape=list(key.shape), key_stride=list(key.stride()),
        value_shape=list(value.shape), value_stride=list(value.stride()),
        live_pages=physical_ids.numel(), query_stride=list(query.stride()),
        source="Actual model forward, before native FA3 attention; original physical page IDs and strides retained",
        workload_release_time=barrier.stat().st_mtime,
        capture_source_sha256=_source_sha256,
    )
    captured = dict(
        geometry=geometry, query=query.detach().cpu(),
        key=key.index_select(0, physical_ids).detach().cpu(),
        value=value.index_select(0, physical_ids).detach().cpu(),
        physical_ids=physical_ids.cpu(), page_table=page_table.cpu(),
        cache_lengths=cache_lengths, cumulative_query=cumulative_query,
    )
    destination = root / (save_key + ".pt")
    if destination.exists():
        raise RuntimeError("Refusing to overwrite a completed capture")
    temporary = destination.with_suffix(".tmp")
    torch.save(captured, temporary)
    temporary.replace(destination)
    digest = hashlib.file_digest(destination.open("rb"), "sha256").hexdigest()
    (destination.with_suffix(".json")).write_text(json.dumps(
        dict(**geometry, file=destination.name, sha256=digest, bytes=destination.stat().st_size), indent=2
    ) + "\n")
    state["saved"].add(save_key)
    print("CAPTURE_MIXED_SAVED", mode, category, geometry["query_lengths"], geometry["kv_lengths"], digest, flush=True)
