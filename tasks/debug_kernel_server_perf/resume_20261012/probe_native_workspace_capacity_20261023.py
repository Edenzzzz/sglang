"""Metadata-only capacity checks; these are not performance workloads."""

import itertools
import json
from pathlib import Path

import torch

from native_fa3_fa2_pdl_20261023 import NativePdlExecutor


def main():
    records = []
    for rows, context in ((24, 65536), (47, 524288)):
        queries = [128] + [4] * rows
        prefixes = [0] + [context - 4] * rows
        pages = torch.arange(context, device="cuda", dtype=torch.int32).expand(rows + 1, -1)
        lengths = torch.tensor([128] + [context] * rows, device="cuda", dtype=torch.int32)
        cumulative = torch.tensor([0, *itertools.accumulate(queries)], device="cuda", dtype=torch.int32)
        executor = NativePdlExecutor(28, 4, torch.device("cuda"), adaptive_splits=True, policy_version="geometry_v3", fp32_partials=True, direct_bf16_output=True, workspace_bytes=512 << 20)
        try:
            plan = executor.plan(queries, prefixes, pages, lengths, cumulative)
            record = dict(rows=rows, kv_length=context, planned=True, plan=list(executor.wrapper._plan_info))
        except Exception as error:
            record = dict(rows=rows, kv_length=context, planned=False, error=str(error))
        records.append(record)
        print(json.dumps(record), flush=True)
        del executor, pages, lengths, cumulative
        torch.cuda.empty_cache()
    Path(__file__).with_name("results.json").write_text(json.dumps(dict(scope=__doc__, records=records), indent=2) + "\n")


if __name__ == "__main__":
    main()
