import argparse
import ast
from pathlib import Path


def patch_source(source):
    if "native_pdl_model_shadow_20261023 import reference_forwards" in source:
        if "check_committed_sample(native_report, ver_result, pre_next_token_ids)" not in source:
            raise ValueError("Partially installed model shadow")
        return source
    anchor = '''        batch_output = self.target_worker.forward_batch_generation(
            batch=None, forward_batch=fb, is_verify=True, pp_proxy_tensors=pp_proxy_tensors
        )
        if debug_mix_sync:'''
    replacement = '''        from native_pdl_model_shadow_20261023 import reference_forwards, compare_logits, check_committed_sample

        native_references = reference_forwards(self.target_worker, fb, pp_proxy_tensors)
        batch_output = self.target_worker.forward_batch_generation(
            batch=None, forward_batch=fb, is_verify=True, pp_proxy_tensors=pp_proxy_tensors
        )
        native_report = compare_logits(
            native_references, batch_output, fb, n_pre_rows, running, verify_input,
            self.target_worker, merged.mixed_pre_sampling_info,
        )
        if debug_mix_sync:'''
    if source.count(anchor) != 1:
        raise ValueError("Unexpected mixed-forward source")
    result = source.replace(anchor, replacement, 1)
    committed_anchor = "        finally:\n            fb.sampling_info = _saved_si\n        result = GenerationBatchResult(\n"
    if result.count(committed_anchor) != 1:
        raise ValueError("Unexpected post-verify sampling seam")
    result = result.replace(
        committed_anchor,
        committed_anchor.replace(
            "        result = GenerationBatchResult(\n",
            "        check_committed_sample(native_report, ver_result, pre_next_token_ids)\n        result = GenerationBatchResult(\n",
        ),
        1,
    )
    ast.parse(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("worker", type=Path)
    arguments = parser.parse_args()
    original = arguments.worker.read_text()
    patched = patch_source(original)
    if patched != original:
        arguments.worker.write_text(patched)
