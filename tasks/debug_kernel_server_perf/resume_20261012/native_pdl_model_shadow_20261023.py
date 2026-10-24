"""Diagnostic same-batch target-logit comparison; never used for timings."""

import copy
import json
import logging
import os
from pathlib import Path

import torch


logger = logging.getLogger(__name__)
_forward_ordinal = 0


def probability_metrics(reference, actual):
    reference = reference.float()
    actual = actual.float()
    reference_logprob = reference.log_softmax(dim=-1)
    actual_logprob = actual.log_softmax(dim=-1)
    reference_prob = reference_logprob.exp()
    actual_prob = actual_logprob.exp()
    divergence = (reference_prob * (reference_logprob - actual_logprob)).sum(dim=-1)
    variation = (reference_prob - actual_prob).abs().sum(dim=-1) * 0.5
    reference_centered = reference - reference.mean(dim=-1, keepdim=True)
    actual_centered = actual - actual.mean(dim=-1, keepdim=True)
    centered_relative = (reference_centered - actual_centered).norm(dim=-1) / reference_centered.norm(dim=-1).clamp_min(1e-6)
    return dict(
        mean_kl_ref_to_candidate=float(divergence.mean()), max_kl_ref_to_candidate=float(divergence.max()),
        mean_total_variation=float(variation.mean()), max_total_variation=float(variation.max()),
        worst_centered_relative=float(centered_relative.max()),
    )


def decode_acceptance(predict, accept_index, accept_lens):
    if len(accept_index) != len(accept_lens):
        raise ValueError("Acceptance row counts differ")
    emitted, selected = [], []
    for indices, count in zip(accept_index, accept_lens):
        if type(count) is not int or not 1 <= count <= len(indices):
            raise ValueError("Invalid accepted-token count")
        live = indices[:count]
        if any(type(index) is not int or not 0 <= index < len(predict) for index in live):
            raise ValueError("Invalid accepted-token index")
        emitted.append([predict[index] for index in live])
        selected.append(live)
    return dict(tokens=emitted, indices=selected, accept_lens=accept_lens)


def greedy_trace(logits, prototype, running, verify_input, prefill_rows):
    from sglang.srt.speculative.eagle_utils import eagle_sample

    if not running.sampling_info.is_all_greedy or running.has_grammar:
        raise RuntimeError("Shadow sampling requires greedy requests without grammars")
    sample_output = copy.copy(prototype)
    sample_output.next_token_logits = logits[prefill_rows:].clone()
    sample_output.hidden_states = None
    predict, accepted, indices = eagle_sample(verify_input, running, sample_output)
    trace = decode_acceptance(predict.cpu().tolist(), indices.cpu().tolist(), accepted.cpu().tolist())
    trace["penalized_argmax"] = sample_output.next_token_logits.argmax(dim=-1).cpu().tolist()
    return trace


def prefill_trace(logits, prototype, forward_batch, prefill_rows, worker, sampling_info):
    if sampling_info is None or not sampling_info.is_all_greedy:
        raise RuntimeError("Prefill shadow requires an explicit greedy sampling snapshot")
    if getattr(sampling_info, "penalizer_orchestrator", None) is not None:
        raise RuntimeError("Prefill shadow cannot consume a live penalty orchestrator")
    runner = worker.model_runner
    if runner.sampling_observer is not None and runner.sampling_observer.is_active(sampling_info):
        raise RuntimeError("Prefill shadow does not support active sampling observers")
    isolated_info = copy.copy(sampling_info)
    isolated_output = copy.copy(prototype)
    isolated_output.next_token_logits = logits[:prefill_rows].clone()
    isolated_output.hidden_states = None
    isolated_output.auxiliary_device_output = None
    runner._preprocess_logits(isolated_output, isolated_info)
    predicted = runner.sampler(
        isolated_output, isolated_info, forward_batch.return_logprob,
        forward_batch.top_logprobs_nums, forward_batch.token_ids_logprobs,
        forward_batch.seq_lens - 1,
    )
    return predicted.cpu().tolist()


def write_report(report):
    with Path(os.environ["NATIVE_PDL_MODEL_SHADOW"]).open("a") as output:
        output.write(json.dumps(report) + "\n")


def reference_forwards(worker, forward_batch, pp_proxy_tensors):
    destination = os.environ.get("NATIVE_PDL_MODEL_SHADOW")
    if not destination:
        return None
    if torch.cuda.is_current_stream_capturing() or forward_batch.forward_metadata_ready:
        raise RuntimeError("Model shadow requires a fresh eager mixed forward")
    if not getattr(forward_batch, "mixed_verify_tokens", 0):
        raise RuntimeError("Model shadow requires actual merged verify rows")
    enabled = os.environ.get("SGLANG_NATIVE_PDL")
    if enabled != "1":
        raise RuntimeError("Model shadow must precede an enabled native PDL candidate")
    references = []
    previous_serial = os.environ.get("SGLANG_NATIVE_PDL_SERIAL")
    previous_tile = os.environ.get("SGLANG_NATIVE_PDL_ORIGINAL_ROW_TILE")
    previous_partials = os.environ.get("SGLANG_NATIVE_PDL_FP32_PARTIALS")
    previous_direct = os.environ.get("SGLANG_NATIVE_PDL_DIRECT_BF16_OUTPUT")
    original_tile_logits = None
    bf16_partial_logits = None
    try:
        os.environ["SGLANG_NATIVE_PDL"] = "0"
        for repeat in range(2):
            repeated_batch = copy.copy(forward_batch)
            result = worker.forward_batch_generation(
                batch=None, forward_batch=repeated_batch, is_verify=True,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            references.append(result.logits_output.next_token_logits.detach().clone())
        os.environ["SGLANG_NATIVE_PDL"] = "1"
        os.environ["SGLANG_NATIVE_PDL_SERIAL"] = "1"
        if os.environ.get("NATIVE_PDL_BF16_PARTIAL_CONTROL") == "1":
            if previous_partials != "1":
                raise RuntimeError("BF16 partial control requires an FP32-partial candidate")
            os.environ["SGLANG_NATIVE_PDL_FP32_PARTIALS"] = "0"
            os.environ["SGLANG_NATIVE_PDL_DIRECT_BF16_OUTPUT"] = "0"
            original = worker.forward_batch_generation(
                batch=None, forward_batch=copy.copy(forward_batch), is_verify=True,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            bf16_partial_logits = original.logits_output.next_token_logits.detach().clone()
            os.environ["SGLANG_NATIVE_PDL_FP32_PARTIALS"] = previous_partials
            if previous_direct is None:
                os.environ.pop("SGLANG_NATIVE_PDL_DIRECT_BF16_OUTPUT", None)
            else:
                os.environ["SGLANG_NATIVE_PDL_DIRECT_BF16_OUTPUT"] = previous_direct
        if os.environ.get("NATIVE_PDL_ORIGINAL_ROW_CONTROL") == "1":
            os.environ["SGLANG_NATIVE_PDL_ORIGINAL_ROW_TILE"] = "1"
            original = worker.forward_batch_generation(
                batch=None, forward_batch=copy.copy(forward_batch), is_verify=True,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            original_tile_logits = original.logits_output.next_token_logits.detach().clone()
            if previous_tile is None:
                os.environ.pop("SGLANG_NATIVE_PDL_ORIGINAL_ROW_TILE", None)
            else:
                os.environ["SGLANG_NATIVE_PDL_ORIGINAL_ROW_TILE"] = previous_tile
        if os.environ.get("NATIVE_PDL_SHADOW_SERIAL_SEPARATE_CONVERSION") == "1":
            if previous_direct != "1" or previous_partials != "1":
                raise RuntimeError("Separate-conversion control requires direct FP32 partials")
            os.environ["SGLANG_NATIVE_PDL_DIRECT_BF16_OUTPUT"] = "0"
        result = worker.forward_batch_generation(
            batch=None, forward_batch=copy.copy(forward_batch), is_verify=True,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        references.append(result.logits_output.next_token_logits.detach().clone())
        if original_tile_logits is not None or bf16_partial_logits is not None:
            references.append(original_tile_logits)
        if bf16_partial_logits is not None:
            references.append(bf16_partial_logits)
    finally:
        os.environ["SGLANG_NATIVE_PDL"] = enabled
        if previous_serial is None:
            os.environ.pop("SGLANG_NATIVE_PDL_SERIAL", None)
        else:
            os.environ["SGLANG_NATIVE_PDL_SERIAL"] = previous_serial
        if previous_tile is None:
            os.environ.pop("SGLANG_NATIVE_PDL_ORIGINAL_ROW_TILE", None)
        else:
            os.environ["SGLANG_NATIVE_PDL_ORIGINAL_ROW_TILE"] = previous_tile
        if previous_partials is None:
            os.environ.pop("SGLANG_NATIVE_PDL_FP32_PARTIALS", None)
        else:
            os.environ["SGLANG_NATIVE_PDL_FP32_PARTIALS"] = previous_partials
        if previous_direct is None:
            os.environ.pop("SGLANG_NATIVE_PDL_DIRECT_BF16_OUTPUT", None)
        else:
            os.environ["SGLANG_NATIVE_PDL_DIRECT_BF16_OUTPUT"] = previous_direct
    return references


def compare_logits(references, candidate, forward_batch, prefill_rows, running=None, verify_input=None, worker=None, prefill_sampling_info=None):
    if references is None:
        return
    global _forward_ordinal
    _forward_ordinal += 1
    reference, repeated, serial = references[:3]
    original_tile = references[3] if len(references) >= 4 else None
    bf16_partial = references[4] if len(references) == 5 else None
    actual = candidate.logits_output.next_token_logits
    if reference.shape != actual.shape or reference.shape != repeated.shape or reference.shape != serial.shape:
        raise RuntimeError("Model shadow logit shapes differ")
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(repeated).all() and torch.isfinite(actual).all())
    reference_float, actual_float = reference.float(), actual.float()
    relative = (reference_float - actual_float).norm(dim=-1) / reference_float.norm(dim=-1).clamp_min(1e-6)
    reference_values, reference_ids = reference_float.topk(2, dim=-1)
    actual_values, actual_ids = actual_float.topk(2, dim=-1)
    reference_argmax = reference_float.argmax(dim=-1)
    actual_argmax = actual_float.argmax(dim=-1)
    changed = (reference_argmax != actual_argmax).nonzero().flatten().tolist()
    differences = (reference_float - actual_float).abs().amax(dim=-1)
    report = dict(
        forward_ordinal=_forward_ordinal,
        rows=reference.shape[0], prefill_rows=prefill_rows,
        verify_tokens=forward_batch.mixed_verify_tokens,
        query_lengths=forward_batch.extend_seq_lens_cpu,
        prefix_lengths=forward_batch.extend_prefix_lens_cpu,
        finite=finite, baseline_repeat_exact=torch.equal(reference, repeated),
        baseline_repeat_max_abs=float((reference.float() - repeated.float()).abs().max()),
        native_serial_exact=torch.equal(serial, actual),
        native_serial_max_abs=float((serial.float() - actual.float()).abs().max()),
        serial_control="separate_fp32_conversion" if os.environ.get("NATIVE_PDL_SHADOW_SERIAL_SEPARATE_CONVERSION") == "1" else "matched_native",
        worst_relative=float(relative.max()), maximum_abs=float(differences.max()),
        model_relative_within_0p02=bool(relative.max() < 0.02),
        argmax_flips=len(changed), prefill_argmax_flips=sum(index < prefill_rows for index in changed),
        flipped_rows=[dict(
            row=index, reference_ids=reference_ids[index].tolist(),
            reference_argmax=int(reference_argmax[index]), candidate_argmax=int(actual_argmax[index]),
            reference_logits=reference_values[index].tolist(), candidate_ids=actual_ids[index].tolist(),
            candidate_logits=actual_values[index].tolist(), maximum_abs=float(differences[index]),
        ) for index in changed],
    )
    report.update(probability_metrics(reference_float, actual_float))
    if original_tile is not None:
        report["original_row_tile_control"] = dict(
            exact_to_native=torch.equal(original_tile, actual),
            prefix_exact_to_native=torch.equal(original_tile[:prefill_rows], actual[:prefill_rows]),
            argmax_diff_from_native=int((original_tile.argmax(-1) != actual.argmax(-1)).sum()),
            metrics_vs_native=probability_metrics(original_tile, actual),
        )
    if bf16_partial is not None:
        report["bf16_partial_control"] = dict(
            exact_to_native=torch.equal(bf16_partial, actual),
            argmax_diff_from_native=int((bf16_partial.argmax(-1) != actual.argmax(-1)).sum()),
            metrics_vs_native=probability_metrics(bf16_partial, actual),
        )
    if running is not None:
        traces = {
            name: greedy_trace(logits, candidate.logits_output, running, verify_input, prefill_rows)
            for name, logits in (("reference", reference), ("serial", serial), ("native", actual))
        }
        consumed = {
            index for name in ("reference", "native")
            for indices in traces[name]["indices"] for index in indices
        }
        report["greedy_sampling"] = traces
        if original_tile is not None:
            traces["original_tile"] = greedy_trace(original_tile, candidate.logits_output, running, verify_input, prefill_rows)
            report["original_row_tile_control"]["emission_changes_from_reference"] = sum(
                expected != observed for expected, observed in zip(traces["reference"]["tokens"], traces["original_tile"]["tokens"])
            )
        if bf16_partial is not None:
            traces["bf16_partial"] = greedy_trace(bf16_partial, candidate.logits_output, running, verify_input, prefill_rows)
            report["bf16_partial_control"]["emission_changes_from_reference"] = sum(
                expected != observed for expected, observed in zip(traces["reference"]["tokens"], traces["bf16_partial"]["tokens"])
            )
        prefill = {
            name: prefill_trace(logits, candidate.logits_output, forward_batch, prefill_rows, worker, prefill_sampling_info)
            for name, logits in (("reference", reference), ("native", actual))
        }
        report["prefill_sampling"] = prefill
        report["prefill_requests_with_changed_emission"] = sum(
            expected != observed for expected, observed in zip(prefill["reference"], prefill["native"])
        )
        report["verify_requests_with_changed_emission"] = sum(
            expected != observed for expected, observed in zip(traces["reference"]["tokens"], traces["native"]["tokens"])
        )
        report["penalized_argmax_flips_consumed"] = sum(
            traces["reference"]["penalized_argmax"][index] != traces["native"]["penalized_argmax"][index]
            for index in consumed
        )
        if traces["serial"] != traces["native"]:
            write_report(report)
            raise RuntimeError("Serial and PDL greedy traces differ despite identical logits")
        disputed = bool(
            report["verify_requests_with_changed_emission"] or report["prefill_requests_with_changed_emission"]
            or report.get("original_row_tile_control", {}).get("emission_changes_from_reference", 0)
            or report.get("bf16_partial_control", {}).get("emission_changes_from_reference", 0)
        )
        if os.environ.get("NATIVE_PDL_FP32_ADJUDICATE") == "1" and (disputed or _forward_ordinal % 16 == 1):
            from fp32_attention_adjudication_20261023 import golden_and_restore

            golden, adjudication = golden_and_restore(worker, forward_batch, candidate)
            if not torch.isfinite(golden).all():
                raise RuntimeError("FP32 target reference has nonfinite logits")
            golden_verify = greedy_trace(golden, candidate.logits_output, running, verify_input, prefill_rows)
            golden_prefill = prefill_trace(golden, candidate.logits_output, forward_batch, prefill_rows, worker, prefill_sampling_info)
            adjudication.update(
                reason="emitted_token_disagreement" if disputed else "periodic_control",
                fa3_vs_fp32=probability_metrics(golden, reference),
                native_vs_fp32=probability_metrics(golden, actual),
                golden_verify=golden_verify, golden_prefill=golden_prefill,
                fa3_raw_argmax_diff_from_fp32=int((reference.argmax(-1) != golden.argmax(-1)).sum()),
                native_raw_argmax_diff_from_fp32=int((actual.argmax(-1) != golden.argmax(-1)).sum()),
            )
            if original_tile is not None:
                adjudication["original_row_tile_vs_fp32"] = probability_metrics(golden, original_tile)
                adjudication["original_row_tile_argmax_diff_from_fp32"] = int((original_tile.argmax(-1) != golden.argmax(-1)).sum())
            if bf16_partial is not None:
                adjudication["bf16_partial_vs_fp32"] = probability_metrics(golden, bf16_partial)
                adjudication["bf16_partial_argmax_diff_from_fp32"] = int((bf16_partial.argmax(-1) != golden.argmax(-1)).sum())
                adjudication["verify_emission_changes_from_fp32"] = {
                    name: sum(expected != observed for expected, observed in zip(golden_verify["tokens"], traces[name]["tokens"]))
                    for name in ("reference", "native", "bf16_partial")
                }
            decisions = []
            for index, (expected, observed, gold) in enumerate(zip(traces["reference"]["tokens"], traces["native"]["tokens"], golden_verify["tokens"])):
                if expected != observed:
                    decisions.append(dict(request_index=index, reference=expected, native=observed, fp32=gold,
                                          agrees_with_fa3=gold == expected, agrees_with_native=gold == observed))
            adjudication["disputed_verify_decisions"] = decisions
            report["fp32_adjudication"] = adjudication
            logger.warning("FP32_ADJUDICATION ordinal=%d reason=%s disputes=%d fa3_kl=%.6g native_kl=%.6g",
                           _forward_ordinal, adjudication["reason"], len(decisions),
                           adjudication["fa3_vs_fp32"]["mean_kl_ref_to_candidate"],
                           adjudication["native_vs_fp32"]["mean_kl_ref_to_candidate"])
    logger.warning(
        "NATIVE_PDL_MODEL_SHADOW rows=%d repeat_exact=%s serial_exact=%s relative=%.6g argmax_flips=%d prefill_flips=%d",
        report["rows"], report["baseline_repeat_exact"], report["native_serial_exact"], report["worst_relative"],
        report["argmax_flips"], report["prefill_argmax_flips"],
    )
    if not finite or not report["baseline_repeat_exact"] or not report["native_serial_exact"]:
        write_report(report)
        raise RuntimeError("Native PDL model shadow failed; inspect recorded same-batch logits")
    if running is None:
        write_report(report)
    return report


def check_committed_sample(report, actual_result, actual_prefill):
    if report is None:
        return
    actual = decode_acceptance(
        actual_result.next_token_ids.cpu().tolist(), actual_result.accept_index.cpu().tolist(),
        actual_result.accept_lens.cpu().tolist(),
    )
    expected = report["greedy_sampling"]["native"]
    report["committed_sample_exact"] = all(actual[key] == expected[key] for key in actual)
    report["committed_prefill_exact"] = actual_prefill.cpu().tolist() == report["prefill_sampling"]["native"]
    write_report(report)
    if not report["committed_sample_exact"] or not report["committed_prefill_exact"]:
        raise RuntimeError("Shadow sampling probe disagrees with actual committed tokens")
