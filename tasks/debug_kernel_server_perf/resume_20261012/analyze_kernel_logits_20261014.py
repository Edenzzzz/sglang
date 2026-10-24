import argparse
import json
from pathlib import Path


def emitted_records(records, outputs, request_id, position):
    matches = []
    for record in records:
        if record["rid"] != request_id or record["output_index"] != position:
            continue
        if record["role"] == "verify":
            begin = record["proposal_begin"]
            if begin < 0 or record["proposal_tokens"] != outputs[begin:position]:
                continue
        elif record["role"] != "prefill" or position != 0:
            continue
        values = record["top_logits"]
        matches.append({
            **record,
            "top_margin": values[0] - values[1],
            "argmax_matches_emitted": record["argmax"] == outputs[position],
        })
    return matches


def compare_outputs(baseline, candidate, baseline_records, candidate_records):
    report = {}
    for request_id in sorted(key for key in baseline if not key.startswith("_")):
        left = baseline[request_id]
        right = candidate[request_id]
        position = next((index for index, (first, second) in enumerate(zip(left, right)) if first != second), min(len(left), len(right)))
        different = position < min(len(left), len(right))
        entry = {"common_prefix": position, "baseline_length": len(left), "candidate_length": len(right)}
        if different:
            first = emitted_records(baseline_records, left, request_id, position)
            second = emitted_records(candidate_records, right, request_id, position)
            hashes = {record["prompt_sha256"] for record in first + second}
            if len(hashes) > 1:
                raise ValueError(f"Mismatched prompts for {request_id}")
            entry.update(
                baseline_token=left[position], candidate_token=right[position],
                baseline_records=first, candidate_records=second,
                identical_prefix_scores_available=bool(first and second),
            )
        report[request_id] = entry
    return report


def load_records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("baseline")
    parser.add_argument("candidates", nargs="+")
    args = parser.parse_args()
    baseline = json.loads((args.directory / f"{args.baseline}.json").read_text())
    baseline_records = load_records(args.directory / f"{args.baseline}.logits.jsonl")
    reports = {}
    for candidate in args.candidates:
        result = json.loads((args.directory / f"{candidate}.json").read_text())
        records = load_records(args.directory / f"{candidate}.logits.jsonl")
        report = compare_outputs(baseline, result, baseline_records, records)
        reports[candidate] = report
        for request_id, values in report.items():
            if not values.get("identical_prefix_scores_available"):
                continue
            print(candidate, request_id, "position", values["common_prefix"],
                  "tokens", values["baseline_token"], values["candidate_token"],
                  "baseline", [(record["top_margin"], record["argmax_matches_emitted"]) for record in values["baseline_records"]],
                  "candidate", [(record["top_margin"], record["argmax_matches_emitted"]) for record in values["candidate_records"]])
    (args.directory / "identical_prefix_logit_comparison.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
