"""Plot complete captured-kernel results without inferring workload-level speedups."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(source, output, native_arm="native_direct_pdl"):
    report = json.loads(source.read_text())
    rows = []
    for capture in report["cases"]:
        measured = [row for row in report["records"] if row["capture"] == capture["name"] and row["style"] == "graph"]
        means = {arm: statistics.mean(row["milliseconds"] for row in measured if row["arm"] == arm) for arm in {row["arm"] for row in measured}}
        control = min((arm for arm in means if "fa3" in arm), key=means.get)
        native = means[native_arm]
        model, workload, bucket = capture["name"].split("/")
        rows.append(dict(
            capture=capture["name"], model=model, mode=workload.rsplit("_", 1)[1],
            context=workload.rsplit("_", 1)[0], bucket=bucket.removesuffix(".pt"),
            control=control, fa3_ms=means[control], native_ms=native,
            speedup=means[control] / native, sha256=capture["sha256"],
        ))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    model_names = {"qwen25": "Qwen2.5-7B", "qwen3": "Qwen3-8B", "glm": "GLM-4.5-Air TP4 rank 0"}
    present = {row["model"] for row in rows}
    if not present <= model_names.keys():
        raise ValueError("Unrecognized model in capture plot")
    models = [model for model in model_names if model in present]
    figure, axes = plt.subplots(len(models), 2, figsize=(13, 4.5 * len(models) + 1), sharex=True, squeeze=False)
    context_names = {"mix32": "32–64K", "mix64": "64–128K", "mix128": "128–256K", "fanout384": "~394K cold"}
    context_order = {"mix32": 0, "mix64": 1, "mix128": 2, "fanout384": 3}
    bucket_order = {"fresh_small": 0, "cached_small": 1, "fresh_large": 2, "cached_large": 3}
    for axis, (model, mode) in zip(axes.flat, ((model, mode) for model in models for mode in ("two", "four"))):
        selected = sorted((row for row in rows if row["model"] == model and row["mode"] == mode), key=lambda row: (context_order[row["context"]], bucket_order[row["bucket"]]))
        labels = [context_names[row["context"]] + " | " + row["bucket"].replace("_", " ") for row in selected]
        values = [row["speedup"] for row in selected]
        axis.barh(range(len(values)), values, color=["#267a9e" if value >= 1 else "#c76448" for value in values], height=0.65)
        axis.set_yticks(range(len(values)), labels=labels, fontsize=8)
        axis.invert_yaxis()
        axis.axvline(1, color="#333333", linewidth=1, linestyle="--")
        for position, value in enumerate(values):
            axis.text(value + 0.025, position, f"{value:.2f}×", va="center", fontsize=8)
        mode_name = "two-mode" if mode == "two" else ("four-mode NEXTN" if model == "glm" else "four-mode EAGLE3")
        axis.set_title(model_names[model] + " · " + mode_name)
        axis.set_xlim(0, max(2.0, max(row["speedup"] for row in rows) + 0.2))
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel("Best tested FA3 / native FA3+FA2 kernel latency")
    figure.suptitle(f"Actual serving Q/K/V replays · H20 · all {len(rows)} captures · {native_arm}", fontsize=14)
    note = report.get("figure_note", "Graph replay; stronger FA3 split, packing and SM-grid controls; FP32 partials + fused BF16 store.\nShape-coverage sample, not workload-weighted. Cold ~394K fan-out; serving-output parity remains open.")
    figure.text(0.5, 0.025, note, ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.065, 1, 0.96))
    figure.savefig(output.with_suffix(".svg"))
    figure.savefig(output.with_suffix(".png"), dpi=140)
    plt.close(figure)
    output.with_suffix(".json").write_text(json.dumps(dict(
        source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        cases=len(rows), native_arm=native_arm, latency_aggregation="mean of opposite-order medians; best remeasured FA3 control per capture",
        scope="No workload-level average or serving performance inference",
    ), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--native-arm", default="native_direct_pdl")
    arguments = parser.parse_args()
    plot(arguments.source, arguments.output, arguments.native_arm)
