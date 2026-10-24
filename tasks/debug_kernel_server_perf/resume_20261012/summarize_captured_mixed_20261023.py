import argparse
import json
from pathlib import Path
import statistics


def fixed_split_policy(geometry, sms=78):
    queries = geometry["query_lengths"][:geometry["prefix_rows"]]
    lengths = geometry["kv_lengths"][:geometry["prefix_rows"]]
    tiles = sum((length + 127) // 128 for length in queries) * geometry["heads"]
    splits = 1
    if max(lengths) >= 8192:
        while splits < 8 and tiles * splits < sms:
            splits *= 2
    return f"native_pdl_ps{splits}_rs4096"


def summarize(root):
    summaries = []
    for path in sorted(root.glob("*/kernel_results.json")):
        report = json.loads(path.read_text())
        if len(report["records"]) != report.get("expected_records", len(report["captures"]) * 40):
            continue
        for capture in report["captures"]:
            geometry = capture["geometry"]
            if geometry["timestamp"] < geometry["workload_release_time"]:
                raise ValueError("Warmup capture must not be reported as workload data")
            for style in ("eager", "graph"):
                records = [row for row in report["records"] if row["capture"] == capture["file"] and row["style"] == style]
                arms = {}
                for arm in {row["arm"] for row in records}:
                    selected = [row for row in records if row["arm"] == arm]
                    if len(selected) != 2 or {row["order"] for row in selected} != {"forward", "reverse"}:
                        raise ValueError("Incomplete opposite-order timing pair")
                    arms[arm] = statistics.mean(row["milliseconds"] for row in selected)
                best_single = min((arm for arm in arms if arm.startswith("fa3_full")), key=arms.get)
                best_dual = min((arm for arm in arms if arm.startswith("fa3_dual")), key=arms.get)
                summary = dict(
                    case=path.parent.name, capture=capture["file"], mode=geometry["mode"], style=style,
                    sha256=capture["sha256"], query_lengths=geometry["query_lengths"],
                    kv_lengths=geometry["kv_lengths"], prefix_rows=geometry["prefix_rows"],
                    default_fa3_ms=arms["fa3_full_default"], best_single_arm=best_single,
                    best_single_ms=arms[best_single], best_dual_arm=best_dual,
                    best_dual_ms=arms[best_dual], native_serial_ms=arms["native_serial"],
                    native_pdl_ms=arms["native_pdl"],
                    speedup_vs_default=arms["fa3_full_default"] / arms["native_pdl"],
                    speedup_vs_best_single=arms[best_single] / arms["native_pdl"],
                    speedup_vs_best_dual=arms[best_dual] / arms["native_pdl"],
                    speedup_vs_serial=arms["native_serial"] / arms["native_pdl"],
                    worst_head_relative_error=max(row["worst_head_relative_error"] for row in records),
                )
                if any(arm.startswith("native_pdl_ps") for arm in arms):
                    best_native = min((arm for arm in arms if arm.startswith("native_pdl")), key=arms.get)
                    fixed_native = fixed_split_policy(geometry)
                    summary.update(
                        best_native_arm=best_native, best_native_ms=arms[best_native],
                        fixed_native_arm=fixed_native, fixed_native_ms=arms[fixed_native],
                        fixed_speedup_vs_single=arms[best_single] / arms[fixed_native],
                        fixed_speedup_vs_dual=arms[best_dual] / arms[fixed_native],
                        oracle_speedup_vs_single=arms[best_single] / arms[best_native],
                        oracle_speedup_vs_dual=arms[best_dual] / arms[best_native],
                    )
                if "native_pdl_geometry_v1" in arms:
                    summary.update(
                        geometry_policy=capture["geometry_policy"],
                        geometry_pdl_ms=arms["native_pdl_geometry_v1"],
                        geometry_serial_ms=arms["native_serial_geometry_v1"],
                        geometry_speedup_vs_single=arms[best_single] / arms["native_pdl_geometry_v1"],
                        geometry_speedup_vs_dual=arms[best_dual] / arms["native_pdl_geometry_v1"],
                        geometry_speedup_vs_serial=arms["native_serial_geometry_v1"] / arms["native_pdl_geometry_v1"],
                    )
                summaries.append(summary)
    return dict(scope="Per-capture kernel measurements; not workload-weighted serving performance", records=summaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    report = summarize(arguments.root)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in report["records"]:
        if row["style"] == "graph":
            print(
                row["case"], row["capture"], "query", row["query_lengths"], "kv", row["kv_lengths"],
                "ms", *(f"{row[key]:.5f}" for key in ("default_fa3_ms", "best_single_ms", "best_dual_ms", "native_pdl_ms")),
                "speedups", *(f"{row[key]:.3f}x" for key in ("speedup_vs_best_single", "speedup_vs_best_dual")),
            )
            if "fixed_native_ms" in row:
                print("  FIXED", row["fixed_native_arm"], row["fixed_native_ms"],
                      "vs best FA3", row["fixed_speedup_vs_single"], row["fixed_speedup_vs_dual"],
                      "ORACLE", row["best_native_arm"], row["best_native_ms"])
            if "geometry_pdl_ms" in row:
                print("  GEOMETRY_V1", row["geometry_policy"], row["geometry_pdl_ms"],
                      "vs best FA3", row["geometry_speedup_vs_single"], row["geometry_speedup_vs_dual"])
