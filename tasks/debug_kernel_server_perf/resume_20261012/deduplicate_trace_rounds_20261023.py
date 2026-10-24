"""Remove only identical logical trace records, rejecting identity conflicts."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def unique_rounds(records):
    seen = {}
    duplicates = 0
    unique = []
    for record in records:
        identity = (record["provider"], record["session_id"], record["round_index"])
        if identity in seen:
            if record != seen[identity]:
                raise ValueError(f"Conflicting trace records for identity {identity}")
            duplicates += 1
            continue
        seen[identity] = record
        unique.append(record)
    return unique, duplicates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("manifest", type=Path)
    arguments = parser.parse_args()
    if arguments.destination.exists() or arguments.manifest.exists():
        raise ValueError("Fresh normalized trace and manifest required")
    with gzip.open(arguments.source, "rt") as source:
        rows, duplicates = unique_rounds(json.loads(line) for line in source)
    with arguments.destination.open("wb") as output:
        with gzip.GzipFile(fileobj=output, mode="wb", mtime=0, filename="") as compressed:
            for row in rows:
                compressed.write((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())
    report = dict(
        source=str(arguments.source), source_sha256=hashlib.sha256(arguments.source.read_bytes()).hexdigest(),
        destination=str(arguments.destination), destination_sha256=hashlib.sha256(arguments.destination.read_bytes()).hexdigest(),
        original_records=len(rows) + duplicates, unique_records=len(rows), identical_duplicates_removed=duplicates,
        conflicting_records=0, order="first occurrence retained; no length/provider filtering or renumbering",
        scope="Duplicate logical identities with identical complete JSON records only; original source remains untouched",
    )
    arguments.manifest.write_text(json.dumps(report, indent=2) + "\n")
    print("TRACE_DEDUPLICATION", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
