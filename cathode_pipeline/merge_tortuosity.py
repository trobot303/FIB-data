"""Merge a phase-restricted tortuosity run into the previous full tortuosity.json.

    python merge_tortuosity.py <sample_output_dir> --phases llzto

After `run --stage tortuosity --force` with tortuosity.phases = [llzto], the sample's
tortuosity.json holds only llzto.  This script takes the other phases from the backup
tortuosity_downsampled.json (made before the run), records the downsample factor per
phase, and writes the combined tortuosity.json so that report / figures see all phases.
"""
import argparse
import json
import shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("sample_dir")
ap.add_argument("--phases", nargs="+", required=True, help="phases solved in the new run")
a = ap.parse_args()

d = Path(a.sample_dir)
new_p, old_p = d / "tortuosity.json", d / "tortuosity_downsampled.json"
if not new_p.exists():
    raise SystemExit(f"{new_p} missing")
new = json.load(open(new_p))
if not old_p.exists():
    print(f"{d.name}: no tortuosity_downsampled.json to merge with; keeping the new result as is")
    raise SystemExit(0)
old = json.load(open(old_p))
shutil.copy(new_p, d / f"tortuosity_{'_'.join(a.phases)}_full.json")

merged = dict(old)
merged["phases"] = {}
for ph, entry in old["phases"].items():
    e = dict(entry); e["downsample"] = old.get("downsample")
    merged["phases"][ph] = e
for ph in a.phases:
    if ph in new["phases"]:
        e = dict(new["phases"][ph]); e["downsample"] = new.get("downsample")
        merged["phases"][ph] = e
merged["downsample"] = {ph: e.get("downsample") for ph, e in merged["phases"].items()}
merged["note"] = "phases " + ", ".join(a.phases) + f" solved at downsample {new.get('downsample')}; others at downsample {old.get('downsample')}"
json.dump(merged, open(new_p, "w"), indent=2)
print(f"{d.name}: merged -> " + ", ".join(f"{ph} (ds {e['downsample']})" for ph, e in merged["phases"].items()))
