"""Thin public entry points; original estimators, targets and checks are reused.

These commands do not emulate the private historical execution receipts. New
outputs are local reproductions, not replacements for the published evidence.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

def fresh_output(relative):
    output = (ROOT / relative).resolve()
    allowed = ROOT / "outputs"
    if output == allowed or allowed not in output.parents:
        raise ValueError("Output must be a new child directory of outputs/")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    return output

def rq1(args):
    from analysis.rq1_speed_reliability_review_v1.scripts import rq1_data, rq1_stats, rq1_io
    config = json.loads((ROOT / "analysis/rq1_speed_reliability_review_v1/RQ1_SPEED_RELIABILITY_FROZEN_CONFIG.json").read_text())
    all_orders, reviewed, audit = rq1_data.build_analysis_frames(args.data_dir, config)
    tables = rq1_data.build_data_audit_tables(all_orders, reviewed, audit, config)
    stats = rq1_stats.run_statistical_analysis(reviewed, config)
    output = fresh_output(args.output)
    # Same deterministic CSV serializer, without private historical reporting.
    from analysis.rq1_speed_reliability_review_v1.scripts.rq1_reporting import STAT_TABLE_FILES
    for filename, frame in tables.items():
        rq1_io.write_csv(output / filename, frame)
    for key, filename in STAT_TABLE_FILES.items():
        rq1_io.write_csv(output / filename, stats[key])
    rq1_io.write_csv(output / "RQ1_FIT_DIAGNOSTICS.csv", stats["diagnostics"])
    rq1_io.write_json(output / "REPRODUCTION_DECISION.json", stats["decision"])
    print(f"RQ1 tables written to {output.relative_to(ROOT)}")

def profiles(args):
    from analysis.profile_pivot_phase2a.scripts import data_pipeline as dp
    from analysis.dynamic_profile_profile_validation_v1.scripts import profile_core as core, selected_daily
    if dp.raw_file_sha256s(args.data_dir) != core.EXPECTED_RAW_HASHES:
        raise ValueError("Raw-data hashes differ from the frozen profile inputs")
    config = core.load_config()
    freeze = json.loads((core.OUT / "PROFILE_SELECTION_FREEZE.json").read_text())
    frame, canonical, _ = core.build_analysis_frame(args.data_dir)
    dp.audit_order_base(canonical, enforce_reference_counts=True)
    thresholds = core.frozen_tail_thresholds(frame)
    frame = core.attach_tail_targets(frame, thresholds)
    # This reconstructs frozen selected profiles, not the candidate-selection study.
    daily, parents = selected_daily.generate_selected_daily_profiles(frame, config, freeze["promoted_candidates"])
    output = fresh_output(args.output)
    daily.to_parquet(output / "selected_daily.parquet", index=False)
    parents.to_parquet(output / "selected_parents.parquet", index=False)
    (output / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")
    print(f"Selected profiles written to {output.relative_to(ROOT)}; not historical byte-format receipts")

def direct(args):
    from analysis.direct_promise_profile_extension_v1.scripts import direct_experiment as experiment
    config = experiment.load_config()
    # Only relocate the exact frame path; preserve its frozen expected SHA-256.
    frame_path = Path(args.model_frame).resolve()
    config["sources"]["order_model_frame"][0] = str(frame_path)
    frame = experiment.load_and_validate_frame(config)
    selection = json.loads((experiment.WORKSPACE / "DIRECT_MODEL_SELECTION_FREEZE.json").read_text())
    output = fresh_output(args.output)
    tables = experiment.evaluate_direct_extension(frame, selection, config)
    # Reuse the original serializers, but write to a new public output directory.
    # Do not mutate the module's output confinement or its original write API.
    for key, table in tables.items():
        filename = experiment.OUTPUT_FILES.get(key) or experiment.WORKING_OUTPUT_FILES.get(key)
        if filename:
            experiment._atomic_csv(table, output / filename, config["determinism"]["float_format"])
    print(f"Direct-promise frozen evaluation written to {output.relative_to(ROOT)}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("rq1", "Fit the frozen observational review analysis"),
                            ("profiles", "Rebuild selected daily profiles (large, not a full candidate search)")):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--data-dir", type=Path, default=Path(os.environ.get("OLIST_DATA_DIR", "data/olist_data")))
        sub.add_argument("--output", default=f"outputs/{name}")
    sub = commands.add_parser("direct", help="Evaluate frozen LR/XGBoost direct-promise models; exact Order V1 frame required")
    sub.add_argument("--model-frame", type=Path, required=True)
    sub.add_argument("--output", default="outputs/direct")
    args = parser.parse_args()
    # Reject accidental use from another working directory before data discovery.
    if Path.cwd().resolve() != ROOT:
        parser.error("Run from the public repository root")
    {"rq1": rq1, "profiles": profiles, "direct": direct}[args.command](args)

if __name__ == "__main__":
    main()
