"""Explicit portable subset of existing tests; no raw data or experiment reruns."""
from pathlib import Path
import ast
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def selected_nodes(relative, predicate):
    tree = ast.parse((ROOT / relative).read_text())
    return [relative + "::" + n.name for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name.startswith("test_") and predicate(n.name)]

def main():
    nodes = ["tests", "analysis/profile_pivot_phase2a/scripts/test_data_pipeline.py"]
    nodes += selected_nodes(
        "analysis/dynamic_profile_profile_validation_v1/scripts/test_profile_validation.py",
        lambda name: name[5:8].isdigit() and (12 <= int(name[5:8]) <= 67 or int(name[5:8]) == 2),
    )
    rq1_ids = {1, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 30, 31, 32, 33, 34, 35}
    nodes += selected_nodes(
        "analysis/rq1_speed_reliability_review_v1/scripts/test_rq1_speed_reliability.py",
        lambda name: name[5:7].isdigit() and int(name[5:7]) in rq1_ids,
    )
    excluded_order = (
        "artifact_if_available", "receipt_if_available", "hash_trust_anchors",
        "local_frozen_control_hashes", "manifest_output_hashes",
        "model_selection_hashes_if_available", "persisted_result_schema_if_available",
        "endpoint_profile_is_resolved_by_preconfirmation_development_rank",
        "no_prior_protected_output_modification_contract",
    )
    nodes += selected_nodes(
        "analysis/order_breach_severity_v1/scripts/test_order_breach_severity.py",
        lambda name: not any(part in name for part in excluded_order),
    )
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg",
               MPLCONFIGDIR=str(ROOT / ".cache/matplotlib"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONPATH=str(ROOT))
    print("Portable synthetic/contract subset; historical raw-data/governance receipt tests are not selected.", flush=True)
    raise SystemExit(subprocess.call([sys.executable, "-B", "-m", "pytest", "-q",
                                    "-c", str(ROOT / "pytest.ini"),
                                    "--confcutdir", str(ROOT), "-p", "no:cacheprovider", *nodes], cwd=ROOT, env=env))

if __name__ == "__main__":
    main()
