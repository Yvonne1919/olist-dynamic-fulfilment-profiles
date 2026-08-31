from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.all_mature_history_sensitivity_v1.scripts import integrity
from analysis.all_mature_history_sensitivity_v1.scripts import order_sensitivity
from analysis.all_mature_history_sensitivity_v1.scripts import reporting
from analysis.all_mature_history_sensitivity_v1.scripts import sensitivity_core
from analysis.all_mature_history_sensitivity_v1.scripts import validate_sensitivity


OUT = ROOT / "analysis/all_mature_history_sensitivity_v1"
WORK = OUT / "working"
MANIFEST_PATH = OUT / "RUN_MANIFEST.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    cfg = sensitivity_core.load_config()
    return {
        "analysis_id": cfg["analysis_id"],
        "status": "created",
        "created_at_utc": utc_now(),
        "manifest_sequence": 0,
        "stages": {},
        "command_history": [],
        "scope": cfg["scope"],
        "direct_order_level_branch": "not_yet_evaluated",
    }


def write_manifest(manifest: dict) -> None:
    manifest["manifest_sequence"] = int(manifest.get("manifest_sequence", 0)) + 1
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "COMMAND_LOG.json").write_text(
        json.dumps(manifest.get("command_history", []), indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def script_inventory() -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for path in sorted((OUT / "scripts").glob("*.py")):
        rows[path.relative_to(ROOT).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    for path in (OUT / "SENSITIVITY_FROZEN_CONFIG.json", OUT / "EXACT_HISTORY_DEFINITIONS.md"):
        rows[path.relative_to(ROOT).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return rows


def write_hash_inventory() -> dict[str, dict[str, object]]:
    inventory: dict[str, dict[str, object]] = {}
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.name in {"HASH_INVENTORY.txt", "RUN_MANIFEST.json"}:
            continue
        relative = path.relative_to(OUT).as_posix()
        inventory[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    lines = [
        f"{record['sha256']}  {record['bytes']}  {relative}"
        for relative, record in inventory.items()
    ]
    lines.extend([
        "",
        "# RUN_MANIFEST.json is excluded because it is finalized after this inventory.",
        "# HASH_INVENTORY.txt excludes itself to avoid a circular digest.",
    ])
    (OUT / "HASH_INVENTORY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return inventory


def run_stage(stage: str) -> object:
    manifest = load_manifest()
    invocation = list(sys.argv)
    record = {
        "stage": stage,
        "started_at_utc": utc_now(),
        "actual_invocation": invocation,
        "working_directory": str(ROOT),
    }
    manifest.setdefault("command_history", []).append(record)
    manifest["status"] = f"{stage}_running"
    write_manifest(manifest)
    try:
        if stage == "preflight":
            result = integrity.preflight()
            manifest["repository_preflight"] = result["repository"]
            manifest["environment"] = result["environment"]
            manifest["raw_file_hashes"] = result["raw_file_hashes"]
            manifest["protected_preflight"] = {
                "root_count": result["protected_root_count"],
                "leaf_count": result["protected_leaf_count"],
                "total_bytes": result["protected_total_bytes"],
            }
            manifest["direct_extension_gate"] = result["direct_extension_gate"]
            manifest["direct_order_level_branch"] = (
                "required" if result["direct_extension_gate"]["available"] else "skipped_gate_not_met"
            )
        elif stage == "standalone":
            result = sensitivity_core.run_standalone()
        elif stage == "order":
            gate = json.loads((WORK / "DIRECT_EXTENSION_GATE.json").read_text(encoding="utf-8"))
            if not gate.get("available"):
                raise RuntimeError("the frozen direct-extension gate is false; order branch is not authorised")
            result = order_sensitivity.run_order_sensitivity()
            manifest["direct_order_level_branch"] = "complete"
        elif stage == "report":
            result = reporting.generate_reports()
        elif stage == "test":
            command = [
                sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q",
                "analysis/all_mature_history_sensitivity_v1/scripts/test_sensitivity.py",
            ]
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["MPLCONFIGDIR"] = str(WORK / "mplconfig")
            completed = subprocess.run(
                command, cwd=ROOT, env=env, text=True, capture_output=True,
            )
            (WORK / "PYTEST_STDOUT.txt").write_text(completed.stdout, encoding="utf-8")
            (WORK / "PYTEST_STDERR.txt").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(f"pytest failed with code {completed.returncode}: {completed.stdout}\n{completed.stderr}")
            result = {
                "command": command,
                "exit_code": completed.returncode,
                "stdout_path": "working/PYTEST_STDOUT.txt",
                "stderr_path": "working/PYTEST_STDERR.txt",
            }
        elif stage == "validate":
            result = validate_sensitivity.validate()
        elif stage == "finalize":
            integrity_receipt = integrity.finalize_integrity()
            inventory = write_hash_inventory()
            result = {
                "integrity": integrity_receipt,
                "hash_inventory_entries": len(inventory),
                "hash_inventory_sha256": sha256_file(OUT / "HASH_INVENTORY.txt"),
            }
            manifest["protected_integrity_verdict"] = integrity_receipt
            manifest["output_inventory"] = inventory
            manifest["hash_inventory_sha256"] = result["hash_inventory_sha256"]
            manifest["executed_scripts"] = script_inventory()
        else:
            raise ValueError(f"unsupported stage: {stage}")
    except Exception as exc:
        record["completed_at_utc"] = utc_now()
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        manifest["status"] = f"{stage}_failed"
        manifest.setdefault("stages", {})[stage] = {
            "status": "failed", "completed_at_utc": record["completed_at_utc"],
            "error": record["error"],
        }
        write_manifest(manifest)
        raise
    record["completed_at_utc"] = utc_now()
    record["status"] = "complete"
    manifest.setdefault("stages", {})[stage] = {
        "status": "complete",
        "completed_at_utc": record["completed_at_utc"],
        "result": result,
    }
    manifest["status"] = "complete" if stage == "finalize" else f"{stage}_complete"
    if stage == "finalize":
        gate = manifest.get("direct_extension_gate", {})
        order_required = bool(gate.get("available"))
        manifest["completion_checks"] = {
            "standalone_complete": manifest.get("stages", {}).get("standalone", {}).get("status") == "complete",
            "report_complete": manifest.get("stages", {}).get("report", {}).get("status") == "complete",
            "tests_complete": manifest.get("stages", {}).get("test", {}).get("status") == "complete",
            "validation_pass": manifest.get("stages", {}).get("validate", {}).get("result", {}).get("status") == "PASS",
            "protected_integrity_pass": result["integrity"]["status"] == "PASS",
            "hash_inventory_created": (OUT / "HASH_INVENTORY.txt").is_file(),
            "direct_branch_complete_if_required": (
                manifest.get("stages", {}).get("order", {}).get("status") == "complete"
                if order_required else
                manifest.get("direct_order_level_branch") == "skipped_gate_not_met"
            ),
        }
        if not all(manifest["completion_checks"].values()):
            raise AssertionError(f"final completion checks failed: {manifest['completion_checks']}")
    write_manifest(manifest)
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_sensitivity.py {preflight|standalone|order|report|test|validate|finalize}")
    result = run_stage(sys.argv[1])
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, default=str))
