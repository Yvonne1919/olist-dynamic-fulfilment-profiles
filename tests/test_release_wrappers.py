"""Release adapter safeguards only; no empirical model rerun."""
import json
import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[1] / "scripts/reproduce.py"
_spec = importlib.util.spec_from_file_location("public_reproduce", _path)
reproduce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reproduce)
from analysis.direct_promise_profile_extension_v1.scripts import direct_experiment


def test_public_output_refuses_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(reproduce, "ROOT", tmp_path)
    with pytest.raises(ValueError):
        reproduce.fresh_output("../escaped")
    with pytest.raises(ValueError):
        reproduce.fresh_output("outputs")


def test_public_output_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(reproduce, "ROOT", tmp_path)
    created = reproduce.fresh_output("outputs/rq1")
    assert created.is_dir()
    with pytest.raises(FileExistsError):
        reproduce.fresh_output("outputs/rq1")


def test_exported_direct_selection_keeps_frozen_model_contract():
    config = direct_experiment.load_config()
    selection = json.loads((direct_experiment.WORKSPACE / "DIRECT_MODEL_SELECTION_FREEZE.json").read_text())
    direct_experiment._validate_selection(selection, config)
    assert config["breach"]["baseline_feature"] == "promised_delivery_days"
    assert selection["later_or_terminal_outcomes_used"] is False


def test_direct_features_do_not_admit_current_order_outcomes():
    for feature_map in (direct_experiment.breach_feature_map(), direct_experiment.severity_feature_map()):
        for numeric, categorical in feature_map.values():
            assert categorical == []
            assert numeric[0] == "promised_delivery_days"
            assert all(name == "promised_delivery_days" or name.startswith(("S1_", "S2_", "R1_", "R2_")) for name in numeric)
