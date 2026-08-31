from src.features.availability import feature_availability_audit
from src.features.blocks import feature_sets


def test_model_features_are_audited_and_kept():
    audit = feature_availability_audit().set_index("feature")
    model_features = {feature for numeric, categorical in feature_sets().values() for feature in numeric + categorical}
    assert model_features <= set(audit.index)
    assert (audit.loc[list(model_features), "keep"] == "yes").all()
    assert (audit.loc[list(model_features), "available_at_promise"] == "yes").all()
