import numpy as np
import pandas as pd
import pytest

import smarttab


def _make_classification_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + X[:, 1] * 0.5 > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(4)])
    df["category"] = rng.choice(["red", "green", "blue"], size=n)
    df["label"] = y
    return df


@pytest.fixture(scope="module")
def voting_model():
    df = _make_classification_df()
    return smarttab.fit(
        df, target="label", ensemble="voting", n_trials=6, cv=3, verbose=0, report=False,
    )


def test_voting_selects_a_strategy(voting_model):
    assert voting_model.model_name == "voting"
    assert voting_model.ensemble_info is not None
    assert voting_model.ensemble_info["strategy"] == "voting"
    members = voting_model.ensemble_info["members"]
    algorithms = {member["algorithm"] for member in members}
    assert {"catboost", "lightgbm"}.issubset(algorithms)
    assert 2 <= len(members) <= 5
    assert len(voting_model.ensemble_info["base_params"]) == len(members)
    assert len(voting_model.ensemble_info["base_scores"]) >= len(members)
    assert np.isclose(sum(voting_model.ensemble_info["voting_weights"].values()), 1.0)
    assert voting_model.ensemble_info["candidates"]
    assert voting_model.ensemble_info["diversity_matrix"]


def test_voting_predict_and_proba(voting_model):
    df = _make_classification_df(n=20, seed=5)
    X = df.drop(columns=["label"])
    preds = voting_model.predict(X)
    proba = voting_model.predict_proba(X)
    assert len(preds) == 20
    assert proba.shape == (20, 2)


def test_voting_report_includes_ensemble_section(voting_model, tmp_path):
    df = _make_classification_df(n=30, seed=6)
    folder = tmp_path / "voting_report"
    report_dict = voting_model.report(str(folder), df.drop(columns=["label"]), df["label"])
    assert report_dict["ensemble_info"]["strategy"] == voting_model.model_name
    content = (folder / "report.html").read_text(encoding="utf-8")
    assert "Ensemble Details" in content


def test_voting_save_load_roundtrip(voting_model, tmp_path):
    df = _make_classification_df(n=15, seed=9)
    X = df.drop(columns=["label"])
    original_preds = voting_model.predict(X)

    bundle_path = voting_model.save(str(tmp_path / "voting.smarttab"))
    loaded = smarttab.load(bundle_path, trusted=True)
    loaded_preds = loaded.predict(X)

    np.testing.assert_array_equal(original_preds, loaded_preds)
    assert loaded.ensemble_info["strategy"] == voting_model.model_name


def test_stacking_strategy_explicit():
    df = _make_classification_df(n=250, seed=11)
    model = smarttab.fit(df, target="label", ensemble="stacking", n_trials=6, cv=3, verbose=0, report=False)
    assert model.model_name == "stacking"
    assert model.ensemble_info["strategy"] == "stacking"
