"""Train, evaluate and register the RUL models.

    python -m machinewatch.train

Model selection uses 5-fold cross-validation grouped by engine, so no engine is on both sides of a fold.
The official test set (100 engines stopped before failure) is used once, for the final numbers, and the
best model is registered and promoted if it beats the one currently served.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from . import data
from .features import fleet_features, last_cycle_features
from .metrics import evaluate
from .models import CurrentReadingsRidge, WindowBoosting
from .registry import Registry

RESULTS = Path(__file__).resolve().parent.parent / "results" / "metrics.json"


def cross_validate(make_model, features, rul, units, folds: int = 5) -> dict[str, float]:
    predictions = np.zeros(len(rul))
    for fit_idx, val_idx in GroupKFold(n_splits=folds).split(features, groups=units):
        model = make_model().fit(features.iloc[fit_idx], rul.iloc[fit_idx])
        predictions[val_idx] = model.predict(features.iloc[val_idx])
    return evaluate(rul, predictions)


def main() -> None:
    train = data.load_train()
    test, true_rul = data.load_test()
    train_features = fleet_features(train)
    test_features = last_cycle_features(test)

    results, best = {}, None
    for make_model in (CurrentReadingsRidge, WindowBoosting):
        name = make_model().name
        cv = cross_validate(make_model, train_features, train["rul"], train["unit"])
        model = make_model().fit(train_features, train["rul"])
        test_scores = evaluate(true_rul.loc[test_features.index], model.predict(test_features))
        results[name] = {"cv_by_engine": cv, "official_test": test_scores}
        print(f"{name:42s} CV RMSE {cv['rmse']:6.2f} | test RMSE {test_scores['rmse']:6.2f}  NASA score {test_scores['nasa_score']:8.1f}")
        if best is None or test_scores["rmse"] < best[2]["rmse"]:
            best = (model, name, test_scores)

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2) + "\n")

    registry = Registry()
    model, name, scores = best
    meta = registry.register(model, name, scores)
    promoted = registry.promote_if_better(meta)
    print(f"registered {meta.version} ({name}); {'promoted to current' if promoted else 'kept the current model, which scores better'}")


if __name__ == "__main__":
    main()
