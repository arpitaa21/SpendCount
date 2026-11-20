"""Train SpendCount categorisation and save leakage-aware synthetic benchmark metrics."""
from __future__ import annotations

import json
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from app.ml.categorisation import (
    MODEL_PATH,
    MODEL_VERSION,
    _rule_prediction,
    build_model,
    load_sample_examples,
    train_model,
    SAMPLE_PATH,
)

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "dataset" / "model_evaluation.json"


def score_split(train_examples, test_examples, include_rules=False):
    model = build_model(sample_examples=train_examples)
    truth, predictions = [], []
    for description, category in test_examples:
        rule = _rule_prediction(description) if include_rules else None
        if rule:
            predicted = rule["category"]
        else:
            predicted = str(model.predict([description])[0])
        truth.append(category)
        predictions.append(predicted)
    return truth, predictions


def grouped_evaluation(examples):
    if not examples:
        return {"status": "no_sample_data"}
    descriptions = np.asarray([desc for desc, _ in examples])
    labels = np.asarray([label for _, label in examples])
    groups = descriptions.copy()
    n_splits = min(5, len(set(groups)))
    if n_splits < 2:
        return {"status": "insufficient_unique_descriptions", "unique_examples": len(examples)}

    results = {"ml_only": {"truth": [], "predictions": []}, "rules_plus_ml": {"truth": [], "predictions": []}}
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(descriptions, labels, groups):
        train = [examples[int(i)] for i in train_idx]
        test = [examples[int(i)] for i in test_idx]
        for name, use_rules in (("ml_only", False), ("rules_plus_ml", True)):
            truth, predictions = score_split(train, test, use_rules)
            results[name]["truth"].extend(truth)
            results[name]["predictions"].extend(predictions)

    summary = {"split": f"{n_splits}-fold GroupKFold by normalized description"}
    for name, values in results.items():
        summary[name] = {
            "accuracy": round(float(accuracy_score(values["truth"], values["predictions"])), 4),
            "macro_f1": round(float(f1_score(values["truth"], values["predictions"], average="macro", zero_division=0)), 4),
        }
    return summary


def main():
    examples = load_sample_examples()
    metrics = grouped_evaluation(examples)
    valid_expense_rows = 0
    if SAMPLE_PATH.exists():
        with SAMPLE_PATH.open(encoding="utf-8-sig", newline="") as stream:
            valid_expense_rows = sum(
                1 for row in csv.DictReader(stream)
                if str(row.get("transaction_type", "")).strip().casefold() == "expense"
                and str(row.get("description", "")).strip()
                and str(row.get("category", "")).strip()
            )
    metrics.update({
        "model_version": MODEL_VERSION,
        "source": "synthetic student_transactions_6_months.csv; invalid and anomaly fixtures excluded",
        "valid_expense_rows_in_sample": valid_expense_rows,
        "unique_training_examples": len(examples),
        "unique_normalized_expense_descriptions": len(examples),
        "unique_examples_by_category": dict(sorted(Counter(label for _, label in examples).items())),
        "training_policy": "Exact duplicate descriptions are deduplicated; all matching test descriptions stay in one group.",
        "limitations": "Synthetic benchmark only. The lexical rules are broad student-spending cues and have not been independently validated on real users. Do not treat these metrics as real-world accuracy.",
    })
    model = train_model()
    metrics["artifact"] = str(MODEL_PATH.relative_to(ROOT))
    metrics["artifact_classes"] = sorted(str(x) for x in model.classes_)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
