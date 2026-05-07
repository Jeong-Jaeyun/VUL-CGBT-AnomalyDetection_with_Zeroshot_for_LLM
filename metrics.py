from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class EvaluationRecord:
    model_name: str
    sample_id: str
    budget: int
    true_label: int
    original_prediction: Optional[int]
    transformed_prediction: Optional[int]
    original_confidence: Optional[float]
    transformed_confidence: Optional[float]
    transformation_valid: bool
    parsing_failed_original: bool
    parsing_failed_transformed: bool


def compute_binary_classification_metrics(
    true_labels: Iterable[int],
    predicted_labels: Iterable[int],
) -> Dict[str, float | int]:
    tp = tn = fp = fn = 0
    predicted_positive = 0
    actual_positive = 0

    for truth, pred in zip(true_labels, predicted_labels):
        truth = int(truth)
        pred = int(pred)
        if truth == 1:
            actual_positive += 1
        if pred == 1:
            predicted_positive += 1
        if truth == 1 and pred == 1:
            tp += 1
        elif truth == 0 and pred == 0:
            tn += 1
        elif truth == 0 and pred == 1:
            fp += 1
        elif truth == 1 and pred == 0:
            fn += 1

    total = tp + tn + fp + fn
    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) else 0.0
    specificity = (tn / (tn + fp)) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = ((tp + tn) / total) if total else 0.0
    balanced_accuracy = (recall + specificity) / 2.0

    return {
        "num_records": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "accuracy": accuracy,
        "predicted_positive_rate": (predicted_positive / total) if total else 0.0,
        "actual_positive_rate": (actual_positive / total) if total else 0.0,
    }


class RobustnessMetricsComputer:
    def compute(
        self,
        evaluations: List[EvaluationRecord],
        validity_counts: Dict[int, Dict[str, int]],
    ) -> Dict[str, Dict]:
        by_model: Dict[str, Dict[int, List[EvaluationRecord]]] = {}
        for record in evaluations:
            by_model.setdefault(record.model_name, {}).setdefault(record.budget, []).append(record)

        result: Dict[str, Dict] = {"models": {}, "transformation_validity": {}}
        for budget, counts in sorted(validity_counts.items()):
            planned = counts.get("planned", 0)
            valid = counts.get("valid", 0)
            result["transformation_validity"][str(budget)] = {
                "planned_transformations": planned,
                "valid_transformations": valid,
                "validity_rate": (valid / planned) if planned else 0.0,
            }

        for model_name, budget_map in sorted(by_model.items()):
            model_summary: Dict[str, Dict[str, float | int | Dict[str, int]]] = {}
            all_records: List[EvaluationRecord] = []
            for budget, records in sorted(budget_map.items()):
                all_records.extend(records)
                model_summary[str(budget)] = self._compute_for_records(records)
            model_summary["overall"] = self._compute_for_records(all_records)
            result["models"][model_name] = model_summary

        return result

    def _compute_for_records(self, records: List[EvaluationRecord]) -> Dict:
        unique_original_records = self._unique_original_records(records)
        comparable = [
            record
            for record in records
            if record.transformation_valid
            and record.original_prediction is not None
            and record.transformed_prediction is not None
        ]

        prediction_changes = sum(
            1
            for record in comparable
            if record.original_prediction != record.transformed_prediction
        )
        pcr = (prediction_changes / len(comparable)) if comparable else 0.0

        clean_correct = sum(
            1 for record in comparable if record.original_prediction == record.true_label
        )
        clean_acc = (clean_correct / len(comparable)) if comparable else 0.0
        original_clean_correct = sum(
            1
            for record in unique_original_records
            if record.original_prediction == record.true_label
        )
        original_clean_acc = (
            original_clean_correct / len(unique_original_records)
            if unique_original_records
            else 0.0
        )
        original_positive_rate = (
            sum(
                1
                for record in unique_original_records
                if record.original_prediction == 1
            )
            / len(unique_original_records)
            if unique_original_records
            else 0.0
        )
        robust_correct = sum(
            1
            for record in comparable
            if record.transformed_prediction == record.true_label
        )
        robust_acc = (robust_correct / len(comparable)) if comparable else 0.0
        robustness_gap = clean_acc - robust_acc

        originally_correct = [
            record for record in comparable if record.original_prediction == record.true_label
        ]
        attacked_to_wrong = sum(
            1
            for record in originally_correct
            if record.transformed_prediction != record.true_label
        )
        asr = (
            attacked_to_wrong / len(originally_correct)
            if originally_correct
            else 0.0
        )

        transitions = {
            "tp_to_fn": 0,
            "tn_to_fp": 0,
            "fp_to_tn": 0,
            "fn_to_tp": 0,
            "other": 0,
        }
        for record in comparable:
            transitions[self._transition_key(record)] += 1

        parsing_failures = sum(
            1 for record in records if record.parsing_failed_original or record.parsing_failed_transformed
        )
        confidence_stats = self._compute_confidence_stats(comparable)
        original_classification = self._classification_stats(
            records=unique_original_records,
            prediction_attr="original_prediction",
        )
        clean_classification = self._classification_stats(
            records=comparable,
            prediction_attr="original_prediction",
        )
        robust_classification = self._classification_stats(
            records=comparable,
            prediction_attr="transformed_prediction",
        )

        return {
            "num_records": len(records),
            "num_comparable_records": len(comparable),
            "num_unique_original_samples": len(unique_original_records),
            "original_clean_accuracy_unique": original_clean_acc,
            "original_prediction_positive_rate_unique": original_positive_rate,
            "original_classification_unique": original_classification,
            "clean_accuracy": clean_acc,
            "clean_classification": clean_classification,
            "prediction_change_rate": pcr,
            "robust_accuracy": robust_acc,
            "robust_classification": robust_classification,
            "robustness_gap": robustness_gap,
            "attack_success_rate": asr,
            "num_originally_correct": len(originally_correct),
            "error_transitions": transitions,
            "parsing_failures": parsing_failures,
            "confidence_shift": confidence_stats,
        }

    def _unique_original_records(
        self,
        records: List[EvaluationRecord],
    ) -> List[EvaluationRecord]:
        unique: "OrderedDict[str, EvaluationRecord]" = OrderedDict()
        for record in records:
            if record.original_prediction is None:
                continue
            unique.setdefault(record.sample_id, record)
        return list(unique.values())

    def _transition_key(self, record: EvaluationRecord) -> str:
        assert record.original_prediction is not None
        assert record.transformed_prediction is not None
        true_label = record.true_label
        orig = record.original_prediction
        transformed = record.transformed_prediction

        if true_label == 1 and orig == 1 and transformed == 0:
            return "tp_to_fn"
        if true_label == 0 and orig == 0 and transformed == 1:
            return "tn_to_fp"
        if true_label == 0 and orig == 1 and transformed == 0:
            return "fp_to_tn"
        if true_label == 1 and orig == 0 and transformed == 1:
            return "fn_to_tp"
        return "other"

    def _compute_confidence_stats(self, records: List[EvaluationRecord]) -> Dict[str, float | int | None]:
        pairs = [
            (record, record.original_confidence, record.transformed_confidence)
            for record in records
            if record.original_confidence is not None and record.transformed_confidence is not None
        ]
        if not pairs:
            return {
                "num_confidence_pairs": 0,
                "mean_delta_confidence": None,
                "mean_abs_delta_confidence": None,
                "clean_calibration_error": None,
                "robust_calibration_error": None,
                "calibration_drift": None,
            }

        deltas = [float(transformed - original) for _, original, transformed in pairs]
        abs_deltas = [abs(delta) for delta in deltas]

        clean_calibration_errors: List[float] = []
        robust_calibration_errors: List[float] = []
        for record, original, transformed in pairs:
            assert record.original_prediction is not None
            assert record.transformed_prediction is not None
            orig_correct = 1.0 if record.original_prediction == record.true_label else 0.0
            trans_correct = 1.0 if record.transformed_prediction == record.true_label else 0.0
            clean_calibration_errors.append(abs(float(original) - orig_correct))
            robust_calibration_errors.append(abs(float(transformed) - trans_correct))

        clean_error = sum(clean_calibration_errors) / len(clean_calibration_errors)
        robust_error = sum(robust_calibration_errors) / len(robust_calibration_errors)

        return {
            "num_confidence_pairs": len(pairs),
            "mean_delta_confidence": sum(deltas) / len(deltas),
            "mean_abs_delta_confidence": sum(abs_deltas) / len(abs_deltas),
            "clean_calibration_error": clean_error,
            "robust_calibration_error": robust_error,
            "calibration_drift": robust_error - clean_error,
        }

    def _classification_stats(
        self,
        records: List[EvaluationRecord],
        prediction_attr: str,
    ) -> Dict[str, float | int]:
        labels: List[int] = []
        predictions: List[int] = []
        for record in records:
            prediction = getattr(record, prediction_attr)
            if prediction is None:
                continue
            labels.append(int(record.true_label))
            predictions.append(int(prediction))
        return compute_binary_classification_metrics(labels, predictions)
