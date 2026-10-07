"""Metrics computation for HaluEval and SimpleQA."""
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd


def halueval_metrics(y_true, y_pred):
    """
    y_true, y_pred: lists of 0/1 where 1 = hallucination (YES).
    Returns dict with accuracy, precision, recall, f1, yes_ratio.
    """
    y_true = [int(v) for v in y_true]
    y_pred = [int(v) for v in y_pred]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "yes_ratio": sum(y_pred) / len(y_pred) if y_pred else 0.0,
    }


def simpleqa_metrics(verdicts):
    """
    verdicts: list of strings in {CORRECT, INCORRECT, NOT_ATTEMPTED}.
    Returns dict with accuracy, attempted_accuracy, hallucination_rate,
    abstention_rate, f_score.
    """
    n = len(verdicts)
    if n == 0:
        return {k: 0.0 for k in
                ["accuracy", "attempted_accuracy", "hallucination_rate",
                 "abstention_rate", "f_score"]}

    n_correct = sum(1 for v in verdicts if v == "CORRECT")
    n_incorrect = sum(1 for v in verdicts if v == "INCORRECT")
    n_not_attempted = sum(1 for v in verdicts if v == "NOT_ATTEMPTED")
    n_attempted = n_correct + n_incorrect

    accuracy = n_correct / n
    attempted_acc = n_correct / n_attempted if n_attempted else 0.0
    hallucination_rate = n_incorrect / n
    abstention_rate = n_not_attempted / n

    # F-score: harmonic mean of accuracy and attempted_accuracy
    if accuracy + attempted_acc > 0:
        f_score = 2 * accuracy * attempted_acc / (accuracy + attempted_acc)
    else:
        f_score = 0.0

    return {
        "accuracy": accuracy,
        "attempted_accuracy": attempted_acc,
        "hallucination_rate": hallucination_rate,
        "abstention_rate": abstention_rate,
        "f_score": f_score,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_not_attempted": n_not_attempted,
    }
