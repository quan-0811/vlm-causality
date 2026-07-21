"""
Evaluate a model's answers against POPE ground-truth labels.

Matches the standard POPE evaluation in evaluate.py: loads a model answer file
(JSONL with "question" and "answer" fields) and a label file (JSONL with
"question_id", "image", "text", "label"), then computes accuracy, precision,
recall, F1, and yes-ratio.

Usage:
    python scripts/eval_pope.py \
        --ans-file results/my_model_answers.json \
        --label-file pope/coco_pope_random.json
"""

import argparse
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate POPE answers")
    parser.add_argument("--ans-file", required=True,
                        help="Path to model answer file (JSONL or JSON list)")
    parser.add_argument("--label-file", required=True,
                        help="Path to POPE label file (JSONL)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-question breakdown")
    return parser.parse_args()


def load_answers(ans_path):
    """Load model answers from JSONL (one per line) or a JSON list."""
    with open(ans_path) as f:
        raw = f.read().strip()

    if raw.startswith("["):
        return json.loads(raw)
    else:
        return [json.loads(line) for line in raw.splitlines() if line.strip()]


def load_labels(label_path):
    """Load POPE labels from JSONL."""
    with open(label_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def binarize_answer(text):
    """
    Convert model answer text to 'yes' or 'no'.
    Matches the POPE evaluate.py logic: first sentence, strip commas,
    'no'/'not' → 'no', else 'yes'.
    """
    # Take first sentence only
    text = text.split(".")[0]
    text = text.replace(",", "").lower()

    words = text.split()
    if "no" in words or "not" in words:
        return "no"
    return "yes"


def compute_metrics(preds, labels):
    """Compute POPE metrics from binarized predictions and labels."""
    # Convert to binary: yes=1, no=0
    pred_bin = [1 if p == "yes" else 0 for p in preds]
    label_bin = [1 if l == "yes" else 0 for l in labels]

    TP = sum(1 for p, l in zip(pred_bin, label_bin) if p == 1 and l == 1)
    TN = sum(1 for p, l in zip(pred_bin, label_bin) if p == 0 and l == 0)
    FP = sum(1 for p, l in zip(pred_bin, label_bin) if p == 1 and l == 0)
    FN = sum(1 for p, l in zip(pred_bin, label_bin) if p == 0 and l == 1)

    accuracy = (TP + TN) / len(preds) if preds else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    yes_ratio = sum(pred_bin) / len(pred_bin) if pred_bin else 0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "yes_ratio": round(yes_ratio, 4),
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "total": len(preds),
    }


def main():
    args = parse_args()

    # Load data
    answers = load_answers(args.ans_file)
    labels = load_labels(args.label_file)

    if len(answers) != len(labels):
        print(f"ERROR: answer count ({len(answers)}) != label count ({len(labels)})")
        sys.exit(1)

    # Binarize
    preds = [binarize_answer(a.get("answer", a.get("text", ""))) for a in answers]
    labs = [l["label"] for l in labels]

    # Compute overall metrics
    metrics = compute_metrics(preds, labs)

    print("=" * 50)
    print("POPE Evaluation Results")
    print("=" * 50)
    print(f"  Total questions:   {metrics['total']}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  Recall:            {metrics['recall']:.4f}")
    print(f"  F1 score:          {metrics['f1']:.4f}")
    print(f"  Yes ratio:         {metrics['yes_ratio']:.4f}")
    print(f"  ---")
    print(f"  TP: {metrics['TP']}, TN: {metrics['TN']}, "
          f"FP: {metrics['FP']}, FN: {metrics['FN']}")
    print("=" * 50)

    # Optionally show per-question errors
    if args.verbose:
        print("\nPer-question details (errors only):")
        for i, (p, l, lab) in enumerate(zip(preds, labs, labels)):
            if p != l:
                print(f"  Q{i+1}: pred={p}, label={l} | {lab['text']} ({lab['image']})")


if __name__ == "__main__":
    main()
