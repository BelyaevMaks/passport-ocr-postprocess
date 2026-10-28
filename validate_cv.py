#!/usr/bin/env python3
"""
Comprehensive Cross-Validation and Sensitivity Verification Script.
Usage: python validate_cv.py
"""

import json
import random
from evaluate import norm, FIELDS
import postprocess

def run_cross_validation(k=5, seed=42):
    subdiv_codes, subdiv_vocab = postprocess.load_reference_subdivisions('reference/subdivisions.csv')

    with open('data/train_predictions.jsonl', encoding='utf-8') as fp, open('data/train_labels.jsonl', encoding='utf-8') as fl:
        preds = [json.loads(line) for line in fp]
        labels = {d['doc_id']: d['fields'] for d in map(json.loads, fl)}

    random.seed(seed)
    indices = list(range(len(preds)))
    random.shuffle(indices)

    fold_size = len(indices) // k
    print(f"=== 5-Fold Cross-Validation (seed={seed}) ===")
    total_accepted = 0
    total_errors = 0

    for fold in range(k):
        start = fold * fold_size
        end = (fold + 1) * fold_size if fold < k - 1 else len(indices)
        val_idx = set(indices[start:end])
        val_preds = [preds[i] for i in val_idx]

        accepted = 0
        errors = 0
        field_errs = {f: 0 for f in FIELDS}

        for doc in val_preds:
            doc_id = doc['doc_id']
            dec, fields = postprocess.process_document(doc, subdiv_codes, subdiv_vocab)
            if dec == 'accept':
                accepted += 1
                lbl = labels[doc_id]
                wrong = [f for f in FIELDS if norm(fields.get(f)) != norm(lbl.get(f))]
                if wrong:
                    errors += 1
                    for f in wrong:
                        field_errs[f] += 1

        prec = ((accepted - errors) / accepted * 100) if accepted else 100.0
        total_accepted += accepted
        total_errors += errors
        err_details = ", ".join(f"{f}={c}" for f, c in field_errs.items() if c) or "none"
        print(f"Fold {fold+1}: size={len(val_preds)}, accepted={accepted} ({accepted/len(val_preds):.1%}), errors={errors}, precision={prec:.2f}%, err_fields={err_details}")

    overall_prec = ((total_accepted - total_errors) / total_accepted * 100) if total_accepted else 100.0
    print(f"Out-of-Fold Total: accepted={total_accepted} ({total_accepted/len(preds):.1%}), errors={total_errors}, precision={overall_prec:.2f}%\n")

def run_repeated_holdout(n_splits=10, holdout_ratio=0.2):
    subdiv_codes, subdiv_vocab = postprocess.load_reference_subdivisions('reference/subdivisions.csv')
    with open('data/train_predictions.jsonl', encoding='utf-8') as fp, open('data/train_labels.jsonl', encoding='utf-8') as fl:
        preds = [json.loads(line) for line in fp]
        labels = {d['doc_id']: d['fields'] for d in map(json.loads, fl)}

    print(f"=== Repeated Train/Holdout Validation ({n_splits} splits, holdout={holdout_ratio:.0%}) ===")
    print("Split | Seed | Holdout Size | Accepted | Errors | Precision")
    seeds = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    holdout_size = int(len(preds) * holdout_ratio)
    zero_error_count = 0

    for idx, s in enumerate(seeds[:n_splits]):
        random.seed(s)
        indices = list(range(len(preds)))
        random.shuffle(indices)
        val_idx = set(indices[:holdout_size])

        val_acc = 0
        val_err = 0
        for i in val_idx:
            doc = preds[i]
            dec, fields = postprocess.process_document(doc, subdiv_codes, subdiv_vocab)
            if dec == 'accept':
                val_acc += 1
                lbl = labels[doc['doc_id']]
                if any(norm(fields.get(f)) != norm(lbl.get(f)) for f in FIELDS):
                    val_err += 1

        prec = ((val_acc - val_err) / val_acc * 100) if val_acc else 100.0
        if val_err == 0:
            zero_error_count += 1
        print(f" {idx+1:2d}   | {s:4d} |     {holdout_size:4d}     |    {val_acc:2d}    |   {val_err:2d}   |  {prec:6.2f}%")

    print(f"Holdouts with 0 errors: {zero_error_count}/{n_splits} ({zero_error_count/n_splits:.0%})\n")

def run_sensitivity_analysis():
    subdiv_codes, subdiv_vocab = postprocess.load_reference_subdivisions('reference/subdivisions.csv')
    with open('data/train_predictions.jsonl', encoding='utf-8') as fp, open('data/train_labels.jsonl', encoding='utf-8') as fl:
        preds = [json.loads(line) for line in fp]
        labels = {d['doc_id']: d['fields'] for d in map(json.loads, fl)}

    orig_bp = postprocess.MIN_CONF_BP
    orig_org = postprocess.MIN_CONF_ORG

    print("=== Guardrail Threshold Sensitivity Grid (Train) ===")
    print("MIN_CONF_BP | MIN_CONF_ORG | Accepted (Train) | Errors | Precision")
    for bp in [0.9950, 0.9980, 0.9985, 0.9990]:
        for org in [0.980, 0.985, 0.990]:
            postprocess.MIN_CONF_BP = bp
            postprocess.MIN_CONF_ORG = org
            acc = 0
            err = 0
            for doc in preds:
                dec, fields = postprocess.process_document(doc, subdiv_codes, subdiv_vocab)
                if dec == 'accept':
                    acc += 1
                    lbl = labels[doc['doc_id']]
                    if any(norm(fields.get(f)) != norm(lbl.get(f)) for f in FIELDS):
                        err += 1
            prec = ((acc - err) / acc * 100) if acc else 100.0
            print(f"   {bp:.4f}   |    {org:.3f}     |      {acc:3d}         |   {err:2d}   |  {prec:6.2f}%")

    # Restore originals
    postprocess.MIN_CONF_BP = orig_bp
    postprocess.MIN_CONF_ORG = orig_org
    print()

if __name__ == '__main__':
    run_cross_validation(k=5, seed=42)
    run_repeated_holdout(n_splits=10, holdout_ratio=0.2)
    run_sensitivity_analysis()
