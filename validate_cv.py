import json
import random
from postprocess import process_document, load_reference_subdivisions
from evaluate import norm, FIELDS

def run_cross_validation(k=5, seed=42):
    subdiv_codes, subdiv_vocab = load_reference_subdivisions('reference/subdivisions.csv')

    with open('data/train_predictions.jsonl', encoding='utf-8') as fp, open('data/train_labels.jsonl', encoding='utf-8') as fl:
        preds = [json.loads(line) for line in fp]
        labels = {d['doc_id']: d['fields'] for d in map(json.loads, fl)}

    random.seed(seed)
    indices = list(range(len(preds)))
    random.shuffle(indices)

    fold_size = len(indices) // k
    fold_results = []

    print(f"=== Cross-Validation ({k} folds, seed={seed}) ===")
    total_val_docs = 0
    total_val_accepted = 0
    total_val_errors = 0

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
            dec, fields = process_document(doc, subdiv_codes, subdiv_vocab)
            if dec == 'accept':
                accepted += 1
                lbl = labels[doc_id]
                wrong = [f for f in FIELDS if norm(fields.get(f)) != norm(lbl.get(f))]
                if wrong:
                    errors += 1
                    for f in wrong:
                        field_errs[f] += 1
        
        prec = ((accepted - errors) / accepted * 100) if accepted else 100.0
        total_val_docs += len(val_preds)
        total_val_accepted += accepted
        total_val_errors += errors
        
        err_flds = {f: c for f, c in field_errs.items() if c}
        print(f"Fold {fold+1}: size={len(val_preds)}, accepted={accepted} ({accepted/len(val_preds):.1%}), errors={errors}, precision={prec:.2f}%, err_fields={err_flds}")

    mean_prec = (total_val_accepted - total_val_errors) / total_val_accepted * 100
    print(f"\nOverall Out-of-Fold: total={total_val_docs}, accepted={total_val_accepted} ({total_val_accepted/total_val_docs:.1%}), errors={total_val_errors}, precision={mean_prec:.2f}%")

if __name__ == '__main__':
    run_cross_validation(k=5, seed=42)
