"""
Baseline fidelity check for the CRAG reproduction (Objective 1).
 
Verifies that the reproduced pipeline behaves like the published CRAG,
using criteria that do NOT depend on the substituted generator.
 
  Check 1  Score distribution      -- is the released evaluator loaded and scoring?
  Check 2  Action distribution     -- is threshold routing behaving as published?
  Check 3  Accuracy and delta      -- does the corrective mechanism produce the reported effect?
 
Run from inside scripts/ so that `from metrics import match` resolves.
Writes <out>.json and <out>.md alongside a console summary.
"""
 
import argparse
import json
import os
from collections import Counter
 
import numpy as np
 
try:
    from metrics import match          # repo's own substring matcher
    _MATCH_SRC = "metrics.match (repo)"
except ImportError:                    # fallback, identical logic
    def match(prediction, ground_truth):
        for gt in ground_truth:
            if gt in prediction:
                return 1
        return 0
    _MATCH_SRC = "local fallback"
 
 
# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return f.read().split("\n")
 
 
def count_queries(input_file):
    """Replicate data_preprocess()'s consecutive-dedup to get the query count."""
    queries = []
    for line in read_lines(input_file):
        c = line.strip()
        if c.endswith("[SEP]"):
            c += " "
        if " [SEP] " not in c:
            continue
        q = c.split(" [SEP] ", 1)[0]
        if not queries or q != queries[-1]:
            queries.append(q)
    return len(queries)
 
 
def accuracy_of(output_file, gold_answers):
    """Positional zip, exactly as eval.py does."""
    preds = read_lines(output_file)
    n = min(len(preds), len(gold_answers))
    hits = [match(preds[i].strip(), gold_answers[i]) for i in range(n)]
    return {
        "file": output_file,
        "n_scored": n,
        "n_predictions": len(preds),
        "correct": int(sum(hits)),
        "accuracy": round(float(np.mean(hits)), 4) if n else None,
        "empty_predictions": int(sum(1 for p in preds[:n] if not p.strip())),
    }
 
 
# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def check_scores(scores, input_file, ndocs):
    lines = [l for l in read_lines(input_file) if l.strip()]
    empties = sum(1 for l in lines if l.strip().endswith("[SEP]"))
    n_q = count_queries(input_file)
 
    return {
        "n_scores": int(len(scores)),
        "n_queries": n_q,
        "expected_n_scores": n_q * ndocs,
        "count_matches_expected": bool(len(scores) == n_q * ndocs),
        "min": round(float(scores.min()), 4),
        "max": round(float(scores.max()), 4),
        "mean": round(float(scores.mean()), 4),
        "std": round(float(scores.std()), 4),
        "percentiles": {str(p): round(float(np.percentile(scores, p)), 4)
                        for p in (1, 5, 25, 50, 75, 95, 99)},
        "within_unit_interval": bool(scores.min() >= -1.0 and scores.max() <= 1.0),
        "exact_minus_one_sentinels": int((scores == -1.0).sum()),
        "empty_passages_in_input": int(empties),
        "degenerate": bool(scores.std() < 0.01),   # near-constant => broken head
    }
 
 
def check_actions(scores, ndocs, upper, lower):
    flags = np.where(scores >= upper, 2, np.where(scores >= lower, 1, 0))
    if len(flags) % ndocs:
        raise ValueError(f"{len(flags)} scores is not divisible by ndocs={ndocs}")
    per_query = flags.reshape(-1, ndocs).max(axis=1)
 
    c, n = Counter(per_query.tolist()), len(per_query)
    dist = {name: {"n": int(c.get(k, 0)), "pct": round(100 * c.get(k, 0) / n, 2)}
            for k, name in ((2, "Correct"), (1, "Ambiguous"), (0, "Incorrect"))}
 
    top = dist["Correct"]["pct"], dist["Ambiguous"]["pct"], dist["Incorrect"]["pct"]
    return {
        "upper_threshold": upper,
        "lower_threshold": lower,
        "n_queries": n,
        "distribution": dist,
        "passage_level_flags": {name: int((flags == k).sum())
                                for k, name in ((2, "Correct"), (1, "Ambiguous"), (0, "Incorrect"))},
        "collapsed_into_one_action": bool(max(top) > 85.0),
    }
 
 
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help=".scores.npy from CRAG_Inference.py")
    ap.add_argument("--input_file", required=True, help="test_<dataset>.txt")
    ap.add_argument("--gold", help="eval JSONL with an 'answers' field")
    ap.add_argument("--crag_output", help="CRAG predictions")
    ap.add_argument("--rag_output", help="vanilla RAG predictions")
    ap.add_argument("--ndocs", type=int, default=10)
    ap.add_argument("--upper_threshold", type=float, default=0.592)
    ap.add_argument("--lower_threshold", type=float, default=0.995,
                    help="given positive; negated internally, as CRAG_Inference.py does")
    ap.add_argument("--ref_crag", type=float, default=0.544, help="Yalavarthi (2026) CRAG PopQA")
    ap.add_argument("--ref_rag", type=float, default=0.514, help="Yalavarthi (2026) RAG PopQA")
    ap.add_argument("--out", default="fidelity_report")
    args = ap.parse_args()
 
    lower = -args.lower_threshold
    scores = np.load(args.scores)
 
    report = {
        "config": {
            "scores": args.scores, "input_file": args.input_file,
            "ndocs": args.ndocs, "upper_threshold": args.upper_threshold,
            "lower_threshold": lower, "matcher": _MATCH_SRC,
        },
        "check_1_score_distribution": check_scores(scores, args.input_file, args.ndocs),
        "check_2_action_distribution": check_actions(scores, args.ndocs,
                                                     args.upper_threshold, lower),
        "check_3_accuracy": {},
    }
 
    # ---- check 3 -----------------------------------------------------------
    gold = None
    if args.gold and os.path.exists(args.gold):
        gold = [json.loads(l)["answers"] for l in read_lines(args.gold) if l.strip()]
        report["check_3_accuracy"]["n_gold"] = len(gold)
 
    acc = {}
    for key, path in (("crag", args.crag_output), ("rag", args.rag_output)):
        if path and os.path.exists(path) and gold:
            acc[key] = accuracy_of(path, gold)
        elif path:
            acc[key] = {"file": path, "status": "not found yet"}
    report["check_3_accuracy"].update(acc)
 
    if "crag" in acc and "rag" in acc and acc["crag"].get("accuracy") and acc["rag"].get("accuracy"):
        d = acc["crag"]["accuracy"] - acc["rag"]["accuracy"]
        report["check_3_accuracy"]["delta_pp"] = round(100 * d, 2)
        report["check_3_accuracy"]["reference_delta_pp"] = round(100 * (args.ref_crag - args.ref_rag), 2)
 
    # ---- console -----------------------------------------------------------
    c1, c2, c3 = (report["check_1_score_distribution"],
                  report["check_2_action_distribution"],
                  report["check_3_accuracy"])
 
    print("=" * 66)
    print("BASELINE FIDELITY CHECK  (Objective 1)")
    print("=" * 66)
 
    print("\n[CHECK 1] Score distribution")
    print(f"  scores            {c1['n_scores']} (expected {c1['expected_n_scores']} "
          f"= {c1['n_queries']} x {args.ndocs})  {'OK' if c1['count_matches_expected'] else 'MISMATCH'}")
    print(f"  range             [{c1['min']}, {c1['max']}]   mean {c1['mean']}   sd {c1['std']}")
    print(f"  within [-1, 1]    {c1['within_unit_interval']}")
    print(f"  median            {c1['percentiles']['50']}")
    print(f"  empty passages    {c1['empty_passages_in_input']} "
          f"({100*c1['empty_passages_in_input']/max(c1['n_scores'],1):.1f}%), "
          f"sentinels at -1.0: {c1['exact_minus_one_sentinels']}")
    print(f"  degenerate?       {c1['degenerate']}  <- True would mean a broken scoring head")
 
    print("\n[CHECK 2] Action distribution "
          f"(upper {c2['upper_threshold']}, lower {c2['lower_threshold']})")
    for k in ("Correct", "Ambiguous", "Incorrect"):
        v = c2["distribution"][k]
        print(f"  {k:10s} {v['n']:6d}  {v['pct']:6.2f}%")
    print(f"  collapsed into one action? {c2['collapsed_into_one_action']}")
 
    print("\n[CHECK 3] Accuracy")
    if not c3:
        print("  no outputs yet")
    for k in ("crag", "rag"):
        if k in c3:
            v = c3[k]
            if "accuracy" in v:
                print(f"  {k.upper():5s} {v['accuracy']:.4f}  "
                      f"({v['correct']}/{v['n_scored']}), empty preds: {v['empty_predictions']}")
            else:
                print(f"  {k.upper():5s} {v['status']}")
    if "delta_pp" in c3:
        print(f"  delta            {c3['delta_pp']:+.2f} pp   "
              f"(reference {c3['reference_delta_pp']:+.2f} pp)")
 
    # ---- write -------------------------------------------------------------
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
 
    with open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write("# Baseline fidelity check\n\n")
        f.write(f"Scores: `{args.scores}`  \nInput: `{args.input_file}`  \n")
        f.write(f"Thresholds: upper {args.upper_threshold}, lower {lower}\n\n")
        f.write("## Check 1 - score distribution\n\n")
        f.write(f"| n | min | max | mean | sd | median |\n|---|---|---|---|---|---|\n")
        f.write(f"| {c1['n_scores']} | {c1['min']} | {c1['max']} | {c1['mean']} | "
                f"{c1['std']} | {c1['percentiles']['50']} |\n\n")
        f.write("## Check 2 - action distribution\n\n| Action | n | % |\n|---|---|---|\n")
        for k in ("Correct", "Ambiguous", "Incorrect"):
            v = c2["distribution"][k]
            f.write(f"| {k} | {v['n']} | {v['pct']} |\n")
        f.write("\n## Check 3 - accuracy\n\n| Config | Accuracy | n |\n|---|---|---|\n")
        for k in ("crag", "rag"):
            if k in c3 and "accuracy" in c3[k]:
                f.write(f"| {k.upper()} | {c3[k]['accuracy']} | {c3[k]['n_scored']} |\n")
        if "delta_pp" in c3:
            f.write(f"\nDelta: **{c3['delta_pp']:+.2f} pp** "
                    f"(reference {c3['reference_delta_pp']:+.2f} pp)\n")
 
    print(f"\nwritten: {args.out}.json, {args.out}.md")
 
 
if __name__ == "__main__":
    main()
 