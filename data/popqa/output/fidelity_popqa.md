# Baseline fidelity check

Scores: `../data/popqa/output/crag_popqa.txt.scores.npy`  
Input: `../data/popqa/test_popqa.txt`  
Thresholds: upper 0.592, lower -0.995

## Check 1 - score distribution

| n | min | max | mean | sd | median |
|---|---|---|---|---|---|
| 13990 | -1.0634 | 1.0953 | -0.7972 | 0.5931 | -1.005 |

## Check 2 - action distribution

| Action | n | % |
|---|---|---|
| Correct | 754 | 53.9 |
| Ambiguous | 426 | 30.45 |
| Incorrect | 219 | 15.65 |

## Check 3 - accuracy

| Config | Accuracy | n |
|---|---|---|
| CRAG | 0.6355 | 1399 |
| RAG | 0.6104 | 1399 |

Delta: **+2.51 pp** (reference +3.00 pp)
