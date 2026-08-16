# Baseline fidelity check

Scores: `../data/pubqa/output/crag_pubqa_c2.txt.scores.npy`  
Input: `../data/pubqa/test_pubqa.txt`  
Thresholds: upper 0.5, lower -0.915

## Check 1 - score distribution

| n | min | max | mean | sd | median |
|---|---|---|---|---|---|
| 9870 | -1.0233 | 1.0088 | -0.8397 | 0.2881 | -0.9196 |

## Check 2 - action distribution

| Action | n | % |
|---|---|---|
| Correct | 117 | 11.85 |
| Ambiguous | 811 | 82.17 |
| Incorrect | 59 | 5.98 |

## Check 3 - accuracy

| Config | Accuracy | n |
|---|---|---|
