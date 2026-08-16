# Baseline fidelity check

Scores: `../data/pubqa/output/crag_pubqa_c2.txt.scores.npy`  
Input: `../data/pubqa/test_pubqa.txt`  
Thresholds: upper 0.592, lower -0.995

## Check 1 - score distribution

| n | min | max | mean | sd | median |
|---|---|---|---|---|---|
| 9870 | -1.0233 | 1.0088 | -0.8397 | 0.2881 | -0.9196 |

## Check 2 - action distribution

| Action | n | % |
|---|---|---|
| Correct | 109 | 11.04 |
| Ambiguous | 878 | 88.96 |
| Incorrect | 0 | 0.0 |

## Check 3 - accuracy

| Config | Accuracy | n |
|---|---|---|
