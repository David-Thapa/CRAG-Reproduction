# Corrective RAG — Reproduction and Threshold Transfer Analysis

A controlled, zero-cost reproduction of **CRAG** (Yan et al., 2024), with a quantitative
analysis of what happens to its fixed confidence thresholds under domain transfer.

**David Thapa** — MSc Artificial Intelligence, University of East London
Supervised by Dr Azhar Mahmood

This repository is the verified baseline for a dissertation on uncertainty-calibrated
corrective retrieval. Everything here is held fixed for that comparison.

---

## Overview

CRAG places a fine-tuned T5-large **retrieval evaluator** between the retriever and the
generator. It scores each retrieved passage, takes the maximum across the query, and
compares that against two fixed thresholds to select a corrective action.

| Action | Evidence passed to the generator |
|---|---|
| Correct | Retrieved passages, refined |
| Incorrect | Passages discarded; web search used instead |
| Ambiguous | Both, combined |

This reproduction rebuilds that mechanism from released artefacts, substituting three
components outside the mechanism with free alternatives.

| Component | Original | Here |
|---|---|---|
| Generator | LLaMA-2-7B (fine-tuned) | Phi-3-mini-4k-instruct |
| Web search | Google Search API (paid) | Wikipedia (precomputed) |
| Keyword extraction | GPT-3.5 Turbo | Rule-based |
| Retrieval evaluator | T5-large (fine-tuned) | Same checkpoint |

**Datasets.** This reproduction evaluates on **PopQA (long-tail)** and **PubHealth**.
Yalavarthi (2026) reproduced CRAG on PopQA and **ARC-Challenge**; ARC-Challenge is not
used here, and neither is Biography. PopQA is the in-domain set — the distribution the
evaluator was fine-tuned on, and the only one of the four with passage-level reliability
labels. PubHealth is the transfer set: its claims are not keyed to biographical entities,
which makes it the site of the domain-shift analysis below. The PopQA results are
therefore directly comparable with the published reproduction, while the PubHealth
transfer analysis is new to this work.

## Key results

**PopQA long-tail** — 1,399 instances

| Configuration | Accuracy | Reference |
|---|---|---|
| Vanilla RAG | 0.6104 | 0.514 |
| CRAG | 0.6355 | 0.544 |
| **Corrective effect** | **+2.51 pp** | +3.0 pp |

Comparable with Yalavarthi (2026), who reports the same PopQA long-tail split.

**Transfer to PubHealth** — 987 claims, no parameter refitted. Not examined in the
reproduction this work follows, which used ARC-Challenge as its transfer set.

| | Correct | Ambiguous | Incorrect | Score s.d. |
|---|---|---|---|---|
| PopQA, in domain | 53.9% | 30.5% | 15.7% | 0.593 |
| PubHealth, transferred thresholds | 11.0% | **89.0%** | **0.0%** | 0.288 |
| PubHealth, tuned thresholds | 11.9% | 82.2% | 6.0% | 0.288 |

## Findings

1. **The corrective mechanism reproduces.** +2.51 pp against +3.0 pp published. Absolute
   accuracy is higher because evaluation uses substring matching, which rewards verbose
   answers; the effect size is the comparable quantity, and it is preserved.

2. **The threshold collapse appears on a second, independent dataset.** Yalavarthi (2026)
   observed 88.3% of ARC-Challenge queries collapsing into a single action. Measured here
   on **PubHealth** — a different transfer domain, and one not examined in that work —
   89.0% collapse the same way, under the same evaluator. Across all 987 claims, web
   search is never selected once. Two unrelated transfer datasets producing the same
   concentration turns a single observation into a pattern.

3. **The mechanism is visible in the scores.** Score dispersion halves under transfer
   (s.d. 0.593 → 0.288). Thresholds that partitioned a wide distribution into three
   populated regions no longer partition a narrow one.

4. **Per-dataset tuning does not repair it.** PubHealth's own published thresholds still
   route 82.2% to a single action. The problem is not the choice of cut-points but the
   loss of the distributional shape they were chosen against.

5. **Every decision rests on one passage.** The median passage scores −1.005, below the
   lower threshold, yet 53.9% of PopQA queries route to Correct. Max aggregation means a
   single passage above the upper threshold determines the action for the whole query.

**Caveats.** The transfer changes task as well as domain (QA → claim verification).
Separately, the released preprocessing prepends passage titles on PubHealth but not on
PopQA, so part of the compression may be a formatting effect. Both are inherited from the
baseline and reported rather than controlled.

## Fidelity checks

Generator-independent, since the generator is a declared substitution.

| Check | Observed | Status |
|---|---|---|
| Score distribution | 13,990 = 1,399 × 10; range [−1.06, +1.10]; s.d. 0.593 | pass |
| Action distribution | 53.9 / 30.5 / 15.7; no collapse | pass |
| Corrective effect | +2.51 pp vs +3.0 pp | pass |

Scores are unbounded regression logits, not confined to [−1, 1], though consistent in
scale with the published thresholds (0.592, −0.995).

## Setup

```bash
git clone https://github.com/David-Thapa/CRAG-Reproduction
cd CRAG-Reproduction
pip install -r requirements.txt
```

Download the T5 evaluator checkpoint from the
[original CRAG repository](https://github.com/HuskyInSalt/CRAG) into
`models/finetuned_t5_evaluator/`. The generator is pulled from HuggingFace on first run.

> Do not install or pin `torch`, `triton`, `xformers` or any `nvidia-*` package on a
> hosted GPU environment — they ship with the image, matched to the driver.

## Running

```bash
bash run_crag_inference.sh     # CRAG and vanilla RAG on PopQA
bash run_eval.sh               # accuracy for both
bash run_fidelity_check.sh     # the three fidelity checks
```

Knowledge preparation is **not** re-run — the `ref/` files are reused as released.
Regenerating them requires a paid API key and would query a different Wikipedia snapshot,
breaking comparability with the reproduction being checked against.

Approximate cost on one 16GB T4: ~1 h evaluator scoring, ~2.5 h generation per
configuration.

## Data

| | PopQA (long-tail) | PubHealth (test) |
|---|---|---|
| Instances | 1,399 (1,385 distinct questions) | 987 |
| Passages per instance | 10, of 25 available | 10 |
| Pairs scored | 13,990 | 9,870 |
| Empty passages | 1,321 (9.4%) | 0 |
| Passage-level labels | yes | no |

Reliability labels are derived using the released rule —
`label = 1 if ctx["title"] == item["s_wiki_title"]` — which records provenance rather than
answer support. Empty passages receive a hardcoded `-1.0` rather than a model score, and
are excluded before any calibration is fitted.

## Repository structure

```
├── data/
│   ├── popqa/          sources · test_popqa.txt · ref/{correct,incorrect,ambiguous} · output/
│   └── pubqa/          sources · test_pubqa.txt · retrieved_psgs · ref/ · output/
├── models/             T5 evaluator checkpoint (not tracked)
├── retrieval_lm/       evaluation JSONL with gold answers
├── scripts/
│   ├── CRAG_Inference.py       pipeline: score → route → generate
│   ├── fidelity_check.py       the three fidelity criteria
│   ├── data_process.py         builds test_*.txt from the evaluation JSONL
│   └── eval.py, metrics.py     accuracy
└── run_*.sh            pipeline stages
```

## Provenance

| Component | Source | Status |
|---|---|---|
| Retrieval evaluator | Yan et al. (2024) | Reused unmodified, inference only |
| Retrieved passages | Asai et al. (2024) | Reused unmodified; no retrieval performed here |
| Knowledge refinement | Yan et al. (2024) | Reused unmodified |
| `ref/` evidence files | Yalavarthi (2026) | Reused unmodified |
| Generator, web search, keyword extraction | — | Free substitutions, following Yalavarthi (2026) |
| `fidelity_check.py` | — | Written for this study |

Upstream commit `568612e` (17 Mar 2026), from
[suryayalavarthi/crag-reproduction](https://github.com/suryayalavarthi/crag-reproduction).

PopQA evaluation data was taken from the mirror
[`awinml/popqa_longtail_w_gs`](https://huggingface.co/datasets/awinml/popqa_longtail_w_gs),
the official release being inaccessible. Verified by exact ordered string match of all
1,399 questions against the released `sources` file.

### Modifications

None alters the decision logic.

| File | Change | Reason |
|---|---|---|
| `CRAG_Inference.py` | Evaluator released from memory before the vLLM engine is built | vLLM reserves ~90% of VRAM; the evaluator could not otherwise load on a 16GB T4 |
| `CRAG_Inference.py` | `gpu_memory_utilization=0.80` | Same constraint |
| `CRAG_Inference.py` | Raw scores cached to `.scores.npy` | Avoids re-scoring per configuration |
| `data_process.py` | Unused dataset paths disabled | Only PopQA and PubHealth in scope |
| `requirements.txt` | Rewritten | Upstream pins `torch`, the CUDA runtime and a `transformers` version predating the generator it uses |

## Next

These results motivate **UC-CRAG**, developed separately: the threshold comparison is
replaced by a calibrated probability and an uncertainty estimate, with the evaluator,
retriever, generator and corrective actions unchanged. Setting the uncertainty tolerance
to infinity and the calibration mapping to the identity recovers CRAG exactly, so the
baseline reproduced here is a strict special case.

## Citation

```bibtex
@inproceedings{yan2024corrective,
  title     = {Corrective Retrieval Augmented Generation},
  author    = {Yan, Shi-Qi and Gu, Jia-Chen and Zhu, Yun and Ling, Zhen-Hua},
  booktitle = {International Conference on Learning Representations},
  year      = {2024}
}
 
@misc{yalavarthi2026crag,
  title         = {Open-Source Reproduction and Explainability Analysis of
                   Corrective Retrieval Augmented Generation},
  author        = {Yalavarthi, Surya Vardhan},
  year          = {2026},
  eprint        = {2603.16169},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
 
@inproceedings{mallen2023popqa,
  title     = {When Not to Trust Language Models: Investigating Effectiveness of
               Parametric and Non-Parametric Memories},
  author    = {Mallen, Alex and Asai, Akari and Zhong, Victor and Das, Rajarshi and
               Khashabi, Daniel and Hajishirzi, Hannaneh},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for
               Computational Linguistics},
  year      = {2023}
}
 
@inproceedings{asai2024selfrag,
  title     = {Self-RAG: Learning to Retrieve, Generate and Critique through Self-Reflection},
  author    = {Asai, Akari and Wu, Zeqiu and Wang, Yizhong and Sil, Avirup and
               Hajishirzi, Hannaneh},
  booktitle = {International Conference on Learning Representations},
  year      = {2024}
}
```

## Acknowledgements

Builds on CRAG by Yan et al., the Self-RAG evaluation data released by Asai et al., and
the open-source reproduction and explainability analysis by Yalavarthi. The T5 retrieval
evaluator checkpoint is the original authors'.
