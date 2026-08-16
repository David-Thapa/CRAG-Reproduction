# CRAG Reproduction

A controlled reproduction of **CRAG — Corrective Retrieval Augmented Generation**
(Yan et al., 2024), built on the open-source reproduction of Yalavarthi (2026)
and run entirely on free components.

MSc Artificial Intelligence dissertation, University of East London.
**David Thapa** · supervised by Dr Azhar Mahmood

This repository is the **baseline**. It exists to establish, with evidence, that
the corrective retrieval mechanism has been reproduced faithfully before anything
is compared against it. What it feeds into is described at the bottom.

---

## What CRAG does

CRAG places a fine-tuned T5-large **retrieval evaluator** between the retriever
and the generator. It scores each retrieved passage against the query, aggregates
those scores to the query level, and compares the result against an upper and a
lower threshold to select one of three corrective actions:

| Action | Evidence passed to the generator |
|---|---|
| **Correct** | The retrieved passages, refined (decompose → filter → recompose) |
| **Incorrect** | The passages are discarded; web search results are used instead |
| **Ambiguous** | Both, combined |

Aggregation is the maximum: a query is routed to Correct if *any single* passage
clears the upper threshold. The decisive passage is the highest-scoring one.

## What this reproduction does

Reproduces that mechanism from released artefacts, with three components outside
the mechanism replaced by free alternatives (declared below), and verifies
fidelity using criteria that do not depend on those substitutions.

---

## Results

### PopQA long-tail

| Configuration | Accuracy | n |
|---|---|---|
| Vanilla RAG | 0.6104 | 854 / 1,399 |
| CRAG | **0.6355** | 889 / 1,399 |
| **Corrective delta** | **+2.51 pp** | reference +3.0 pp |

The corrective delta reproduces Yalavarthi (2026) within 0.5 pp.

Absolute accuracy is higher than the 0.544 reported there. This is expected and
is **not** used as a fidelity criterion: evaluation uses substring matching, which
is sensitive to answer verbosity, which in turn depends on generator version,
prompt formatting and decoding library — none of which affect the retrieval
evaluation under test. The comparable quantity is the delta.

### Fidelity checks

Three generator-independent criteria, since the generator is a declared substitution.

| Check | Result | Status |
|---|---|---|
| 1. Score distribution | 13,990 scores = 1,399 × 10; range [−1.06, +1.10]; mean −0.797; sd 0.593; not degenerate | pass |
| 2. Action distribution | 53.9% Correct / 30.5% Ambiguous / 15.7% Incorrect; no collapse | pass |
| 3. Corrective delta | +2.51 pp vs +3.0 pp published | pass |

The raw scores are **not** strictly bounded by [−1, 1]. They are unbounded
regression logits spanning approximately that range, consistent in scale with the
published thresholds (0.592, −0.995).

### Transfer to PubHealth

The evaluator and thresholds are applied to PubHealth with **no refitting**.

| | Correct | Ambiguous | Incorrect | score sd |
|---|---|---|---|---|
| PopQA (in-domain) | 53.9% | 30.5% | 15.7% | 0.593 |
| PubHealth, PopQA thresholds | 11.0% | **89.0%** | **0.0%** | 0.288 |
| PubHealth, PubHealth thresholds | 11.9% | 82.2% | 6.0% | 0.288 |

1. **The collapse reproduces.** 89.0% of PubHealth queries route to a single
   action, closely matching the 88.3% Yalavarthi (2026) reported on
   ARC-Challenge. Two different transfer datasets, same evaluator, same collapse.
2. **The mechanism is visible.** Score standard deviation halves under transfer
   (0.593 → 0.288). The distribution compresses, so thresholds that partitioned a
   wide distribution into three meaningful regions now sit outside a narrow one.
   Under transferred thresholds the system never selects web search — the
   corrective mechanism has effectively switched itself off.
3. **Per-dataset tuning does not fix it.** Using PubHealth's own published
   thresholds still routes 82.2% to a single action. If the score distribution has
   lost its shape, no choice of two cut-points recovers it.

**Stated caveats.** The transfer changes task as well as domain (short-answer QA →
claim verification), so this measures robustness to transfer in general rather
than domain shift with task held constant. Separately, `data_process.py` formats
passages differently per dataset — PopQA uses `ctx["text"]` alone while PubHealth
uses `ctx["title"] + ' // ' + ctx["text"]` — so the evaluator sees Wikipedia
titles on one dataset and not the other. Both are inherited from the baseline and
are reported rather than treated as controlled.

---

## Provenance

| Component | Source | Status |
|---|---|---|
| Retrieval evaluator (T5-large) | Released checkpoint, Yan et al. (2024) | Reused unmodified, inference only |
| Retrieved passages | `eval_data`, Asai et al. (2024) | Reused unmodified; no retrieval performed here |
| Knowledge refinement | Released procedure, Yan et al. (2024) | Reused unmodified |
| `ref/{correct,incorrect,ambiguous}` | Precomputed, Yalavarthi (2026) | Reused unmodified |
| Generator | Phi-3-mini-4k-instruct | Free substitution, following Yalavarthi (2026) |
| Web search | Wikipedia API | Free substitution, following Yalavarthi (2026) |
| Keyword extraction | Rule-based | Free substitution for the GPT-3.5 step |
| `fidelity_check.py` | — | Written for this study |

Upstream commit: `568612e80bd17665f76ffeeccb93c056f87880df` (17 Mar 2026),
from [suryayalavarthi/crag-reproduction](https://github.com/suryayalavarthi/crag-reproduction).

PopQA evaluation data was obtained from the HuggingFace mirror
[`awinml/popqa_longtail_w_gs`](https://huggingface.co/datasets/awinml/popqa_longtail_w_gs),
the official Google Drive release being unavailable. The mirror was verified by
exact string match of all 1,399 questions, in order, against the `sources` file
distributed with the reproduction.

### Modifications to upstream code

All changes are declared. **None alters the decision logic.**

| File | Change | Reason |
|---|---|---|
| `CRAG_Inference.py` | Evaluator freed (`del model` + `empty_cache`) before the vLLM engine is constructed | vLLM reserves ~90% of VRAM by default; on a 16GB T4 the evaluator could not then be loaded |
| `CRAG_Inference.py` | `gpu_memory_utilization=0.80` | Same 16GB constraint |
| `CRAG_Inference.py` | Raw evaluator scores cached to `.scores.npy` | Avoids re-scoring for every configuration |
| `data_process.py` | Unused dataset paths commented out | Only PopQA and PubHealth are in scope |
| `requirements.txt` | Rewritten | The upstream file is a `pip freeze` pinning `torch`, `triton`, `xformers` and the CUDA runtime, which is unresolvable on current hosted GPU environments |

---

## Data

| Dataset | Items | Passages | Lines in `test_*.txt` |
|---|---|---|---|
| PopQA (long-tail) | 1,399 (1,385 distinct questions; 13 repeated) | 10 per item, top-10 of 25 available | 13,990 |
| PubHealth (test) | 987 | 10 per item | 9,870 |

Passage-level reliability labels exist only for PopQA, derived using the released
procedure:

```python
label = 1 if ctx["title"] == item["s_wiki_title"] else 0
```

This records whether a passage comes from the intended entity's Wikipedia page,
not whether it supports the answer — a provenance proxy rather than a direct
reliability judgement. PubHealth supplies labels only for the final verdict.

**1,321 of 13,990 PopQA passages (9.4%) are empty.** `inference()` assigns these a
hardcoded `-1.0` rather than a model output. Since the PopQA median score is
−1.005, this sentinel spike sits close to the centre of the distribution.
PubHealth has no empty passages.

---

## Setup

```bash
git clone https://github.com/David-Thapa/CRAG-Reproduction
cd CRAG-Reproduction
pip install -r requirements.txt
```

Do **not** install or pin `torch`, `triton`, `xformers` or any `nvidia-*` package
on a hosted GPU environment — they ship with the image, matched to the driver.

Download the T5 evaluator checkpoint from the
[original CRAG repository](https://github.com/HuskyInSalt/CRAG) into
`models/finetuned_t5_evaluator/`. The generator is pulled from HuggingFace
automatically on first run.

## Running

```bash
bash run_crag_inference.sh     # CRAG and vanilla RAG on PopQA
bash run_eval.sh               # accuracy for both
bash run_fidelity_check.sh     # the three fidelity checks
```

`data_process.py` only needs to be re-run to regenerate `test_popqa.txt` **with**
the `\t0/1` reliability labels, which the shipped file does not carry.

Knowledge preparation is **not** re-run. The `ref/` files are reused as released:
`external_knowledge_preparation.py` calls `extract_keywords()` in `utils.py`,
which requires a paid OpenAI key, and regenerating against a current Wikipedia
snapshot would break comparability with the reproduction being checked against.

Approximate cost on a single 16GB T4: ~1 hour of evaluator scoring for PopQA plus
~2.5 hours of generation per configuration. Generation is issued one prompt at a
time, which forgoes vLLM's continuous batching; batching the calls produces
identical output under greedy decoding and is under consideration for subsequent
runs.

## Repository layout

```
data/
  popqa/    sources · test_popqa.txt · ref/{correct,incorrect,ambiguous} · output/
  pubqa/    sources · test_pubqa.txt · retrieved_psgs · ref/{...} · output/
models/     T5 evaluator checkpoint (not tracked)
retrieval_lm/eval_data/   evaluation JSONL with gold answers
scripts/    pipeline and analysis code
run_*.sh    pipeline stages
```

---

## Status

- [x] CRAG baseline reproduced on PopQA; three fidelity checks pass
- [x] Vanilla RAG baseline
- [x] Evaluator scores cached for PopQA and PubHealth
- [x] Transfer action distributions under both threshold settings
- [ ] PubHealth gold labels — the released `health_claims_processed.jsonl` is
      unavailable; to be reconstructed from the original PUBHEALTH release by
      claim-text matching
- [ ] Prompt truncation for PubHealth generation (some prompts exceed the
      4,096-token context of Phi-3-mini-4k-instruct)
- [ ] Closed-book diagnostic, to bound the headroom available to any retrieval
      evaluation strategy

---

## What this feeds into

This reproduction is the baseline for **UC-CRAG (Uncertainty-Calibrated Corrective
Retrieval-Augmented Generation)**, developed in a separate repository.

The results above motivate it. CRAG's evaluator returns a raw score, not a
probability, and nothing in the framework establishes what that score means. The
thresholds are set per dataset, which requires labelled data from the target
domain — exactly what is missing when a system meets an unfamiliar one. The
transfer results here show the consequence: the score distribution compresses,
fixed thresholds stop partitioning it, and 89% of queries collapse into a single
action while the system registers no problem. Retuning the thresholds recovers
little.

UC-CRAG replaces the threshold comparison with a calibrated probability *p* and an
uncertainty estimate *σ*, selecting the corrective action from the pair. The
evaluator, the retriever, the generator and the three corrective actions are all
left unchanged — the intervention is confined to the step that turns an
evaluator score into a decision. Setting the uncertainty tolerance to infinity and
the calibration mapping to the identity recovers CRAG's rule exactly, so the
baseline reproduced here is a strict special case of it.

Everything in this repository is held fixed for that comparison: the same cached
evaluator scores, the same retrieved passages, the same knowledge files, the same
generator. Only the decision rule changes.

---

## References

```bibtex
@inproceedings{yan2024corrective,
  title     = {Corrective Retrieval Augmented Generation},
  author    = {Yan, Shi-Qi and Gu, Jia-Chen and Zhu, Yun and Ling, Zhen-Hua},
  booktitle = {International Conference on Learning Representations},
  year      = {2024}
}

@article{yalavarthi2026crag,
  title   = {Open-Source Reproduction and Explainability Analysis of
             Corrective Retrieval Augmented Generation},
  author  = {Yalavarthi, Surya Vardhan},
  journal = {arXiv preprint arXiv:2603.16169},
  year    = {2026}
}

@inproceedings{asai2024selfrag,
  title     = {Self-RAG: Learning to Retrieve, Generate and Critique through
               Self-Reflection},
  author    = {Asai, Akari and Wu, Zeqiu and Wang, Yizhong and Sil, Avirup and
               Hajishirzi, Hannaneh},
  booktitle = {International Conference on Learning Representations},
  year      = {2024}
}

@inproceedings{mallen2023popqa,
  title     = {When Not to Trust Language Models: Investigating Effectiveness of
               Parametric and Non-Parametric Memories},
  author    = {Mallen, Alex and Asai, Akari and Zhong, Victor and Das, Rajarshi
               and Khashabi, Daniel and Hajishirzi, Hannaneh},
  booktitle = {ACL},
  year      = {2023}
}
```

## Acknowledgements

This work builds directly on CRAG by Yan et al., the Self-RAG evaluation data
released by Asai et al., and the open-source reproduction and explainability
analysis by Yalavarthi. The T5 retrieval evaluator checkpoint is the original
authors'. Reuse is central to this methodology rather than incidental to it, and
is cited accordingly.
