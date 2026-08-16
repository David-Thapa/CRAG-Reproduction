cd scripts

python eval.py \
  --input_file ../retrieval_lm/eval_data/popqa_longtail_w_gs.jsonl \
  --eval_file ../data/popqa/output/crag_popqa.txt \
  --metric match --task popqa

python eval.py \
  --input_file ../retrieval_lm/eval_data/popqa_longtail_w_gs.jsonl \
  --eval_file ../data/popqa/output/rag_popqa.txt \
  --metric match --task popqa