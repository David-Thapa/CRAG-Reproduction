#!/bin/sh
# Baseline fidelity check (Objective 1).
# Safe to run before rag_popqa.txt exists -- missing outputs are reported, not fatal.
 
cd scripts
 
dataset=popqa
 
python check_fidelity.py \
--scores ../data/$dataset/output/crag_$dataset.txt.scores.npy \
--input_file ../data/$dataset/test_$dataset.txt \
--gold ../retrieval_lm/eval_data/popqa_longtail_w_gs.jsonl \
--crag_output ../data/$dataset/output/crag_$dataset.txt \
--rag_output ../data/$dataset/output/rag_$dataset.txt \
--ndocs 10 --upper_threshold 0.592 --lower_threshold 0.995 \
--ref_crag 0.544 --ref_rag 0.514 \
--out ../data/$dataset/output/fidelity_$dataset
 
# PubHealth, for the transfer condition later
# python fidelity_check.py \
# --scores ../data/pubqa/output/crag_pubqa.txt.scores.npy \
# --input_file ../data/pubqa/test_pubqa.txt \
# --crag_output ../data/pubqa/output/crag_pubqa.txt \
# --ndocs 10 --upper_threshold 0.5 --lower_threshold 0.915 \
# --out ../data/pubqa/output/fidelity_pubqa
 