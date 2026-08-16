#!/bin/sh

cd scripts

dataset=popqa

# Vanilla RAG (C1 — needed for the +3pp delta)
python CRAG_Inference.py \
--generator_path microsoft/Phi-3-mini-4k-instruct \
--evaluator_path ../models/finetuned_t5_evaluator \
--input_file ../data/$dataset/test_$dataset.txt \
--output_file ../data/$dataset/output/rag_popqa.txt \
--task $dataset --method rag --device cuda:0 \
--ndocs 10 --batch_size 8