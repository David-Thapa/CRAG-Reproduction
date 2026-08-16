#!/bin/sh

cd scripts

dataset=popqa
# CRAG (C2 — baseline)
python CRAG_Inference.py \
--generator_path microsoft/Phi-3-mini-4k-instruct \
--evaluator_path ../models/finetuned_t5_evaluator \
--input_file ../data/$dataset/test_$dataset.txt \
--output_file ../data/$dataset/output/crag_popqa.txt \
--internal_knowledge_path ../data/$dataset/ref/correct \
--external_knowledge_path ../data/$dataset/ref/incorrect \
--combined_knowledge_path ../data/$dataset/ref/ambiguous \
--task $dataset --method crag --device cuda:0 \
--ndocs 10 --batch_size 8 --upper_threshold 0.592 --lower_threshold 0.995

# python CRAG_Inference.py \
# --generator_path YOUR_GENERATOR_PATH \
# --evaluator_path YOUR_EVALUATOR_PATH \
# --input_file ../data/pubqa/test_pubqa.txt \
# --output_file ../data/pubqa/output/YOUR_OUTPUT_FILE \
# --internal_knowledge_path ../data/pubqa/ref/correct \
# --external_knowledge_path ../data/pubqa/ref/incorrect \
# --combined_knowledge_path ../data/pubqa/ref/ambiguous \
# --task pubqa --method crag --device cuda:0 \
# --ndocs 10 --batch_size 8 --upper_threshold 0.5 --lower_threshold 0.915

# Vanilla RAG (C1 — needed for the +3pp delta)
python CRAG_Inference.py \
--generator_path microsoft/Phi-3-mini-4k-instruct \
--evaluator_path ../models/finetuned_t5_evaluator \
--input_file ../data/$dataset/test_$dataset.txt \
--output_file ../data/$dataset/output/rag_popqa.txt \
--task $dataset --method rag --device cuda:0 \
--ndocs 10 --batch_size 8