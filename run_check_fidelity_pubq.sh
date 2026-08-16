cd scripts

# C2 — PopQA thresholds, no retuning
python fidelity_check.py \
--scores ../data/pubqa/output/crag_pubqa_c2.txt.scores.npy \
--input_file ../data/pubqa/test_pubqa.txt \
--ndocs 10 --upper_threshold 0.592 --lower_threshold 0.995 \
--out ../data/pubqa/output/fidelity_pubqa_c2

# C2b — PubHealth published thresholds
python fidelity_check.py \
--scores ../data/pubqa/output/crag_pubqa_c2.txt.scores.npy \
--input_file ../data/pubqa/test_pubqa.txt \
--ndocs 10 --upper_threshold 0.5 --lower_threshold 0.915 \
--out ../data/pubqa/output/fidelity_pubqa_c2b