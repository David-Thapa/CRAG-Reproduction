# ============================================
# Popqa Dataset
# ================================================
# from datasets import load_dataset
# ds = load_dataset("awinml/popqa_longtail_w_gs")
# print(ds)                      # check the split name

# split = list(ds.keys())[0]     # usually "train"
# out = "data/popqa_longtail_w_gs.jsonl"

# import os; 
# os.makedirs(os.path.dirname(out), exist_ok=True)
# ds[split].to_json(out, orient="records", lines=True)

# ================================================
# Checking the whether json and the source are the same data
# ================================================
# import json
# src = [l.rstrip("\n") for l in open("data/popqa/sources", encoding="utf-8")]
# jsn = [json.loads(l)["question"] for l in open("retrieval_lm\eval_data\popqa_longtail_w_gs.jsonl", encoding="utf-8")]
# print(len(src), len(jsn))
# print("aligned:", src == jsn)