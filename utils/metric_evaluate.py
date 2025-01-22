import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from tqdm import tqdm
import json
import metrics
import sema_score_generator



candidates_path = sys.argv[1]
references_path = sys.argv[2]
task = sys.argv[3]


with open(candidates_path, "r") as f:
    candidates = json.load(f)

with open(references_path, "r") as f:
    references = json.load(f)

results = {}

wer_results = []
normalized_wer_results = []
bleu_results = []
bertscore_results = []
semascore_results = []
similarity_results = []

# def default_value():
#     return 1.0

# model_type = "microsoft/deberta-large-mnli"
# num_layers = 18
# tokenizer = sema_score_generator.get_tokenizer(model_type, use_fast=False)
# model = sema_score_generator.get_model(model_type, num_layers)
# device = "cpu"


# idf_dict = sema_score_generator.defaultdict(default_value)
# # set idf for [SEP] and [CLS] to 0
# idf_dict[tokenizer.sep_token_id] = 0
# idf_dict[tokenizer.cls_token_id] = 0

for i in tqdm(range(len(candidates))):
    candidate = candidates[i]
    reference = references[i]

    wer = metrics.wer_score([candidate], [reference])
    wer_results.append(wer)

    normalized_wer = metrics.normalized_wer_score([candidate], [reference])
    normalized_wer_results.append(normalized_wer)

    bleu = metrics.bleu_score([candidate], [reference])
    bleu_results.append(bleu)

bertscore_results = metrics.bert_score(candidates, references)

# semascore_results = metrics.sema_score(candidates, references)

similarity_results = metrics.sentence_similarity_score(candidates, references)

results["WER"] = wer_results
results["Normalized WER"] = normalized_wer_results
results["BLEU"] = bleu_results
results["BERTScore"] = bertscore_results
# results["SEMAScore"] = semascore_results
results["Similarity"] = [float(x) for x in similarity_results]

with open(f"results/evaluation_{task}.json", "w") as f:
    json.dump(results, f)
