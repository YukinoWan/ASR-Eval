import json
from utils.metrics import normalized_wer_score
from tqdm import tqdm

datasets = ["earning22", "voxpopuli", "tedlium"]
# datasets = ["medasr"]
q_llm = "gemini-1.5-flash"
a_llm = "gpt-4o"
models = ["whisper_v3_1best"]

for dataset in datasets:
    for model in models:
        print(f"dataset: {dataset}, model: {model}") 

        with open(f"/mnt/home/zhenwan.nlp/ASR-Eval/QA_eval/task-{dataset}-{q_llm}-qa-200-asr-gold-answer-{a_llm}.json", "r") as f:
            gold_data = json.load(f)

        with open(f"/mnt/home/zhenwan.nlp/ASR-Eval/QA_eval/task-{dataset}-{q_llm}-qa-200-asr-{model}-answer-{a_llm}.json", "r") as f:
            model_data = json.load(f)
        # with open(f"/mnt/home/zhenwan.nlp/ASR-Eval/QA_results/{dataset}-whisper-large-v2.json", "r") as f:
        #     v2_data = json.load(f)

        assert len(gold_data) == len(model_data)
        model_hard_accuracy = []    
        model_mid_accuracy = []
        model_easy_accuracy = []

        gold_hard_accuracy = []
        gold_mid_accuracy = []
        gold_easy_accuracy = []

        gold_preds = []
        model_preds = []
        for i in tqdm(range(len(gold_data))):
            gold_preds.append(gold_data[i]["gold"])
            model_preds.append(model_data[i][f"{model}"][0])
            for j in range(len(gold_data[i]["qa"])):
                if gold_data[i]["qa"][j][f"gold_{a_llm}_easy_accuracy"] == 0:
                    continue
                else:
                    gold_hard_accuracy.append(gold_data[i]["qa"][j][f"gold_{a_llm}_hard_accuracy"])
                    gold_mid_accuracy.append(gold_data[i]["qa"][j][f"gold_{a_llm}_mid_accuracy"])
                    gold_easy_accuracy.append(gold_data[i]["qa"][j][f"gold_{a_llm}_easy_accuracy"])

                    model_easy_accuracy.append(model_data[i]["qa"][j][f"{model}_{a_llm}_easy_accuracy"])
                    model_mid_accuracy.append(model_data[i]["qa"][j][f"{model}_{a_llm}_mid_accuracy"])
                    model_hard_accuracy.append(model_data[i]["qa"][j][f"{model}_{a_llm}_hard_accuracy"])

        print("gold hard accuracy: ", sum(gold_hard_accuracy) / len(gold_hard_accuracy))
        print("gold mid accuracy: ", sum(gold_mid_accuracy) / len(gold_mid_accuracy))
        print("gold easy accuracy: ", sum(gold_easy_accuracy) / len(gold_easy_accuracy))

        print("model hard accuracy: ", sum(model_hard_accuracy) / len(model_hard_accuracy))
        print("model mid accuracy: ", sum(model_mid_accuracy) / len(model_mid_accuracy))
        print("model easy accuracy: ", sum(model_easy_accuracy) / len(model_easy_accuracy))

        print("wer: ", normalized_wer_score(model_preds, gold_preds))
        


