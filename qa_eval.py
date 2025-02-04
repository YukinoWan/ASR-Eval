import json
from utils.metrics import normalized_wer_score
from tqdm import tqdm

datasets = ["earning22", "voxpopuli", "tedlium", "medasr"]
# datasets = ["medasr"]
q_llm = "gpt-4o"
a_llm = "meta-llama/Meta-Llama-3-8B-Instruct"
models = ["whisper-large-v2"]
asr_key = "whisper_v2_1best"

def compare_answer(correct, answer):
    # print(correct, answer)
    if f"({correct})" in answer or f"[{correct}]" in answer:
        # assert False
        return 1
    else:
        return 0

def eval_answer(correct, answers):
    acc_dict ={"hard" : 0, "mid": 0, "easy": 0}
    acc = [compare_answer(correct, answer) for answer in answers]
    if sum(acc) == len(acc):
        acc_dict["hard"] = 1
    if sum(acc) >= len(acc) / 2:
        acc_dict["mid"] = 1
    if sum(acc) >= 1:
        acc_dict["easy"] = 1
    return acc_dict

for dataset in datasets:
    for model in models:
        print(f"dataset: {dataset}, model: {model}") 

        # with open(f"/mnt/home/zhenwan.nlp/ASR-Eval/QA_eval/subset/task-{dataset}-{q_llm}-qa-asr-gold-answer-gpt-4o.json", "r") as f:
        #     gold_data = json.load(f)

        with open(f"/mnt/home/zhenwan.nlp/ASR-Eval/QA_eval/subset/task-{dataset}-{q_llm}-qa-asr-{model}-answer-llama3-8b.json", "r") as f:
            model_data = json.load(f)
        
        with open(f"/mnt/home/zhenwan.nlp/ASR-Eval/ASR_results/subset/{dataset}-{model}.json", "r") as f:
            asr_data = json.load(f)

        with open(f"/mnt/home/zhenwan.nlp/ASR-Eval/llm_respond_results/subset/llm_eval_{dataset}_{model}.json", "r") as f:
            llm_data = json.load(f)
        # with open(f"/mnt/home/zhenwan.nlp/ASR-Eval/QA_results/{dataset}-whisper-large-v2.json", "r") as f:
        #     v2_data = json.load(f)

        assert len(asr_data) == len(model_data)
        model_hard_accuracy = []    
        model_mid_accuracy = []
        model_easy_accuracy = []

        back_list = []
        summarize_list = []

        # gold_hard_accuracy = []
        # gold_mid_accuracy = []
        # gold_easy_accuracy = []

        gold_preds = []
        model_preds = []
        for i in tqdm(range(len(model_data))):
            gold_preds.append(asr_data[i]["gold"])
            model_preds.append(asr_data[i][f"{asr_key}"][0])
            back_list.append(llm_data[i]["back_layer_last"])
            summarize_list.append(llm_data[i]["summarize_layer_last"])
            for j in range(len(model_data[i]["qa"])):
                question = model_data[i]["qa"][j]
                # if gold_data[i]["qa"][j][f"gold_{a_llm}_easy_accuracy"] == 0:
                #     continue
                # else:
                    # gold_hard_accuracy.append(gold_data[i]["qa"][j][f"gold_{a_llm}_hard_accuracy"])
                    # gold_mid_accuracy.append(gold_data[i]["qa"][j][f"gold_{a_llm}_mid_accuracy"])
                    # gold_easy_accuracy.append(gold_data[i]["qa"][j][f"gold_{a_llm}_easy_accuracy"])
                answer = model_data[i]["qa"][j][f"{model}_{a_llm}_answer"]
                question["{}_{}_hard_accuracy".format(model, a_llm)] = eval_answer(question["correct answer"], answer)["hard"]
                question["{}_{}_mid_accuracy".format(model, a_llm)] = eval_answer(question["correct answer"], answer)["mid"]
                question["{}_{}_easy_accuracy".format(model, a_llm)] = eval_answer(question["correct answer"], answer)["easy"]

                model_easy_accuracy.append(model_data[i]["qa"][j][f"{model}_{a_llm}_easy_accuracy"])
                model_mid_accuracy.append(model_data[i]["qa"][j][f"{model}_{a_llm}_mid_accuracy"])
                model_hard_accuracy.append(model_data[i]["qa"][j][f"{model}_{a_llm}_hard_accuracy"])

        # print("gold hard accuracy: ", sum(gold_hard_accuracy) / len(gold_hard_accuracy))
        # print("gold mid accuracy: ", sum(gold_mid_accuracy) / len(gold_mid_accuracy))
        # print("gold easy accuracy: ", sum(gold_easy_accuracy) / len(gold_easy_accuracy))

        print("model hard accuracy: ", sum(model_hard_accuracy) / len(model_hard_accuracy))
        print("model mid accuracy: ", sum(model_mid_accuracy) / len(model_mid_accuracy))
        print("model easy accuracy: ", sum(model_easy_accuracy) / len(model_easy_accuracy))
        print("back layer last: ", sum(back_list) / len(back_list))
        print("summarize layer last: ", sum(summarize_list) / len(summarize_list))

        print("wer: ", normalized_wer_score(model_preds, gold_preds))

        with open(f"/mnt/home/zhenwan.nlp/ASR-Eval/QA_eval/subset/task-{dataset}-{q_llm}-qa-asr-{model}-answer-llama3-8b-filtered.json", "w") as f:
            json.dump(model_data, f, indent=1)
        


