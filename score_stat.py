import json
from utils.metrics import normalized_wer_score
from tqdm import tqdm

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


if __name__ == "__main__":
    datasets = ["earning22", "voxpopuli", "tedlium", "medasr"]
    asr_model1 = "whisper_v2_nbest_gpt4o"
    asr_input1 = "whisper_v2_nbest_gpt4o"
    llm = "qwen2-7b"
    # asr_model2 = "whisper-large-v2"
    # asr_input2 = "whisper_v2_1best"

    asr_list = []


    for dataset in datasets:
        with open(f"llm_respond_results/subset/llm_eval_{dataset}_{asr_model1}.json", "r") as f:
            asr1_data = json.load(f)

        # with open(f"llm_respond_results/subset/llm_eval_{dataset}_{asr_model2}.json", "r") as f:
        #     asr2_data = json.load(f)
        
        with open(f"QA_eval/subset/task-{dataset}-gpt-4o-qa-asr-{asr_model1}-answer-{llm}.json", "r") as f:
            asr1_qa_data = json.load(f)

        # with open(f"QA_eval/subset/task-{dataset}-gpt-4o-qa-asr-{asr_model2}-answer-gpt-4o.json", "r") as f:
        #     asr2_qa_data = json.load(f)


        for i in tqdm(range(len(asr1_data))):
            tmp = {}
            # if "inaudible" in asr1_data[i]["gold"]:
            #     continue
            asr1_wer = normalized_wer_score(asr1_data[i][asr_input1], [asr1_data[i]["gold"]])

            asr1_llm_back = asr1_data[i]["back_layer_last"]
            asr1_summarize = asr1_data[i]["summarize_layer_last"]


            asr1_llm_avg = (asr1_llm_back + asr1_summarize) / 2

            asr1_qa_hard_list = []
            asr1_qa_mid_list = []
            asr1_qa_easy_list = []
            for q1 in asr1_qa_data[i]["qa"]:
                asr1_qa_hard = eval_answer(q1["correct answer"], q1[f"{asr_model1}_Qwen/Qwen2-7B-Instruct_answer"])["hard"]
                asr1_qa_mid = eval_answer(q1["correct answer"], q1[f"{asr_model1}_Qwen/Qwen2-7B-Instruct_answer"])["mid"]
                asr1_qa_easy = eval_answer(q1["correct answer"], q1[f"{asr_model1}_Qwen/Qwen2-7B-Instruct_answer"])["easy"]

                asr1_qa_hard_list.append(asr1_qa_hard)
                asr1_qa_mid_list.append(asr1_qa_mid)
                asr1_qa_easy_list.append(asr1_qa_easy)

            asr1_qa_avg_hard = sum(asr1_qa_hard_list) / len(asr1_qa_hard_list)
            asr1_qa_avg_mid = sum(asr1_qa_mid_list) / len(asr1_qa_mid_list)
            asr1_qa_avg_easy = sum(asr1_qa_easy_list) / len(asr1_qa_easy_list)

            tmp = {}
            tmp["gold"] = asr1_data[i]["gold"]

            tmp[f"output"] = asr1_data[i][asr_input1][0]

            tmp[f"WER"] = asr1_wer

            tmp[f"LLM_AVG"] = asr1_llm_avg

            tmp[f"LLM_BACK"] = asr1_llm_back

            tmp[f"LLM_SUMMARIZE"] = asr1_summarize

            tmp[f"QA_Hard"] = asr1_qa_avg_hard
            tmp[f"QA_Mid"] = asr1_qa_avg_mid
            tmp[f"QA_Easy"] = asr1_qa_avg_easy

            tmp["dataset"] = dataset

            asr_list.append(tmp)


    print(len(asr_list))
    with open(f"analysis/subset/{asr_model1}.json", "w") as f:
        json.dump(asr_list, f, indent=1)
