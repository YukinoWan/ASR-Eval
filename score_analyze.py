import json
from utils import metrics
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
    asr_model2 = "whisper-large-v2"
    asr_input2 = "whisper_v2_1best"

    wer_llm_avg_list = []
    llm_back_summarize_list = []
    wer_qa_list = []
    llm_qa_list = []


    for dataset in datasets:
        with open(f"llm_respond_results/subset/llm_eval_{dataset}_{asr_model1}.json", "r") as f:
            asr1_data = json.load(f)

        with open(f"llm_respond_results/subset/llm_eval_{dataset}_{asr_model2}.json", "r") as f:
            asr2_data = json.load(f)
        
        with open(f"QA_eval/subset/task-{dataset}-gpt-4o-qa-asr-{asr_model1}-answer-gpt-4o.json", "r") as f:
            asr1_qa_data = json.load(f)

        with open(f"QA_eval/subset/task-{dataset}-gpt-4o-qa-asr-{asr_model2}-answer-gpt-4o.json", "r") as f:
            asr2_qa_data = json.load(f)

        
        
        assert len(asr1_data) == len(asr2_data)


        for i in tqdm(range(len(asr1_data))):
            tmp = {}
            assert asr1_data[i]["gold"] == asr2_data[i]["gold"]
            if "inaudible" in asr1_data[i]["gold"]:
                continue
            asr1_wer = metrics.normalized_wer_score(asr1_data[i][asr_input1], [asr1_data[i]["gold"]])
            asr2_wer = metrics.normalized_wer_score(asr2_data[i][asr_input2], [asr2_data[i]["gold"]])

            asr1_llm_back = asr1_data[i]["back_layer_last"]
            asr1_summarize = asr1_data[i]["summarize_layer_last"]

            asr2_llm_back = asr2_data[i]["back_layer_last"]
            asr2_summarize = asr2_data[i]["summarize_layer_last"]

            asr1_llm_avg = (asr1_llm_back + asr1_summarize) / 2
            asr2_llm_avg = (asr2_llm_back + asr2_summarize) / 2

            asr1_qa_list = []
            asr2_qa_list = []
            for q1 in asr1_qa_data[i]["qa"]:
                asr1_qa = eval_answer(q1["correct answer"], q1[f"{asr_model1}_gpt-4o_answer"])["easy"]
                asr1_qa_list.append(asr1_qa)
            for q2 in asr2_qa_data[i]["qa"]:
                asr2_qa = eval_answer(q2["correct answer"], q2[f"{asr_model2}_gpt-4o_answer"])["easy"]
                asr2_qa_list.append(asr2_qa)
            asr1_qa_avg = sum(asr1_qa_list) / len(asr1_qa_list)
            asr2_qa_avg = sum(asr2_qa_list) / len(asr2_qa_list)

            tmp = {}
            tmp["gold"] = asr1_data[i]["gold"]

            tmp[f"{asr_model1}_output"] = asr1_data[i][asr_input1][0]
            tmp[f"{asr_model2}_output"] = asr2_data[i][asr_input2][0]

            tmp[f"{asr_model1}_WER"] = asr1_wer
            tmp[f"{asr_model2}_WER"] = asr2_wer

            tmp[f"{asr_model1}_LLM_AVG"] = asr1_llm_avg
            tmp[f"{asr_model2}_LLM_AVG"] = asr2_llm_avg 

            tmp[f"{asr_model1}_LLM_BACK"] = asr1_llm_back
            tmp[f"{asr_model2}_LLM_BACK"] = asr2_llm_back

            tmp[f"{asr_model1}_LLM_SUMMARIZE"] = asr1_summarize
            tmp[f"{asr_model2}_LLM_SUMMARIZE"] = asr2_summarize

            tmp[f"{asr_model1}_QA_AVG"] = asr1_qa_avg
            tmp[f"{asr_model2}_QA_AVG"] = asr2_qa_avg

            tmp["dataset"] = dataset

            if asr1_wer < asr2_wer and asr1_llm_avg < asr2_llm_avg:
                wer_llm_avg_list.append(tmp)
            elif asr1_wer > asr1_wer and asr1_llm_avg > asr2_llm_avg:
                wer_llm_avg_list.append(tmp)
            elif asr1_wer < asr2_wer and asr1_qa_avg < asr2_qa_avg:
                wer_qa_list.append(tmp)
            elif asr1_wer > asr2_wer and asr1_qa_avg > asr2_qa_avg:
                wer_qa_list.append(tmp)
            elif asr1_llm_avg > asr2_llm_avg and asr1_qa_avg < asr2_qa_avg:
                llm_qa_list.append(tmp)
            elif asr1_llm_avg < asr2_llm_avg and asr1_qa_avg > asr2_qa_avg:
                llm_qa_list.append(tmp)
            elif asr1_llm_back > asr2_llm_back and asr1_summarize < asr2_summarize:
                llm_back_summarize_list.append(tmp)
            elif asr1_llm_back < asr2_llm_back and asr1_summarize > asr2_summarize:
                llm_back_summarize_list.append(tmp)




    with open(f"analysis/wer_llm_avg/{asr_model1}-{asr_model2}.json", "w") as f:
        json.dump(wer_llm_avg_list, f, indent=1)
    with open(f"analysis/llm_back_summarize/{asr_model1}-{asr_model2}.json", "w") as f:
        json.dump(llm_back_summarize_list, f, indent=1)
    with open(f"analysis/wer_qa/{asr_model1}-{asr_model2}.json", "w") as f:
        json.dump(wer_qa_list, f, indent=1)
    with open(f"analysis/llm_qa/{asr_model1}-{asr_model2}.json", "w") as f:
        json.dump(llm_qa_list, f, indent=1)