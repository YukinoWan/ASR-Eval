import json
from APIs.gemini import generate_gemini_response
from APIs.gpt import generate_gpt_response
from datasets import load_dataset, Dataset, Audio, load_from_disk
import torch
import sys
import time
from tqdm import tqdm
import os
from utils.data_loader import load_data


def get_answer_prompt(question, context):
    prompt = (
        "Here is an English test, given the question, please select the correct answer based on the context. "
        "The format will be: \n"
        "Context: Jone is a student.\n"
        "Question: What is occupation of Jone?\n"
        "(A) Doctor\n"
        "(B) Student\n"
        "(C) Lawyer\n"
        "(D) Police officer\n"
        "(E) None of above\n"
        "Correct Answer: [[B]]\n"
        "Note that if it is confusing to select the correct answer based on the context or the information is lost in the context, please select [[E]].\n"
        f"Context: {context}\n"
        f"{question}\n"
        "(E) None of above\n"
        "Correct Answer: "
    )
    return prompt


def generate_answer(question, context, model_type, model_name):
    question = f"Question: {question['question']}\n{question['options']}"
    prompt = get_answer_prompt(question, context)
    # print("Prompt:\n", prompt)
    # assert False
    if model_type == "gpt":
        content = generate_gpt_response(model_name, prompt)
    elif model_type == "gemini":
        content = generate_gemini_response(model_name, prompt)

    # print("Generation: ", content)
    return content

def generate_n_answer(question, context, model_type, model_name, n):
    contents = []
    for i in range(n):
        while(True):
            try:
                contents.append(generate_answer(question, context, model_type, model_name))
                break
            except:
                continue
    return contents

def compare_answer(correct, answer):
    if correct in answer:
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

    dataset = sys.argv[1]
    asr_model = sys.argv[2]
    asr_input = sys.argv[3]
    answer_model = sys.argv[4]
    model_name = sys.argv[5]
    qa_data_path = "/mnt/home/zhenwan.nlp/ASR-Eval/QA_results/subset/{}-gpt-4o-qa.json".format(dataset)
    context_data_path = "/mnt/home/zhenwan.nlp/ASR-Eval/GER_results/subset/{}_{}.json".format(dataset, asr_model)

    hard_answer_stat = []
    mid_answer_stat = []
    easy_answer_stat = []
    with open(context_data_path, "r") as f:
        context_outputs = json.load(f)
    with open(qa_data_path, "r") as f:
        qa_outputs = json.load(f)

    print(len(qa_outputs))
    for i in tqdm(range(len(qa_outputs))):
        for j in range(len(qa_outputs[i]["qa"])):
            question = qa_outputs[i]["qa"][j]
            if "gold" not in asr_input:
                context = context_outputs[i][asr_input][0]
            else:
                context = context_outputs[i][asr_input]
            qa_outputs[i][asr_model] = [context]
            answer = generate_n_answer(question, context, answer_model, model_name, 5)
            question["{}_{}_answer".format(asr_model, model_name)] = answer
            question["{}_{}_hard_accuracy".format(asr_model, model_name)] = eval_answer(question["correct answer"], answer)["hard"]
            question["{}_{}_mid_accuracy".format(asr_model, model_name)] = eval_answer(question["correct answer"], answer)["mid"]
            question["{}_{}_easy_accuracy".format(asr_model, model_name)] = eval_answer(question["correct answer"], answer)["easy"]

            hard_answer_stat.append(question["{}_{}_hard_accuracy".format(asr_model, model_name)])
            mid_answer_stat.append(question["{}_{}_mid_accuracy".format(asr_model, model_name)])
            easy_answer_stat.append(question["{}_{}_easy_accuracy".format(asr_model, model_name)])

    print("Hard Accuracy: ", sum(hard_answer_stat) / len(hard_answer_stat))
    print("Mid Accuracy: ", sum(mid_answer_stat) / len(mid_answer_stat))
    print("Easy Accuracy: ", sum(easy_answer_stat) / len(easy_answer_stat))

    with open("QA_eval/subset/task-{}-gpt-4o-qa-asr-{}-answer-{}.json".format(dataset, asr_model, model_name), "w") as f:
        json.dump(qa_outputs, f, indent=1)