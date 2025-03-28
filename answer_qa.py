import json
from APIs.gemini import generate_gemini_response
from APIs.gpt import generate_gpt_response
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, Audio, load_from_disk
import torch
import sys
import time
from tqdm import tqdm
import os
from utils.data_loader import load_data
from vllm import LLM, SamplingParams
import argparse


def get_answer_prompt(question, context):
    prompt = (
        "Here is an English test, given the question, please select the correct answer based on the context. "
        "Note that if it is confusing to select the correct answer based on the context or the information is lost in the context, please select [[E]].\n\n"
        "Context: Jone is a student.\n"
        "Question: What is occupation of Jone?\n"
        "(A) Doctor\n"
        "(B) Student\n"
        "(C) Lawyer\n"
        "(D) Police officer\n"
        "(E) None of above\n"
        "Correct Answer: [[B]]\n\n"  
        f"Context: {context}\n"
        f"{question}\n"
        "(E) None of above\n"
        "Correct Answer:"
    )
    return prompt


def generate_answer(question, context, model_type, model_name):
    question = f"Question: {question['question']}\n{question['options']}"
    prompt = get_answer_prompt(question, context)
    # print("Prompt:\n", prompt)
    # assert False
    if "gpt" in model_type:
        content = generate_gpt_response(model_name, prompt)
    elif "gemini" in model_type:
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
    if f"({correct})" in answer or f"[{correct}]" in answer:
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

def get_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    llm = LLM(model=model_id, tensor_parallel_size=4, trust_remote_code=True)

    # model = AutoModelForCausalLM.from_pretrained(model_id, use_flash_attention_2=True, torch_dtype=torch.bfloat16, device_map="auto")
    # print(model.hf_device_map)

    return llm, tokenizer

def genenerate_n_answer_vllm(question, context, llm, tokenizer, n):
    question = f"Question: {question['question']}\n{question['options']}"
    prompt = get_answer_prompt(question, context)
    try:
        prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': f"{prompt}"}],
        tokenize=False, 
        add_generation_prompt=True 
        )]
        sampling_params = SamplingParams(n=5, temperature=0.4, max_tokens=512)
    except:

        prompts = ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n' + prompt + '<|im_end|>\n<|im_start|>assistant\n']
        sampling_params = SamplingParams(n=5, temperature=0.4, max_tokens=512, stop=["<|im_end|>", "<|im_start|>",])

    
    outputs = llm.generate(prompts, sampling_params)
    contents = [outputs[0].outputs[i].text for i in range(n)]

    return contents
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="earning22")
    parser.add_argument("--asr_model", type=str, default="whisper_v3_1best")
    parser.add_argument("--asr_input", type=str, default="gold")
    parser.add_argument("--answer_model", type=str, default="gpt-4o")
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    args = parser.parse_args()

    qa_data_path = "./QA_data/subset/{}-gpt-4o-qa.json".format(args.dataset)
    context_data_path = "./ASR_results/subset/{}-{}.json".format(args.dataset, args.asr_model)

    hard_answer_stat = []
    mid_answer_stat = []
    easy_answer_stat = []
    with open(context_data_path, "r") as f:
        context_outputs = json.load(f)
    with open(qa_data_path, "r") as f:
        qa_outputs = json.load(f)

    if "gpt" not in args.answer_model and "gemini" not in args.answer_model:
        llm, tokenizer = get_model(args.model_name)

    subset = len(qa_outputs)
    # subset = 5

    print(len(qa_outputs))
    for i in tqdm(range(subset)):
        for j in range(len(qa_outputs[i]["qa"])):
            question = qa_outputs[i]["qa"][j]
            if "gold" not in args.asr_input:
                context = context_outputs[i][args.asr_input][0]
            else:
                context = context_outputs[i][args.asr_input]
            qa_outputs[i][args.asr_model] = [context]
            if "gpt" not in args.answer_model and "gemini" not in args.answer_model:
                answer = genenerate_n_answer_vllm(question, context, llm, tokenizer, 5)
            else:
                answer = generate_n_answer(question, context, args.answer_model, args.model_name, 5)
            question["{}_{}_answer".format(args.asr_model, args.model_name)] = answer
            question["{}_{}_hard_accuracy".format(args.asr_model, args.model_name)] = eval_answer(question["correct answer"], answer)["hard"]
            question["{}_{}_mid_accuracy".format(args.asr_model, args.model_name)] = eval_answer(question["correct answer"], answer)["mid"]
            question["{}_{}_easy_accuracy".format(args.asr_model, args.model_name)] = eval_answer(question["correct answer"], answer)["easy"]

            hard_answer_stat.append(question["{}_{}_hard_accuracy".format(args.asr_model, args.model_name)])
            mid_answer_stat.append(question["{}_{}_mid_accuracy".format(args.asr_model, args.model_name)])
            easy_answer_stat.append(question["{}_{}_easy_accuracy".format(args.asr_model, args.model_name)])

    print("Hard Accuracy: ", sum(hard_answer_stat) / len(hard_answer_stat))
    print("Mid Accuracy: ", sum(mid_answer_stat) / len(mid_answer_stat))
    print("Easy Accuracy: ", sum(easy_answer_stat) / len(easy_answer_stat))

    with open("QA_Results/subset/task-{}-asr-{}-answer-{}.json".format(args.dataset, args.asr_model, args.answer_model), "w") as f:
        json.dump(qa_outputs, f, indent=1)