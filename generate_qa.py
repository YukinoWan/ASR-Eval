import json
from APIs.gemini import generate_gemini_response
from APIs.gpt import generate_gpt_response
from datasets import load_dataset, Dataset, Audio, load_from_disk
import sys
from tqdm import tqdm
import os
from utils.data_loader import load_data

sys.setrecursionlimit(3000)
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import answer_qa as eval




def get_qa_prompt(text):
    prompt = (
        "A listening test is to answer the questions based on the audio context, "
        "and the questions reflect the core information in the audio context. "
        "Below is the audio context, please generate 3 different questions (depending on the length of the context), each should be a 4-option question including the correct answer. "
        "The format will be: \n"
        "Question: What is 1 + 1?\n"
        "(A) 1\n"
        "(B) 2\n"
        "(C) 3\n"
        "(D) 4\n"
        "Correct Answer: [[B]]\n"
        "Note that to avoid the random guess of the correct answer, other 3 options should be similar and confusing in listening.\n"
        "Audio context: {}\n".format(text)
    )
    return prompt


def generate_qa(text, model_type, model_name):
    prompt = get_qa_prompt(text)
    # print("Prompt:\n", prompt)
    # assert False
    if model_type == "gpt":
        content = generate_gpt_response(model_name, prompt)
        content = gpt_qa_to_dict(content)
    elif model_type == "gemini":
        content = generate_gemini_response(model_name, prompt)
        content = gemini_qa_to_dict(content)
    # print("Generation: ", content)

    return content

def gpt_qa_to_dict(model_output):
    questions = model_output.strip().split("\n\n")

    dicts = []
    for q in questions:
        lines = q.split("\n")
        question = lines[0].split(": ")[1].strip()  # 提取问题文本
        options = "\n".join(lines[1:5])               # 合并选项
        correct_answer = lines[-1].split(": ")[1].strip()  # 提取正确答案
        correct_answer = correct_answer.strip("[]").strip("()")
        # 构建字典
        dicts.append({"question": f"{question}", "options": f"{options}", "correct answer": correct_answer})
    return dicts

def gemini_qa_to_dict(model_output):
    questions = model_output.strip().split("\n\n**")

    dicts = []
    for q in questions:
        try:
            lines = q.split("\n\n")
            question = lines[1].strip()  # 提取问题文本
            options = lines[2].strip()             # 合并选项
            correct_answer = lines[3].split(": ")[1].strip()  # 提取正确答案
            correct_answer = correct_answer.strip("[]")
            dicts.append({"question": f"{question}", "options": f"{options}", "correct answer": correct_answer})
        except:
            continue
        # 构建字典
    return dicts



if __name__ == "__main__":


    dataset = sys.argv[1]
    model_type = sys.argv[2]
    model_name = sys.argv[3]


    data_list = []

    whisper_outputs = load_data(dataset, False)

    for i in tqdm(range(len(whisper_outputs))):
        print(len(data_list))
        # qa_pairs = generate_qa(data[i]["text"], "gpt", "gpt-4o")
        # qa_list = qa_to_dict(qa_pairs)
        while(True):
            try:
                qa_list = generate_qa(whisper_outputs[i]["gold"], model_type, model_name)
                acc_list = []
                for question in qa_list:
                    answer = eval.generate_n_answer(question, whisper_outputs[i]["gold"], model_type, model_name, 3)
                    acc = eval.eval_answer(question["correct answer"], answer)["easy"]
                    acc_list.append(acc)
                if sum(acc_list) / len(acc_list) == 1:
                    break
                else:
                    continue
            except:
                continue
        
        tmp_dict = {}
        tmp_dict["gold"] = whisper_outputs[i]["gold"]
        tmp_dict["qa"] = qa_list
        data_list.append(tmp_dict)


    with open("QA_results/subset/{}-{}-qa.json".format(dataset, model_name), "w") as f:
        json.dump(data_list, f, indent=1)











