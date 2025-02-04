import json
from datasets import load_dataset, Dataset, Audio, load_from_disk
from APIs.gemini import generate_gemini_audio_response
import torch
import sys
import time
from tqdm import tqdm
import soundfile as sf
import os
from utils.data_loader import load_data
from transformers import AutoModel
from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration



def get_qa_answer_prompt(question):
    prompt = (
        "Given the audio, what is the correct answer to the question?\n"
        "Note that if it is confusing to select the correct answer based on the context or the information is lost in the context, please select [[E]].\n"
        f"{question}\n"
        "(E) None of above\n"
        "Correct Answer: "
    )
    return prompt

def get_asr_answer_prompt():
    prompt = (
        "The transcription of the audio is"
    )
    return prompt

# def generate_answer(question, context, model_type, model_name):
#     question = f"Question: {question['question']}\n{question['options']}"
#     prompt = get_qa_answer_prompt(question, context)
#     # print("Prompt:\n", prompt)
#     # assert False
#     if model_type == "gpt":
#         content = generate_gpt_response(model_name, prompt)
#     elif model_type == "gemini":
#         content = generate_gemini_response(model_name, prompt)

#     # print("Generation: ", content)
#     return content

def generate_n_answer(question, asr_model, model, audio_path, n):
    question = f"Question: {question['question']}\n{question['options']}"
    prompt = get_qa_answer_prompt(question)
    contents = []
    for i in range(n):

        if asr_model == "desta2":
            content = get_desta2_respond(model, audio_path, prompt, 512)
        elif asr_model == "qwen2-audio":
            content = get_qwen2_respond(model, audio_path, prompt, 512)
        elif asr_model == "gemini-1.5-flash" or asr_model == "gemini-1.5-pro":
            while(True):
                try:
                    content = generate_gemini_audio_response(asr_model, audio_path, prompt)
                    break
                except:
                    continue
            # content = generate_gemini_audio_response(asr_model, audio_path, prompt)
        contents.append(content)

    return contents


def generate_asr_answer(asr_model, model, audio_path):
    prompt = get_asr_answer_prompt()
    if asr_model == "desta2":
        content = get_desta2_respond(model, audio_path, prompt, 512).strip("\n")
    elif asr_model == "qwen2-audio":
        content = get_qwen2_respond(model, audio_path, prompt, 512).strip("\n")
    elif asr_model == "gemini-1.5-flash" or asr_model == "gemini-1.5-pro":
        while(True):
            try:
                content = generate_gemini_audio_response(asr_model, audio_path, prompt).strip("\n")
                break
            except:
                continue
        # content = generate_gemini_audio_response(asr_model, audio_path, prompt).strip("\n")
    # content = get_desta2_respond(model, audio_path, prompt, 512).strip("\n")
    return [content]

def read_audio(file_or_url):
    if os.path.isfile(file_or_url):  # 检查是否是本地文件
        with open(file_or_url, "rb") as f:
            return BytesIO(f.read())
    else:  # 否则假定是 URL
        return BytesIO(urlopen(file_or_url).read())

def get_qwen2_respond(model, audio_path, prompt, max_lenth):
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct" ,trust_remote_code=True)
    
    conversation = [{'role': 'system', 'content': 'You are a helpful assistant.'}, 
        {"role": "user", "content": [
            {"type": "audio", "audio_url": f"{audio_path}"},
            {"type": "text", "text": f"{prompt}"},
        ]}]
    
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(
                        librosa.load(
                            read_audio(ele['audio_url']), 
                            sr=processor.feature_extractor.sampling_rate)[0]
                    )

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    inputs.input_ids = inputs.input_ids.to("cuda")

    generate_ids = model.generate(**inputs, max_length=max_lenth)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]

    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(response)
    return response


def get_desta2_respond(model, audio_path, prompt, max_lenth):

    messages = [
            {"role": "system", "content": "You are a helpful voice assistant."},
            {"role": "audio", "content": f"{audio_path}"},
            {"role": "user", "content": f"{prompt}"}
        ]

    generated_ids = model.chat(
        messages, 
        max_new_tokens=max_lenth, 
        do_sample=True, 
        temperature=0.6, 
        top_p=0.9
    )

    response = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)

    return response

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

    if asr_model == "desta2":
        # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        model = AutoModel.from_pretrained("DeSTA-ntu/DeSTA2-8B-beta", trust_remote_code=True, attn_implementation="flash_attention_2", device_map="auto")
    elif asr_model == "qwen2-audio":
        model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", trust_remote_code=True, device_map="auto")
    elif asr_model == "gemini-1.5-flash" or asr_model == "gemini-1.5-pro":
        model = None


    #load qa data
    qa_data_path = "/mnt/home/zhenwan.nlp/ASR-Eval/QA_results/subset/{}-gpt-4o-qa.json".format(dataset)
    with open(qa_data_path, "r") as f:
        qa_outputs = json.load(f)

    # load audio data
    sample = load_data(dataset, True)
    output_dir = f"/mnt/home/zhenwan.nlp/ASR-Eval/canary_infer/{dataset}"
    os.makedirs(output_dir, exist_ok=True)

    
    # subset = len(qa_outputs)
    print(sample[0])
    subset =len(qa_outputs)
    # subset = 1
    audio_path = []
    for i in range(subset):
        audio = sample[i]["audio"]  # 获取音频数据
        audio_array = audio["array"]  # 获取音频数组
        sample_rate = audio["sampling_rate"]  # 获取采样率
        output_path = os.path.join(output_dir, f"sample_{i}.wav")
        audio_path.append(output_path)
        sf.write(output_path, audio_array, sample_rate)

    # assert False

    hard_answer_stat = []
    mid_answer_stat = []
    easy_answer_stat = []

    asr_list = []


    print(len(qa_outputs))
    for i in tqdm(range(subset)):
        asr_input = generate_asr_answer(asr_model, model, audio_path[i])
        tmp_dict = {}
        tmp_dict["gold"] = sample[i]["text"]
        tmp_dict[asr_model] = asr_input
        asr_list.append(tmp_dict)

        for j in range(len(qa_outputs[i]["qa"])):
            question = qa_outputs[i]["qa"][j]

            qa_outputs[i][asr_model] = asr_input
            answer = generate_n_answer(question, asr_model, model, audio_path[i], 5)
            question["{}_{}_answer".format(asr_model, asr_model)] = answer
            question["{}_{}_hard_accuracy".format(asr_model, asr_model)] = eval_answer(question["correct answer"], answer)["hard"]
            question["{}_{}_mid_accuracy".format(asr_model, asr_model)] = eval_answer(question["correct answer"], answer)["mid"]
            question["{}_{}_easy_accuracy".format(asr_model, asr_model)] = eval_answer(question["correct answer"], answer)["easy"]

            hard_answer_stat.append(question["{}_{}_hard_accuracy".format(asr_model, asr_model)])
            mid_answer_stat.append(question["{}_{}_mid_accuracy".format(asr_model, asr_model)])
            easy_answer_stat.append(question["{}_{}_easy_accuracy".format(asr_model, asr_model)])

    print("Hard Accuracy: ", sum(hard_answer_stat) / len(hard_answer_stat))
    print("Mid Accuracy: ", sum(mid_answer_stat) / len(mid_answer_stat))
    print("Easy Accuracy: ", sum(easy_answer_stat) / len(easy_answer_stat))

    with open("/mnt/home/zhenwan.nlp/ASR-Eval/ASR_results/subset/{}-{}.json".format(dataset, asr_model), "w") as f:
        json.dump(asr_list, f, indent=1)

    with open("QA_eval/subset/task-{}-gpt-4o-qa-asr-{}-answer-{}.json".format(dataset, asr_model, asr_model), "w") as f:
        json.dump(qa_outputs, f, indent=1)