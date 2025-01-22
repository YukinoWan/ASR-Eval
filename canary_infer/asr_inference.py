import json
from datasets import load_dataset, Dataset, Audio, load_from_disk
import torch
import sys
from tqdm import tqdm
import soundfile as sf
import os
from nemo.collections.asr.models import EncDecMultiTaskModel

def load_voxpopuli(do_infer=False):
    if do_infer:
        voxpopuli = load_dataset("facebook/voxpopuli", "en", split="test")
        voxpopuli = voxpopuli.map(lambda x: {"length": len(x["raw_text"])})
        voxpopuli = voxpopuli.map(lambda x: {"text": x["raw_text"]})
        voxpopuli = voxpopuli.sort("length", reverse=True)
    else:
        with open("/mnt/home/zhenwan.nlp/ASR-Eval/ASR_results/subset/voxpopuli-whisper-large-v3.json", "r") as f:
            voxpopuli = json.load(f)
    # print(voxpopuli)
    return voxpopuli

def load_tedlium(do_infer=False):
    if do_infer:
        tedlium = load_dataset("LIUM/tedlium", "release1", split="test")
        tedlium = tedlium.map(lambda x: {"length": len(x["text"])})
        tedlium = tedlium.sort("length", reverse=True)
    else:
        with open("/mnt/home/zhenwan.nlp/ASR-Eval/ASR_results/subset/tedlium-whisper-large-v3.json", "r") as f:
            tedlium = json.load(f)
    return tedlium

def load_earning22(do_infer=False):
    if do_infer:
        earning22 = load_dataset("anton-l/earnings22_robust", "all", split="test")
        earning22 = earning22.map(lambda x: {"text": x["sentence"]})
        earning22 = earning22.map(lambda x: {"length": len(x["text"])})
        earning22 = earning22.sort("length", reverse=True)
    else:
        with open("/mnt/home/zhenwan.nlp/ASR-Eval/ASR_results/subset/earning22-whisper-large-v3.json", "r") as f:
            earning22 = json.load(f)
    print(len(earning22))
    return earning22


def load_medasr(do_infer=False):
    if do_infer:
        medasr = load_dataset("jarvisx17/Medical-ASR-EN", split="train")
        medasr = medasr.map(lambda x: {"text": x["transcription"]})
        medasr = medasr.map(lambda x: {"length": len(x["text"])})
        medasr = medasr.sort("length", reverse=True)
    else:
        with open("/mnt/home/zhenwan.nlp/ASR-Eval/ASR_results/subset/medasr-whisper-large-v3.json", "r") as f:
            medasr = json.load(f)
    print(len(medasr))
    return medasr

def load_data(dataset, do_infer=False):
    if dataset == "voxpopuli":
        return load_voxpopuli(do_infer)
    elif dataset == "tedlium":
        return load_tedlium(do_infer)
    elif dataset == "earning22":
        return load_earning22(do_infer)
    elif dataset == "medasr":
        return load_medasr(do_infer)
    else:
        raise ValueError("Invalid dataset name")    

def get_canary_outputs(sample, num_data, num_beam, num_return_sequences):

    # load model
    canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')

    # update dcode params
    decode_cfg = canary_model.cfg.decoding
    decode_cfg.beam.beam_size = 1
    canary_model.change_decoding_strategy(decode_cfg)

    print(sample[0])

    output_dir = "./tmp"
    os.makedirs(output_dir, exist_ok=True)

    data = []
    for i in range(num_data):
        audio = sample[i]["audio"]  # 获取音频数据
        audio_array = audio["array"]  # 获取音频数组
        sample_rate = audio["sampling_rate"]  # 获取采样率
        output_path = os.path.join(output_dir, f"sample_{i}.wav")
        data.append(output_path)
        sf.write(output_path, audio_array, sample_rate)

    # data = [x["path"] for x in sample["audio"][:subset]]
    # print(data)
    # assert False
    results = canary_model.transcribe(
        paths2audio_files=data,
        batch_size=16,  # batch size to run the inference with
    )
    return results



if __name__ == "__main__":

    dataset = "medasr"
    model_name = "canary"
    subset = 500

    whisper_outputs = []
    data = load_data(dataset, True)
    print(type(data))
    if model_name == "whisper-large-v2":
        whisper_v2_nbest_outputs = get_whisper_outputs("openai/whisper-large-v2", data, subset, 20, 5)
        whisper_v2_1best_outputs = get_whisper_outputs("openai/whisper-large-v2", data, subset, 1, 1)
    elif model_name == "whisper-large-v3":
        whisper_v3_1best_outputs = get_whisper_outputs("openai/whisper-large-v3", data, subset, 1, 1)
    elif model_name == "canary":
        canary_outputs = get_canary_outputs(data, subset, 1, 1)
    # print(whisper_1best_outputs)
    # whisper_nbest_outputs = get_whisper_v3_outputs(data, 20, 5)
    # assert len(whisper_1best_outputs) == len(whisper_nbest_outputs)
    for i in tqdm(range(subset)):
        tmp = {}
        tmp["gold"] = data[i]["text"]
        if model_name == "whisper-large-v2":
            tmp["whisper_v2_nbest"] = whisper_v2_nbest_outputs[i]["nbest"]
            tmp["whisper_v2_1best"] = whisper_v2_1best_outputs[i]["1best"]
        elif model_name == "whisper-large-v3":
            tmp["whisper_v3_1best"] = whisper_v3_1best_outputs[i]["1best"]
        elif model_name == "canary":
            tmp["canary_1best"] = [canary_outputs[i]]
        # tmp["whisper_v3_1best"] = whisper_v3_1best_outputs[i]["1best"]
        # tmp["whisper_v3_nbest"] = whisper_nbest_outputs[i]["whisper_v3_nbest"]
        whisper_outputs.append(tmp)
    
    print(whisper_outputs[0])
    print(len(whisper_outputs))
    with open("/mnt/home/zhenwan.nlp/ASR-Eval/ASR_results/subset/{}-{}.json".format(dataset, model_name), "w") as f:
        json.dump(whisper_outputs, f, indent=1)

