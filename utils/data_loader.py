
from datasets import load_dataset, Dataset, Audio, load_from_disk
import json

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

def load_closed(do_infer=False):
    if do_infer:
        with open("./closed-set/target_wav_data/generation_metadata.json", "r") as f:
            meta_data = json.load(f)
            meta_data = meta_data["generations"]
        for tmp in meta_data:
            tmp["gold"] = tmp["text"]
            tmp["audio_path"] = "./closed-set/target_wav_data/" + tmp["audio_file"]
            tmp["audio"] = tmp["audio_path"]
    else:
        with open("./closed-set/target_wav_data/generation_metadata.json", "r") as f:
            meta_data = json.load(f)
            meta_data = meta_data["generations"]
        for tmp in meta_data:
            tmp["gold"] = tmp["text"]
            tmp["audio_path"] = "./closed-set/target_wav_data/" + tmp["audio_file"]
            tmp["audio"] = tmp["audio_path"]
    return meta_data

def load_data(dataset, do_infer=False):
    if dataset == "voxpopuli":
        return load_voxpopuli(do_infer)
    elif dataset == "tedlium":
        return load_tedlium(do_infer)
    elif dataset == "earning22":
        return load_earning22(do_infer)
    elif dataset == "medasr":
        return load_medasr(do_infer)
    elif dataset == "closed":
        return load_closed(do_infer)
    else:
        raise ValueError("Invalid dataset name")    