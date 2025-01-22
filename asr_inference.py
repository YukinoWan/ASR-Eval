import json
from datasets import load_dataset, Dataset, Audio, load_from_disk
import torch
import sys
from tqdm import tqdm
import os
from utils.remove_repetition import remove_repetitive_text
from utils.data_loader import load_data
from nemo.collections.asr.models import EncDecMultiTaskModel

sys.setrecursionlimit(3000)
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def get_whisper_v3_outputs(sample, num_beam, num_return_sequences):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="flash_attention_2", device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,                               
            )

    num_beams = num_beam
    num_return_sequences = num_return_sequences

    results = []

    generate_kwargs = {
        "max_new_tokens": 400,
        "num_beams": num_beams,
        "num_return_sequences": num_return_sequences,
        "condition_on_prev_tokens": False,
        "compression_ratio_threshold": 1.35,  # zlib compression ratio threshold (in token space)
        # "temperature": 0.1,
        # "logprob_threshold": -1.0,
        # "no_speech_threshold": 0.6,
        "return_timestamps": True,
        "language": "english",
    }
    batch_size = 1

    for i in tqdm(range(0, len(sample), batch_size), desc="Processing batches"):
        batch = sample[i]  # 获取当前 batch
        result = pipe(batch["audio"], generate_kwargs=generate_kwargs)["text"]

        tmp = {}
        if num_return_sequences == 1:
            tmp["1best"] = [remove_repetitive_text("".join(result))]
            results.append(tmp)
            continue
        if len(result) % num_return_sequences != 0:
            print(result)
            print(len(result))
            tmp["nbest"] = []
            results.append(tmp)
            continue
        num_stamps = len(result) // num_return_sequences
        result = ["".join(result[i * num_stamps:(i + 1) * num_stamps]) for i in range(num_return_sequences)]
        # print(result)
        # assert False
        tmp["nbest"] = [remove_repetitive_text(s) for s in result]
        # tmp["whisper"] = remove_repetitive_text(result)
        # assert False
        results.append(tmp) 
    return results

def get_whisper_v2_outputs(ds, num_beam, num_return_sequences):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # load model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", language="en")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2", device_map="auto")
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")



    # load sample
    results = []
    for i in tqdm(range(200)):

        tmp = {}
        tmp["gold"] = ds[i]["text"]

        sample = ds[i]["audio"]
        input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

        # generate token ids
        predicted_ids = model.generate(input_features.to(device), forced_decoder_ids=forced_decoder_ids)

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        # print(type(transcription))
        tmp["1best"] = [transcription]
        results.append(tmp)
    return results


def get_whisper_outputs(model_id, sample, num_data, num_beam, num_return_sequences):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="flash_attention_2", device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,                               
            )

    num_beams = num_beam
    num_return_sequences = num_return_sequences

    results = []

    generate_kwargs = {
        "max_new_tokens": 400,
        "num_beams": num_beams,
        "num_return_sequences": num_return_sequences,
        "condition_on_prev_tokens": False,
        "compression_ratio_threshold": 1.35,  # zlib compression ratio threshold (in token space)
        # "temperature": 0.1,
        # "logprob_threshold": -1.0,
        # "no_speech_threshold": 0.6,
        "return_timestamps": True,
        "language": "english",
    }
    batch_size = 1

    for i in tqdm(range(0, num_data, batch_size), desc="Processing batches"):
        batch = sample[i]  # 获取当前 batch
        result = pipe(batch["audio"], generate_kwargs=generate_kwargs)["text"]

        tmp = {}
        # tmp["gold"] = batch["text"]
        if num_return_sequences == 1:
            tmp["1best"] = [remove_repetitive_text("".join(result))]
            results.append(tmp)
            continue
        if len(result) % num_return_sequences != 0:
            print(result)
            print(len(result))
            tmp["nbest"] = []
            results.append(tmp)
            continue
        num_stamps = len(result) // num_return_sequences
        result = ["".join(result[i * num_stamps:(i + 1) * num_stamps]) for i in range(num_return_sequences)]
        # print(result)
        # assert False
        tmp["nbest"] = [remove_repetitive_text(s) for s in result]
        # tmp["whisper"] = remove_repetitive_text(result)
        # assert False
        results.append(tmp) 
    return results

def get_canary_outputs(sample, num_data, num_beam, num_return_sequences):

    # load model
    canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')

    # update dcode params
    decode_cfg = canary_model.cfg.decoding
    decode_cfg.beam.beam_size = 1
    canary_model.change_decoding_strategy(decode_cfg)

    data = sample["audio"][:subset]
    results = canary_model.transcribe(
        paths2audio_files=data,
        atch_size=16,  # batch size to run the inference with
    )
    return results



if __name__ == "__main__":

    dataset = "voxpopuli"
    model_name = "canary"
    subset = 200

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
    with open("ASR_results/subset/{}-{}.json".format(dataset, model_name), "w") as f:
        json.dump(whisper_outputs, f, indent=1)

