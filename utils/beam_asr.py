import json
from datasets import load_dataset, Dataset, Audio, load_from_disk
import torch
import sys
from utils.remove_repetition import remove_repetitive_text
from utils.data_loader import load_data

import os

sys.setrecursionlimit(3000)
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def get_whisper_outputs(model_id, sample, num_beam, num_return_sequences):

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


  # 获取当前 batch
    result = pipe(sample, generate_kwargs=generate_kwargs)["text"]

    tmp = {}
    # tmp["gold"] = batch["text"]
    if num_return_sequences == 1:
        tmp["1best"] = [remove_repetitive_text("".join(result))]
        results.append(tmp)
    
    if len(result) % num_return_sequences != 0:
        print(result)
        print(len(result))
        tmp["nbest"] = []
        results.append(tmp)

    num_stamps = len(result) // num_return_sequences
    result = ["".join(result[i * num_stamps:(i + 1) * num_stamps]) for i in range(num_return_sequences)]
    # print(result)
    # assert False
    tmp["nbest"] = [remove_repetitive_text(s) for s in result]
    # tmp["whisper"] = remove_repetitive_text(result)
    # assert False
    results.append(tmp) 
    return results


sample = "/mnt/home/zhenwan.nlp/multimodal_reasoning/test_clap.mp3"

# sample = "/mnt/home/zhenwan.nlp/ASR-Eval/canary_infer/earning22/sample_195.wav"
model_id = "openai/whisper-large-v2"


results = get_whisper_outputs(model_id, sample, 60, 10)

for result in results[0]["nbest"]:
    print(result)
    print("-"*100)