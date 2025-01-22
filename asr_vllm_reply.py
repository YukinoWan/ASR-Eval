from transformers import AutoTokenizer
import json
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import jiwer
from utils.metrics import normalized_wer_score



checkpoint = "meta-llama/Meta-Llama-3-8B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(checkpoint) 

# messages = [
#         {
#             "role": "system",
#             "content": "You are a friendly chatbot who always responds in the style of a pirate",
#             },
#         {"role": "user", "content": "How many helicopters can a human eat in one sitting?"}, 
#         ] 
# chat_str = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True 
#         )

# Loading dataset from HF 

from datasets import load_dataset

with open("/mnt/home/zhenwan.nlp/ASR-Eval/llm_respond_results/llm_eval_medasr_neko_normalized.json", "r", encoding="utf-8") as f:
    data = json.load(f)
normalize = True
updated_data = []

for tmp in data:
    if tmp["whisper_kl"] == tmp["neko_kl"]:
        continue
    else:
        updated_data.append(tmp)

# updated_data = updated_data[:10]
if normalize:
    normalizer = BasicTextNormalizer()
    prompts_whisper = [
            tokenizer.apply_chat_template( [{'role': 'user', 'content': normalizer(x["whisper"][0])}],
                tokenize=False, 
                add_generation_prompt=True 
                ) for x in updated_data
            ]
    prompts_neko = [
            tokenizer.apply_chat_template( [{'role': 'user', 'content': normalizer(x["clean_neko"])}],
                tokenize=False, 
                add_generation_prompt=True 
                ) for x in updated_data
            ]
    prompts_gold = [
            tokenizer.apply_chat_template( [{'role': 'user', 'content': normalizer(x["gold"])}],
                tokenize=False, 
                add_generation_prompt=True 
                ) for x in updated_data
            ]
    

assert(len(prompts_whisper) == len(prompts_neko))

from vllm import LLM, SamplingParams # Sample prompts.

#prompts = [
#        chat_str,
#        ] # Create a sampling params object.

sampling_params = SamplingParams(temperature=0.5, n=5, max_tokens=64)

# Create an LLM.

llm = LLM(model=checkpoint, tensor_parallel_size=8) # Generate texts from the prompts. The output is a list of RequestOutput objects that contain the prompt, generated text, and other information. 

outputs_whisper = llm.generate(prompts_whisper, sampling_params) 

outputs_neko = llm.generate(prompts_neko, sampling_params) 

outputs_gold = llm.generate(prompts_gold, sampling_params) 


for i in range(len(updated_data)):
    updated_data[i]["whisper_5_respond"] = [x.text for x in outputs_whisper[i].outputs]
    updated_data[i]["neko_5_respond"] = [x.text for x in outputs_neko[i].outputs]
    updated_data[i]["gold_5_respond"] = [x.text for x in outputs_gold[i].outputs]
    updated_data[i]["whisper_wer"] = normalized_wer_score([updated_data[i]["whisper"][0]], [updated_data[i]["gold"]])
    updated_data[i]["neko_wer"] = normalized_wer_score([updated_data[i]["clean_neko"]], [updated_data[i]["gold"]])


# Print the outputs.
print(updated_data[0])
with open("llm_respond_results/llm_respond_medasr_neko_normalized.json", "w", encoding="utf-8") as f:
    json.dump(updated_data, f, ensure_ascii=False, indent=1)
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

