import os
import json
import sys
from tqdm import tqdm
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# llama = "meta-llama/Llama-2-7b-hf"
#llama = "mistralai/Mistral-7B-v0.1"

# def get_current_temperature(location: str) -> float:
#     """
#     Get the current temperature at a location.
    
#     Args:
#         location: The location to get the temperature for, in the format "City, Country"
#     Returns:
#         The current temperature at the specified location in the specified units, as a float.
#     """
#     return 22.  # A real function should probably actually get the temperature!

# # Next, create a chat and apply the chat template


def get_hidden_states(model, tokenizer, input_sentence):
    # 加载模型和tokenizer
    # 输入句子
    messages = [
    {"role": "system", "content": "You are a bot that responds to queries."},
    {"role": "user", "content": "{}".format(input_sentence)},
    ]
    # input_sentence = tokenizer.apply_chat_template(messages, tools=[get_current_temperature], add_generation_prompt=True)
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')

    # print(input_ids)

    # input_ids = tokenizer.encode(inputs, return_tensors='pt')
    

    # 获取模型输出
    with torch.no_grad():
        outputs = model(input_ids)

    # 获取每一层的hidden states
    hidden_states = outputs.hidden_states

    emb = model.model.norm(outputs.hidden_states[-1][:, -1, :])
    logits = model.lm_head(emb)
    probs = torch.softmax(logits, dim=-1)
    max_id = torch.argmax(probs, dim=1)
    # print(max_id)
    # print(tokenizer.decode(max_id))
    # print(probs)

    # 获取最后一个token在每一层的hidden states
    last_token_hidden_states = [state[:, -1, :] for state in hidden_states]

    return last_token_hidden_states, probs

def get_background(text):
    normalizer = BasicTextNormalizer()
    return f"Please use one word to describe in which background scenario will the following conversation happen.\nSpeaker: {normalizer(text)}\nWord:"

def get_summarize(text):
    normalizer = BasicTextNormalizer()
    return f"This sentence \'{normalizer(text)}\' means in one word:"


if __name__ == "__main__":
    dataset = sys.argv[1]
    asr_model = sys.argv[2]
    asr_input = sys.argv[3]

    # llama = "meta-llama/Meta-Llama-3.1-8B"
    llama = "meta-llama/Llama-3.1-8B-Instruct"

    # 加载模型和tokenizer
    model_name = llama
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True, device_map='auto')


    # input1 = "The background domain of the conversation \'Uh, um, uh, and at the same time, basically, for one year, they're also supporting inaudible merchants, um, uh, in terms of giving them the possibility that from the taxes the, the commissions that they pay\' is most propabaly"
    # input2 = "The background domain of the conversation \'and at the same time basically for one year they're also supporting new-to-commerchants in terms of giving them the possibility to deduct from their taxes the commissions that they pay\' is most propabaly "
    # input1 = "Please use one word to describe in which background scenario will the following conversation happen.\nSpeaker: Uh, um, uh, and at the same time, basically, for one year, they're also supporting merchants, um, uh, in terms of giving them the possibility that from the taxes the, the commissions that they pay.\nWord:"
    # input2 = "Please use one word to describe in which background scenario will the following conversation happen.\nSpeaker: and at the same time basically for one year they're also supporting new-to-commerchants in terms of giving them the possibility to deduct from their taxes the commissions that they pay.\nWord:"
    # input1 = "This sentence \'I have a pain in my trapes.\' means in one word:"
    # input2 = "This sentence \'I have a pain in my traps.\' means in one word:"

    # # input1 = "Please use one word to describe in which background scenario will the following conversation happen.\nSpeaker: I have a pain in my trapes.\nWord:"
    # # input2 = "Please use one word to describe in which background scenario will the following conversation happen.\nSpeaker: I have a pain in my traps.\nWord:"


    # # input2 = "This is a good book that"
    # # 获取hidden states
    # last_token_hidden_states1, logits1 = get_hidden_states(model, tokenizer, input1)
    # last_token_hidden_states2, logits2 = get_hidden_states(model, tokenizer, input2)

    # # compare two hidden states by cosine similarity
    # for i, (state1, state2) in enumerate(zip(last_token_hidden_states1, last_token_hidden_states2)):
    #     print(f"Layer {i+1} cosine similarity: {torch.nn.functional.cosine_similarity(state1, state2, dim=-1)}")

    # print(torch.nn.functional.cosine_similarity(last_token_hidden_states1[19], last_token_hidden_states2[19], dim=-1).detach().item())

    
    # compare two logits by kl divergence
    # print(f"KL divergence: {torch.nn.functional.kl_div(logits1, logits2)}")

    # # 打印每一层的hidden states的形状
    # for i, layer_hidden_state in enumerate(last_token_hidden_states):
    #     print(f"Layer {i+1} hidden state shape: {layer_hidden_state.shape}")


    data_path = f"/mnt/home/zhenwan.nlp/ASR-Eval/ASR_results/subset/{dataset}-{asr_model}-filtered.json"

    data_list = []
    with open(data_path, "r") as f:
        data = json.load(f)

    for i in tqdm(range(len(data))):
        asr = data[i][asr_input]
        gold = data[i]["gold"]
        tmp = {}
        tmp["gold"] = gold
        tmp[asr_input] = asr

        back_last_token_hidden_states1, _ = get_hidden_states(model, tokenizer, get_background(asr[0]))
        back_last_token_hidden_states2, _ = get_hidden_states(model, tokenizer, get_background(gold))

        summarize_last_token_hidden_states1, _ = get_hidden_states(model, tokenizer, get_summarize(asr[0]))
        summarize_last_token_hidden_states2, _ = get_hidden_states(model, tokenizer, get_summarize(gold)) 

        tmp["back_layer_mid"] = torch.nn.functional.cosine_similarity(back_last_token_hidden_states1[19], back_last_token_hidden_states2[19], dim=-1).detach().item()
        tmp["summarize_layer_mid"] = torch.nn.functional.cosine_similarity(summarize_last_token_hidden_states1[19], summarize_last_token_hidden_states2[19], dim=-1).detach().item()

        tmp["back_layer_last"] = torch.nn.functional.cosine_similarity(back_last_token_hidden_states1[-1], back_last_token_hidden_states2[-1], dim=-1).detach().item()
        tmp["summarize_layer_last"] = torch.nn.functional.cosine_similarity(summarize_last_token_hidden_states1[-1], summarize_last_token_hidden_states2[-1], dim=-1).detach().item()

        data_list.append(tmp)

    with open(f"/mnt/home/zhenwan.nlp/ASR-Eval/llm_respond_results/subset/llm_eval_{dataset}_{asr_model}.json", "w") as f:
        json.dump(data_list, f, indent=1)



