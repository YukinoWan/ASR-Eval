import os
import json
import sys
from tqdm import tqdm
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM



def get_hidden_states(model, tokenizer, input_sentence):

    messages = [
    {"role": "system", "content": "You are a bot that responds to queries."},
    {"role": "user", "content": "{}".format(input_sentence)},
    ]
    # input_sentence = tokenizer.apply_chat_template(messages, tools=[get_current_temperature], add_generation_prompt=True)
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')


    with torch.no_grad():
        outputs = model(input_ids)

    hidden_states = outputs.hidden_states

    emb = model.model.norm(outputs.hidden_states[-1][:, -1, :])
    logits = model.lm_head(emb)
    probs = torch.softmax(logits, dim=-1)
    max_id = torch.argmax(probs, dim=1)
    # print(max_id)
    # print(tokenizer.decode(max_id))
    # print(probs)

    last_token_hidden_states = [state[:, -1, :] for state in hidden_states]

    return last_token_hidden_states, probs

def get_background(text):
    normalizer = BasicTextNormalizer()
    return f"Please use one word to describe in which background scenario will the following conversation happen.\nSpeaker: {normalizer(text)}\nWord:"

def get_summarize(text):
    normalizer = BasicTextNormalizer()
    return f"This sentence \'{normalizer(text)}\' means in one word:"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="earning22")
    parser.add_argument("--asr_model", type=str, default="whisper_v3_1best")
    parser.add_argument("--asr_input", type=str, default="gold")
    args = parser.parse_args()

    dataset = args.dataset
    asr_model = args.asr_model
    asr_input = args.asr_input

    # llama = "meta-llama/Meta-Llama-3.1-8B"
    llama = "meta-llama/Llama-3.1-8B-Instruct"

    model_name = llama
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True, device_map='auto')


    data_path = f"./ASR_results/subset/{dataset}-{asr_model}.json"

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

    with open(f"./llm_respond_results/subset/llm_eval_{dataset}_{asr_model}.json", "w") as f:
        json.dump(data_list, f, indent=1)



