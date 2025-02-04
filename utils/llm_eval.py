import torch
from tqdm import tqdm
import json
from filter import st_filter
from transformers import AutoModelForCausalLM, AutoTokenizer
from metrics import llm_eval
from transformers.models.whisper.english_normalizer import BasicTextNormalizer



# 加载 LLaMA 模型和 tokenizer
def get_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        return_dict_in_generate=True,
    )
    return model, tokenizer

def get_greedy_respond(model, tokenizer, asr_input):
    max_new_length = 64
    temperature = 0.0
    messages = [
        {"role": "system", "content": "You are a doctor and I am a patient, please reply to me!"},
        {"role": "user", "content": "{}".format(asr_input)},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_length,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True
        )
    generated_tokens = outputs.sequences[:, input_ids.shape[-1]:] 
    logits = torch.stack(outputs.scores, dim=1)  # 每个生成的 token 的 logits
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return input_ids, outputs, generated_tokens, logits, generated_text

# 输入文本
with open("/mnt/home/zhenwan.nlp/ASR-Eval/llm_eval_medasr.json", "r", encoding="utf-8") as f:
    data = json.load(f)
neko_losses = []
neko_kls = []
whisper_losses = []
whisper_kls = []
updated_data = []
normalize = True
return_pobs = True
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # 替换为 Hugging Face 提供的模型 ID

model, tokenizer = get_model(model_name)
for tmp in tqdm(data[:5]):
    gold_input = tmp["gold"]
    neko_input = st_filter(tmp["neko"])
    whisper_input = tmp["whisper"][0]
    tmp["clean_neko"] = neko_input


    if normalize:
        normalizer = BasicTextNormalizer()
        gold_input = normalizer(gold_input)
        whisper_input = normalizer(whisper_input)
        neko_input = normalizer(neko_input)

    # gold = "Below is a conversation between a patient and a doctor, please"
    # test = "I am happy and."

    # 设置生成参数

    # top_p = 0.9

    if return_pobs:
        gold_input_ids, gold_outputs, gold_generated_tokens, gold_logits, gold_generated_text = get_greedy_respond(model, tokenizer, gold_input)
        neko_input_ids, neko_outputs, neko_generated_tokens, neko_logits, neko_generated_text = get_greedy_respond(model, tokenizer, neko_input)
        whisper_input_ids, whisper_outputs, whisper_generated_tokens, whisper_logits, whisper_generated_text = get_greedy_respond(model, tokenizer, whisper_input)

        # print("Gold Generated Text:", gold_generated_text)
        # print("Test Generated Text:", test_generated_text)

        tmp["gold_generation"] = gold_generated_text
        tmp["whisper_generation"] = whisper_generated_text
        tmp["neko_generation"] = neko_generated_text

        whisper_eval_dict = llm_eval(gold_generated_tokens, gold_logits, whisper_generated_tokens, whisper_logits, True, True, True)
        neko_eval_dict = llm_eval(gold_generated_tokens, gold_logits, neko_generated_tokens, neko_logits, True, True, True)

        tmp["neko_loss"] = neko_eval_dict["loss_eval"]
        tmp["neko_kl"] = neko_eval_dict["kl_eval"]
        tmp["neko_swd"] = neko_eval_dict["res_mean"]
        tmp["whisper_loss"] = whisper_eval_dict["loss_eval"]
        tmp["whisper_kl"] = whisper_eval_dict["kl_eval"]
        tmp["whisper_swd"] = whisper_eval_dict["res_mean"]
        

        neko_losses.append(neko_eval_dict["loss_eval"])
        whisper_losses.append(whisper_eval_dict["loss_eval"])
        neko_kls.append(neko_eval_dict["kl_eval"])
        whisper_kls.append(whisper_eval_dict["kl_eval"])
        updated_data.append(tmp)
    else:
        if tmp["neko_kl"] == tmp["whisper_kl"]:
            continue


with open("test.json", "w", encoding="utf-8") as f:
    json.dump(updated_data, f, ensure_ascii=False, indent=1)
    # print(f"KL Divergence: {kl_divergence.item()}")
if return_pobs:
    print("AVG whisper loss: ", sum(whisper_losses)/len(whisper_losses))
    print("AVG whisper kl: ", sum(whisper_kls)/len(whisper_kls))
    print("AVG neko loss: ", sum(neko_losses)/len(neko_losses))
    print("AVG neko kl: ", sum(neko_kls)/len(neko_kls))