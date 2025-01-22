import json
from utils.metrics import normalized_wer_score
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_sts(model, tokenizer, input_texts):
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = (embeddings[:1] @ embeddings[1:].T) * 100
    # print(scores)
    return scores
    
tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
model = AutoModel.from_pretrained('intfloat/e5-base-v2')

with open("/mnt/home/zhenwan.nlp/ASR-Eval/llm_respond_results/llm_respond_medasr_neko_normalized.json","r",encoding="utf-8") as f:
    data = json.load(f)

updated_data = []
for tmp in tqdm(data[:1000]):
    if tmp["neko_wer"] == 0.0 or tmp["whisper_wer"] == 0.0:
        continue
    if tmp["neko_kl"] > tmp["whisper_kl"] and tmp["neko_wer"] > tmp["whisper_wer"]:
        tmp["whisper_respond_wer"] = normalized_wer_score([tmp["whisper_generation"]], [tmp["gold_generation"]])
        tmp["neko_respond_wer"] = normalized_wer_score([tmp["neko_generation"]], [tmp["gold_generation"]])

        whisper_texts_1 = ['query: {}'.format(tmp["gold"]),
                    "query: {}".format(tmp["whisper"][0])]
        neko_texts_1 = ['query: {}'.format(tmp["gold"]),
                    "query: {}".format(tmp["clean_neko"])]

        whisper_texts_2 = ['query: {}'.format(tmp["gold_generation"]),
                    "query: {}".format(tmp["whisper_generation"])]
        neko_texts_2 = ['query: {}'.format(tmp["gold_generation"]),
                    "query: {}".format(tmp["neko_generation"])]
        whisper_output_1 = get_sts(model, tokenizer, whisper_texts_1).cpu().detach().numpy()
        whisper_output_2 = get_sts(model, tokenizer, whisper_texts_2).cpu().detach().numpy()
        neko_output_1 = get_sts(model, tokenizer, neko_texts_1).cpu().detach().numpy()
        neko_output_2 = get_sts(model, tokenizer, neko_texts_2).cpu().detach().numpy()

        tmp["asr_sts"] = {"whisper": whisper_output_1[0][0].item(), "neko": neko_output_1[0][0].item()}
        tmp["respond_sts"] = {"whisper": whisper_output_2[0][0].item(), "neko": neko_output_2[0][0].item()}
        updated_data.append(tmp)

# print(updated_data)
with open("llm_respond_results/medasr_whisper_better_respond_whisper_better.json", "w", encoding="utf-8") as f:
    json.dump(updated_data, f, indent=1)