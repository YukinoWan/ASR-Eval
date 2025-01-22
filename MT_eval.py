import json
import re
from metrics import bleu_score
from tqdm import tqdm

with open("/mnt/home/zhenwan.nlp/Med-GER/mt_outputs/ejmmt_neko.json", "r", encoding="utf-8") as f:
    data = json.load(f)


gold = []
mt = []
neko = []

def neko_filter(output):
    if isinstance(output, list):
        output = output[0]
    output = output.strip("\n")
    match = re.search(r"\['(.*?)'\]", output)
    if match:
        output = match.group(1)
    return output

for i in tqdm(range(len(data))):
    tmp = data[i]
    gold.append(tmp["japanese"])
    mt.append(neko_filter(tmp["swallow"]))
    neko.append(neko_filter(tmp["neko"]))

print("swallow: ", bleu_score(mt, gold))
print("neko: ", bleu_score(neko, gold))