import json
import re
from tqdm import tqdm
import ast

def match_top(output):
    output = output.strip("\n")
    match = re.search(r"\['(.*?)'\]", output)
    if match:
        output = match.group(1)
    return output

def get_string(output):
    arsed_list = ast.literal_eval(data)

    # 提取列表的第一个字符串
    first_string = parsed_list[0]
    return first_string


def neko_filter(output):
    print("===================================")
    print(output)
    try:
       output = get_string(output)

    except:
        if isinstance(output, list):
            output = output[0]
            output = match_top(output)
            return output
        if "correct" not in output:
            match = re.search(r"\['(.*?)'\]", output.split("\n")[0])
            if match:
                output = match.group(1)
                return output
            else:
                try:
                    output = get_string(output.split("\n")[0])
                    return output
                except:
                    output = output
        
        try:
            output = output.split(":")[1].strip()
            output = output.split("\n")
            # print(output)
            output = output[0]
            # print(output)
            output = match_top(output)
            return output
        except:
            match = re.search(r"\['(.*?)'\]", output.split("\n")[0])
            if match:
                output = match.group(1)
                return output
            else:
                try:
                    output = get_string(output.split("\n")[0])

                    return output

                except:
                    return "false"

dataset = "voxpopuli"

with open(f"/mnt/home/zhenwan.nlp/ASR-Eval/Neko_results/{dataset}_neko.json", "r") as f:
    data = json.load(f)
    # data = data[:10]

for i in tqdm(range(len(data))):
    if "identical" in data[i]["whisper_v3_nbest_neko"]:
        data[i]["whisper_v3_nbest_neko"] = data[i]["whisper_v3_1best"]
    elif data[i]["whisper_v3_nbest"] == []:
        data[i]["whisper_v3_nbest_neko"] = data[i]["whisper_v3_1best"]
    else:
        output = neko_filter(data[i]["whisper_v3_nbest_neko"])
        if output == "false" or "[" in output:
            data[i]["whisper_v3_nbest_neko"] = data[i]["whisper_v3_1best"]
        else:
            data[i]["whisper_v3_nbest_neko"] = [output.strip('\"')]
    print("----------------------------------")
    print(data[i]["whisper_v3_nbest_neko"])

with open(f"/mnt/home/zhenwan.nlp/ASR-Eval/Neko_results/{dataset}_clean_neko.json", "w") as f: 
    json.dump(data, f, indent=1)