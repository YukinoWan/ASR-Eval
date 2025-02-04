import json

dataset = "voxpopuli-qwen2-audio"
with open(dataset+".json", "r") as f:
    data = json.load(f)

# for i in range(len(data)):
#     asr = data[i]["desta2"][0]
#     try:
#         asr = asr.split(":\n\n")[1].strip("\"")
#         try:
#             asr = asr.split("]\n")[1].strip()
#             # asr = asr.split(": ")[1].strip()
#             data[i]["desta2"] = [asr]
#         except:
#             # asr = asr.split(":")[1].strip()
#             data[i]["desta2"] = [asr]
#     except:
#         data[i]["desta2"] = [asr]
    
for i in range(len(data)):
    asr = data[i]["qwen2-audio"][0]
    try:
        asr = asr.split(":")[1].strip("\"").strip("'")

        # asr = asr.split(":")[1].strip()
        data[i]["qwen2-audio"] = [asr]
    except:
        data[i]["qwen2-audio"] = [asr]


with open(dataset+"-filtered.json", "w") as f:
    json.dump(data, f, indent=1)