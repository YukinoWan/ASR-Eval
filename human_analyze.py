import pandas as pd
import json

# 假设数据如下，每个 example 有两个模型的两个 metrics

with open("/home/zhen/ASR-Eval/analysis/subset/whisper-large-v2.json", "r", encoding="utf-8") as f:
    model1_data = json.load(f)
with open("/home/zhen/ASR-Eval/analysis/subset/whisper_v2_nbest_gpt4o.json", "r", encoding="utf-8") as f:
    model2_data = json.load(f)

data = {
    "example_id": [model1_data[i]["gold"] for i in range(len(model1_data))],
    "model_A_metric1": [model1_data[i]["WER"] for i in range(len(model1_data))],
    "model_A_metric2": [model1_data[i]["QA_Mid"] for i in range(len(model1_data))],
    "model_B_metric1": [model2_data[i]["WER"] for i in range(len(model2_data))],
    "model_B_metric2": [model2_data[i]["QA_Mid"] for i in range(len(model2_data))],
    "Model_A": [model1_data[i]["output"] for i in range(len(model1_data))],
    "Model_B": [model2_data[i]["output"] for i in range(len(model2_data))],
}

# 转换为 Pandas DataFrame
df = pd.DataFrame(data)

# 1. 判断在哪些 examples 中 A 在 Metric 1 更好，但 B 在 Metric 2 更好
df["A_better_metric1"] = df["model_A_metric1"] < df["model_B_metric1"]  # A 在 Metric 1 更好
df["B_better_metric2"] = df["model_A_metric2"] < df["model_B_metric2"]  # B 在 Metric 2 更好
df["B_better_metric1"] = df["model_A_metric1"] > df["model_B_metric1"]  # B 在 Metric 1 更好
df["A_better_metric2"] = df["model_A_metric2"] > df["model_B_metric2"]  # A 在 Metric 2 更好

# 2. 只保留 A 在 Metric 1 更好，而 B 在 Metric 2 更好的例子
df_filtered = df[(df["A_better_metric1"] & df["B_better_metric2"]) | (df["B_better_metric1"] & df["A_better_metric2"])].copy()

# 3. 计算 A 和 B 在两个 metrics 下的差距大小
df_filtered["metric1_diff"] = abs(df_filtered["model_A_metric1"] - df_filtered["model_B_metric1"])
df_filtered["metric2_diff"] = abs(df_filtered["model_A_metric2"] - df_filtered["model_B_metric2"])

# 4. 计算最大分歧程度
df_filtered["max_discrepancy"] = df_filtered[["metric2_diff"]].max(axis=1)

# 5. 按分歧程度降序排序
df_filtered = df_filtered.sort_values(by="max_discrepancy", ascending=False)

# 6. 导出为 Excel
excel_filename = "metrics_conflict_examples.xlsx"
df_filtered.to_excel(excel_filename, index=False)

print(f"已保存 {excel_filename}，请检查分歧最大的 examples！")
