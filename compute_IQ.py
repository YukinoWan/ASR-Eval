import numpy as np
import json

# def compute_variance_weights(scores):
#     variances = np.var(scores, axis=1)
#     return variances / np.sum(variances)  # 归一化，使权重之和为 1

# def compute_weighted_scores(scores, weights):
#     return np.sum(scores * weights[:, np.newaxis], axis=0)

# def standardize_scores(scores, invert=False):
#     if invert:
#         scores = -scores  # 取反，使得 WER 越低得分越高
#     mean = np.mean(scores)
#     std = np.std(scores)
#     return (scores - mean) / std

# def convert_to_iq(standardized_scores):
#     return 100 + 15 * standardized_scores

# def compute_size_adjusted_scores(scores, model_sizes):
#     """
#     根据模型 Size 进行分组标准化，再进行 Size 之间的调整。
#     """
#     unique_sizes = np.unique(model_sizes)
#     standardized_scores = np.zeros_like(scores)
    
#     # Step 1: 在每个 Size 组内进行标准化
#     for size in unique_sizes:
#         size_indices = np.where(model_sizes == size)[0]
#         if len(size_indices) > 1:  # 只有多个模型时才标准化
#             standardized_scores[size_indices] = standardize_scores(scores[size_indices])
#         else:  # 只有一个模型的情况，保持原始得分
#             standardized_scores[size_indices] = 0  

#     # Step 2: 计算 Size 组的均值和标准差
#     size_means = {}
#     size_stds = {}
#     for size in unique_sizes:
#         size_indices = np.where(model_sizes == size)[0]
#         size_means[size] = np.mean(standardized_scores[size_indices])
#         size_stds[size] = np.std(standardized_scores[size_indices]) if len(size_indices) > 1 else 1
    
#     # Step 3: 在所有 Size 组间进行标准化
#     adjusted_scores = np.zeros_like(scores)
#     for size in unique_sizes:
#         size_indices = np.where(model_sizes == size)[0]
#         adjusted_scores[size_indices] = (standardized_scores[size_indices] - size_means[size]) / size_stds[size]
    
#     return adjusted_scores


def compute_variance_weights(scores):
    variances = np.var(scores, axis=1)
    return variances / np.sum(variances)  # 归一化，使权重之和为 1

def compute_weighted_scores(scores, weights):
    return np.sum(scores * weights[:, np.newaxis], axis=0)

def standardize_scores(scores, invert=False):
    if invert:
        scores = -scores  # 取反，使得 WER 越低得分越高
    mean = np.mean(scores)
    std = np.std(scores)
    return (scores - mean) / std

def convert_to_iq(standardized_scores):
    return 100 + 15 * standardized_scores

def compute_size_adjusted_scores(scores, model_sizes):
    """
    根据模型 Size 进行分组标准化，再进行 Size 之间的调整。
    """
    unique_sizes = np.unique(model_sizes)
    standardized_scores = np.zeros_like(scores)
    
    # Step 1: 在每个 Size 组内进行标准化
    for size in unique_sizes:
        size_indices = np.where(model_sizes == size)[0]
        if len(size_indices) > 1:  # 只有多个模型时才标准化
            standardized_scores[size_indices] = standardize_scores(scores[size_indices])
        else:  # 只有一个模型的情况，保持原始得分
            standardized_scores[size_indices] = 0  

    # Step 2: 计算 Size 组的均值和标准差
    size_means = {}
    size_stds = {}
    for size in unique_sizes:
        size_indices = np.where(model_sizes == size)[0]
        size_means[size] = np.mean(standardized_scores[size_indices])
        size_stds[size] = np.std(standardized_scores[size_indices]) if len(size_indices) > 1 else 1
    
    # Step 3: 在所有 Size 组间进行标准化
    adjusted_scores = np.zeros_like(scores)
    for size in unique_sizes:
        size_indices = np.where(model_sizes == size)[0]
        adjusted_scores[size_indices] = (standardized_scores[size_indices] - size_means[size]) / size_stds[size]
    
    return adjusted_scores

def compute_dynamic_weights_by_variance(scores_list):
    """
    通过数据的方差计算自适应权重
    参数:
        scores_list: 包含不同维度得分的列表，例如 [WER_scores, SIM_scores, QA_scores]
    返回:
        归一化后的权重数组
    """
    variances = np.array([np.var(scores) for scores in scores_list])
    weights = 1 / (variances + 1e-6)  # 避免除以 0
    # weights = variances
    return weights / np.sum(weights)  # 归一化

def compute_dynamic_weights_by_correlation(scores_list, iq_scores):
    """
    通过相关性计算权重
    参数:
        scores_list: 包含不同维度得分的列表，例如 [WER_scores, SIM_scores, QA_scores]
        iq_scores: 最终 IQ 评分
    返回:
        归一化后的权重数组
    """
    correlations = np.array([abs(np.corrcoef(scores, iq_scores)[0, 1]) for scores in scores_list])
    return correlations / np.sum(correlations)  # 归一化
def select_lower(a, b):
    return a if a < b else b

def select_higher(a, b):
    return a if a > b else b

def select_avg(a, b, c):
    return (a + b + c) / 3
if __name__ == "__main__":
    models = ["canary", "desta2", "qwen2-audio", "whisper_v2_nbest_gpt4o", "whisper-large-v2", "whisper-large-v3"]
    datasets = ["earning22"]
    # 假设我们有 3 组模型 Size（小、中、大），以及 3 个维度（WER, Similarity, QA Accuracy）
    WER = []
    SIM = []
    QA = []
    compute_variance_weights_dimensions = False
    for model in models:
        data = json.load(open(f"/mnt/home/zhenwan.nlp/ASR-Eval/analysis/subset/{model}.json", "r", encoding="utf-8"))
        WER.append([-d["WER"] for d in data if d["dataset"] in datasets])
        SIM.append([select_lower(d["LLM_BACK"], d["LLM_SUMMARIZE"])for d in data if d["dataset"] in datasets])
        # SIM.append([d["LLM_AVG"] for d in data if d["dataset"] in datasets])

        # QA.append([select_avg(d["QA_Hard"], d["QA_Mid"], d["QA_Easy"]) for d in data if d["dataset"] in datasets])
        QA.append([d["QA_Mid"] for d in data if d["dataset"] in datasets])


    WER = np.array(WER).T
    print(datasets)
    print(WER.shape)
    SIM = np.array(SIM).T
    QA  = np.array(QA).T

    # 假设模型 Size（0: 小, 1: 中, 2: 大）
    # model_sizes = np.array([1, 0, 0, 1, 1, 1])  # 6 个模型分别属于 3 组 Size

    model_sizes_WER = np.array([1, 1, 1, 1, 1, 1])  # WER 维度
    model_sizes_SIM = np.array([1, 1, 1, 1, 1, 1])  # SIM 维度
    model_sizes_QA  = np.array([2, 2, 2, 2, 2, 2])  # QA 维度


    # 计算各维度的区分度（方差权重）
    WER_weights = compute_variance_weights(WER)
    SIM_weights = compute_variance_weights(SIM)
    QA_weights  = compute_variance_weights(QA)

    # 计算加权得分
    WER_scores = compute_weighted_scores(WER, WER_weights)
    SIM_scores = compute_weighted_scores(SIM, SIM_weights)
    QA_scores  = compute_weighted_scores(QA, QA_weights)

    # 计算标准化得分
    WER_standardized = standardize_scores(WER_scores)
    SIM_standardized = standardize_scores(SIM_scores)
    QA_standardized  = standardize_scores(QA_scores)

    # 计算 Size 调整（不同维度使用不同的 Size 组）
    WER_size_adjusted = compute_size_adjusted_scores(WER_standardized, model_sizes_WER)
    SIM_size_adjusted = compute_size_adjusted_scores(SIM_standardized, model_sizes_SIM)
    QA_size_adjusted  = compute_size_adjusted_scores(QA_standardized, model_sizes_QA)
    # final_scores = WER_standardized * 0.2 + SIM_standardized * 0.3 + QA_standardized * 0.5
    # size_adjusted_scores = compute_size_adjusted_scores(final_scores, model_sizes)

    # 使用方差计算权重
    if compute_variance_weights_dimensions:
        weights_var = compute_dynamic_weights_by_variance([WER, SIM, QA])
        print(f"基于方差的权重: {weights_var}")

        # 计算最终 IQ 评分（第一次计算）
        final_scores_var = weights_var[0] * WER_size_adjusted + weights_var[1] * SIM_size_adjusted + weights_var[2] * QA_size_adjusted
    else:
        # w1, w2, w3 = 0.03, 0.72, 0.25  # 维度权重
        w1, w2, w3 = 0.2, 0.3, 0.5 # 维度权重
        final_scores_var = w1 * WER_size_adjusted + w2 * SIM_size_adjusted + w3 * QA_size_adjusted
    IQ_scores_var = convert_to_iq(final_scores_var)

    # # 计算最终 IQ 评分
    # w1, w2, w3 = 0.05, 0.6, 0.35  # 维度权重
    # ori_scores = w1 * WER_standardized + w2 * SIM_standardized + w3 * QA_standardized
    # final_scores = w1 * WER_size_adjusted + w2 * SIM_size_adjusted + w3 * QA_size_adjusted

    # 计算基于相关性的权重
    # weights_corr = compute_dynamic_weights_by_correlation([WER_size_adjusted, SIM_size_adjusted, QA_size_adjusted], IQ_scores_var)
    # print(f"基于相关性的权重: {weights_corr}")

    # # 计算最终 IQ 评分（基于相关性调整的权重）
    # final_scores_corr = weights_corr[0] * WER_size_adjusted + weights_corr[1] * SIM_size_adjusted + weights_corr[2] * QA_size_adjusted
    # IQ_scores_corr = convert_to_iq(final_scores_corr)

    # 输出最终 IQ 评分
    # for i, (iq_var, iq_corr) in enumerate(zip(IQ_scores_var, IQ_scores_corr)):
    #     print(f"模型 {models[i]} 的 IQ 评分（方差权重）: {iq_var:.2f}, （相关性权重）: {iq_corr:.2f}")
    for i, iq_var in enumerate(IQ_scores_var):
        print(f"模型 {models[i]} 的 IQ 评分（方差权重）: {iq_var:.2f}")
    # ori_iq = convert_to_iq(ori_scores)
    # IQ_scores = convert_to_iq(final_scores)
    # # 计算最终 IQ
    # ori_iq = convert_to_iq(final_scores)
    # IQ_scores = convert_to_iq(size_adjusted_scores)

    # 输出结果
    # print(len(ori_iq))
    # for i, iq in enumerate(ori_iq):
    #     print(f"模型 {models[i]} 的 原始IQ 评分: {iq:.2f}")
    # for i, iq in enumerate(IQ_scores):
    #     print(f"模型 {models[i]} 的 IQ 评分（考虑 Size 调整）: {iq:.2f}")
