import sacrebleu
import jiwer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import sema_score_generator
from multiprocessing import Pool
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
# import neko.metric_evaluate as metric_evaluate
from functools import partial
import random






# BLEU 分数计算
def bleu_score(predictions, references):
    assert len(predictions) == len(references)
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score

# ROUGE 分数计算
# def rouge_score(predictions, references):
#     assert len(predictions) == len(references)
#     rouge = metric_evaluate.load("rouge")

#     results = rouge.compute(predictions=predictions, references=references)

#     return results['rougeL']

# 计算 WER
def wer_score(predictions, references):
    assert len(predictions) == len(references)
    wer = jiwer.wer(references, predictions)
    return wer

# 计算 CER
def cer_score(predictions, references):
    assert len(predictions) == len(references)
    cer = jiwer.cer(references, predictions)
    return cer

def normalized_wer_score(predictions, references):
    assert len(predictions) == len(references)
    normalizer = BasicTextNormalizer()
    normalized_predictions = [normalizer(p) for p in predictions]
    normalized_references = [normalizer(r) for r in references]
    wer = jiwer.wer(normalized_references, normalized_predictions)
    return wer

def normalized_cer_score(predictions, references):
    assert len(predictions) == len(references)
    normalizer = BasicTextNormalizer()
    normalized_predictions = [normalizer(p) for p in predictions]
    normalized_references = [normalizer(r) for r in references]
    cer = jiwer.cer(normalized_references, normalized_predictions)
    return cer

def calculate_bert_similarity(pair):
    return sema_score_generator.generate_bert_score_v1(pair[0], pair[1])

def bert_score(predictions, references):
    assert len(predictions) == len(references)
    if len(predictions) == 1:
        return sema_score_generator.generate_bert_score_v1(predictions[0], references[0])

    sentence_pairs = list(zip(predictions, references))
    with Pool(processes=12) as pool:  # processes 设置为可用 CPU 核心数
        similarities = list(tqdm(pool.imap(calculate_bert_similarity, sentence_pairs),total=len(sentence_pairs)))

        # 计算平均相似度
        # average_similarity = sum(similarities) / len(similarities)
        return similarities
    # all_bert_score = []
    # for i in range(len(predictions)):
    #     candidate = predictions[i]
    #     reference = references[i]
    #     bert_score = sema_score_generator.generate_bert_score_v1(candidate, reference)   

    #     all_bert_score.append(bert_score)
    # return sum(all_bert_score) / len(all_bert_score)

def calculate_similarity(pair, model, tokenizer, idf_dict, device):
    return sema_score_generator.generate_sema_score(pair[0], pair[1], model, tokenizer, idf_dict, device)[0]

def default_value():
    return 1.0

def sema_score(predictions, references):
    assert len(predictions) == len(references)

    model_type = "microsoft/deberta-large-mnli"
    num_layers = 18
    tokenizer = sema_score_generator.get_tokenizer(model_type, use_fast=False)
    model = sema_score_generator.get_model(model_type, num_layers)
    device = "cpu"


    idf_dict = sema_score_generator.defaultdict(default_value)
    # set idf for [SEP] and [CLS] to 0
    idf_dict[tokenizer.sep_token_id] = 0
    idf_dict[tokenizer.cls_token_id] = 0

    if len(predictions) == 1:
        return sema_score_generator.generate_sema_score(predictions[0], references[0],  model, tokenizer, idf_dict, device)[0]

    sentence_pairs = random.sample(list(zip(predictions, references)), 100)
    calculate_similarity_partial = partial(calculate_similarity, model=model, tokenizer=tokenizer, idf_dict=idf_dict, device=device)
    with Pool(processes=12) as pool:  # processes 设置为可用 CPU 核心数
        similarities = list(tqdm(pool.imap(calculate_similarity_partial, sentence_pairs), total=len(sentence_pairs)))

        # 计算平均相似度
        # average_similarity = sum(similarities) / len(similarities)
        return similarities
    
    # candidate = "smoke kills"
    # reference = "smoking kills"
    # all_sema_score = []
    # for i in range(len(predictions)):
    #     candidate = predictions[i]
    #     reference = references[i]
    #     sema_score = sema_score_generator.generate_sema_score(candidate, reference)
    #     all_sema_score.append(sema_score[0]) 
    # return sum(all_sema_score) / len(all_sema_score)


def sentence_similarity_score(predictions, references):

    # 加载 GPU 上的预训练模型
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')  # 使用 GPU

    # 检查句子数量是否一致
    assert len(predictions) == len(references), "Sentence lists must have the same length!"

    # 批量处理的函数
    def batch_cosine_similarity(sentences1, sentences2, batch_size=64):
        similarities = []

        # 分批处理句子对
        for i in tqdm(range(0, len(sentences1), batch_size), desc="Processing batches"):
            # 获取当前批次的句子对
            batch_sentences1 = sentences1[i:i + batch_size]
            batch_sentences2 = sentences2[i:i + batch_size]

            # 使用 GPU 生成嵌入
            embeddings1 = model.encode(batch_sentences1, convert_to_tensor=True)
            embeddings2 = model.encode(batch_sentences2, convert_to_tensor=True)

            # 计算余弦相似度
            batch_similarities = cosine_similarity(embeddings1.cpu().numpy(), embeddings2.cpu().numpy())
            
            # 提取对角线的值（逐对相似度）
            similarities.extend(batch_similarities.diagonal())

        return similarities

    # 执行批量处理
    batch_size = 64  # 设置批量大小
    similarities = batch_cosine_similarity(predictions, references, batch_size=batch_size)

    # 计算平均相似度
    # average_similarity = np.mean(similarities)

    return similarities


# def sentiment_score(predictions, references):

def compute_all_scores(predictions, references):
    # 计算 BLEU 分数
    print("start to calculate bleu")
    bleu = bleu_score(predictions, references)
    print("bleu score is ", bleu)

    # 计算 ROUGE 分数
    # print("start to calculate rouge")
    # rouge = rouge_score(predictions, references)
    # print("rouge score is ", rouge)

    # 计算 WER
    print("start to calculate wer")
    wer = wer_score(predictions, references)
    print("wer score is ", wer)

    # 计算 CER
    print("start to calculate cer")
    cer = cer_score(predictions, references)
    print("cer score is ", cer)

    # 计算归一化 WER
    print("start to calculate normalized wer")
    normalized_wer = normalized_wer_score(predictions, references)
    print("normalized wer score is ", normalized_wer)

    # 计算归一化 CER
    print("start to calculate normalized cer")
    normalized_cer = normalized_cer_score(predictions, references)
    print("normalized cer score is ", normalized_cer)

    # 计算 BERT 分数
    print("start to calculate bert")
    bert = bert_score(predictions, references)
    print("bert score is ", bert)

    # 计算 SemaScore
    print("start to calculate sema")
    sema = sema_score(predictions, references)
    print("sema score is ", sema)

    # 计算句子相似度
    print("start to calculate sentence similarity")
    sentence_similarity = sentence_similarity_score(predictions, references)
    print("sentence similarity score is ", sentence_similarity)

    return {
        "BLEU": bleu,
        # "ROUGE": rouge,
        "WER": wer,
        "CER": cer,
        "Normalized WER": normalized_wer,
        "Normalized CER": normalized_cer,
        "BERT": bert,
        "SemaScore": sema,
        "Sentence Similarity": sentence_similarity
    }