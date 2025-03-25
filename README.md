# Speech IQ Test (SIQ)

This repository supports the SIQ test for voice understanding LLMs, including two main types: cascaded (ASR+LLM) and mulitmodal LLMs.
SIQ involves three levels of tests: remember, understand and apply and the final SIQ is computed and normalized among three levels.

## Evaluation of cascaded (ASR+LLM)
We use `whisper-large-v3` as a demo.

### Level 1: Remember
```
python asr_inference.py --dataset "medasr" --model_name "whisper-large-v3"
```
This will save the ASR results to `ASR_results/subset` and we use WER as the score.

### Level 2: Understand

We prompt `meta-llama/Llama-3.1-8B-Instruct` to respond to both groud-truth transcriptions and ASR results to get two last layer hidden states and compute the cosien similarity between them as the score.
```
python llm_respond.py --dataset "medasr" --asr_model "whisper-large-v3" --asr_input "whisper_v3_1best"

```
This will save the llm_respond results to `llm_respond_results/subset`.

### Level 3: Apply
Apply level denotes how well can models answer the input speech-related questions. We already generated question-answer (QA) pairs for each input audio at `QA_data`. For cascaded ASR + LLM, we use `Qwen2-7b-instruct` to answer questions based on ASR results:

```
python answer_qa.py --dataset "medasr" --asr_model "whisper-large-v3" --asr_input "whisper_v3_1best" --answer_model "qwen2-7b" "Qwen/Qwen2-7B-Instruct"
```
This will save the QA results to `QA_results/subset`.


## Evaluation for multimodal LLMs
We show a demo with `Qwen2-audio-instruct`.

### Level 1 and level 3
We use the end-to-end framework for level 1 and level 3, different from cascaded ASR + LLM, 

```
python end2end_asr.py --dataset "medasr" --asr_model "qwen2-audio"
```
This will save both ASR results and QA results

### Level 2
For level 2, the code is same as the cascaded one:
```
python llm_respond.py --dataset "medasr" --asr_model "qwen2-audio" --asr_input "qwen2-audio"
```
## Final SIQ

Final SIQ is computed considering the normalization among levels and among models (e.g., the difficulties of each example will be dicided by the model performance)

To compute the final SIQ, we first preprocess all-level results by:
```
python score_stat.py --dataset "medasr" --asr_model "whisper-large-v3" --asr_input "whisper_v3_1best" --answer_model "qwen2-7b" "Qwen/Qwen2-7B-Instruct"
```
Here is the demo for `whisper-large-v3`. After we preprocess all models' results, we can run:
```
python compute_IQ.py
```
to derive the final SIQ.

