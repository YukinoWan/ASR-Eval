# python llm_respond.py "closed" "canary" "canary_1best"
# python llm_respond.py "closed" "desta2" "desta2"

# python llm_respond.py "closed" "espnet" "espnet_1best"
# python llm_respond.py "closed" "gemini-1.5-flash" "gemini-1.5-flash"
# python llm_respond.py "closed" "gemini-1.5-pro" "gemini-1.5-pro"
# python llm_respond.py --dataset "closed" --asr_model "whisper-large-v2" --asr_input "whisper_v2_1best"
python llm_respond.py --dataset "closed" --asr_model "whisper-large-v3" --asr_input "whisper_v3_1best"
# python llm_respond.py --dataset "closed" --asr_model "whisper_v3_1best" --answer_model "gpt-4o"

# python llm_respond.py "medasr" "baichuan" "baichuan"
# # python llm_respond.py "tedlium" "anygpt" "anygpt"
# python llm_respond.py "voxpopuli" "baichuan" "baichuan"
