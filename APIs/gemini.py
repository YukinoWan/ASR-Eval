import google.generativeai as genai


def generate_gemini_response(model, input_text):
    GEMINI_API_KEY = ""
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(model)
    response = model.generate_content(
        input_text,
        generation_config = genai.GenerationConfig(
        max_output_tokens=1000,
        temperature=0.1,
        )
    )
    return response.text

def generate_gemini_audio_response(model, audio_path, input_text):
    GEMINI_API_KEY = ""
    genai.configure(api_key=GEMINI_API_KEY)
    myfile = genai.upload_file(audio_path)
    model = genai.GenerativeModel(model)
    result = model.generate_content([myfile, input_text])
    print(f"{result.text=}")

    return result.text




if __name__ == "__main__":

    # print(generate_gemini_response("gemini-2.0-flash-exp", "请给我讲3个笑话"))
    generate_gemini_audio_response("gemini-1.5-flash", "/mnt/home/zhenwan.nlp/ASR-Eval/canary_infer/voxpopuli/sample_0.wav", "The transcription of the audio is")