from openai import OpenAI
def generate_gpt_response(model, input_text):
    client = OpenAI(api_key="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")
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": input_text
            }
        ]
    )
    return completion.choices[0].message.content



