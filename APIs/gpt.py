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



