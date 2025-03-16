from openai import AzureOpenAI
 
client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://llm-proxy.perflab.nvidia.com",
    api_key="eyJhbGciOiJIUzI1NiJ9.eyJpZCI6IjQzMTZmNzdiLTI0M2EtNGRhYi04NDdiLWY5NDNjMzRhNDkyNyIsInNlY3JldCI6IkZqeWpPdEd3azZKakdaRnlyYWZpY3RTcUQwNUt1cGRNTUtNZXpMTUw2Qkk9In0.XfLTsQF_byb286BMY7gYLyn5nMR_ySDPcExla5JGazA",
)

t_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me a joke."},
    {"role": "assistant", "content": "Why don't skeletons fight each other? They don't have the guts."}
]

response = client.chat.completions.create(
    model="gpt-4o-20241120", # model = "deployment_name".
    messages=t_messages
)

print("output", response.choices[0].message.content)
