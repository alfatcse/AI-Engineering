from dotenv import load_dotenv
from openai import OpenAI
import ollama
from importlib.metadata import version
print("ollama version:", version("ollama"))
MODEL_GPT = 'phi3'
MODEL_LLAMA = 'llama3.2'
load_dotenv()
# Route the OpenAI client at the local Ollama server (OpenAI-compatible API).
# No OpenAI key or credits needed; api_key is required but ignored by Ollama.
openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
# here is the question; type over this to ask something new

question = """
Please explain what this code does and why:
what is the difference between a list and a tuple in Python, and when should I use one over the other?
"""
# prompts

system_prompt = "You are a helpful technical tutor who answers questions about python code, software engineering, data science and LLMs"
user_prompt = "Please give a detailed explanation to the following question: " + question
# messages

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]
# Get gpt-4o-mini to answer, with streaming

stream = openai.chat.completions.create(model=MODEL_GPT, messages=messages,stream=True)

print(f"\n===== {MODEL_GPT} (via Ollama) =====\n")
response = ""
for chunk in stream:
    token = chunk.choices[0].delta.content or ''
    response += token
    print(token, end='', flush=True)   # stream tokens to the terminal as they arrive
print()
# Get Llama 3.2 to answer

print(f"\n===== {MODEL_LLAMA} =====\n")
response = ollama.chat(model=MODEL_LLAMA, messages=messages)
reply = response['message']['content']
print(reply)
