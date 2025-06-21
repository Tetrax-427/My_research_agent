import os
import openai
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
def get_response_groq_text(system_prompt, user_query, model = "llama-3.3-70b-versatile", temp = 0, top_p = 1, max_new_tokens = 1024):
    client = Groq()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_query
            }
        ],
        temperature= temp,
        max_completion_tokens=max_new_tokens,
        top_p=top_p,
        stream=False,
        stop=None,
    )
    return completion.choices[0].message.content

def get_response_open_ai(system_prompt, user_query,model = "gpt-4o-mini", temperature = 0, top_p = 1, max_new_tokens = 1024):
    pass

def get_response(system_prompt, user_query, model = "llama-3.3-70b-versatile", temperature = 0, top_p = 1, max_new_tokens = 1024):
    return get_response_groq_text(system_prompt, user_query, model, temperature, top_p, max_new_tokens )

#print(get_response("Reply as a indian person", "What is the name of capital of india"))