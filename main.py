import openai
import csv
import json
import os
import numpy as np
from collections import defaultdict
import tiktoken
import gradio as gr
from openai import OpenAI

client = OpenAI(api_key='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')

training_response = client.files.create(
    file=open("checkpoint2_f24.jsonl", "rb"), purpose="fine-tune"
)
print(type(training_response))
print(training_response)

training_file_id = training_response.id

#Gives training file id
print("Training file id:", training_file_id)
suffix_name = "Webtech-bot"

response = client.fine_tuning.jobs.create(
    training_file=training_file_id,
    model="ft:gpt-3.5-turbo-0613:personal::8HF18kYo",
    suffix=suffix_name,
)
job_id = response.id


#Getting model response
def get_model_response(user_input):
    try:
         conversation_history = [
            {"role": "system", "content": "Act as expert in webt technologies, Now your goal is to provide feedback to students in clear, understanding and encouraging way"},
            {"role": "user", "content": user_input}
        ]

         response = client.chat.completions.create(
            model='ft:gpt-3.5-turbo-0613:personal::8HF18kYo',  
            messages=conversation_history,
            max_tokens=1000
        )
         messages= response.choices[0].message.content
         print(messages)
         return messages
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"Sorry, I couldn't process your request. Error: {e}"
    
# displaying by using gradio Interface
def chat_interface(user_question):
 model_response = get_model_response(user_question)
 return model_response

iface = gr.Interface(
    fn=chat_interface,
    inputs="text",
    outputs= "text"
)
iface.launch()














































# import requests

# def chat_with_model(api_key, model):
#     url = "https://api.openai.com/v1/chat/completions"
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {api_key}"
#     }
#     messages = [{"role": "system", "content": "Act as expert in Webtechnology subject with great knowledge on HTML/CSS javascript,PHP and all technologies related to webtechnologies ."}]

#     while True:
#         user_input = input("You: ")
#         if user_input.lower() in ['exit', 'quit']:
#             break

#         messages.append({"role": "user", "content": user_input})

#         data = {
#             "model": model,
#             "messages": messages
#         }

#         response = requests.post(url, headers=headers, json=data)
#         model_response = response.json()['choices'][0]['message']['content']
#         print("Assistant:", model_response)

#         messages.append({"role": "assistant", "content": model_response})

# # Replace 'your_api_key' and 'your_model_name' with your actual API key and model name
# chat_with_model("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", "ft:gpt-3.5-turbo-0613:personal::7xPsahmg")


