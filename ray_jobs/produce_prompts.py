from transformers import AutoTokenizer, AutoModelForCausalLM
from rabbit_mq import RabbitMQ_Client
import torch

# Putting Prompt messages to MQ server. Use the Consumer to process them

torch.cuda.empty_cache()
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="auto")

messages = [
    {"role": "system","content": "You are a friendly chatbot who always responds in the style of a pirate"},
    {"role": "user", "content": "Where is Ganges"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
rabbit_client = RabbitMQ_Client(queue_name="llama-queue")
rabbit_client.produce([prompt]*1000)
