from transformers import AutoTokenizer, AutoModelForCausalLM
from rabbit_mq import RabbitMQ_Client
import torch
import ray


# To install ray, use pip install "ray[default]" 
# You MUST install with Default otherwise CLI is NOT available to submit jobs
# To start the server use the following 
# ray start --head --port=6379 --dashboard-host=0.0.0.0 
# To submit this JOB ... 
# ray job submit --address=127.0.0.1:6379 --submission-id llama-batch5 --working-dir <DIR>\ray_jobs -- python consume_prompts.py

ray.init(address="auto")

@ray.remote
def process_messages_from_queue(num_messages:int = 100):
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="auto")

    rabbit_client_consume = RabbitMQ_Client(queue_name="llama-queue")
    rabbit_client_result = RabbitMQ_Client(queue_name="llama-queue_result")
    messages_list = rabbit_client_consume.consume_fixed(num_messages=num_messages)

    decoded_messages = [m.decode() for m in messages_list] 
    
    print(decoded_messages)
    if decoded_messages != None:
        prompt_tokens = tokenizer(decoded_messages,return_tensors="pt").to("cuda")
        outputs = model.generate(**prompt_tokens, max_length=256, do_sample=True, temperature=0.1, top_k=50, top_p=0.95)
        print('Model Outputs ->>> ', outputs)
        for output in outputs:
            decoded_output = tokenizer.decode(output, skip_special_tokens=True)
            if (decoded_output != None):
                print('Decoded Out ->>>> ', decoded_output)
                rabbit_client_result.produce(decoded_output)
    return True
# process_messages_from_queue(2)

if __name__ == "__main__":
    future = process_messages_from_queue.options(num_gpus=1,num_cpus=1).remote(96)
    print(" Remote Return -> ", ray.get(future))
    ray.shutdown()
