The folder rayjobs contains code that uses ray for distributed execution of MQ Consumer. You need to install RabbitMQ to publish/consume messages
We need to install torch, transformers to execute the consumer code. 
The following description is from consumer. 
To install ray, use pip install "ray[default]" 
You MUST install with Default otherwise CLI is NOT available to submit jobs
To start the server use the following 
ray start --head --port=6379 --dashboard-host=0.0.0.0 
To submit this JOB ... 
# ray job submit --address=127.0.0.1:6379 --submission-id llama-batch5 --working-dir <DIR>\ray_jobs -- python consume_prompts.py
