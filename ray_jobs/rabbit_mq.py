import pika
import atexit

# Assuming that you have a Rabbit MQ installation and default UID/PWD 
# pip install pika 

RABBIT_USER = "guest"
RABBIT_PWD = "guest"
RABBIT_IP = "127.0.0.1"
RABBIT_PORT = 5672

class RabbitMQ_Client:
    def __init__(self, queue_name:str):
        self.queue_name = queue_name
        self.credentials = pika.PlainCredentials(RABBIT_USER,RABBIT_PWD)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBIT_IP,
                                                                            port=RABBIT_PORT,
                                                                            virtual_host="/",
                                                                            credentials=self.credentials))
        self.channel = self.connection.channel()
        self.queue = self.channel.queue_declare(queue=self.queue_name,durable=True)

        atexit.register(self.cleanup)


    def produce(self, messages:list[str]):
        for message in messages:
            self.channel.basic_publish(exchange='', 
                                       routing_key=self.queue_name, 
                                       body=message,
                                       properties=pika.BasicProperties(delivery_mode=2))

    def consume_fixed(self, num_messages:int):
        messages = []
        for _ in range(num_messages):
            method_frame, header_frame, body = self.channel.basic_get(queue=self.queue_name)
            if method_frame:
                messages.append(body)
                self.channel.basic_ack(method_frame.delivery_tag)
        return messages

    def cleanup(self):
        if self.connection != None and self.connection.is_open:
            print('-> closing connection ->')
            self.connection.close()

#mq_client = RabbitMQ_Client("test_queue")
#mq_client.produce(["Hare Krisha", "Hare Rama"])
#print(mq_client.consume_fixed(num_messages=2))

