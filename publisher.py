import paho.mqtt.client as mqtt
import json
import time

class Publisher:
    def __init__(self, broker, port, topic):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client = mqtt.Client()
        self.client.connect(self.broker, self.port, keepalive=60)

    def publish(self, objects):
        for obj in objects:
            message = json.dumps(obj)
            self.client.publish(self.topic, message)
            print(f"Published message: {message}")

# Sample object detection data
def get_object_data():
    # In a real application, replace this with actual detection logic
    return [
        {
            "object": "example_object_1",
            "coordinates": {
                "x": 100,
                "y": 200,
                "z": 300
            }
        },
        {
            "object": "example_object_2",
            "coordinates": {
                "x": 400,
                "y": 500,
                "z": 600
            }
        }
    ]

if __name__ == "__main__":
    broker = 'broker.hivemq.com'  # Public broker for demonstration
    port = 1883
    topic = "robotic_arm/object_detection"

    pub = Publisher(broker, port, topic)

    while True:
        object_data = get_object_data()
        pub.publish(object_data)
        time.sleep(2)  # Publish every 2 seconds for demonstration purposes

    # Disconnect from the broker when done
    pub.client.disconnect()
