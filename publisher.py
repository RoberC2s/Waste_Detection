import paho.mqtt.client as mqtt
import json
import time
import csv

class Publisher:
    def __init__(self, broker, port, topic):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client = mqtt.Client()
        self.client.connect(self.broker, self.port, keepalive=60)
        
        # Open the CSV file and write the header if it doesn't exist
        self.csv_file = open('results_to_compare.csv', 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        if self.csv_file.tell() == 0:
            self.csv_writer.writerow(['timestamp', 'object type', 'object ID', 'confidence', 'x', 'y', 'z', 'spd_X', 'spd_Y', 'spd_Z'])

    def publish(self, objects):
        for obj in objects:
            message = json.dumps(obj)
            self.client.publish(self.topic, message)
            print(f"Published message: {message}")
            self.save_to_csv(obj)

    def save_to_csv(self, obj):
        print(obj)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S.%s.%2f')[:-3]
        timestamp = obj.get('idf_time', 'unknown')
        object_class = obj.get('class_id', 'unknown')
        object_ID = obj.get('tracker_id', 'unknown')
        speed_X = obj.get('speed_x', 'unknown')
        speed_y = obj.get('speed_y', 'unknown')
        speed_z = obj.get('speed_z', 'unknown')
        
        confidence = obj.get('confidence', {})
        x = obj.get('x', 'unknown')
        y = obj.get('y', 'unknown')
        z = obj.get('z', 'unknown')
        self.csv_writer.writerow([timestamp, object_class, object_ID, confidence, x, y, z, speed_X, speed_y, speed_z])

    def close_csv(self):
        self.csv_file.close()

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

    try:
        while True:
            object_data = get_object_data()
            pub.publish(object_data)
            time.sleep(2)  # Publish every 2 seconds for demonstration purposes
    except KeyboardInterrupt:
        pass
    finally:
        # Disconnect from the broker and close the CSV file when done
        pub.client.disconnect()
        pub.close_csv()
