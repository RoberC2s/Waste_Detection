import paho.mqtt.client as mqtt
import json

# Define the MQTT broker details
broker = 'broker.hivemq.com'
port = 1883
topic = "robotic_arm/object_detection"

# Callback function on connect
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
        client.subscribe(topic)
    else:
        print(f"Failed to connect, return code {rc}")

# Callback function on message
def on_message(client, userdata, msg):
    message = msg.payload.decode()
    object_data = json.loads(message)
    print(f"Received message: {object_data}")
    # Here you would add logic to control the robotic arm based on the coordinates

# Create an MQTT client instance
client = mqtt.Client()

# Attach the on_connect and on_message callback functions
client.on_connect = on_connect
client.on_message = on_message

# Connect to the MQTT broker
client.connect(broker, port, keepalive=60)

# Blocking call that processes network traffic, dispatches callbacks, and handles reconnecting.
client.loop_forever()
