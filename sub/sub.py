import os
from kafka import KafkaProducer
import nats
import json
import numpy as np


async def nats_connect():
    nats_server = os.getenv('NATS_SERVER', '10.48.24.73')
    nats_port = os.getenv('NATS_PORT', '30022')
    return await nats.connect(f"nats://{nats_server}:{nats_port}")


async def kafka_producer():
    kafka_broker = os.getenv('KAFKA_BROKER', '210.125.85.62')
    broker_port = os.getenv('BROKER_PORT', '9094')
    return KafkaProducer(bootstrap_servers=f"{kafka_broker}:{broker_port}",  value_serializer=lambda v: json.dumps(v).encode('utf-8'))


async def main():
    # Nats 연결
    try:
        nc = await nats_connect()
        print("Nats connected")
        sub = await nc.subscribe("Falcon.ternal.Group.A")
        print("Nats subcribed")
    except Exception as e:
        print(f"Failed to connect to NATS server: {e}")
        return

    # kafka 연결
    try:
        producer = await kafka_producer()
        print("Kafka broker connected")
    except Exception as e:
        print(f"Failed to connect to Kafka broker: {e}")
        return

    # coordinates = []
    topic = "falcon-xy"  # kafka topic

    while True:
        try:
            msg = await sub.next_msg()
            print("received: ")
            json_obj = json.loads(msg.data.decode())  # msg.data is in bytes
            # Ensure coordinates are converted to float

            # demo
            detected_objects = json_obj['objects']
            kafka_msg_data = []

            for object in detected_objects:

                kafka_msg_data.append({
                    "x": object['position_x'],
                    "y": object['position_y'],
                })

            producer.send(topic, value=kafka_msg_data)
            producer.flush()

            # position_x = float(json_obj['position_x'])
            # position_y = float(json_obj['position_y'])
            # coordinates.append([position_x, position_y])
            # if len(result) >= 10:
            #     print("Grouped and averaged coordinates:", coordinates)

            #     # kafka-broker로 좌표 데이터 전송
            #     for coord in coordinates:
            #         coord_json = {"x": coord[0], "y": coord[1]}

            #         producer.send(topic, value=coord_json)
            #         print(f"sent: {coord_json}")
            #     producer.flush()

            #     result = []  # refresh
        except:
            pass

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
