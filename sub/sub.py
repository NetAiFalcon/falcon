from kafka import KafkaProducer
import nats
import json
import numpy as np

async def nats_connect():
    return await nats.connect("nats://210.125.85.31:31773")

async def kafka_producer():
    return KafkaProducer(bootstrap_servers="10.80.0.3:9094",  value_serializer=lambda v: json.dumps(v).encode('utf-8'))

def group_and_average_coordinates(coords):
    n = len(coords)
    visited = [False] * n
    new_coords = []

    def distance(coord1, coord2):
        return np.linalg.norm(np.array(coord1) - np.array(coord2))

    for i in range(n):
        if not visited[i]:
            group = [coords[i]]
            visited[i] = True
            for j in range(i + 1, n):
                if not visited[j] and distance(coords[i], coords[j]) <= 5:
                    group.append(coords[j])
                    visited[j] = True
            if group:
                avg_coord = np.mean(group, axis=0).tolist()
                new_coords.append(avg_coord)

    return new_coords


async def main():
    try:
        nc = await nats_connect()
        print("Nats connected")
        sub = await nc.subscribe("Falcon.ternal.Group.A")
        print("Nats subcribed")
    except Exception as e:
        print(f"Failed to connect to NATS server: {e}")
        return

    try:
        producer = await kafka_producer()
        print("Kafka broker connected")
    except Exception as e:
        print(f"Failed to connect to Kafka broker: {e}")
        return

    result = []
    topic = "falcon-xy"

    while True:
        try:
            msg = await sub.next_msg()
            print("received: ")
            json_obj = json.loads(msg.data.decode())  # msg.data is in bytes
            # Ensure coordinates are converted to float
            position_x = float(json_obj['position_x'])
            position_y = float(json_obj['position_y'])
            result.append([position_x, position_y])
            if len(result) >= 10:
                coordinate = group_and_average_coordinates(result)
                print("Grouped and averaged coordinates:", coordinate)
                
                # kafka-broker로 좌표 데이터 전송
                for coord in coordinate:
                    coord_json = {"x": coord[0], "y": coord[1]}
                    
                    producer.send(topic, value=coord_json)
                    print(f"sent: {coord_json}")
                producer.flush()

                result = [] # refresh
        except:
            pass

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
