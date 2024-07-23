import asyncio
import cv2
import nats
import torch
import numpy as np
import json
import base64
import time
import subprocess

# time check
import math
import time

# uwb
import psycopg2
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
# uwb 경로 추가
uwb_dir = os.path.abspath(os.path.join(current_dir, '../uwb'))
sys.path.append(uwb_dir)
from Websocket_raw import SewioWebSocketClient_v2
from kafka import KafkaProducer

# unidepth
from PIL import Image
import argparse

from unidepth.models import UniDepthV1, UniDepthV2
from unidepth.utils import colorize, image_grid



start = time.time()
math.factorial(100000)

# YOLO 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# time check
end_point_yolo = time.time()
print(f"YOLO model load time {end_point_yolo - start:.5f} sec")

# UniDepthV2 모델 로드
name_uni = "unidepth-v2-vitl14"
model_uni = UniDepthV2.from_pretrained(f"lpiccinelli/{name_uni}")
# time check
end_point_unidepth = time.time()
print(f"depth model load time {end_point_unidepth - end_point_yolo:.5f} sec")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_uni = model_uni.to(device)

# NATS 서버에 연결
async def connect_to_nats():
    return await nats.connect("nats://210.125.85.31:31773")

# Depth 값을 가져오는 함수
def get_depth_value(frame, cx, cy):
    # 임시 파일에 프레임을 저장
    temp_filename = 'temp_frame.jpg'
    cv2.imwrite(temp_filename, frame)
    
    rgb = np.array(Image.open(temp_filename))

    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    intrinsics_torch = torch.from_numpy(np.load("/home/user/falcon/UniDepth/assets/demo/intrinsics.npy"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_torch = rgb_torch.to(device)
    intrinsics_torch = intrinsics_torch.to(device)
    # predict
    predictions = model_uni.infer(rgb_torch, intrinsics_torch)

    # get GT and pred
    depth_pred = predictions["depth"].squeeze().cpu().numpy()

    # 특정 좌표(x, y)의 깊이 출력
    depth_at_point = depth_pred[cy, cx] - 1
    print(f"Depth at ({cx}, {cy}): {depth_at_point:.4f} meters")
    return float(depth_at_point)

# 카메라 스트림 캡처 및 NATS 전송
async def capture_and_send_video():
    nc = await connect_to_nats()
    cap = cv2.VideoCapture(0)
    
    while True:
        
        # end_point_capture_st = time.time() time check

        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO 감지
        results = model(frame)
        if len(results.xyxy[0]) > 0:  # 사람이 감지된 경우
            detected = False
            centers = []
            depths = []
            for result in results.xyxy[0]:
                if int(result[5]) == 0:  # 0번 클래스가 'person'인 경우
                    detected = True
                    # 바운딩 박스 중심 좌표 계산
                    xmin, ymin, xmax, ymax = result[0], result[1], result[2], result[3]
                    cx = (xmin + xmax) / 2
                    cy = (ymin + ymax) / 2
                    centers.append((cx.item(), cy.item()))

                    # 깊이 값 계산  
                    depth = get_depth_value(frame, int(cx.item()), int(cy.item()))
                    depths.append(depth)
            
            if detected:
                # JPEG로 인코딩하여 손실 압축 적용 (품질 80%)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                _, buffer = cv2.imencode('.jpg', frame, encode_param)
                
                # 파일명 생성
                filename = f"frame_{int(time.time())}.jpg"
                
                # JSON 데이터 생성
                data = {
                    "filename": filename,
                    "image": base64.b64encode(buffer).decode('utf-8'),
                    "centers": centers,
                    "depths": depths
                }
                message = json.dumps(data)
                
                await nc.publish("Falcon.ternal.Group.A", message.encode('utf-8'))
        
        await asyncio.sleep(0.1)  # 100ms 딜레이

    cap.release()
    await nc.close()








"""

MVC 신경 안쓰고 오르지 Sewio RLTS 에서 데이터 넘어오면 DB만 적재하도록 작성된 코드

"""
class DataManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.load_config()

        self.producer = None
        self.topic_name = None
        self.db_connect()
        self.kafka_connect()



    def load_config(self):
        with open(self.config_path, 'r') as file:
            self.config = json.load(file)

    def db_connect(self):
        try:
            self.conn = psycopg2.connect(
                dbname=self.config['db_name'],
                user=self.config['db_user'],
                password=self.config['db_password'],
                host=self.config['db_host'],
                port=self.config['db_port']
            )
            self.cursor = self.conn.cursor()
            print("Database connection successfully established.")
        except Exception as e:
            print(f"Failed to connect to the database: {e}")

    def kafka_connect(self):
        try:
            self.producer = KafkaProducer(bootstrap_servers=self.config['kafka_server'],
                                          value_serializer=lambda v: json.dumps(v).encode('utf-8'))
            self.topic_name = self.config['topic_name']

            print("Kafka connection successfully established.")
        except Exception as e:
            print(f"Failed to connect to Kafka: {e}")

    def send_data_to_kafka(self, tag_id, posX, posY):
        # 데이터를 JSON 문자열로 변환합니다.
        coord_data = {'id': tag_id, 'latitude': posX, 'longitude': posY}
        self.producer.send(self.topic_name, coord_data)
        self.producer.flush()

    def close_producer(self):
        if self.producer is not None:
            self.producer.close()

    def store_data_in_db(self, tag_id, posX, posY, timestamp, anchor_info):
        query = """
        INSERT INTO uwb_raw (tag_id, x_position, y_position, timestamp, anchor_info) VALUES (%s, %s, %s, %s, %s)
        """
        self.cursor.execute(query, (tag_id, posX, posY, timestamp, anchor_info))
        self.conn.commit()

    def handle_data(self, tag_id, posX, posY, timestamp, anchor_info):
        #print(f"Data received: Tag ID={tag_id}, Position X={posX}, Position Y={posY}, Timestamp={timestamp}")
        self.store_data_in_db(tag_id, posX, posY, timestamp, anchor_info)
        self.send_data_to_kafka(tag_id, posX, posY)


def uwb():
    url = "ws://10.76.20.88/sensmapserver/api"
    config_path = os.getenv('CONFIG_PATH', '/home/user/falcon/falcon/uwb/config.json')

    manager = DataManager(config_path)

    client = SewioWebSocketClient_v2(url, data_callback=manager.handle_data)
    try:
        client.run_forever()
    finally:
        manager.close_producer()






















# 메인 함수
if __name__ == "__main__":
    uwb()
    # asyncio.run(capture_and_send_video())
