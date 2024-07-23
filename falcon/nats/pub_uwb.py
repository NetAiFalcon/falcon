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

import requests


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
        
        uwb = requests.get('http://127.0.0.1:8000/items/info')

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

                    print(uwb.text) # uwb text

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

        # time check
        # end_point_capture_done = time.time()
        # print(f"Funtion run time {end_point_capture_done - end_point_capture_st:.5f} sec")

    cap.release()
    await nc.close()

# 메인 함수
if __name__ == "__main__":
    asyncio.run(capture_and_send_video())
