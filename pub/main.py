import os
import datetime
import asyncio
import json
import base64
import requests
import time
import math
import argparse

import cv2
import nats
import torch
import numpy as np
from PIL import Image

from unidepth.models import UniDepthV2

#############################################
######### Set Up for Yolo, Unidepth #########
#############################################

start_point_yolo = time.time()
math.factorial(100000)

# YOLO 모델 로드 from Pytorch Hub
# YOLO는 따로 GPU 연산 안함: v5는 경량화 모델, uniDepth 연산에 영향 고려
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Time check
end_point_yolo = time.time()
print(f"YOLO model load time: {end_point_yolo - start_point_yolo:.5f} sec")

# UniDepthV2 모델 로드
name_uni = "unidepth-v2-vitl14"
model_uni = UniDepthV2.from_pretrained(f"lpiccinelli/{name_uni}")

# Time check
end_point_unidepth = time.time()
print(f"depth model load time: {end_point_unidepth - end_point_yolo:.5f} sec")

# CUDA 연결 for Unidepth Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_uni = model_uni.to(device)


#############################################
################ Main Logics ################
#############################################


# NATS 서버에 연결 (환경 변수 사용)
async def connect_to_nats():
    nats_server = os.getenv('NATS_SERVER', 'localhost')
    nats_port = os.getenv('NATS_PORT', '4222')
    return await nats.connect(f"nats://{nats_server}:{nats_port}")


# Unidepth로 Camera frame의 이차원 좌표에 따른 Depth 추정
def pred_depth_frame(frame):
    temp_filename = 'temp_frame.jpg'
    cv2.imwrite(temp_filename, frame)

    rgb = np.array(Image.open(temp_filename))

    rgb_torch = torch.from_numpy(rgb).permute(
        2, 0, 1).unsqueeze(0).float() / 255.0
    intrinsics_torch = torch.from_numpy(np.load("assets/demo/intrinsics.npy"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_torch = rgb_torch.to(device)
    intrinsics_torch = intrinsics_torch.to(device)
    # predict
    predictions = model_uni.infer(rgb_torch, intrinsics_torch)

    # get GT and pred
    depth_pred = predictions["depth"].squeeze().cpu().numpy()

    return depth_pred


# 카메라 스트림 캡처 및 NATS 전송
async def capture_and_send_video(tag_id, subject):
    nc = await connect_to_nats()
    cap = cv2.VideoCapture(0)

    while True:
        capture_start = time.time()

        # ret: boolean
        # frame: current frame (image)
        ret, frame = cap.read()
        if not ret:
            break

        # current frame이 찍힌 시간
        time_cap = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        # tag_id의 uwb 센서의 현재 좌표
        # TODO: uwb 정보 가져오기 위한 POD 재배포하기
        uwb = str(requests.get('http://210.125.85.31:31008/uwb/' +
                  str(tag_id)).text)[1:-1].split("_")

        results = model(frame)  # YOLO object detection
        if len(results.xyxy[0]) > 0:  # 사람이 감지된 경우
            detected = False
            centers = []
            depths = []
            x_position = uwb[1]

            # Camera frame에 대한 Depth 추정 (UniDepth)
            depth_pred = pred_depth_frame(frame)

            # 모든 Yolo frames에 대해
            for result in results.xyxy[0]:
                # 0번 클래스가 'person'인 경우
                if int(result[5]) == 0:
                    detected = True
                    # 바운딩 박스 중심 좌표 계산
                    xmin, ymin, xmax, ymax = result[0], result[1], result[2], result[3]
                    cx = (xmin + xmax) / 2
                    cy = (ymin + ymax) / 2
                    centers.append((cx.item(), cy.item()))

                    # Yolo frame에 대한 depth 가져오기
                    depth = float(
                        depth_pred[int(cx.item()), int(cy.item())] - 1)
                    depths.append(depth)

            # 'Person' frame이 적어도 1개 이상 있었을 경우
            if detected:

                # 간단 보정
                # TODO: Depth의 방향 추정 제대로 구현할 것
                if centers[0][0] < 170:
                    x_position = float(x_position) - 1
                elif centers[0][0] > 470:
                    x_position = float(x_position) + 1

                # JPEG로 인코딩하여 손실 압축 적용 (품질 80%)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                _, buffer = cv2.imencode('.jpg', frame, encode_param)

                # 파일명 생성
                filename = f"frame_{int(time.time())}.jpg"
                # JSON 데이터 생성
                data = {
                    "filename": filename,
                    "image": base64.b64encode(buffer).decode('utf-8'),
                    "tag_id": uwb[4],  # position_17.78_-21.76_tagid_15
                    "position_x": x_position,
                    "position_y": float(uwb[2]) + float(depths[0]),
                    "time": time_cap
                }

                data_print = {
                    "filename": filename,
                    "image": "r4lwEXtvKB95SM1w5Ug9K9I+I8",
                    "tag_id": uwb[4],  # position_17.78_-21.76_tagid_15
                    "position_x": x_position,
                    "position_y": float(uwb[2]) + float(depths[0]),
                    "time": time_cap
                }
                print(data_print)
                message = json.dumps(data)

                await nc.publish(subject, message.encode('utf-8'))

        await asyncio.sleep(0.2)  # 200ms 딜레이 안하면 초당 4장, 100ms 딜레이는 초당 2장

        capture_end = time.time()

        print(
            f"Funtion run time {capture_end - capture_start:.5f} sec")

    cap.release()
    await nc.close()

# 메인 함수
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='falcon Demo')
    parser.add_argument('--tag_id', type=int, required=True,
                        help='to the input tag_id')  # tag_id: UWB 센서 ID
    parser.add_argument('--subject', type=str, required=False, default='Falcon.ternal.Group.A',
                        help='to the input tag_id')  # subject: nats subject
    args = parser.parse_args()
    asyncio.run(capture_and_send_video(args.tag_id, args.subject))
