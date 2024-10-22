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
    print("NATS connecting...")
    nats_server = os.getenv('NATS_SERVER', 'localhost')
    nats_port = os.getenv('NATS_PORT', '4222')
    ns = await nats.connect(f"nats://{nats_server}:{nats_port}")
    print("NATS connected")
    return ns


# # Unidepth로 Camera frame의 이차원 좌표에 따른 Depth 추정
# def pred_depth_frame(frame):
#     temp_filename = 'temp_frame.jpg'
#     cv2.imwrite(temp_filename, frame)

#     rgb = np.array(Image.open(temp_filename))

#     rgb_torch = torch.from_numpy(rgb).permute(
#         2, 0, 1).unsqueeze(0).float() / 255.0
#     intrinsics_torch = torch.from_numpy(np.load("assets/demo/intrinsics.npy"))

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     rgb_torch = rgb_torch.to(device)
#     intrinsics_torch = intrinsics_torch.to(device)
#     # predict
#     predictions = model_uni.infer(rgb_torch, intrinsics_torch)

#     # get GT and pred
#     depth_pred = predictions["depth"].squeeze().cpu().numpy()

#     return depth_pred

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
    with torch.no_grad():  # 추가
        predictions = model_uni.infer(rgb_torch, intrinsics_torch)

    # tensor를 numpy로 변환하고 device에서 분리
    depth_pred = predictions["depth"].squeeze().cpu().numpy()

    # numpy array가 맞는지 확인
    assert isinstance(
        depth_pred, np.ndarray), f"depth_pred is not numpy array: {type(depth_pred)}"

    return depth_pred


def print_type_info(var_name, var):
    print(f"{var_name} type: {type(var)}")
    if torch.is_tensor(var):
        print(f"{var_name} device: {var.device}")
        print(f"{var_name} requires_grad: {var.requires_grad}")

# 카메라 스트림 캡처 및 NATS 전송


async def capture_and_send_video(tag_id, subject, uwb, direction):

    nc = await connect_to_nats()

    cap = cv2.VideoCapture(0)

    # 카메라 해상도 640 x 360
    # 1920 x 1080, 1280 x 720, 960 x 540도 가능함
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    uwb_x = uwb[0]
    uwb_y = uwb[1]

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
        # uwb = str(requests.get('http://210.125.85.31:31008/uwb/' +
        #           str(tag_id)).text)[1:-1].split("_")

        results = model(frame)  # YOLO object detection
        if len(results.xyxy[0]) > 0:  # 사람이 감지된 경우
            # detected = False # 필요 없어짐
            centers = []
            # depths = []

            # 모든 Yolo 결과 중 'person' 클래스만 필터링하고 신뢰도 높은 순으로 정렬
            person_detections = [
                result for result in results.xyxy[0] if int(result[5]) == 0]
            person_detections = sorted(person_detections, key=lambda x: x[4], reverse=True)[
                :3]  # 신뢰도 높은 순으로 상위 3개 선택

            # 선택된 'Person' 객체들에 대해 처리
            if len(person_detections) > 0:

                # Camera frame에 대한 Depth 추정 (UniDepth)
                # 'Person' 객체가 0개면 굳이 Depth 추정을 할 이유가 없어서 현재 indent level로 코드 옮김
                depth_pred = pred_depth_frame(frame)

                for result in person_detections:

                    # 바운딩 박스 중심 좌표 계산
                    xmin, ymin, xmax, ymax = result[0], result[1], result[2], result[3]
                    cx = (xmin + xmax) / 2
                    cy = (ymin + ymax) / 2

                    # Yolo frame에 대한 depth 가져오기
                    depth = float(
                        depth_pred[int(cy.item()), int(cx.item())] - 1)
                    x_pos = 0
                    y_pos = 0

                    # 좌표 간단 보정
                    # 해상도 20 = 좌표 0.125라고 가정
                    if direction == 'x+':
                        x_pos = float(uwb_x + depth)
                        y_pos = float(uwb_y)
                        if cx.item() < 320:
                            gap = 320 - cx.item()
                            adjust = (gap / 20) * 0.125
                            y_pos -= adjust
                        else:
                            gap = cx.item() - 320
                            adjust = (gap / 20) * 0.125
                            y_pos += adjust
                    elif direction == 'x-':
                        x_pos = float(uwb_x - depth)
                        y_pos = float(uwb_y)
                        if cx.item() < 320:
                            gap = 320 - cx.item()
                            adjust = (gap / 20) * 0.125
                            y_pos += adjust
                        else:
                            gap = cx.item() - 320
                            adjust = (gap / 20) * 0.125
                            y_pos -= adjust
                    elif direction == 'y+':
                        y_pos = float(uwb_y + depth)
                        x_pos = float(uwb_x)
                        if cx.item() < 320:
                            gap = 320 - cx.item()
                            adjust = (gap / 20) * 0.125
                            x_pos += adjust
                        else:
                            gap = cx.item() - 320
                            adjust = (gap / 20) * 0.125
                            x_pos -= adjust
                    else:  # y-
                        y_pos = float(uwb_y - depth)
                        x_pos = float(uwb_x)
                        if cx.item() < 320:
                            gap = 320 - cx.item()
                            adjust = (gap / 20) * 0.125
                            x_pos -= adjust
                        else:
                            gap = cx.item() - 320
                            adjust = (gap / 20) * 0.125
                            x_pos += adjust

                    x_pos = float(
                        x_pos.item() if torch.is_tensor(x_pos) else x_pos)
                    y_pos = float(
                        y_pos.item() if torch.is_tensor(y_pos) else y_pos)

                    # JPEG로 인코딩하여 손실 압축 적용 (품질 80%)
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                    _, buffer = cv2.imencode('.jpg', frame, encode_param)

                    print(f"x_pos type: {type(x_pos)}")
                    print(f"y_pos type: {type(y_pos)}")

                    if torch.is_tensor(x_pos):
                        x_pos = float(x_pos.item())
                    if torch.is_tensor(y_pos):
                        y_pos = float(y_pos.item())

                    # 파일명 생성
                    filename = f"frame_{int(time.time())}.jpg"
                    # JSON 데이터 생성
                    data = {
                        "filename": filename,
                        "image": base64.b64encode(buffer).decode('utf-8'),
                        "position_x": x_pos,
                        "position_y": y_pos,
                        "time": time_cap
                    }

                    data_print = {
                        "filename": filename,
                        "position_x": x_pos,
                        "position_y": y_pos,
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

    parser.add_argument('--xpos', type=float, required=True,
                        help='to the uwb x position')
    parser.add_argument('--ypos', type=float, required=True,
                        help='to the uwb y position')

    parser.add_argument('--direction', type=str, required=True,
                        help='to the station direction, x+, x-, y+, y-')
    args = parser.parse_args()
    asyncio.run(capture_and_send_video(
        args.tag_id, args.subject, [args.xpos, args.ypos], args.direction))
