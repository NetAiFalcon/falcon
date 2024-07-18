import asyncio
import cv2
import nats
import torch
import numpy as np
import json
import base64
import time
import subprocess

# YOLO 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# NATS 서버에 연결
async def connect_to_nats():
    return await nats.connect("nats://210.125.85.31:31773")

# Depth 값을 가져오는 함수
def get_depth_value(frame, cx, cy):
    # 임시 파일에 프레임을 저장
    temp_filename = 'temp_frame.jpg'
    cv2.imwrite(temp_filename, frame)
    
    # unidepth.py 스크립트를 호출
    result = subprocess.run(['python3', '../depth.py', '--image', temp_filename, '--x', str(cx), '--y', str(cy)], capture_output=True, text=True)
    
    # 출력된 깊이 값을 파싱
    depth = float(result.stdout.strip())
    return depth

# 카메라 스트림 캡처 및 NATS 전송
async def capture_and_send_video():
    nc = await connect_to_nats()
    cap = cv2.VideoCapture(0)
    
    while True:
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
                    depth = get_depth_value(frame, cx, cy)
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

# 메인 함수
if __name__ == "__main__":
    asyncio.run(capture_and_send_video())
