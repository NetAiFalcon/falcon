# 1. 사진 찍기
# 2. YOLO 사람 인식 + 바운딩 박스의 중간 좌표 찾기
# 3. Depth Estimation으로 해당 좌표 길이 추산
# 4. UWB 좌표 가지고 특정하기

import asyncio
import cv2
import nats
import torch
import numpy as np
import json
import base64
import time
import subprocess
from gi.repository import Gst, GLib

# GStreamer 초기화
Gst.init(None)

# YOLO 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# NATS 서버에 연결
async def connect_to_nats():
    return await nats.connect("nats://210.125.85.31:31773")

# GStreamer를 사용하여 카메라 스트림 캡처 및 NATS 전송
async def capture_and_send_video():
    nc = await connect_to_nats()

    # GStreamer 파이프라인 구성
    pipeline = Gst.parse_launch(
        "v4l2src ! videoconvert ! videoscale ! video/x-raw,width=1280,height=720,format=BGR ! appsink name=sink"
    )
    
    sink = pipeline.get_by_name("sink")
    sink.set_property("emit-signals", True)
    sink.connect("new-sample", on_new_sample, nc)

    # 파이프라인 시작
    pipeline.set_state(Gst.State.PLAYING)
    
    # GStreamer 메인 루프 실행
    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)
        await nc.close()

def on_new_sample(sink, nc):
    sample = sink.emit("pull-sample")
    buf = sample.get_buffer()
    result, mapinfo = buf.map(Gst.MapFlags.READ)
    
    if result:
        frame_data = np.frombuffer(mapinfo.data, np.uint8)
        frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
        
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
                    
                    # unidepth.py 스크립트를 호출하여 깊이 값 계산
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
                
                asyncio.run_coroutine_threadsafe(nc.publish("Falcon.ternal.Group.A", message.encode('utf-8')), asyncio.get_event_loop())
    
    buf.unmap(mapinfo)
    return Gst.FlowReturn.OK

def get_depth_value(frame, cx, cy):
    # 임시 파일에 프레임을 저장
    temp_filename = 'temp_frame.jpg'
    cv2.imwrite(temp_filename, frame)
    
    # unidepth.py 스크립트를 호출
    result = subprocess.run(['python3', 'unidepth.py', '--image', temp_filename, '--x', str(cx), '--y', str(cy)], capture_output=True, text=True)
    
    # 출력된 깊이 값을 파싱
    depth = float(result.stdout.strip())
    return depth

# 메인 함수
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(capture_and_send_video())
